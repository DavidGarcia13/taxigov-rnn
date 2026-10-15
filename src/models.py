"""
models.py — arquiteturas recorrentes e utilitarios para a previsao de demanda TaxiGOV.

As tres familias (SimpleRNN, GRU, LSTM) sao construidas pela MESMA funcao
`build_model`, variando apenas o tipo de celula. Isso garante o "protocolo
equivalente" exigido: mesma profundidade, largura, dropout, otimizador e saida
para todas, de modo que a comparacao reflita a arquitetura e nao detalhes de
implementacao.

Fornece tambem:
  - carregamento do dataset .npz + parametros do scaler
  - baselines (naive sazonal e media movel) para referencia
  - metricas (MAE, RMSE, R2) calculadas na ESCALA ORIGINAL (corridas/dia)
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)                      # raiz do projeto (um nivel acima de src/)
DATA_PROC = os.path.join(RAIZ, "data", "processed")
NPZ = os.path.join(DATA_PROC, "taxigov-rnn-dataset.npz")
SCALER = os.path.join(DATA_PROC, "scaler_params.json")

ARQUITETURAS = {
    "SimpleRNN": layers.SimpleRNN,
    "GRU": layers.GRU,
    "LSTM": layers.LSTM,
}


# ---------------------------------------------------------------------------
# dados
# ---------------------------------------------------------------------------
def carregar_dados():
    """Retorna dict com X/y de treino/val/teste, feature_cols e scaler."""
    d = np.load(NPZ, allow_pickle=True)
    with open(SCALER, encoding="utf-8") as f:
        scaler = json.load(f)
    return {
        "X_train": d["X_train"], "y_train": d["y_train"],
        "X_val": d["X_val"],     "y_val": d["y_val"],
        "X_test": d["X_test"],   "y_test": d["y_test"],
        "feature_cols": [str(c) for c in d["feature_cols"]],
        "scaler": scaler,
    }


def desnormaliza_alvo(y_norm, scaler):
    """Converte o alvo normalizado de volta para corridas/dia."""
    return np.asarray(y_norm) * scaler["alvo_sd"] + scaler["alvo_mu"]


# ---------------------------------------------------------------------------
# construcao de modelo (protocolo equivalente p/ as 3 familias)
# ---------------------------------------------------------------------------
def build_model(arch, input_shape, n_camadas=1, unidades=64, dropout=0.0,
                recurrent_dropout=0.0, l2=0.0, lr=1e-3, bidirecional=False,
                otimizador="adam", horizonte=1):
    """
    Constroi uma rede recorrente.

    Parametros
    ----------
    arch : str  -> "SimpleRNN" | "GRU" | "LSTM"
    input_shape : (janela, n_features)
    n_camadas : nº de camadas recorrentes empilhadas
    unidades : nº de unidades por camada
    dropout / recurrent_dropout : regularizacao
    l2 : coef. de regularizacao L2 no kernel recorrente
    lr : taxa de aprendizado
    bidirecional : empacota as camadas em Bidirectional
    otimizador : "adam" | "rmsprop"
    horizonte : nº de passos previstos (saida)
    """
    if arch not in ARQUITETURAS:
        raise ValueError(f"arch invalida: {arch}. Use {list(ARQUITETURAS)}")
    Cell = ARQUITETURAS[arch]

    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    model = models.Sequential(name=f"{arch}_{n_camadas}x{unidades}")
    model.add(layers.Input(shape=input_shape))

    for i in range(n_camadas):
        return_seq = i < (n_camadas - 1)   # ultima camada retorna so o ultimo passo
        cell = Cell(unidades, return_sequences=return_seq,
                    dropout=dropout, recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=reg)
        model.add(layers.Bidirectional(cell) if bidirecional else cell)

    model.add(layers.Dense(horizonte))   # regressao (saida linear)

    opt = (optimizers.Adam(learning_rate=lr) if otimizador == "adam"
           else optimizers.RMSprop(learning_rate=lr))
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# metricas (na escala original: corridas/dia)
# ---------------------------------------------------------------------------
def metricas(y_true_norm, y_pred_norm, scaler):
    yt = desnormaliza_alvo(np.ravel(y_true_norm), scaler)
    yp = desnormaliza_alvo(np.ravel(y_pred_norm), scaler)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# baselines (referencia obrigatoria do enunciado)
# ---------------------------------------------------------------------------
def baseline_naive_sazonal(X, feature_cols, scaler, lag_feature="lag_7"):
    """
    Previsao = demanda de 7 dias atras (mesmo dia da semana anterior).
    Usa a feature 'lag_7' ja presente no ultimo passo da janela.
    Retorna previsao normalizada (comparavel a y_*_norm).
    Observacao: lag_7 foi normalizado pela media/desvio da PROPRIA coluna;
    convertemos para a escala do alvo para comparar corretamente.
    """
    idx = feature_cols.index(lag_feature)
    ultimo = X[:, -1, idx]                       # valor normalizado de lag_7
    # desnormaliza na escala da coluna lag_7 e renormaliza na escala do alvo
    mu_c = scaler["mu"][lag_feature]; sd_c = scaler["sd"][lag_feature]
    valor_corridas = ultimo * sd_c + mu_c
    return (valor_corridas - scaler["alvo_mu"]) / scaler["alvo_sd"]


def baseline_media_movel(X, feature_cols, scaler, feature="media_movel_7"):
    """Previsao = media movel de 7 dias (feature ja calculada e defasada)."""
    idx = feature_cols.index(feature)
    ultimo = X[:, -1, idx]
    mu_c = scaler["mu"][feature]; sd_c = scaler["sd"][feature]
    valor_corridas = ultimo * sd_c + mu_c
    return (valor_corridas - scaler["alvo_mu"]) / scaler["alvo_sd"]


# ---------------------------------------------------------------------------
# treino padronizado (mesmos callbacks p/ as 3 familias)
# ---------------------------------------------------------------------------
def treinar(model, data, epochs=100, batch_size=32, patience=12, verbose=0):
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(3, patience // 3),
                                             min_lr=1e-5),
    ]
    hist = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=verbose,
    )
    return hist


def contar_parametros(model):
    return int(model.count_params())
