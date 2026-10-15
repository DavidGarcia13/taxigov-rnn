"""
window_experiment.py — efeito do TAMANHO DA JANELA (lookback) na previsao.

Fecha o requisito da frente 3.2:
  "Investigar diferentes tamanhos de janela coerentes com a dinamica do fenomeno"
  "Avaliar dependencias de longo prazo, discutindo se janelas maiores melhoram
   a previsao e como afetam custo, estabilidade e overfitting."

Protocolo controlado: tudo fixo (mesma serie, mesma arquitetura GRU com os
melhores hiperparametros do Optuna, mesmo split cronologico 70/15/15, mesmo
horizonte=1), variando SOMENTE a janela. Assim a diferenca observada e atribuivel
ao lookback.

Janelas testadas: 7, 14, 28, 56 dias (1 a 8 semanas).

Saidas:
  results/window_experiment.json       (tabela comparativa)
  results/figuras/janela_vs_metrica.png
"""
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import models as M

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
SERIE_CSV = os.path.join(RAIZ, "data", "processed", "taxigov-demanda-diaria-DF.csv")
RESULTS = os.path.join(RAIZ, "results")
FIGS = os.path.join(RESULTS, "figuras")
os.makedirs(FIGS, exist_ok=True)

JANELAS = [7, 14, 28, 56]
HORIZONTE = 1
FRACAO_TREINO = 0.70
FRACAO_VAL = 0.15

# arquitetura fixa = melhor GRU do Optuna (lida de best_params; fallback se faltar)
def gru_params():
    p = os.path.join(RESULTS, "best_params_GRU.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))["best_params"]
    return dict(n_camadas=2, unidades=64, dropout=0.0, recurrent_dropout=0.0,
                l2=3.9e-6, lr=4.15e-4, batch_size=32, otimizador="adam", bidirecional=True)


def preparar_features(df):
    """Aplica lags + exogenas defasadas + covid (mesma logica do preprocess.py)."""
    df = df.copy()
    for L in [1, 7, 14, 28]:
        df[f"lag_{L}"] = df["qtd_corridas"].shift(L)
    df["media_movel_7"] = df["qtd_corridas"].shift(1).rolling(7).mean()
    for c in ["km_medio", "valor_medio", "duracao_media", "pct_cancelada"]:
        df[f"{c}_lag1"] = df[c].shift(1)
        df.drop(columns=[c], inplace=True)
    df["covid"] = ((df.index >= "2020-03-15") & (df.index <= "2022-06-30")).astype(int)
    return df.dropna()


def montar_para_janela(agg, janela):
    """Split cronologico + normalizacao (fit no treino) + janelamento para dada janela."""
    n = len(agg)
    n_tr = int(n * FRACAO_TREINO)
    n_val = int(n * FRACAO_VAL)
    tr = agg.iloc[:n_tr]
    val = agg.iloc[n_tr - janela: n_tr + n_val]
    te = agg.iloc[n_tr + n_val - janela:]

    cols = list(agg.columns)
    alvo = "qtd_corridas"
    mu = tr[cols].mean()
    sd = tr[cols].std().replace(0, 1.0)

    def montar(d):
        Z = ((d[cols] - mu) / sd).values
        yv = ((d[alvo] - mu[alvo]) / sd[alvo]).values
        X, y = [], []
        for i in range(len(d) - janela - HORIZONTE + 1):
            X.append(Z[i:i + janela]); y.append(yv[i + janela: i + janela + HORIZONTE])
        return np.array(X, "float32"), np.array(y, "float32")

    data = dict(zip(["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
                    [*montar(tr), *montar(val), *montar(te)]))
    scaler = {"alvo_mu": float(mu[alvo]), "alvo_sd": float(sd[alvo])}
    return data, scaler


def main():
    df = pd.read_csv(SERIE_CSV, parse_dates=["dia"]).set_index("dia")
    agg = preparar_features(df)
    hp = gru_params()
    print("janelas:", JANELAS, "| arquitetura: GRU", hp)

    linhas = []
    for J in JANELAS:
        data, scaler = montar_para_janela(agg, J)
        tf.keras.backend.clear_session()
        model = M.build_model(
            "GRU", data["X_train"].shape[1:],
            n_camadas=hp["n_camadas"], unidades=hp["unidades"],
            dropout=hp["dropout"], recurrent_dropout=hp["recurrent_dropout"],
            l2=hp["l2"], lr=hp["lr"], bidirecional=hp["bidirecional"],
            otimizador=hp["otimizador"], horizonte=HORIZONTE,
        )
        t0 = time.time()
        hist = M.treinar(model, data, epochs=120, batch_size=hp["batch_size"],
                         patience=15, verbose=0)
        t_treino = time.time() - t0

        pred = model.predict(data["X_test"], verbose=0)
        met = M.metricas(data["y_test"], pred, scaler)
        # gap treino-val como sinal de overfitting (loss final)
        gap = float(hist.history["val_loss"][-1] - hist.history["loss"][-1])
        linhas.append({
            "janela": J, "MAE": met["MAE"], "RMSE": met["RMSE"], "R2": met["R2"],
            "n_params": M.contar_parametros(model), "tempo_treino_s": round(t_treino, 1),
            "epocas": len(hist.history["loss"]),
            "gap_val_treino": round(gap, 4),
            "amostras_treino": int(data["X_train"].shape[0]),
        })
        print(f"J={J:>3}  MAE={met['MAE']:.1f}  RMSE={met['RMSE']:.1f}  R2={met['R2']:.3f}  "
              f"params={M.contar_parametros(model)}  treino={t_treino:.1f}s  "
              f"epocas={linhas[-1]['epocas']}  gap={gap:+.3f}")

    with open(os.path.join(RESULTS, "window_experiment.json"), "w", encoding="utf-8") as f:
        json.dump(linhas, f, ensure_ascii=False, indent=2)

    # figura: janela x (RMSE, R2)
    J = [l["janela"] for l in linhas]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(J, [l["RMSE"] for l in linhas], "o-", color="tab:red", label="RMSE")
    ax1.set_xlabel("tamanho da janela (dias)"); ax1.set_ylabel("RMSE (corridas/dia)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.set_xticks(J)
    ax2 = ax1.twinx()
    ax2.plot(J, [l["R2"] for l in linhas], "s--", color="tab:blue", label="R2")
    ax2.set_ylabel("R2", color="tab:blue"); ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title("Efeito do tamanho da janela (GRU) — conjunto de teste")
    fig.tight_layout()
    p = os.path.join(FIGS, "janela_vs_metrica.png")
    plt.savefig(p, dpi=120); plt.close()

    print("\n" + "="*70)
    print(f"{'Janela':>7}{'MAE':>9}{'RMSE':>9}{'R2':>8}{'params':>9}{'treino(s)':>11}{'gap':>9}")
    for l in linhas:
        print(f"{l['janela']:>7}{l['MAE']:>9.1f}{l['RMSE']:>9.1f}{l['R2']:>8.3f}"
              f"{l['n_params']:>9}{l['tempo_treino_s']:>11}{l['gap_val_treino']:>9.3f}")
    print("="*70)
    print(f"figura: {p}\njson: {os.path.join(RESULTS,'window_experiment.json')}")


if __name__ == "__main__":
    main()
