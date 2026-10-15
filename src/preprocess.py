"""
Pre-processamento para RNN de previsao de demanda TaxiGOV.

Decisoes (acordadas):
  - Granularidade: DIARIA
  - Demanda: TODAS as solicitacoes (todos os status)
  - Escopo espacial: DF (serie mais densa/consistente, ~87% dos dados)
  - Modelo multivariado: calendario ciclico + feriados + lags + exogenas defasadas
  - Dias sem corrida -> 0; scaler ajustado SO no treino; janelamento cronologico

Saidas:
  - taxigov-demanda-diaria-DF.csv  : serie diaria com todas as features (inspecionavel)
  - taxigov-rnn-dataset.npz        : tensores X/y de treino/val/teste + metadados
  - scaler_params.json             : media/desvio do treino (para reverter previsoes)
"""
import pandas as pd
import numpy as np
import json
import os

try:
    import holidays as holidays_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz do projeto
DATA_RAW = os.path.join(RAIZ, "data", "raw")
DATA_PROC = os.path.join(RAIZ, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)
ENTRADA = os.path.join(DATA_RAW, "taxigov-corridas-unificado.csv")
SAIDA_CSV = os.path.join(DATA_PROC, "taxigov-demanda-diaria-DF.csv")
SAIDA_NPZ = os.path.join(DATA_PROC, "taxigov-rnn-dataset.npz")
SAIDA_SCALER = os.path.join(DATA_PROC, "scaler_params.json")

# ----- hiperparametros de preparacao -----
UF_ALVO = "DF"
# recorte do inicio da serie: descarta 2017-2019, que contem o buraco de dados
# de 160 dias zerados (08/mar a 14/ago/2019) e o inicio ralo da operacao em 2017.
# A partir de 2020-01-01 a serie diaria e continua (a pandemia fica sinalizada
# pela flag 'covid').
DATA_INICIO = "2020-01-01"
# lookback: 7 dias (1 semana). O experimento de tamanho de janela
# (window_experiment.py) mostrou que 7 dias e o melhor lookback para a demanda
# diaria — a sazonalidade semanal domina (ACF lag_7 = 0.80) e janelas maiores
# pioram desempenho, aumentam custo e induzem overfitting.
JANELA = 7
HORIZONTE = 1        # previsao one-step (proximo dia); multi-step pode ser avaliado depois
FRACAO_TREINO = 0.70
FRACAO_VAL = 0.15    # resto (0.15) vira teste; split CRONOLOGICO

# =====================================================================
# ETAPA 1 — ingestao + filtro DF + agregacao diaria (todas as solicitacoes)
# =====================================================================
def etapa1_agregar():
    print("[1] Lendo unificado, filtrando DF e agregando por DIA...")
    usecols = ["data_abertura", "origem_uf", "status", "km_total", "valor_corrida",
               "data_inicio", "data_final"]
    partes = []
    for ch in pd.read_csv(ENTRADA, usecols=usecols, dtype=str,
                          chunksize=200_000, low_memory=False):
        uf = ch["origem_uf"].astype(str).str.strip().str.upper()
        ch = ch[uf == UF_ALVO].copy()
        if ch.empty:
            continue
        dt = pd.to_datetime(ch["data_abertura"], errors="coerce", format="mixed")
        ch = ch.assign(dia=dt.dt.normalize())
        ch = ch.dropna(subset=["dia"])
        ch["km_total"] = pd.to_numeric(ch["km_total"], errors="coerce")
        ch["valor_corrida"] = pd.to_numeric(ch["valor_corrida"], errors="coerce")
        di = pd.to_datetime(ch["data_inicio"], errors="coerce", format="mixed")
        df_ = pd.to_datetime(ch["data_final"], errors="coerce", format="mixed")
        ch["duracao_min"] = (df_ - di).dt.total_seconds() / 60.0
        ch["cancelada"] = (ch["status"].astype(str).str.upper() == "CANCELADA").astype(int)
        partes.append(ch[["dia", "km_total", "valor_corrida", "duracao_min", "cancelada"]])
    df = pd.concat(partes, ignore_index=True)
    print(f"    corridas DF com data valida: {len(df):,}")

    agg = df.groupby("dia").agg(
        qtd_corridas=("dia", "size"),               # ALVO: todas as solicitacoes
        km_medio=("km_total", "mean"),
        valor_medio=("valor_corrida", "mean"),
        duracao_media=("duracao_min", "mean"),
        pct_cancelada=("cancelada", "mean"),
    )
    return agg

# =====================================================================
# ETAPA 2 — reindex continuo (dias vazios=0) + features de calendario
# =====================================================================
def etapa2_calendario(agg):
    print("[2] Reindexando serie diaria continua e criando features de calendario...")
    inicio = max(pd.Timestamp(DATA_INICIO), agg.index.min())
    idx = pd.date_range(inicio, agg.index.max(), freq="D")
    print(f"    recorte do inicio da serie: a partir de {inicio.date()}")
    agg = agg.reindex(idx)
    # dias sem NENHUMA corrida -> alvo 0; exogenas ficam NaN (serao preenchidas por ffill)
    agg["qtd_corridas"] = agg["qtd_corridas"].fillna(0).astype(int)
    for c in ["km_medio", "valor_medio", "duracao_media", "pct_cancelada"]:
        agg[c] = agg[c].ffill().bfill()
    agg.index.name = "dia"

    dow = agg.index.dayofweek          # 0=segunda ... 6=domingo
    mes = agg.index.month
    doy = agg.index.dayofyear
    agg["dow_sin"] = np.sin(2*np.pi*dow/7);   agg["dow_cos"] = np.cos(2*np.pi*dow/7)
    agg["mes_sin"] = np.sin(2*np.pi*mes/12);  agg["mes_cos"] = np.cos(2*np.pi*mes/12)
    agg["doy_sin"] = np.sin(2*np.pi*doy/365); agg["doy_cos"] = np.cos(2*np.pi*doy/365)
    agg["fim_de_semana"] = (dow >= 5).astype(int)

    if HAS_HOLIDAYS:
        anos = list(range(agg.index.year.min(), agg.index.year.max()+1))
        fer = holidays_lib.Brazil(years=anos, subdiv="DF")
        agg["feriado"] = agg.index.to_series().apply(lambda d: int(d.date() in fer)).values
    else:
        print("    [aviso] lib 'holidays' ausente -> coluna feriado=0")
        agg["feriado"] = 0
    return agg

# =====================================================================
# ETAPA 3 — lags do alvo, exogenas defasadas e flag COVID
# =====================================================================
def etapa3_lags(agg):
    print("[3] Criando lags do alvo, exogenas defasadas e flag COVID...")
    for L in [1, 7, 14, 28]:
        agg[f"lag_{L}"] = agg["qtd_corridas"].shift(L)
    agg["media_movel_7"] = agg["qtd_corridas"].shift(1).rolling(7).mean()
    # exogenas SEMPRE defasadas (valor do dia anterior) -> anti-leakage
    for c in ["km_medio", "valor_medio", "duracao_media", "pct_cancelada"]:
        agg[f"{c}_lag1"] = agg[c].shift(1)
        agg.drop(columns=[c], inplace=True)   # remove a versao "do proprio dia" (vazaria)
    # regime de pandemia (choque identificado em mar/2020; normalizacao ~meados 2022)
    agg["covid"] = ((agg.index >= "2020-03-15") & (agg.index <= "2022-06-30")).astype(int)
    agg = agg.dropna()   # descarta as primeiras linhas sem lag completo
    return agg

# =====================================================================
# ETAPA 4 — split cronologico, normalizacao (fit so no treino), janelamento
# =====================================================================
def etapa4_janelas(agg):
    print("[4] Split cronologico, normalizacao (fit no treino) e janelamento...")
    n = len(agg)
    n_tr = int(n * FRACAO_TREINO)
    n_val = int(n * FRACAO_VAL)
    # embargo = JANELA dias entre particoes para nenhuma janela cruzar o split
    tr = agg.iloc[:n_tr]
    val = agg.iloc[n_tr - JANELA : n_tr + n_val]
    te = agg.iloc[n_tr + n_val - JANELA :]

    feature_cols = [c for c in agg.columns]           # todas as colunas sao entrada...
    alvo_col = "qtd_corridas"
    # normalizacao: media/desvio estimados SO no treino
    mu = tr[feature_cols].mean()
    sd = tr[feature_cols].std().replace(0, 1.0)

    def normaliza(d):
        return (d[feature_cols] - mu) / sd

    def montar(d):
        Z = normaliza(d).values
        alvo_norm = ((d[alvo_col] - mu[alvo_col]) / sd[alvo_col]).values
        X, y = [], []
        for i in range(len(d) - JANELA - HORIZONTE + 1):
            X.append(Z[i:i+JANELA])
            y.append(alvo_norm[i+JANELA : i+JANELA+HORIZONTE])
        return np.array(X, dtype="float32"), np.array(y, dtype="float32")

    Xtr, ytr = montar(tr)
    Xval, yval = montar(val)
    Xte, yte = montar(te)

    scaler = {
        "alvo_col": alvo_col,
        "alvo_mu": float(mu[alvo_col]), "alvo_sd": float(sd[alvo_col]),
        "feature_cols": feature_cols,
        "mu": {c: float(mu[c]) for c in feature_cols},
        "sd": {c: float(sd[c]) for c in feature_cols},
        "janela": JANELA, "horizonte": HORIZONTE,
    }
    return (Xtr, ytr, Xval, yval, Xte, yte, scaler,
            (tr.index, val.index, te.index))

# =====================================================================
def main():
    agg = etapa1_agregar()
    agg = etapa2_calendario(agg)
    agg.to_csv(SAIDA_CSV, encoding="utf-8")   # serie diaria inspecionavel (antes dos lags)
    agg = etapa3_lags(agg)
    Xtr, ytr, Xval, yval, Xte, yte, scaler, idxs = etapa4_janelas(agg)

    np.savez_compressed(SAIDA_NPZ,
                        X_train=Xtr, y_train=ytr,
                        X_val=Xval, y_val=yval,
                        X_test=Xte, y_test=yte,
                        feature_cols=np.array(scaler["feature_cols"]))
    with open(SAIDA_SCALER, "w", encoding="utf-8") as fp:
        json.dump(scaler, fp, ensure_ascii=False, indent=2)

    print("\n" + "="*64)
    print("PRE-PROCESSAMENTO CONCLUIDO")
    print("="*64)
    print(f"Serie diaria: {idxs[0][0].date()} ... {idxs[2][-1].date()}")
    print(f"Nº de features por passo: {len(scaler['feature_cols'])}")
    print(f"Janela (lookback): {JANELA} dias | Horizonte: {HORIZONTE} dia(s)")
    print(f"  X_train={Xtr.shape}  y_train={ytr.shape}")
    print(f"  X_val  ={Xval.shape}  y_val  ={yval.shape}")
    print(f"  X_test ={Xte.shape}  y_test ={yte.shape}")
    print(f"\nAlvo (treino): media={scaler['alvo_mu']:.1f}  desvio={scaler['alvo_sd']:.1f} corridas/dia")
    print(f"Features: {scaler['feature_cols']}")
    print(f"\nArquivos salvos:\n  {SAIDA_CSV}\n  {SAIDA_NPZ}\n  {SAIDA_SCALER}")

if __name__ == "__main__":
    main()
