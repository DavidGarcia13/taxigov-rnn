"""
eda.py — Analise Exploratoria dos Dados (requisito 2.1).

Caracteriza a variavel alvo (demanda diaria de corridas no DF) e sua estrutura
temporal, com estatisticas descritivas e visualizacoes:
  - distribuicao do alvo (faixa, dispersao, assimetria, curtose, outliers)
  - serie temporal completa + media movel
  - sazonalidade: por dia-da-semana e por mes
  - media diaria por ano (efeito pandemia)
  - autocorrelacao (ACF) para justificar lags/janela

Saidas:
  results/figuras/eda/*.png
  results/eda_stats.json
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
SERIE_CSV = os.path.join(RAIZ, "data", "processed", "taxigov-demanda-diaria-DF.csv")
RESULTS = os.path.join(RAIZ, "results")
EDA = os.path.join(RESULTS, "figuras", "eda")
os.makedirs(EDA, exist_ok=True)

DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]


def acf(x, nlags):
    x = np.asarray(x, float); x = x - x.mean()
    den = np.sum(x * x)
    return [1.0 if k == 0 else float(np.sum(x[k:] * x[:-k]) / den) for k in range(nlags + 1)]


def main():
    df = pd.read_csv(SERIE_CSV, parse_dates=["dia"]).set_index("dia")
    q = df["qtd_corridas"].astype(float)

    # ---- estatisticas descritivas ----
    stats = {
        "n_dias": int(len(q)),
        "periodo": f"{q.index.min().date()} a {q.index.max().date()}",
        "min": float(q.min()), "max": float(q.max()),
        "media": float(q.mean()), "mediana": float(q.median()),
        "desvio_padrao": float(q.std()),
        "assimetria_skew": float(q.skew()),
        "curtose": float(q.kurtosis()),
        "q25": float(q.quantile(.25)), "q75": float(q.quantile(.75)),
        "dias_zero": int((q == 0).sum()),
        "cv_desvio_sobre_media": float(q.std() / q.mean()),
    }
    # outliers por IQR
    iqr = stats["q75"] - stats["q25"]
    lo, hi = stats["q25"] - 1.5 * iqr, stats["q75"] + 1.5 * iqr
    stats["outliers_iqr"] = int(((q < lo) | (q > hi)).sum())
    with open(os.path.join(RESULTS, "eda_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- 1. histograma do alvo ----
    plt.figure(figsize=(7, 4))
    plt.hist(q, bins=40, color="tab:blue", alpha=.8)
    plt.axvline(q.mean(), color="red", ls="--", label=f"media={q.mean():.0f}")
    plt.axvline(q.median(), color="green", ls=":", label=f"mediana={q.median():.0f}")
    plt.title(f"Distribuicao da demanda diaria (skew={q.skew():.2f})")
    plt.xlabel("corridas/dia"); plt.ylabel("frequencia"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(EDA, "01_hist_alvo.png"), dpi=120); plt.close()

    # ---- 2. serie temporal + media movel 30d ----
    plt.figure(figsize=(11, 4))
    plt.plot(q.index, q, lw=.7, alpha=.6, label="diario")
    plt.plot(q.index, q.rolling(30).mean(), color="red", lw=1.5, label="media movel 30d")
    plt.title("Serie temporal da demanda diaria (DF)")
    plt.xlabel("data"); plt.ylabel("corridas/dia"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(EDA, "02_serie_temporal.png"), dpi=120); plt.close()

    # ---- 3. boxplot por dia-da-semana ----
    dados = [q[q.index.dayofweek == d].values for d in range(7)]
    plt.figure(figsize=(7, 4))
    plt.boxplot(dados, tick_labels=DIAS, showfliers=False)
    plt.title("Sazonalidade semanal da demanda")
    plt.ylabel("corridas/dia"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(EDA, "03_boxplot_dow.png"), dpi=120); plt.close()

    # ---- 4. media por mes ----
    media_mes = q.groupby(q.index.month).mean()
    plt.figure(figsize=(7, 4))
    plt.bar(media_mes.index, media_mes.values, color="tab:green", alpha=.8)
    plt.title("Demanda media por mes"); plt.xlabel("mes"); plt.ylabel("corridas/dia")
    plt.xticks(range(1, 13)); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(EDA, "04_media_mes.png"), dpi=120); plt.close()

    # ---- 5. media diaria por ano (pandemia) ----
    media_ano = q.groupby(q.index.year).mean()
    plt.figure(figsize=(7, 4))
    plt.bar(media_ano.index.astype(str), media_ano.values, color="tab:orange", alpha=.85)
    plt.title("Demanda media diaria por ano"); plt.ylabel("corridas/dia"); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(EDA, "05_media_ano.png"), dpi=120); plt.close()

    # ---- 6. ACF ----
    lags = 40
    a = acf(q.values, lags)
    plt.figure(figsize=(8, 4))
    plt.stem(range(lags + 1), a)
    plt.axhline(0, color="k", lw=.8)
    plt.title("Autocorrelacao (ACF) da demanda diaria")
    plt.xlabel("lag (dias)"); plt.ylabel("ACF"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(EDA, "06_acf.png"), dpi=120); plt.close()

    print("ESTATISTICAS DA VARIAVEL ALVO (demanda diaria, DF):")
    for k, v in stats.items():
        print(f"  {k:28} {v}")
    print(f"\nACF: lag1={a[1]:.3f}  lag7={a[7]:.3f}  lag14={a[14]:.3f}  lag28={a[28]:.3f}")
    print(f"\nFiguras: {EDA}\nStats: {os.path.join(RESULTS,'eda_stats.json')}")


if __name__ == "__main__":
    main()
