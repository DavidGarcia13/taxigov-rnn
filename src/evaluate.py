"""
evaluate.py — avaliacao final das 3 arquiteturas recorrentes.

Fluxo:
  1. Le os melhores hiperparametros de results/best_params_<ARCH>.json (Optuna).
     (Se nao existirem, usa um fallback razoavel para a arquitetura.)
  2. Re-treina cada arquitetura (treino + validacao) sob o mesmo protocolo.
  3. Avalia no conjunto de TESTE: MAE, RMSE, R2 (escala de corridas/dia).
  4. Mede custo: tempo de treino, nº de parametros, tempo de inferencia.
  5. Compara com os baselines (naive sazonal e media movel).
  6. Gera figuras (curvas treino×val e previsto×real) e salva metrics_final.json.

Uso:
  python evaluate.py
  python evaluate.py --epochs 120
"""
import os, json, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import models as M

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
RESULTS = os.path.join(RAIZ, "results")
FIGS = os.path.join(RESULTS, "figuras")
os.makedirs(FIGS, exist_ok=True)

# fallback caso o tuning ainda nao tenha rodado para alguma arquitetura
FALLBACK = {
    "SimpleRNN": dict(n_camadas=1, unidades=64, dropout=0.1, recurrent_dropout=0.0,
                      l2=1e-5, lr=1e-3, batch_size=32, otimizador="adam", bidirecional=False),
    "GRU":       dict(n_camadas=1, unidades=64, dropout=0.2, recurrent_dropout=0.0,
                      l2=1e-5, lr=1e-3, batch_size=32, otimizador="adam", bidirecional=False),
    "LSTM":      dict(n_camadas=2, unidades=64, dropout=0.2, recurrent_dropout=0.0,
                      l2=1e-5, lr=1e-3, batch_size=32, otimizador="adam", bidirecional=False),
}


def carregar_best_params(arch):
    caminho = os.path.join(RESULTS, f"best_params_{arch}.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            bp = json.load(f)["best_params"]
        print(f"[{arch}] usando best_params do Optuna: {bp}")
        return bp
    print(f"[{arch}] best_params ausente -> fallback")
    return FALLBACK[arch]


def plota_curvas(hist, arch):
    plt.figure(figsize=(7, 4))
    plt.plot(hist.history["loss"], label="treino")
    plt.plot(hist.history["val_loss"], label="validacao")
    plt.title(f"Curva de perda (MSE) — {arch}")
    plt.xlabel("epoca"); plt.ylabel("loss (MSE norm.)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    p = os.path.join(FIGS, f"curva_loss_{arch}.png")
    plt.savefig(p, dpi=120); plt.close()
    return p


def plota_pred(y_true, y_pred, arch):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="real", linewidth=1.2)
    plt.plot(y_pred, label="previsto", linewidth=1.2, alpha=.8)
    plt.title(f"Teste: previsto × real — {arch}")
    plt.xlabel("dia (conjunto de teste)"); plt.ylabel("corridas/dia")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    p = os.path.join(FIGS, f"pred_vs_real_{arch}.png")
    plt.savefig(p, dpi=120); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()

    data = M.carregar_dados()
    sc = data["scaler"]; fc = data["feature_cols"]
    input_shape = data["X_train"].shape[1:]
    horizonte = data["y_train"].shape[1]

    resultado = {"baselines": {}, "modelos": {}}

    # ---- baselines ----
    for nome, fn in [("naive_sazonal", M.baseline_naive_sazonal),
                     ("media_movel_7", M.baseline_media_movel)]:
        pred = fn(data["X_test"], fc, sc)
        resultado["baselines"][nome] = M.metricas(data["y_test"], pred, sc)

    # ---- modelos ----
    y_test_real = M.desnormaliza_alvo(np.ravel(data["y_test"]), sc)
    for arch in M.ARQUITETURAS:
        bp = carregar_best_params(arch)
        tf.keras.backend.clear_session()
        model = M.build_model(
            arch, input_shape,
            n_camadas=bp.get("n_camadas", 1), unidades=bp.get("unidades", 64),
            dropout=bp.get("dropout", 0.1), recurrent_dropout=bp.get("recurrent_dropout", 0.0),
            l2=bp.get("l2", 0.0), lr=bp.get("lr", 1e-3),
            bidirecional=bp.get("bidirecional", False),
            otimizador=bp.get("otimizador", "adam"), horizonte=horizonte,
        )
        t0 = time.time()
        hist = M.treinar(model, data, epochs=args.epochs,
                         batch_size=bp.get("batch_size", 32), patience=15, verbose=0)
        t_treino = time.time() - t0

        t1 = time.time()
        pred = model.predict(data["X_test"], verbose=0)
        t_infer = time.time() - t1

        met = M.metricas(data["y_test"], pred, sc)
        resultado["modelos"][arch] = {
            **met,
            "n_params": M.contar_parametros(model),
            "tempo_treino_s": round(t_treino, 1),
            "tempo_inferencia_ms": round(1000 * t_infer, 1),
            "epocas_treinadas": len(hist.history["loss"]),
            "hiperparametros": bp,
        }
        plota_curvas(hist, arch)
        plota_pred(y_test_real, M.desnormaliza_alvo(np.ravel(pred), sc), arch)
        print(f"[{arch}] MAE={met['MAE']:.1f} RMSE={met['RMSE']:.1f} R2={met['R2']:.3f} "
              f"| params={resultado['modelos'][arch]['n_params']} "
              f"| treino={t_treino:.1f}s | infer={1000*t_infer:.1f}ms")

    with open(os.path.join(RESULTS, "metrics_final.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # ---- tabela resumo ----
    print("\n" + "="*78)
    print(f"{'Modelo':16}{'MAE':>8}{'RMSE':>9}{'R2':>8}{'params':>9}{'treino(s)':>11}{'infer(ms)':>11}")
    print("-"*78)
    for n, m in resultado["baselines"].items():
        print(f"{n:16}{m['MAE']:>8.1f}{m['RMSE']:>9.1f}{m['R2']:>8.3f}{'-':>9}{'-':>11}{'-':>11}")
    for a, m in resultado["modelos"].items():
        print(f"{a:16}{m['MAE']:>8.1f}{m['RMSE']:>9.1f}{m['R2']:>8.3f}"
              f"{m['n_params']:>9}{m['tempo_treino_s']:>11}{m['tempo_inferencia_ms']:>11}")
    print("="*78)
    print(f"\nFiguras em: {FIGS}\nMetricas em: {os.path.join(RESULTS,'metrics_final.json')}")


if __name__ == "__main__":
    main()
