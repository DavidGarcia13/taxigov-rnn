"""
tune.py — busca de hiperparametros com Optuna para SimpleRNN, GRU e LSTM.

Protocolo:
  - Um estudo Optuna POR arquitetura (mesmo espaco de busca para as 3).
  - Metrica otimizada: val-loss (MSE na validacao) -> minimizar.
  - Early stopping por val_loss e pruning (MedianPruner) para cortar trials ruins.
  - Divisao cronologica ja aplicada no pre-processamento (sem embaralhar).

Saidas (pasta results/):
  - optuna_<ARCH>.db        : storage do estudo (reabrivel)
  - best_params_<ARCH>.json : melhores hiperparametros + val-loss
  - resumo_tuning.json      : consolidado das 3 arquiteturas

Uso:
  python tune.py --arch LSTM --trials 30 --epochs 80
  python tune.py --todas --trials 25            # roda as 3 em sequencia
"""
import os, json, argparse, time
import numpy as np
import optuna
import tensorflow as tf

import models as M

BASE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(BASE)
RESULTS = os.path.join(RAIZ, "results")
os.makedirs(RESULTS, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# espaco de busca (prioriza os hiperparametros de maior impacto — enunciado 2.4)
# ---------------------------------------------------------------------------
def sugerir(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "n_camadas": trial.suggest_int("n_camadas", 1, 3),
        "unidades": trial.suggest_categorical("unidades", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.3, step=0.1),
        "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "otimizador": trial.suggest_categorical("otimizador", ["adam", "rmsprop"]),
        "bidirecional": trial.suggest_categorical("bidirecional", [False, True]),
    }


class PodaKeras(tf.keras.callbacks.Callback):
    """Reporta val_loss ao Optuna por epoca e permite pruning."""
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        vl = (logs or {}).get("val_loss")
        if vl is None:
            return
        self.trial.report(vl, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objetivo(trial, arch, data, input_shape, epochs):
    hp = sugerir(trial)
    tf.keras.backend.clear_session()
    model = M.build_model(
        arch, input_shape,
        n_camadas=hp["n_camadas"], unidades=hp["unidades"],
        dropout=hp["dropout"], recurrent_dropout=hp["recurrent_dropout"],
        l2=hp["l2"], lr=hp["lr"], bidirecional=hp["bidirecional"],
        otimizador=hp["otimizador"], horizonte=data["y_train"].shape[1],
    )
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
        PodaKeras(trial),
    ]
    hist = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs, batch_size=hp["batch_size"], callbacks=cbs, verbose=0,
    )
    return float(np.min(hist.history["val_loss"]))


# ---------------------------------------------------------------------------
def rodar_arquitetura(arch, data, trials, epochs):
    input_shape = data["X_train"].shape[1:]
    storage = f"sqlite:///{os.path.join(RESULTS, f'optuna_{arch}.db')}"
    study = optuna.create_study(
        study_name=f"taxigov_{arch}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        storage=storage, load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(lambda t: objetivo(t, arch, data, input_shape, epochs),
                   n_trials=trials, show_progress_bar=False)
    dur = time.time() - t0

    melhor = {
        "arch": arch,
        "best_val_loss": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_podados": len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.PRUNED]),
        "tempo_s": round(dur, 1),
    }
    with open(os.path.join(RESULTS, f"best_params_{arch}.json"), "w", encoding="utf-8") as f:
        json.dump(melhor, f, ensure_ascii=False, indent=2)
    print(f"[{arch}] melhor val_loss={melhor['best_val_loss']:.4f} "
          f"| trials={melhor['n_trials']} (podados={melhor['n_podados']}) "
          f"| {melhor['tempo_s']}s")
    print(f"        params: {melhor['best_params']}")
    return melhor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(M.ARQUITETURAS))
    ap.add_argument("--todas", action="store_true", help="roda SimpleRNN, GRU e LSTM")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    data = M.carregar_dados()
    alvos = list(M.ARQUITETURAS) if args.todas else [args.arch]
    if alvos == [None]:
        ap.error("informe --arch <ARQUITETURA> ou --todas")

    resumo = {}
    for arch in alvos:
        resumo[arch] = rodar_arquitetura(arch, data, args.trials, args.epochs)

    with open(os.path.join(RESULTS, "resumo_tuning.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print("\nResumo salvo em results/resumo_tuning.json")


if __name__ == "__main__":
    main()
