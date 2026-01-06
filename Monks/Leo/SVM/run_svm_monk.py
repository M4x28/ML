"""
Grid search (con repliche multi-seed) di SVM su MONK (classificazione).

Richieste implementate:
1) Split train/val sul training set:
   - hold-out 80/20 (train/val) con 5 seed fissi: 0,1,2,3,4.
   - il test set originale resta separato ed e' usato solo per la valutazione finale.
2) Normalizzazione degli input (One-Hot + StandardScaler).
3) Grid search in scala logaritmica per C e gamma (valori forniti come potenze di 10).
4) Metriche:
   - training: MSE + accuracy
   - validation/test: MSE + MEE + accuracy
5) TensorBoard: logga solo i migliori risultati (per ogni modello e best complessivo per MONK).
6) Retraining finale ripetuto su tutti i 5 seed; si riportano le medie sui seed e si salvano i modelli di ogni seed.

Dati attesi in `data/monk/`.

Nota:
- SVC ottimizza hinge loss (non MSE). Qui MSE/MEE vengono calcolate come metriche su output continuo.
- Per SVC usiamo `decision_function` e lo trasformiamo con sigmoide (pseudo-probabilità) per avere output in [0,1].
- Per SVR usiamo direttamente l'output continuo di `predict`.
- La regolarizzazione non puo' essere rimossa: `C` controlla la sua forza (C grande = regolarizzazione piu' debole).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR

from monk_data import load_monk_task, make_holdout_split
from metrics import accuracy_from_labels, mee, mse


# Per ridurre rumore su Windows con TensorBoard/TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


ModelKind = Literal["svc_linear", "svc_rbf", "svc_poly", "svr_rbf"]

# Per ogni combinazione di iperparametri facciamo sempre 5 run con questi seed (richiesta del task).
SEEDS = [0, 1, 2, 3, 4]

# Nota su "early stopping" per SVM:
# SVC/SVR (libsvm) non hanno epoche come le reti neurali. Qui l'unico "early stop" possibile è
# fermare l'ottimizzatore dopo un numero massimo di iterazioni (max_iter) e/o con tolleranza (tol).
# Tol un po' più "largo" per evitare terminazioni premature (ConvergenceWarning) su alcune combinazioni (es. gamma molto piccolo).
LIBSVM_TOL = 1e-2
LIBSVM_MAX_ITER = 50_000
LIBSVM_VERBOSE = False

# Grid search "adattiva": prima screening su griglia grossolana, poi raffinamento sui range promettenti.
SCREEN_TOP_K = 5

# Salviamo tutto sotto la cartella `SVM_Monk/` (directory di questo script).
OUTPUT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    train_acc: float
    train_mse: float
    val_acc: float
    val_mse: float
    val_mee: float
    test_acc: float
    test_mse: float
    test_mee: float


@dataclass(frozen=True)
class AggregatedMetrics:
    # medie (richieste) + std (utile per analisi)
    train_acc_mean: float
    train_acc_std: float
    train_mse_mean: float
    train_mse_std: float
    val_acc_mean: float
    val_acc_std: float
    val_mse_mean: float
    val_mse_std: float
    val_mee_mean: float
    val_mee_std: float
    test_acc_mean: float
    test_acc_std: float
    test_mse_mean: float
    test_mse_std: float
    test_mee_mean: float
    test_mee_std: float


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoide numericamente stabile.
    """
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)

    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _predict_continuous(model: Pipeline, X: np.ndarray, *, kind: ModelKind) -> np.ndarray:
    """
    Produce un output continuo da usare per MSE/MEE:
    - SVC: sigmoide(decision_function) -> pseudo-probabilità in [0,1]
    - SVR: predict (output reale)
    """
    if kind.startswith("svc_"):
        scores = model.decision_function(X)
        # Per MONK siamo in binario: shape (n_samples,)
        return _stable_sigmoid(scores)
    if kind.startswith("svr_"):
        return model.predict(X)
    raise ValueError(f"Model kind non supportato: {kind}")


def _predict_labels_from_continuous(y_cont: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """
    Converte output continuo in etichette 0/1 con soglia 0.5.
    """
    return (np.asarray(y_cont) >= threshold).astype(int)


def _build_pipeline(kind: ModelKind, params: dict[str, Any]) -> Pipeline:
    """
    Costruisce una pipeline:
        OneHotEncoder (dense) -> StandardScaler -> SVC/SVR
    """
    # One-hot delle 6 feature categoriali + standardizzazione (SVM è molto sensibile alla scala).
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    if kind == "svc_linear":
        # Per il kernel lineare usiamo LinearSVC (liblinear): è molto più veloce e stabile di SVC(kernel="linear").
        model = LinearSVC(
            C=float(params["C"]),
            tol=LIBSVM_TOL,
            max_iter=LIBSVM_MAX_ITER,
            dual="auto",
            verbose=0,
        )
    elif kind == "svc_rbf":
        model = SVC(
            kernel="rbf",
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            tol=LIBSVM_TOL,
            max_iter=LIBSVM_MAX_ITER,
            verbose=LIBSVM_VERBOSE,
        )
    elif kind == "svc_poly":
        # gamma può essere float oppure 'scale'/'auto' (stringa)
        model = SVC(
            kernel="poly",
            C=float(params["C"]),
            degree=int(params["degree"]),
            gamma=params["gamma"],
            coef0=float(params["coef0"]),
            tol=LIBSVM_TOL,
            max_iter=LIBSVM_MAX_ITER,
            verbose=LIBSVM_VERBOSE,
        )
    elif kind == "svr_rbf":
        model = SVR(
            kernel="rbf",
            C=float(params["C"]),
            gamma=float(params["gamma"]),
            epsilon=float(params["epsilon"]),
            tol=LIBSVM_TOL,
            max_iter=LIBSVM_MAX_ITER,
            verbose=LIBSVM_VERBOSE,
        )
    else:
        raise ValueError(f"Model kind non supportato: {kind}")

    return Pipeline(
        steps=[
            ("onehot", encoder),
            ("scaler", scaler),
            ("model", model),
        ]
    )


def _get_param_space(kind: ModelKind) -> dict[str, list[Any]]:
    """
    Spazio degli iperparametri (liste richieste dal task).
    """
    if kind == "svc_linear":
        return {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
    if kind == "svc_rbf":
        return {
            "C": [1e-2, 1e-1, 1, 10, 100, 1000],
            "gamma": [1e-3, 1e-2, 1e-1, 1, 10],
        }
    if kind == "svc_poly":
        return {
            "C": [0.1, 1, 10, 100],
            "degree": [2, 3, 4, 5],
            "gamma": ["scale", "auto", 1e-2, 1e-1, 1],
            "coef0": [0, 1, 5, 10],
        }
    if kind == "svr_rbf":
        return {
            "C": [1, 10, 100, 500, 1000],
            "gamma": [1e-3, 1e-2, 1e-1, 1],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1],
        }
    raise ValueError(f"Model kind non supportato: {kind}")


def _get_param_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Espande uno spazio in una lista di combinazioni.
    """
    return list(ParameterGrid(space))


def _screen_values(values: list[Any]) -> list[Any]:
    """
    Seleziona pochi valori rappresentativi (griglia grossolana) mantenendo l'ordine originale.
    """
    if len(values) <= 3:
        return list(values)
    if len(values) == 4:
        candidates = [values[0], values[-1]]
    else:
        candidates = [values[0], values[len(values) // 2], values[-1]]

    out: list[Any] = []
    for v in candidates:
        if v not in out:
            out.append(v)
    return out


def _neighbors(values: list[Any], value: Any) -> list[Any]:
    """
    Restituisce value e i suoi vicini (precedente/successivo) nella lista originale.
    """
    if value not in values:
        return [value]
    idx = values.index(value)
    out: list[Any] = []
    if idx > 0:
        out.append(values[idx - 1])
    out.append(values[idx])
    if idx < len(values) - 1:
        out.append(values[idx + 1])
    return out


def _refine_space(full_space: dict[str, list[Any]], top_params: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Costruisce una griglia più ristretta attorno alle migliori configurazioni (unione dei vicini).
    """
    refined: dict[str, list[Any]] = {}
    for name, values in full_space.items():
        selected: set[Any] = set()
        for p in top_params:
            selected.update(_neighbors(values, p[name]))
        refined[name] = [v for v in values if v in selected]
    return refined


def _aggregate(per_seed: list[SeedMetrics]) -> AggregatedMetrics:
    """
    Media e deviazione standard sulle repliche.
    """
    def _mean_std(values: Iterable[float]) -> tuple[float, float]:
        arr = np.asarray(list(values), dtype=float)
        if np.all(np.isnan(arr)):
            return float("nan"), float("nan")
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    train_acc_mean, train_acc_std = _mean_std(m.train_acc for m in per_seed)
    train_mse_mean, train_mse_std = _mean_std(m.train_mse for m in per_seed)
    val_acc_mean, val_acc_std = _mean_std(m.val_acc for m in per_seed)
    val_mse_mean, val_mse_std = _mean_std(m.val_mse for m in per_seed)
    val_mee_mean, val_mee_std = _mean_std(m.val_mee for m in per_seed)
    test_acc_mean, test_acc_std = _mean_std(m.test_acc for m in per_seed)
    test_mse_mean, test_mse_std = _mean_std(m.test_mse for m in per_seed)
    test_mee_mean, test_mee_std = _mean_std(m.test_mee for m in per_seed)

    return AggregatedMetrics(
        train_acc_mean=train_acc_mean,
        train_acc_std=train_acc_std,
        train_mse_mean=train_mse_mean,
        train_mse_std=train_mse_std,
        val_acc_mean=val_acc_mean,
        val_acc_std=val_acc_std,
        val_mse_mean=val_mse_mean,
        val_mse_std=val_mse_std,
        val_mee_mean=val_mee_mean,
        val_mee_std=val_mee_std,
        test_acc_mean=test_acc_mean,
        test_acc_std=test_acc_std,
        test_mse_mean=test_mse_mean,
        test_mse_std=test_mse_std,
        test_mee_mean=test_mee_mean,
        test_mee_std=test_mee_std,
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_grid_search(
    *,
    task_id: int,
    kind: ModelKind,
    run_name: str,
) -> dict[str, Any]:
    """
    Esegue la grid search per un task MONK e un tipo di modello.
    Salva:
    - risultati incrementali (JSONL)
    - modelli per seed (joblib) + metadata
    - checkpoint best-so-far
    """
    # Usiamo il training set per train/val e teniamo il test set separato.
    X_train_full, y_train_full = load_monk_task(task_id, split="train")
    X_test, y_test = load_monk_task(task_id, split="test")
    output_root = OUTPUT_ROOT

    results_dir = output_root / "results" / f"monk{task_id}" / kind
    ckpt_dir = output_root / "checkpoints" / f"monk{task_id}" / kind / run_name
    export_dir = output_root / "exported_models" / f"monk{task_id}" / kind / run_name

    for p in (results_dir, ckpt_dir, export_dir):
        _ensure_dir(p)

    full_space = _get_param_space(kind)
    screen_space = {name: _screen_values(values) for name, values in full_space.items()}
    screen_params = _get_param_grid(screen_space)

    print(
        f"  Grid search adattiva: full={len(ParameterGrid(full_space))} "
        f"screen={len(screen_params)} top_k={SCREEN_TOP_K} seeds={SEEDS}",
        flush=True,
    )
    print(f"  libsvm: tol={LIBSVM_TOL} max_iter={LIBSVM_MAX_ITER}", flush=True)
    results_path = results_dir / f"{run_name}.jsonl"
    summary_path = export_dir / "summary.json"
    models_dir = export_dir / "models_by_seed"
    best_meta_ckpt_path = ckpt_dir / "best_model_so_far.json"

    best_meta: dict[str, Any] | None = None
    # Criterio di selezione: minimizzare MEE su validation, poi massimizzare accuracy.
    best_key: tuple[float, float, float] | None = None  # (val_mse_mean, val_mee_mean, -val_acc_mean)

    def _sig(params: dict[str, Any]) -> str:
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def _eval_params_on_validation(params: dict[str, Any]) -> tuple[list[SeedMetrics], AggregatedMetrics, float]:
        per_seed: list[SeedMetrics] = []
        t0 = time.time()
        for seed in SEEDS:
            # Split 80/20 sul training set: train/val (usato per la grid search).
            X_train, X_val, y_train, y_val = make_holdout_split(X_train_full, y_train_full, seed=seed)

            model = _build_pipeline(kind, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(X_train, y_train)

            core = model.named_steps["model"]
            if hasattr(core, "fit_status_") and getattr(core, "fit_status_") != 0:
                print(
                    f"    WARNING seed={seed}: libsvm stopped early (n_iter={getattr(core, 'n_iter_', '?')})",
                    flush=True,
                )

            # Output continuo -> metriche MSE/MEE; output discreto -> accuracy.
            y_train_cont = _predict_continuous(model, X_train, kind=kind)
            y_val_cont = _predict_continuous(model, X_val, kind=kind)

            y_train_pred = _predict_labels_from_continuous(y_train_cont)
            y_val_pred = _predict_labels_from_continuous(y_val_cont)

            per_seed.append(
                SeedMetrics(
                    seed=seed,
                    train_acc=accuracy_from_labels(y_train, y_train_pred),
                    train_mse=mse(y_train, y_train_cont),
                    val_acc=accuracy_from_labels(y_val, y_val_pred),
                    val_mse=mse(y_val, y_val_cont),
                    val_mee=mee(y_val, y_val_cont),
                    test_acc=float("nan"),
                    test_mse=float("nan"),
                    test_mee=float("nan"),
                )
            )

        agg = _aggregate(per_seed)
        seconds = float(time.time() - t0)
        return per_seed, agg, seconds

    def _final_train_and_test(
        best_params: dict[str, Any],
    ) -> tuple[list[SeedMetrics], AggregatedMetrics, float, dict[int, Pipeline]]:
        """
        Dopo aver scelto gli iperparametri sul validation set:
        - retraining sull'intero training set
        - test sul test set mai visto.
        Ritorna i modelli per ciascun seed (nessuna selezione sul test).
        """
        per_seed: list[SeedMetrics] = []
        t0 = time.time()

        models_by_seed: dict[int, Pipeline] = {}

        for seed in SEEDS:
            model = _build_pipeline(kind, best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model.fit(X_train_full, y_train_full)

            core = model.named_steps["model"]
            if hasattr(core, "fit_status_") and getattr(core, "fit_status_") != 0:
                print(
                    f"    WARNING seed={seed}: libsvm stopped early (n_iter={getattr(core, 'n_iter_', '?')})",
                    flush=True,
                )

            y_dev_cont = _predict_continuous(model, X_train_full, kind=kind)
            y_test_cont = _predict_continuous(model, X_test, kind=kind)
            y_dev_pred = _predict_labels_from_continuous(y_dev_cont)
            y_test_pred = _predict_labels_from_continuous(y_test_cont)

            metrics_seed = SeedMetrics(
                seed=seed,
                train_acc=accuracy_from_labels(y_train_full, y_dev_pred),
                train_mse=mse(y_train_full, y_dev_cont),
                val_acc=float("nan"),
                val_mse=float("nan"),
                val_mee=float("nan"),
                test_acc=accuracy_from_labels(y_test, y_test_pred),
                test_mse=mse(y_test, y_test_cont),
                test_mee=mee(y_test, y_test_cont),
            )
            per_seed.append(metrics_seed)

            models_by_seed[seed] = model

        agg = _aggregate(per_seed)
        seconds = float(time.time() - t0)
        if not models_by_seed:
            raise RuntimeError("Final retraining non riuscito: nessun seed valutato.")
        return per_seed, agg, seconds, models_by_seed

    start_time = time.time()
    seen: set[str] = set()
    screen_rank: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    combo_idx = 0

    with results_path.open("w", encoding="utf-8") as out_f:
        # ----------------
        # STAGE 1: SCREENING
        # ----------------
        for i, params in enumerate(screen_params):
            print(f"  [screen {i + 1}/{len(screen_params)}] params={params}", flush=True)
            per_seed, agg, seconds = _eval_params_on_validation(params)

            print(
                "    mean val:  "
                f"acc={agg.val_acc_mean:.4f} MSE={agg.val_mse_mean:.6f} MEE={agg.val_mee_mean:.6f} "
                f"({seconds:.2f}s)",
                flush=True,
            )

            out_f.write(
                json.dumps(
                    {
                        "stage": "screen",
                        "task_id": task_id,
                        "model_kind": kind,
                        "combo_idx": combo_idx,
                        "params": params,
                        "eval_split": "val",
                        "seeds": SEEDS,
                        "per_seed": [asdict(m) for m in per_seed],
                        "agg": asdict(agg),
                        "seconds": seconds,
                    }
                )
                + "\n"
            )
            out_f.flush()

            sig = _sig(params)
            seen.add(sig)

            key = (agg.val_mse_mean, agg.val_mee_mean, -agg.val_acc_mean)
            screen_rank.append((key, params))

            if best_key is None or key < best_key:
                best_key = key
                print("    -> NEW BEST", flush=True)

                best_meta = {
                    "task_id": task_id,
                    "model_kind": kind,
                    "run_name": run_name,
                    "best_params": params,
                    "best_combo_idx": combo_idx,
                    "selection": {
                        "eval_split": "val",
                        "criterion": {
                            "min_val_mse_mean": agg.val_mse_mean,
                            "min_val_mee_mean": agg.val_mee_mean,
                            "max_val_acc_mean": agg.val_acc_mean,
                        },
                        "metrics": asdict(agg),
                        "per_seed": [asdict(m) for m in per_seed],
                    },
                }
                best_meta_ckpt_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

            combo_idx += 1

        # ----------------
        # STAGE 2: REFINEMENT
        # ----------------
        screen_rank.sort(key=lambda x: x[0])
        top_k = min(SCREEN_TOP_K, len(screen_rank))
        top_params = [p for _, p in screen_rank[:top_k]]

        refine_space = _refine_space(full_space, top_params)
        refine_params = [p for p in _get_param_grid(refine_space) if _sig(p) not in seen]

        if refine_params:
            print(f"  Refinement: {len(refine_params)} combinazioni (top_k={top_k})", flush=True)

        for i, params in enumerate(refine_params):
            print(f"  [refine {i + 1}/{len(refine_params)}] params={params}", flush=True)
            per_seed, agg, seconds = _eval_params_on_validation(params)

            print(
                "    mean val:  "
                f"acc={agg.val_acc_mean:.4f} MSE={agg.val_mse_mean:.6f} MEE={agg.val_mee_mean:.6f} "
                f"({seconds:.2f}s)",
                flush=True,
            )

            out_f.write(
                json.dumps(
                    {
                        "stage": "refine",
                        "task_id": task_id,
                        "model_kind": kind,
                        "combo_idx": combo_idx,
                        "params": params,
                        "eval_split": "val",
                        "seeds": SEEDS,
                        "per_seed": [asdict(m) for m in per_seed],
                        "agg": asdict(agg),
                        "seconds": seconds,
                    }
                )
                + "\n"
            )
            out_f.flush()

            key = (agg.val_mse_mean, agg.val_mee_mean, -agg.val_acc_mean)
            if best_key is None or key < best_key:
                best_key = key
                print("    -> NEW BEST", flush=True)

                best_meta = {
                    "task_id": task_id,
                    "model_kind": kind,
                    "run_name": run_name,
                    "best_params": params,
                    "best_combo_idx": combo_idx,
                    "selection": {
                        "eval_split": "val",
                        "criterion": {
                            "min_val_mse_mean": agg.val_mse_mean,
                            "min_val_mee_mean": agg.val_mee_mean,
                            "max_val_acc_mean": agg.val_acc_mean,
                        },
                        "metrics": asdict(agg),
                        "per_seed": [asdict(m) for m in per_seed],
                    },
                }
                best_meta_ckpt_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

            combo_idx += 1

    total_seconds = float(time.time() - start_time)

    if best_meta is None:
        raise RuntimeError("Nessuna combinazione valutata: controlla la griglia.")

    # ----
    # Final step: retraining su training set e test su test set per ogni seed.
    # ----
    print("  Final retraining su training set e valutazione su test (mai visto)...", flush=True)
    best_params = dict(best_meta["best_params"])
    final_per_seed, final_agg, final_seconds, final_models_by_seed = _final_train_and_test(
        best_params
    )
    best_meta["final"] = {
        "eval_split": "test",
        "metrics": asdict(final_agg),
        "per_seed": [asdict(m) for m in final_per_seed],
        "seconds": final_seconds,
    }

    # Salviamo i modelli di tutti i seed (nessuna selezione sul test).
    _ensure_dir(export_dir)
    _ensure_dir(models_dir)
    model_paths: dict[int, str] = {}
    for seed, model in final_models_by_seed.items():
        model_path = models_dir / f"model_seed_{seed}.joblib"
        joblib.dump(model, model_path)
        model_paths[seed] = str(model_path)

    best_meta["final"]["model_paths"] = model_paths
    summary_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

    # Report finale: migliore combinazione scelta sul validation set + risultati finali su test.
    sel = best_meta["selection"]["metrics"]
    fin = best_meta["final"]["metrics"]
    print(f"  BEST params (from val): {best_meta['best_params']}")
    print(
        "  selection mean train: "
        f"acc={sel['train_acc_mean']:.4f} "
        f"MSE={sel['train_mse_mean']:.6f}"
    )
    print(
        "  selection mean val:   "
        f"acc={sel['val_acc_mean']:.4f} "
        f"MSE={sel['val_mse_mean']:.6f} "
        f"MEE={sel['val_mee_mean']:.6f}"
    )
    print(
        "  final mean train:     "
        f"acc={fin['train_acc_mean']:.4f} "
        f"MSE={fin['train_mse_mean']:.6f}"
    )
    print(
        "  final mean test:      "
        f"acc={fin['test_acc_mean']:.4f} "
        f"MSE={fin['test_mse_mean']:.6f} "
        f"MEE={fin['test_mee_mean']:.6f}"
    )
    print(f"  saved models:         {models_dir}")

    return {
        "task_id": task_id,
        "model_kind": kind,
        "run_name": run_name,
        "results_jsonl": str(results_path),
        "summary_json": str(summary_path),
        "models_dir": str(models_dir),
        "total_seconds": total_seconds,
        "best": best_meta,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search SVM/SVR su MONK con hold-out multi-seed.")
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Quali task MONK eseguire (default: 1 2 3).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["svc_linear", "svc_rbf", "svc_poly"],
        choices=["svc_linear", "svc_rbf", "svc_poly", "svr_rbf"],
        help="Quali modelli eseguire (default: SVC linear/rbf/poly).",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disabilita logging TensorBoard.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    run_name = _now_run_id()
    tensorboard_enabled = not args.no_tensorboard

    def best_key(meta: dict[str, Any]) -> tuple[float, float, float]:
        # Confrontiamo i modelli usando SOLO il test finale (iperparametri scelti su validation).
        metrics = meta["final"]["metrics"]
        return float(metrics["test_mse_mean"]), float(metrics["test_mee_mean"]), -float(metrics["test_acc_mean"])

    def log_best_to_tensorboard(
        task_id: int,
        best_by_model: dict[str, dict[str, Any]],
        overall_best: dict[str, Any],
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = OUTPUT_ROOT / "tb_logs" / f"monk{task_id}" / run_name
        _ensure_dir(tb_dir)
        writer = SummaryWriter(log_dir=str(tb_dir))

        for model_kind, meta in best_by_model.items():
            sel = meta["selection"]["metrics"]
            fin = meta["final"]["metrics"]

            writer.add_scalar(f"{model_kind}/selection/train_acc", float(sel["train_acc_mean"]), 0)
            writer.add_scalar(f"{model_kind}/selection/train_mse", float(sel["train_mse_mean"]), 0)
            writer.add_scalar(f"{model_kind}/selection/val_acc", float(sel["val_acc_mean"]), 0)
            writer.add_scalar(f"{model_kind}/selection/val_mse", float(sel["val_mse_mean"]), 0)
            writer.add_scalar(f"{model_kind}/selection/val_mee", float(sel["val_mee_mean"]), 0)

            writer.add_scalar(f"{model_kind}/final/train_acc", float(fin["train_acc_mean"]), 0)
            writer.add_scalar(f"{model_kind}/final/train_mse", float(fin["train_mse_mean"]), 0)
            writer.add_scalar(f"{model_kind}/final/test_acc", float(fin["test_acc_mean"]), 0)
            writer.add_scalar(f"{model_kind}/final/test_mse", float(fin["test_mse_mean"]), 0)
            writer.add_scalar(f"{model_kind}/final/test_mee", float(fin["test_mee_mean"]), 0)
            writer.add_text(f"{model_kind}/params", json.dumps(meta["best_params"]), 0)

        writer.add_text("overall/model_kind", str(overall_best["model_kind"]), 0)
        writer.add_text("overall/params", json.dumps(overall_best["best_params"]), 0)
        om = overall_best["final"]["metrics"]
        writer.add_scalar("overall/test_acc", float(om["test_acc_mean"]), 0)
        writer.add_scalar("overall/test_mse", float(om["test_mse_mean"]), 0)
        writer.add_scalar("overall/test_mee", float(om["test_mee_mean"]), 0)

        writer.close()

    for task_id in args.tasks:
        best_by_model: dict[str, dict[str, Any]] = {}

        for model_kind in args.models:
            print(f"\n[MONK {task_id}] {model_kind} | run={run_name}")
            info = run_grid_search(
                task_id=task_id,
                kind=model_kind,  # type: ignore[arg-type]
                run_name=run_name,
            )
            best_by_model[model_kind] = info["best"]

            print(
                "  -> salvati:",
                f"results={info['results_jsonl']}",
                f"summary={info['summary_json']}",
                f"models_dir={info['models_dir']}",
                sep="\n     ",
            )

        overall_best = min(best_by_model.values(), key=best_key)
        bm = overall_best["final"]["metrics"]
        print(f"\n[MONK {task_id}] OVERALL BEST = {overall_best['model_kind']} params={overall_best['best_params']}")
        print(
            f"  mean test: acc={float(bm['test_acc_mean']):.4f} "
            f"MSE={float(bm['test_mse_mean']):.6f} "
            f"MEE={float(bm['test_mee_mean']):.6f}"
        )

        if tensorboard_enabled:
            log_best_to_tensorboard(task_id, best_by_model, overall_best)
            print(f"  TensorBoard best logs: {OUTPUT_ROOT / 'tb_logs' / f'monk{task_id}' / run_name}")


if __name__ == "__main__":
    main()
