"""
autogluon_code.py
=================
AutoGluon integration for the Auto-Unrolled PGD experiment.

Two distinct roles:
  1. Bayesian HPO outer-loop  — searches over (num_layers, init_eta, optimizer,
                                 lr_scheduler) to find the best Auto-PGD config.
  2. TabularPredictor baseline — supervised black-box AutoML model trained on ZF
                                 labels; demonstrates the contrast between
                                 "supervised AutoML" and "Auto-PGD".
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim

# ── AutoGluon imports ───────────────────────────────────────────────────────
# autogluon.core.space was removed in AutoGluon ≥ 1.0; define lightweight
# replacements that preserve the .lower/.upper/.log/.data interface used by
# _sample_config() for the random-search fallback.
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Int:
    lower: int
    upper: int


@dataclass
class Real:
    lower: float
    upper: float
    log: bool = False


@dataclass
class Categorical:
    data: tuple

    def __init__(self, *choices):
        self.data = tuple(choices)


from autogluon.tabular import TabularPredictor

from data_generator import (
    compute_sum_rate,
    generate_channel_dataset,
    generate_6g_miso_dataset,
    prepare_autogluon_tabular_data,
    project_frobenius_ball,
    zero_forcing_beamforming,
)
from unrolled import DeepUnrolledNetwork


# ---------------------------------------------------------------------------
# Search space definition (used by both HPO and paper documentation)
# ---------------------------------------------------------------------------

search_space: dict[str, Any] = {
    "num_layers":   Int(lower=3, upper=25),
    "init_eta":     Real(lower=1e-4, upper=1e-1, log=True),
    "optimizer":    Categorical("adam", "sgd"),
    "lr_scheduler": Categorical("cosine", "step"),
    "layer_type":   Categorical("pgd", "custom"),
    "activation":   Categorical("identity", "relu", "tanh"),
}


# ---------------------------------------------------------------------------
# HPO objective function
# ---------------------------------------------------------------------------

def objective_fn(
    config: dict[str, Any],
    H_train: torch.Tensor,
    H_val: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    p_max: float = 1.0,
    noise_var: float = 0.1,
) -> float:
    """Train one Auto-PGD configuration; return validation sum-rate.

    Args:
        config    : dict with keys num_layers, init_eta, optimizer, lr_scheduler
        H_train   : (N_train, K, M) complex — training channels
        H_val     : (N_val,   K, M) complex — validation channels
        device    : torch.device
        epochs    : number of training epochs
        batch_size: mini-batch size

    Returns:
        val_sum_rate : higher is better (bits/s/Hz)
    """
    K_users = H_train.shape[1]
    M       = H_train.shape[2]

    model = DeepUnrolledNetwork(
        num_layers=int(config["num_layers"]),
        init_eta=float(config["init_eta"]),
        p_max=p_max,
        noise_var=noise_var,
        M=M,
        K_users=K_users,
        layer_type=config.get("layer_type", "pgd"),
        activation=config.get("activation", "identity"),
    ).to(device)

    # Optimizer
    if config["optimizer"] == "adam":
        opt = optim.Adam(model.parameters(), lr=1e-3)
    else:
        opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # Scheduler
    if config["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=epochs // 3, gamma=0.5)

    H_train_dev = H_train.to(device)
    N = H_train_dev.shape[0]

    for _ in range(epochs):
        # Shuffle
        perm = torch.randperm(N)
        H_shuf = H_train_dev[perm]

        for start in range(0, N, batch_size):
            H_batch = H_shuf[start : start + batch_size]
            opt.zero_grad()
            loss = model.compute_loss(H_batch)
            loss.backward()
            opt.step()

        scheduler.step()

    # Validation — use ZF init for consistency with training compute_loss
    model.eval()
    H_val_dev = H_val.to(device)
    W_init_val = zero_forcing_beamforming(H_val_dev, p_max=p_max)

    with torch.no_grad():
        W_out = model(W_init_val, H_val_dev)
    val_sum_rate = compute_sum_rate(W_out, H_val_dev, noise_var)
    return val_sum_rate


# ---------------------------------------------------------------------------
# HPO outer-loop  (Bayesian via optuna if available, else random search)
# ---------------------------------------------------------------------------

def _sample_config(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sample one configuration from the search space."""
    cfg: dict[str, Any] = {}
    for key, sp in space.items():
        if isinstance(sp, Int):
            cfg[key] = int(rng.integers(sp.lower, sp.upper + 1))
        elif isinstance(sp, Real):
            if getattr(sp, "log", False):
                log_lo, log_hi = np.log(sp.lower), np.log(sp.upper)
                cfg[key] = float(np.exp(rng.uniform(log_lo, log_hi)))
            else:
                cfg[key] = float(rng.uniform(sp.lower, sp.upper))
        elif isinstance(sp, Categorical):
            cfg[key] = rng.choice(sp.data)
        else:
            cfg[key] = sp  # scalar fallback
    return cfg


def run_autogluon_hpo(
    H_train: torch.Tensor,
    H_val: torch.Tensor,
    num_trials: int = 20,
    time_limit: float = 3600.0,
    device: torch.device | None = None,
    epochs_per_trial: int = 50,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    seed: int = 0,
) -> tuple[dict[str, Any], float, list[dict]]:
    """Bayesian / random HPO over the Auto-PGD search space.

    Uses ``optuna`` for Bayesian optimisation when available; otherwise falls
    back to reproducible random search.

    Args:
        H_train       : (N_train, K, M) complex
        H_val         : (N_val,   K, M) complex
        num_trials    : maximum number of HPO trials
        time_limit    : wall-clock budget in seconds
        device        : torch.device (auto-detected if None)
        epochs_per_trial: training epochs per trial

    Returns:
        best_config      : dict of best hyperparameters
        best_val_rate    : validation sum-rate of best config
        hpo_history      : list of {trial_id, config, val_sum_rate, elapsed_s}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hpo_history: list[dict] = []
    best_config: dict[str, Any] = {}
    best_val_rate = -np.inf
    t0 = time.time()

    # Try Optuna first
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _optuna_objective(trial: "optuna.Trial") -> float:
            cfg = {
                "num_layers":   trial.suggest_int("num_layers", 3, 25),
                "init_eta":     trial.suggest_float("init_eta", 1e-4, 1e-1, log=True),
                "optimizer":    trial.suggest_categorical("optimizer", ["adam", "sgd"]),
                "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["cosine", "step"]),
            }
            return objective_fn(
                cfg, H_train, H_val, device,
                epochs=epochs_per_trial, p_max=p_max, noise_var=noise_var,
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(
            _optuna_objective,
            n_trials=num_trials,
            timeout=time_limit,
            show_progress_bar=False,
        )

        for t in study.trials:
            hpo_history.append({
                "trial_id":     t.number,
                "config":       t.params,
                "val_sum_rate": t.value if t.value is not None else float("nan"),
                "elapsed_s":    (t.datetime_complete - t.datetime_start).total_seconds()
                                if t.datetime_complete else 0.0,
            })

        best_config = study.best_params
        best_val_rate = study.best_value

    except ImportError:
        # Fallback: random search
        print("[HPO] optuna not found — using random search.")
        rng = np.random.default_rng(seed)
        for trial_id in range(num_trials):
            if time.time() - t0 > time_limit:
                print(f"[HPO] Time limit reached after {trial_id} trials.")
                break
            cfg = _sample_config(search_space, rng)
            t_start = time.time()
            val = objective_fn(
                cfg, H_train, H_val, device,
                epochs=epochs_per_trial, p_max=p_max, noise_var=noise_var,
            )
            elapsed = time.time() - t_start
            hpo_history.append({
                "trial_id":     trial_id,
                "config":       cfg,
                "val_sum_rate": val,
                "elapsed_s":    elapsed,
            })
            if val > best_val_rate:
                best_val_rate = val
                best_config = cfg
            print(
                f"[HPO] Trial {trial_id:3d}  val_rate={val:.4f}  "
                f"best={best_val_rate:.4f}  layers={cfg['num_layers']}"
            )

    return best_config, best_val_rate, hpo_history


# ---------------------------------------------------------------------------
# AutoGluon TabularPredictor supervised baseline
# ---------------------------------------------------------------------------

def decode_tabular_prediction(
    pred_array: np.ndarray,
    M: int = 8,
    K: int = 4,
    p_max: float = 1.0,
) -> torch.Tensor:
    """Reshape flat TabularPredictor output → power-feasible complex W.

    Args:
        pred_array : (N, 2*M*K) float numpy array
        M, K       : antenna / user counts
        p_max      : transmit power budget

    Returns:
        W : (N, M, K) complex tensor, Frobenius-projected
    """
    N = pred_array.shape[0]
    half = M * K
    W_real = torch.tensor(pred_array[:, :half], dtype=torch.float32).reshape(N, M, K)
    W_imag = torch.tensor(pred_array[:, half:], dtype=torch.float32).reshape(N, M, K)
    W = torch.complex(W_real, W_imag)
    return project_frobenius_ball(W, p_max)


def run_tabular_baseline(
    N_train: int = 5000,
    N_test: int = 1000,
    M: int = 8,
    K: int = 4,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    time_limit: int = 300,
    save_dir: str = "results",
) -> tuple[float, object]:
    """Train an AutoGluon TabularPredictor on ZF-labeled data; evaluate sum-rate.

    The ZF beamformer provides free analytical labels for supervised training.
    This demonstrates the AutoML-supervised paradigm as a contrast to Auto-PGD.

    Returns:
        tabular_sum_rate : float — mean test sum-rate (bits/s/Hz)
        leaderboard      : pd.DataFrame — AutoGluon model leaderboard
    """
    import pandas as pd

    print(f"[Tabular] Generating ZF-labeled dataset  N_train={N_train}, N_test={N_test}")
    feat_train, lab_train = prepare_autogluon_tabular_data(
        N=N_train, M=M, K=K, p_max=p_max, noise_var=noise_var, seed=42
    )
    feat_test, lab_test = prepare_autogluon_tabular_data(
        N=N_test, M=M, K=K, p_max=p_max, noise_var=noise_var, seed=999
    )

    # AutoGluon needs a single DataFrame with feature + label columns
    # Multi-target regression: one predictor per output column
    lab_cols = list(lab_train.columns)
    train_df = pd.concat([feat_train, lab_train], axis=1)
    test_df  = pd.concat([feat_test,  lab_test],  axis=1)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "tabular_predictor")

    # Fit one predictor per label column (AutoGluon multi-label)
    # For speed we use a single multi-output wrapper approach: one predictor
    # for the entire flattened label vector using a custom eval metric.
    # Simpler: predict each column independently via multi-label.
    from autogluon.tabular import TabularPredictor

    predictions: list[np.ndarray] = []
    leaderboard = None

    for col in lab_cols:
        combined_df = pd.concat([feat_train, lab_train[[col]]], axis=1)
        predictor = TabularPredictor(
            label=col,
            problem_type="regression",
            eval_metric="r2",
            path=os.path.join(model_path, col),
            verbosity=0,
        )
        predictor.fit(combined_df, time_limit=time_limit // len(lab_cols), presets="medium_quality")
        preds = predictor.predict(feat_test).to_numpy()
        predictions.append(preds)
        if leaderboard is None:
            leaderboard = predictor.leaderboard(silent=True)

    pred_array = np.stack(predictions, axis=1)  # (N_test, 2*M*K)
    W_pred = decode_tabular_prediction(pred_array, M=M, K=K, p_max=p_max)

    # Reconstruct H_test from the stored features
    H_test, _ = generate_channel_dataset(N=N_test, M=M, K=K, p_max=p_max, noise_var=noise_var, seed=999)
    tabular_sum_rate = compute_sum_rate(W_pred, H_test, noise_var)

    print(f"[Tabular] Test sum-rate = {tabular_sum_rate:.4f} bits/s/Hz")
    return tabular_sum_rate, leaderboard


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    M, K = 8, 4
    H_tr, _ = generate_channel_dataset(N=200, M=M, K=K, seed=42)
    H_va, _ = generate_channel_dataset(N=100, M=M, K=K, seed=123)

    print("[HPO] Running 3-trial smoke test …")
    best_cfg, best_rate, history = run_autogluon_hpo(
        H_tr, H_va,
        num_trials=3,
        time_limit=300,
        device=device,
        epochs_per_trial=5,
    )
    print(f"Best config     : {best_cfg}")
    print(f"Best val rate   : {best_rate:.4f} bits/s/Hz")
    print(f"HPO trials done : {len(history)}")
    print("autogluon_code.py  OK")
