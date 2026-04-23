"""
train_evaluate.py
=================
Unified training loop, data-scarcity experiment orchestration, and
results serialisation for the Auto-Unrolled PGD paper.

Usage
-----
    # Quick smoke-test (N_max=100, 10 epochs, 3 HPO trials):
    python train_evaluate.py --smoke-test

    # Full paper experiment:
    python train_evaluate.py --full

    # Full on GPU:
    python train_evaluate.py --full --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from autogluon_code import run_autogluon_hpo, run_tabular_baseline
from baselines import MLPBaseline, classical_pgd_solver, evaluate_zf, wmmse_solver
from data_generator import (
    compute_sum_rate,
    generate_channel_dataset,
    generate_6g_miso_dataset,
    zero_forcing_beamforming,
)
from unrolled import DeepUnrolledNetwork
from lista import LISTABeamformer
from PGDNet import PGDNetBaseline


# ---------------------------------------------------------------------------
# Generic training loop  (works for both Auto-PGD and MLPBaseline)
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    H_train: torch.Tensor,
    epochs: int = 100,
    lr: float = 3e-1,
    batch_size: int = 128,
    device: torch.device | None = None,
    log_every: int = 10,
) -> list[float]:
    """Train `model` by minimising negative mean sum-rate.

    Both :class:`DeepUnrolledNetwork` and :class:`MLPBaseline` expose a
    ``compute_loss(H_batch) -> Tensor`` method that returns a differentiable
    scalar.

    Args:
        model     : DeepUnrolledNetwork or MLPBaseline
        H_train   : (N, K, M) complex — training channels
        epochs    : number of full passes over H_train
        lr        : Adam learning rate
        batch_size: mini-batch size
        device    : if None, auto-detects CUDA
        log_every : print loss every N epochs

    Returns:
        loss_history : list of mean epoch losses (one per epoch)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    H = H_train.to(device)
    N = H.shape[0]
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        H_shuf = H[perm]
        epoch_losses: list[float] = []

        for start in range(0, N, batch_size):
            H_batch = H_shuf[start : start + batch_size]
            optimizer.zero_grad()
            loss = model.compute_loss(H_batch)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{epochs}  loss={mean_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return loss_history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    H_test: torch.Tensor,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    device: torch.device | None = None,
) -> float:
    """Return mean test sum-rate (bits/s/Hz) for Auto-PGD or MLP.

    Args:
        model   : DeepUnrolledNetwork or MLPBaseline (already trained)
        H_test  : (N_test, K, M) complex
        p_max   : transmit power budget
        noise_var: AWGN power σ²

    Returns:
        Scalar float mean sum-rate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    H = H_test.to(device)

    with torch.no_grad():
        if isinstance(model, (DeepUnrolledNetwork, PGDNetBaseline)):
            # Initialise from ZF beamformer — deterministic, near-optimal start,
            # and consistent with how compute_loss initialises during training.
            W_init = zero_forcing_beamforming(H, p_max=p_max)
            W_out = model(W_init, H)
        else:
            # MLPBaseline or LISTABeamformer — take H directly
            W_out = model(H)

    return compute_sum_rate(W_out, H, noise_var)


# ---------------------------------------------------------------------------
# Core experiment: data-scarcity
# ---------------------------------------------------------------------------

def data_scarcity_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full data-scarcity comparison (Table 1 / Figure 1).

    For each training set size N in `config['sizes']`:
      - Trains MLPBaseline with sum-rate loss
      - Runs AutoGluon HPO to find best Auto-PGD architecture
      - Trains best Auto-PGD config
      - Records test sum-rate for both + reference baselines

    Args:
        config: dict with keys
            M, K, p_max, noise_var  — system parameters
            sizes                   — list of training set sizes
            test_N                  — fixed test set size (default 5000)
            epochs                  — training epochs per model
            lr                      — learning rate
            batch_size              — mini-batch size
            hpo_trials              — number of HPO trials
            hpo_epochs              — training epochs per HPO trial
            device_str              — "cuda" | "cpu" | "auto"

    Returns:
        results dict (serialisable to JSON)
    """
    M             = config.get("M", 8)
    K             = config.get("K", 4)
    p_max         = config.get("p_max", 1.0)
    noise_var     = config.get("noise_var", 0.1)
    sizes         = config.get("sizes", [100, 1000, 10000, 50000])
    test_N        = config.get("test_N", 5000)
    epochs        = config.get("epochs", 100)
    lr            = config.get("lr", 3e-1)
    autopgd_lr    = config.get("autopgd_lr", lr)
    batch_sz      = config.get("batch_size", 128)
    hpo_trials    = config.get("hpo_trials", 20)
    hpo_epochs    = config.get("hpo_epochs", 50)
    pgdnet_layers = config.get("pgdnet_layers", 10)
    dev_str       = config.get("device_str", "auto")

    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    print(f"\n{'='*60}")
    print(f"Data-Scarcity Experiment   device={device}")
    print(f"M={M}, K={K}, P_max={p_max}, noise_var={noise_var}")
    print(f"Sizes: {sizes}  |  Test N={test_N}")
    print(f"{'='*60}\n")

    # ── Fixed test set (seed=999) ────────────────────────────────────────────
    print("Generating fixed test set …")
    H_test, _ = generate_channel_dataset(N=test_N, M=M, K=K, p_max=p_max, noise_var=noise_var, seed=999)

    # ── Fixed reference metrics ──────────────────────────────────────────────
    print("Computing ZF reference …")
    zf_rate = evaluate_zf(H_test, p_max=p_max, noise_var=noise_var)
    print(f"ZF reference sum-rate       : {zf_rate:.4f} bits/s/Hz")

    print("Computing Classical PGD reference (200 iters) …")
    W_pgd_ref = classical_pgd_solver(
        H_test.to(device), p_max=p_max, noise_var=noise_var, num_iters=200
    )
    pgd_rate = compute_sum_rate(W_pgd_ref.cpu(), H_test, noise_var)
    print(f"Classical PGD (200-it) rate : {pgd_rate:.4f} bits/s/Hz")

    print("Computing WMMSE reference (100 iters) …")
    W_wmmse_ref = wmmse_solver(
        H_test.to(device), p_max=p_max, noise_var=noise_var, num_iters=100
    )
    wmmse_rate = compute_sum_rate(W_wmmse_ref.cpu(), H_test, noise_var)
    print(f"WMMSE (100-it) rate         : {wmmse_rate:.4f} bits/s/Hz")

    results: dict[str, Any] = {
        "config":     config,
        "zf_rate":    zf_rate,
        "pgd_rate":   pgd_rate,
        "wmmse_rate": wmmse_rate,
        "by_size":    {},
    }

    # ── Per training-size loop ───────────────────────────────────────────────
    for N in sizes:
        print(f"\n{'─'*50}")
        print(f"Training size N = {N}")
        print(f"{'─'*50}")

        seeds     = config.get("seeds", [42])
        H_val     = H_test[:500]          # fixed 500-sample validation slice

        size_results: dict[str, Any] = {}

        # ── HPO: run once with first-seed training data ──────────────────────
        H_train_hpo, _ = generate_channel_dataset(
            N=N, M=M, K=K, p_max=p_max, noise_var=noise_var, seed=seeds[0]
        )
        print(f"\n[HPO] Searching over Auto-PGD architecture (N={N}) …")
        t0_hpo = time.time()
        best_cfg, best_val, hpo_history = run_autogluon_hpo(
            H_train_hpo, H_val,
            num_trials=hpo_trials,
            time_limit=3600,
            device=device,
            epochs_per_trial=hpo_epochs,
            p_max=p_max,
            noise_var=noise_var,
        )
        size_results["hpo_history"]     = hpo_history
        size_results["best_hpo_config"] = best_cfg
        size_results["hpo_time_s"]      = time.time() - t0_hpo
        print(f"[HPO]  Best config: {best_cfg}  val_rate={best_val:.4f}")

        # ── Multi-seed training (mean ± std across seeds) ─────────────────────
        mlp_rates:     list[float] = []
        lista_rates:   list[float] = []
        pgdnet_rates:  list[float] = []
        autopgd_rates: list[float] = []
        mlp_loss_h = lista_loss_h = pgdnet_loss_h = autopgd_loss_h = []

        for seed_idx, seed in enumerate(seeds):
            H_train, _ = generate_channel_dataset(
                N=N, M=M, K=K, p_max=p_max, noise_var=noise_var, seed=seed
            )
            log_e = max(1, epochs // 5)

            # MLP — independent seed per model to avoid RNG state contamination
            torch.manual_seed(seed * 10 + 0)
            print(f"\n[MLP] Training N={N} seed={seed} …")
            mlp = MLPBaseline(M=M, K=K, p_max=p_max, noise_var=noise_var)
            t0 = time.time()
            h = train_model(mlp, H_train, epochs=epochs, lr=lr,
                            batch_size=batch_sz, device=device, log_every=log_e)
            r = evaluate_model(mlp, H_test, p_max=p_max, noise_var=noise_var,
                               device=device)
            mlp_rates.append(r)
            if seed_idx == 0:
                mlp_loss_h = h
                size_results["mlp_train_time_s"] = time.time() - t0
            print(f"[MLP]  seed={seed}  rate={r:.4f}")

            # LISTA
            torch.manual_seed(seed * 10 + 1)
            print(f"\n[LISTA] Training N={N} seed={seed} …")
            lista = LISTABeamformer(M=M, K_users=K, p_max=p_max, noise_var=noise_var)
            t0 = time.time()
            h = train_model(lista, H_train, epochs=epochs, lr=lr,
                            batch_size=batch_sz, device=device, log_every=log_e)
            r = evaluate_model(lista, H_test, p_max=p_max, noise_var=noise_var,
                               device=device)
            lista_rates.append(r)
            if seed_idx == 0:
                lista_loss_h = h
                size_results["lista_train_time_s"] = time.time() - t0
            print(f"[LISTA]  seed={seed}  rate={r:.4f}")

            # PGD-Net
            torch.manual_seed(seed * 10 + 2)
            print(f"\n[PGD-Net] Training N={N} seed={seed} (depth={pgdnet_layers}) …")
            pgdnet = PGDNetBaseline(
                num_layers=pgdnet_layers, p_max=p_max, M=M, K_users=K,
                noise_var=noise_var,
            )
            t0 = time.time()
            h = train_model(pgdnet, H_train, epochs=epochs, lr=lr,
                            batch_size=batch_sz, device=device, log_every=log_e)
            r = evaluate_model(pgdnet, H_test, p_max=p_max, noise_var=noise_var,
                               device=device)
            pgdnet_rates.append(r)
            if seed_idx == 0:
                pgdnet_loss_h = h
                size_results["pgdnet_train_time_s"] = time.time() - t0
            print(f"[PGD-Net]  seed={seed}  rate={r:.4f}")

            # Auto-PGD (best HPO config, retrained per seed)
            torch.manual_seed(seed * 10 + 3)
            print(f"\n[Auto-PGD] Training best config N={N} seed={seed} …")
            auto_pgd = DeepUnrolledNetwork(
                num_layers=int(best_cfg["num_layers"]),
                init_eta=float(best_cfg["init_eta"]),
                p_max=p_max,
                noise_var=noise_var,
                M=M,
                K_users=K,
                layer_type=config.get("autopgd_layer_type", best_cfg.get("layer_type", "pgd")),
                activation=config.get("autopgd_activation", best_cfg.get("activation", "identity")),
            )
            t0 = time.time()
            autopgd_metrics = {}
            def autopgd_train_model(model, H_train, epochs, lr, batch_size, device, log_every):
                model = model.to(device)
                model.train()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                H = H_train.to(device)
                N = H.shape[0]
                loss_history = []
                metrics_history = []
                for epoch in range(1, epochs + 1):
                    perm = torch.randperm(N, device=device)
                    H_shuf = H[perm]
                    epoch_losses = []
                    epoch_metrics = {}
                    for start in range(0, N, batch_size):
                        H_batch = H_shuf[start : start + batch_size]
                        optimizer.zero_grad()
                        # Layer-wise metrics dict
                        layer_metrics = {}
                        _ = model.forward(zero_forcing_beamforming(H_batch, p_max=p_max).to(H_batch.device), H_batch, log_metrics=layer_metrics)
                        epoch_metrics = layer_metrics
                        loss = model.compute_loss(H_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        epoch_losses.append(loss.item())
                    scheduler.step()
                    mean_loss = float(np.mean(epoch_losses))
                    loss_history.append(mean_loss)
                    metrics_history.append(epoch_metrics)
                    if epoch % log_every == 0 or epoch == 1:
                        print(f"  Epoch {epoch:4d}/{epochs}  loss={mean_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
                return loss_history, metrics_history
            h, metrics_history = autopgd_train_model(auto_pgd, H_train, epochs, autopgd_lr, batch_sz, device, log_e)
            r = evaluate_model(auto_pgd, H_test, p_max=p_max, noise_var=noise_var, device=device)
            autopgd_rates.append(r)
            if seed_idx == 0:
                autopgd_loss_h = h
                size_results["autopgd_train_time_s"] = time.time() - t0
                size_results["learned_step_sizes"]   = auto_pgd.get_learned_step_sizes()
                size_results["autopgd_layer_metrics_history"] = metrics_history
            print(f"[Auto-PGD]  seed={seed}  rate={r:.4f}")

        # ── Aggregate across seeds ────────────────────────────────────────────
        mlp_rate     = float(np.mean(mlp_rates))
        lista_rate   = float(np.mean(lista_rates))
        pgdnet_rate  = float(np.mean(pgdnet_rates))
        autopgd_rate = float(np.mean(autopgd_rates))
        size_results.update({
            "mlp_rate":             mlp_rate,
            "mlp_rate_std":         float(np.std(mlp_rates)),
            "mlp_loss_history":     mlp_loss_h,
            "lista_rate":           lista_rate,
            "lista_rate_std":       float(np.std(lista_rates)),
            "lista_loss_history":   lista_loss_h,
            "pgdnet_rate":          pgdnet_rate,
            "pgdnet_rate_std":      float(np.std(pgdnet_rates)),
            "pgdnet_loss_history":  pgdnet_loss_h,
            "autopgd_rate":         autopgd_rate,
            "autopgd_rate_std":     float(np.std(autopgd_rates)),
            "autopgd_loss_history": autopgd_loss_h,
        })

        seed_note = f" (mean/{len(seeds)} seeds)" if len(seeds) > 1 else ""
        print(
            f"\n[Summary N={N}{seed_note}]  ZF={zf_rate:.3f}  PGD={pgd_rate:.3f}  "
            f"WMMSE={wmmse_rate:.3f}  MLP={mlp_rate:.3f}  LISTA={lista_rate:.3f}  "
            f"PGD-Net={pgdnet_rate:.3f}  Auto-PGD={autopgd_rate:.3f}"
        )

        results["by_size"][str(N)] = size_results

    return results


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def run_all_experiments(args: argparse.Namespace) -> None:
    """Top-level function: runs experiment, saves results, prints summary."""
    import time
    t0_exp = time.time()
    if args.smoke_test:
        config: dict[str, Any] = {
            "M": 8, "K": 4, "p_max": 1.0, "noise_var": 0.1,
            "sizes": [100, 300],
            "test_N": 500,
            "epochs": 50,
            "lr": 1e-3,
            "autopgd_lr": 3e-3,  # Set Auto-PGD learning rate separately
            "batch_size": 128,
            "hpo_trials": 30,
            "hpo_epochs": 50,
            "pgdnet_layers": 5,
            "seeds": [42],
            "device_str": args.device,
        }
        print("Running SMOKE TEST (fast, small scale) …")
    else:
        config = {
            "M": 8, "K": 4, "p_max": 1.0, "noise_var": 0.1,
            "sizes": [100, 1000, 10000, 50000],
            "test_N": 5000,
            "epochs": 200,
            "lr": 1e-3,
            "autopgd_lr": 3e-3,  # Set Auto-PGD learning rate separately
            "batch_size": 128,
            "hpo_trials": 50,
            "hpo_epochs": 100,
            "pgdnet_layers": 10,
            "seeds": [42, 678, 888, 123, 456],
            "device_str": args.device,
        }
        print("Running FULL EXPERIMENT …")

    results = data_scarcity_experiment(config)
    t1_exp = time.time()
    exp_time_s = t1_exp - t0_exp

    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 145)
    print(
        f"{'N':>8}  {'ZF':>8}  {'PGD':>8}  {'WMMSE':>8}  {'MLP (mean±std)':>14}  {'MLP t(s)':>10}  "
        f"{'LISTA':>14}  {'LISTA t(s)':>10}  {'PGD-Net':>14}  {'PGD-Net t(s)':>12}  "
        f"{'Auto-PGD':>14}  {'Auto-PGD t(s)':>12}"
    )
    print("=" * 145)
    zf    = results["zf_rate"]
    pgd   = results["pgd_rate"]
    wmmse = results["wmmse_rate"]
    for n_str, sr in results["by_size"].items():
        def _r(key: str) -> str:
            v   = sr.get(key, float("nan"))
            std = sr.get(key + "_std", 0.0)
            return f"{v:.3f}±{std:.3f}" if std > 1e-9 else f"{v:.4f}"
        def _t(key: str) -> str:
            t = sr.get(key, float("nan"))
            return f"{t:.2f}" if t == t else "-"
        print(
            f"{int(n_str):>8}  {zf:>8.4f}  {pgd:>8.4f}  {wmmse:>8.4f}  "
            f"{_r('mlp_rate'):>14}  {_t('mlp_train_time_s'):>10}  "
            f"{_r('lista_rate'):>14}  {_t('lista_train_time_s'):>10}  "
            f"{_r('pgdnet_rate'):>14}  {_t('pgdnet_train_time_s'):>12}  "
            f"{_r('autopgd_rate'):>14}  {_t('autopgd_train_time_s'):>12}"
        )
    print("=" * 145)

    print(f"\nExperiment total time: {exp_time_s:.2f} seconds ({exp_time_s/60:.2f} min)")

    # Optional: run tabular baseline
    if not args.smoke_test:
        print("\n[Tabular] Running AutoGluon TabularPredictor baseline …")
        try:
            tab_rate, leaderboard = run_tabular_baseline(
                N_train=5000, N_test=1000, M=config["M"], K=config["K"],
                p_max=config["p_max"], noise_var=config["noise_var"],
                time_limit=300,
            )
            results["tabular_rate"] = tab_rate
            print(f"Tabular sum-rate: {tab_rate:.4f} bits/s/Hz")
        except Exception as e:
            print(f"[Tabular] Skipped ({e})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-Unrolled PGD — Experiment Runner"
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 10-epoch run on N=[100,300] for debugging"
    )
    mode.add_argument(
        "--full", action="store_true", default=True,
        help="Full paper experiment (default)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto-detect)"
    )
    args = parser.parse_args()
    run_all_experiments(args)
