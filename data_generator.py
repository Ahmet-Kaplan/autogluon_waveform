"""
data_generator.py
=================
Data generation utilities for the Auto-Unrolled PGD experiment.

Tensor shape conventions (throughout the entire codebase):
  H  : (B, K, M)  complex  — channel matrix, K users, M antennas
  W  : (B, M, K)  complex  — beamforming matrix
  B  : batch size
  M  : number of BS antennas   (default 8)
  K  : number of single-antenna users (default 4)
"""

import torch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_frobenius_ball(Z: torch.Tensor, p_max: float) -> torch.Tensor:
    """Project Z onto the Frobenius power ball ||W||_F^2 <= p_max.

    Per the paper (Eq. 6):  proj(Z) = sqrt(p_max) / max(||Z||_F, sqrt(p_max)) * Z
    Works on complex tensors; operates over the last two dims (M, K).
    """
    fro = torch.norm(Z.reshape(Z.shape[0], -1), dim=1)          # (B,) real
    scale = np.sqrt(p_max) / torch.clamp(fro, min=np.sqrt(p_max))  # (B,) real
    return Z * scale.view(-1, *([1] * (Z.dim() - 1)))


# ---------------------------------------------------------------------------
# Existing functions (bug-fixed)
# ---------------------------------------------------------------------------

def generate_6g_miso_dataset(
    num_samples: int = 1000,
    num_antennas: int = 8,
    num_users: int = 4,
    p_max: float = 1.0,
    noise_var: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Rayleigh fading channels and power-constrained initial W.

    Returns:
        channels : (num_samples, num_users, num_antennas) complex
        w_init   : (num_samples, num_antennas, num_users) complex, ||W||_F^2 <= p_max
    """
    h_real = torch.randn(num_samples, num_users, num_antennas) / np.sqrt(2)
    h_imag = torch.randn(num_samples, num_users, num_antennas) / np.sqrt(2)
    channels = torch.complex(h_real, h_imag)

    w_real = torch.randn(num_samples, num_antennas, num_users) / np.sqrt(2)
    w_imag = torch.randn(num_samples, num_antennas, num_users) / np.sqrt(2)
    w_init = torch.complex(w_real, w_imag)

    # Strict Frobenius ball projection (fix: was only a lower-clamp, not a projection)
    w_init = project_frobenius_ball(w_init, p_max)

    return channels, w_init


def compute_sum_rate_gradient(
    W: torch.Tensor,
    H: torch.Tensor,
    noise_var: float = 0.1,
    create_graph: bool = False,
) -> torch.Tensor:
    """Return ∇_W (−sum_rate).  Safe to call inside a larger computation graph.

    Bug fix: no longer mutates the input W via requires_grad_(True).
    Uses detach+clone so the outer graph is never corrupted.

    Args:
        W            : (B, M, K) complex  — must NOT already require grad
        H            : (B, K, M) complex
        noise_var    : AWGN power σ²
        create_graph : keep second-order graph (needed if eta ∈ outer graph)

    Returns:
        grad : (B, M, K) complex — ∇_W (−sum_rate), detached unless create_graph=True
    """
    # torch.enable_grad() ensures a local computation graph is built even
    # when called from a torch.no_grad() context (e.g. model eval inference).
    with torch.enable_grad():
        W_var = W.detach().clone().requires_grad_(True)

        h_w = torch.bmm(H, W_var)                                       # (B, K, K)
        signal = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2   # (B, K)
        interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal  # (B, K)
        sinr = signal / (interference + noise_var)
        # Use .sum() not .mean() so the gradient for each W[i] equals the
        # per-sample gradient ∂(-rate_i)/∂W[i], independent of batch size B.
        # With .mean() the gradient was scaled by 1/B, causing a 78× step-size
        # mismatch between training (B=64) and evaluation (B=5000).
        loss = -torch.log2(1.0 + sinr).sum()

        (grad,) = torch.autograd.grad(loss, W_var, create_graph=create_graph)
    return grad


# ---------------------------------------------------------------------------
# New functions
# ---------------------------------------------------------------------------

def compute_sum_rate(
    W: torch.Tensor,
    H: torch.Tensor,
    noise_var: float = 0.1,
) -> float:
    """Compute mean sum-rate (bits/s/Hz) over a batch — no gradient tracking.

    Args:
        W : (B, M, K) complex
        H : (B, K, M) complex

    Returns:
        Scalar float mean sum-rate.
    """
    with torch.no_grad():
        h_w = torch.bmm(H, W)                                          # (B, K, K)
        signal = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2  # (B, K)
        interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal # (B, K)
        sinr = signal / (interference + noise_var)
        sum_rate = torch.log2(1.0 + sinr).sum(dim=1).mean()
    return sum_rate.item()


def zero_forcing_beamforming(
    H: torch.Tensor,
    p_max: float = 1.0,
) -> torch.Tensor:
    """Compute Zero-Forcing beamforming matrix W_ZF = H^H (H H^H)^{-1}.

    The output is projected onto the Frobenius power ball.

    Args:
        H : (B, K, M) complex

    Returns:
        W_zf : (B, M, K) complex,  ||W_zf||_F^2 <= p_max
    """
    # H^H : (B, M, K)
    H_H = H.conj().transpose(-1, -2)
    # H H^H : (B, K, K)
    HHH = torch.bmm(H, H_H)
    # (H H^H)^{-1} : (B, K, K)
    HHH_inv = torch.linalg.inv(HHH)
    # W_zf = H^H (H H^H)^{-1} : (B, M, K)
    W_zf = torch.bmm(H_H, HHH_inv)
    return project_frobenius_ball(W_zf, p_max)


def generate_channel_dataset(
    N: int,
    M: int = 8,
    K: int = 4,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate N i.i.d. Rayleigh channel samples with seeded RNG.

    Returns:
        H      : (N, K, M) complex
        W_init : (N, M, K) complex,  ||W_init||_F^2 <= p_max
    """
    torch.manual_seed(seed)
    return generate_6g_miso_dataset(
        num_samples=N,
        num_antennas=M,
        num_users=K,
        p_max=p_max,
        noise_var=noise_var,
    )


def prepare_autogluon_tabular_data(
    N: int,
    M: int = 8,
    K: int = 4,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tabular dataset with ZF-labeled beamforming vectors.

    Features  : 2*K*M real columns  [Re(H.flatten()), Im(H.flatten())]
    Labels    : 2*M*K real columns  [Re(W_zf.flatten()), Im(W_zf.flatten())]

    Returns:
        features_df : pd.DataFrame  shape (N, 2*K*M)
        labels_df   : pd.DataFrame  shape (N, 2*M*K)
    """
    H, _ = generate_channel_dataset(N, M, K, p_max, noise_var, seed)
    W_zf = zero_forcing_beamforming(H, p_max)

    # Flatten to real vectors
    H_np = H.numpy()    # (N, K, M) complex
    W_np = W_zf.numpy() # (N, M, K) complex

    feat = np.concatenate(
        [H_np.real.reshape(N, -1), H_np.imag.reshape(N, -1)], axis=1
    )  # (N, 2*K*M)
    label = np.concatenate(
        [W_np.real.reshape(N, -1), W_np.imag.reshape(N, -1)], axis=1
    )  # (N, 2*M*K)

    feat_cols = [f"h_re_{i}" for i in range(K * M)] + [f"h_im_{i}" for i in range(K * M)]
    lab_cols  = [f"w_re_{i}" for i in range(M * K)] + [f"w_im_{i}" for i in range(M * K)]

    return pd.DataFrame(feat, columns=feat_cols), pd.DataFrame(label, columns=lab_cols)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, M, K = 100, 8, 4

    H_batch, W_batch = generate_6g_miso_dataset(num_samples=B, num_antennas=M, num_users=K)
    print(f"H shape      : {H_batch.shape}")   # (100, 4, 8)
    print(f"W_init shape : {W_batch.shape}")   # (100, 8, 4)

    # Power constraint check
    fro = torch.norm(W_batch.reshape(B, -1), dim=1)
    assert (fro <= 1.0 + 1e-5).all(), "Power constraint violated in W_init"
    print(f"W_init max Frobenius norm : {fro.max().item():.4f}  (should be ≤ 1.0)")

    # Gradient (bug-fixed, no in-place mutation)
    grad = compute_sum_rate_gradient(W_batch[:1], H_batch[:1])
    print(f"Gradient shape : {grad.shape}")    # (1, 8, 4)

    # Sum-rate evaluation
    sr = compute_sum_rate(W_batch, H_batch)
    print(f"Random W sum-rate : {sr:.4f} bits/s/Hz")

    # Zero-Forcing
    W_zf = zero_forcing_beamforming(H_batch, p_max=1.0)
    fro_zf = torch.norm(W_zf.reshape(B, -1), dim=1)
    assert (fro_zf <= 1.0 + 1e-5).all(), "ZF power constraint violated"
    sr_zf = compute_sum_rate(W_zf, H_batch)
    print(f"ZF sum-rate       : {sr_zf:.4f} bits/s/Hz")

    # Tabular dataset
    feat_df, lab_df = prepare_autogluon_tabular_data(N=200, M=M, K=K)
    print(f"Tabular features : {feat_df.shape}  labels : {lab_df.shape}")
    print("data_generator.py  OK")
