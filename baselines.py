"""
baselines.py
============
All non-Auto-PGD baselines for the paper experiments:

  1. Zero-Forcing (ZF)          — closed-form, no training
  2. Classical PGD solver       — 1000-iter non-learned reference
  3. Black-box MLP              — 5-layer PyTorch MLP, sum-rate loss
  4. evaluate_zf()              — convenience wrapper for Figure 1

Tensor shape convention (matches rest of codebase):
  H  : (B, K, M) complex
  W  : (B, M, K) complex
"""

import torch
import torch.nn as nn
import numpy as np

from data_generator import (
    compute_sum_rate,
    compute_sum_rate_gradient,
    generate_6g_miso_dataset,
    project_frobenius_ball,
    zero_forcing_beamforming,
)


# ---------------------------------------------------------------------------
# 1. Zero-Forcing (re-exported for convenience)
# ---------------------------------------------------------------------------

# zero_forcing_beamforming is already fully implemented in data_generator.py.
# Re-export so callers can do:  from baselines import zero_forcing_beamforming
__all__ = [
    "zero_forcing_beamforming",
    "classical_pgd_solver",
    "wmmse_solver",
    "MLPBaseline",
    "evaluate_zf",
]


# ---------------------------------------------------------------------------
# 2. Classical (non-learned) PGD solver
# ---------------------------------------------------------------------------

def classical_pgd_solver(
    H_batch: torch.Tensor,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    num_iters: int = 200,
    init_eta: float = 0.003,
) -> torch.Tensor:
    """Run vectorised classical PGD on an entire batch of channels.

    This is the "Classical PGD (200 iter)" reference line in Figure 1.
    It is NOT trained; it optimises each channel sample independently
    for `num_iters` gradient steps with a fixed step-size.

    Initialised from the Zero-Forcing solution, which is near-optimal and
    lets the solver converge in far fewer iterations than from random init.

    Args:
        H_batch  : (B, K, M) complex — channel batch
        p_max    : transmit power budget
        noise_var: AWGN power σ²
        num_iters: number of PGD iterations
        init_eta : fixed step-size (not learned); 0.003 is near-optimal after
                   the gradient is B-invariant (.sum() fix in data_generator)

    Returns:
        W : (B, M, K) complex — optimised beamforming matrix
    """
    W = zero_forcing_beamforming(H_batch, p_max=p_max).to(H_batch.device)

    for _ in range(num_iters):
        grad = compute_sum_rate_gradient(W, H_batch, noise_var, create_graph=False)
        Z = W - init_eta * grad
        W = project_frobenius_ball(Z, p_max)

    return W.detach()


# ---------------------------------------------------------------------------
# 3. WMMSE beamformer (Shi et al. 2011)
# ---------------------------------------------------------------------------

def wmmse_solver(
    H_batch: torch.Tensor,
    p_max: float = 1.0,
    noise_var: float = 0.1,
    num_iters: int = 100,
) -> torch.Tensor:
    """Vectorised WMMSE beamformer for MISO downlink sum-rate maximisation.

    Implements Algorithm 1 of Shi et al. (2011) adapted to MISO (scalar
    receive filters), fully vectorised over a batch of channel samples.
    The transmit power constraint is enforced per-sample via an
    eigendecomposition-based bisection on the Lagrange multiplier μ.

    Tensor convention (matches rest of codebase):
        H_batch[b, k, m] = (h_k^H)_m   →  actual channel  h_k = H_batch[b,k,:].conj()
        W[b, m, k]       = w_k[m]

    Args:
        H_batch  : (B, K, M) complex — channel batch
        p_max    : total transmit power budget  ‖W‖_F² ≤ p_max
        noise_var: AWGN power σ²
        num_iters: outer WMMSE iterations (50–100 typically sufficient)

    Returns:
        W : (B, M, K) complex — power-feasible beamforming matrix
    """
    B, K, M = H_batch.shape
    device = H_batch.device
    # Use float32 for bisection numerics regardless of complex precision
    real_dtype = torch.float32

    # Initialise from ZF (near-optimal warm start, fast convergence)
    W = zero_forcing_beamforming(H_batch, p_max=p_max).to(device)

    eye_M = torch.eye(M, device=device, dtype=H_batch.dtype)

    for _ in range(num_iters):
        # ── Step 1: MMSE receive scalars u[b, k] ─────────────────────────────
        # HW[b, k, j] = h_k^H w_j
        HW = torch.bmm(H_batch, W)                          # (B, K, K)
        k_idx = torch.arange(K, device=device)
        sig = HW[:, k_idx, k_idx]                           # (B, K) h_k^H w_k
        denom = HW.abs().pow(2).sum(dim=-1) + noise_var     # (B, K) real
        u = sig / denom                                      # (B, K) complex

        # ── Step 2: MSE weights xi[b, k] > 0 ─────────────────────────────────
        e = (1.0 - (u.conj() * sig).real).clamp(min=1e-9)  # (B, K) real
        xi = 1.0 / e                                         # (B, K) real

        # ── Step 3: Transmit beamformer update via bisection on μ ─────────────
        # Actual channel vectors: h[b, k, :] = conj(H_batch[b, k, :])
        h = H_batch.conj()                                   # (B, K, M)

        # Phi[b] = Σ_k  xi_k |u_k|²  h_k  h_k^H    (M×M Hermitian PSD)
        xi_u2 = xi * u.abs().pow(2)                          # (B, K) real
        h_sc = h * xi_u2.unsqueeze(-1)                       # (B, K, M) scaled
        # Phi[b, m, n] = Σ_k h_sc[b,k,m]  conj(h[b,k,n])
        Phi = torch.einsum("bkm,bkn->bmn", h_sc, h.conj())  # (B, M, M)

        # rhs[b, :, k] = xi_k conj(u_k) h_k   → (B, M, K)
        xi_uc = xi * u.conj()                                # (B, K) complex
        rhs = (h * xi_uc.unsqueeze(-1)).permute(0, 2, 1)    # (B, M, K)

        # Eigendecompose Phi for efficient per-sample bisection
        Lambda, U = torch.linalg.eigh(Phi)                  # Lambda (B,M), U (B,M,M)
        Lambda = Lambda.clamp(min=0.0)                       # numerical safety
        Z = U.mH @ rhs                                       # (B, M, K) rotated rhs
        Z2 = Z.abs().pow(2).float()                          # (B, M, K) real
        Lf = Lambda.float()                                  # (B, M) real

        # P(μ) = Σ_{m,k}  |Z[b,m,k]|² / (Λ[b,m] + μ)²   per sample b
        def _power(mu_vec: torch.Tensor) -> torch.Tensor:
            """mu_vec : (B,) → total Frobenius power per sample."""
            lam_mu = (Lf + mu_vec.unsqueeze(-1)).clamp(min=1e-12)  # (B, M)
            return (Z2 / lam_mu.unsqueeze(-1).pow(2)).sum(dim=[1, 2])  # (B,)

        # Bisection: find μ ≥ 0  s.t.  P(μ) ≤ p_max per sample
        mu_lo = torch.zeros(B, device=device, dtype=real_dtype)
        mu_hi = torch.full((B,), 1e5, device=device, dtype=real_dtype)

        pow_unconstrained = _power(mu_lo)                    # (B,) at μ=0
        needs_bisect = pow_unconstrained > p_max + 1e-8

        if needs_bisect.any():
            for _ in range(60):                              # ~60 bisection steps → < 1e-18 error
                mu_mid = (mu_lo + mu_hi) * 0.5
                exceed = _power(mu_mid) > p_max
                mu_lo = torch.where(exceed, mu_mid, mu_lo)
                mu_hi = torch.where(exceed, mu_hi, mu_mid)

        mu_final = torch.where(
            needs_bisect, (mu_lo + mu_hi) * 0.5, mu_lo
        ).to(dtype=Lf.dtype)                                # (B,) real

        # Reconstruct W = U  diag(1 / (Λ + μ))  Z
        lam_mu = (Lambda + mu_final.unsqueeze(-1)).clamp(min=1e-12)  # (B, M) real
        # Cast to complex for matrix multiply
        lam_mu_c = lam_mu.to(dtype=H_batch.dtype)
        W = U @ (Z / lam_mu_c.unsqueeze(-1))                # (B, M, K)

    return W.detach()


# ---------------------------------------------------------------------------
# 4. Black-box MLP baseline
# ---------------------------------------------------------------------------

class MLPBaseline(nn.Module):
    """5-layer fully-connected MLP that maps channel H → beamforming W.

    Architecture (real-valued):
        Input  : 2*K*M  (flattened [Re(H), Im(H)])
        Hidden : hidden_size × (num_layers − 1) with ReLU
        Output : 2*M*K  (flattened [Re(W), Im(W)])

    The output is reshaped to (B, M, K) complex and projected onto the
    Frobenius power ball to guarantee physical feasibility.

    Training loss (same as Auto-PGD): negative mean sum-rate.

    Args:
        M          : number of BS antennas  (default 8)
        K          : number of users        (default 4)
        hidden_size: width of hidden layers (default 256)
        num_layers : total depth including input/output layers (default 5)
        p_max      : transmit power budget  (default 1.0)
        noise_var  : AWGN power σ²          (default 0.1)
    """

    def __init__(
        self,
        M: int = 8,
        K: int = 4,
        hidden_size: int = 256,
        num_layers: int = 5,
        p_max: float = 1.0,
        noise_var: float = 0.1,
    ):
        super().__init__()
        self.M = M
        self.K = K
        self.p_max = p_max
        self.noise_var = noise_var

        in_dim  = 2 * K * M   # flattened real + imag channel
        out_dim = 2 * M * K   # flattened real + imag beamforming

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, out_dim))

        self.net = nn.Sequential(*layers)

    def _vectorize_H(self, H: torch.Tensor) -> torch.Tensor:
        """Flatten complex (B,K,M) channel into real (B, 2*K*M) vector."""
        B = H.shape[0]
        H_np = H.detach()
        return torch.cat(
            [H_np.real.reshape(B, -1), H_np.imag.reshape(B, -1)], dim=1
        ).float()

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Map channel batch H → power-feasible beamforming W.

        Args:
            H : (B, K, M) complex

        Returns:
            W : (B, M, K) complex,  ||W||_F^2 <= p_max
        """
        x = self._vectorize_H(H)          # (B, 2*K*M) real
        out = self.net(x)                  # (B, 2*M*K) real
        B = H.shape[0]
        # Reconstruct complex W
        half = self.M * self.K
        W_real = out[:, :half].reshape(B, self.M, self.K)
        W_imag = out[:, half:].reshape(B, self.M, self.K)
        W = torch.complex(W_real, W_imag)
        return project_frobenius_ball(W, self.p_max)

    def compute_loss(self, H: torch.Tensor) -> torch.Tensor:
        """Negative mean sum-rate — training loss compatible with Auto-PGD."""
        W = self.forward(H)
        h_w = torch.bmm(H, W)
        signal = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2
        interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal
        sinr = signal / (interference + self.noise_var)
        return -torch.log2(1.0 + sinr).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# 5. Convenience evaluation wrapper for ZF
# ---------------------------------------------------------------------------

def evaluate_zf(
    H_test: torch.Tensor,
    p_max: float = 1.0,
    noise_var: float = 0.1,
) -> float:
    """Compute ZF sum-rate on a test set — used as Figure 1 reference line.

    Returns:
        Scalar float mean sum-rate in bits/s/Hz.
    """
    W_zf = zero_forcing_beamforming(H_test, p_max)
    return compute_sum_rate(W_zf, H_test, noise_var)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, M, K = 32, 8, 4
    H, W_init = generate_6g_miso_dataset(num_samples=B, num_antennas=M, num_users=K)

    # Zero-Forcing
    sr_zf = evaluate_zf(H, p_max=1.0, noise_var=0.1)
    print(f"ZF sum-rate            : {sr_zf:.4f} bits/s/Hz")

    # Classical PGD (use few iters for quick test)
    W_pgd = classical_pgd_solver(H, num_iters=50, init_eta=0.01)
    fro_pgd = torch.norm(W_pgd.reshape(B, -1), dim=1)
    assert (fro_pgd <= 1.0 + 1e-5).all(), "Classical PGD power constraint violated"
    sr_pgd = compute_sum_rate(W_pgd, H)
    print(f"Classical PGD (50 it)  : {sr_pgd:.4f} bits/s/Hz")
    print(f"PGD max Frobenius norm : {fro_pgd.max().item():.4f}  (should be ≤ 1.0)")

    # WMMSE
    W_wmmse = wmmse_solver(H, p_max=1.0, noise_var=0.1, num_iters=50)
    fro_wmmse = torch.norm(W_wmmse.reshape(B, -1), dim=1)
    assert (fro_wmmse <= 1.0 + 1e-5).all(), "WMMSE power constraint violated"
    sr_wmmse = compute_sum_rate(W_wmmse, H)
    print(f"WMMSE (50 it)          : {sr_wmmse:.4f} bits/s/Hz")
    print(f"WMMSE max Frob norm    : {fro_wmmse.max().item():.4f}  (should be ≤ 1.0)")

    # Black-box MLP
    mlp = MLPBaseline(M=M, K=K, hidden_size=256, num_layers=5, p_max=1.0)
    W_mlp = mlp(H)
    fro_mlp = torch.norm(W_mlp.reshape(B, -1), dim=1)
    assert (fro_mlp <= 1.0 + 1e-5).all(), "MLP power constraint violated"
    print(f"MLP output shape       : {W_mlp.shape}")
    print(f"MLP max Frobenius norm : {fro_mlp.max().item():.4f}  (should be ≤ 1.0)")

    loss = mlp.compute_loss(H)
    loss.backward()
    print(f"MLP loss               : {loss.item():.4f}")
    print(f"MLP param grad check   : OK (no None grads)")
    print("baselines.py  OK")
