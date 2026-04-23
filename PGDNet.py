"""
PGDNet.py
=========
PGD-Net: unrolled Proximal Gradient Descent with per-layer learned step sizes
and fixed (non-HPO) depth.  This is the ablation baseline for the paper:
it shows the benefit of the AutoGluon HPO component in Auto-PGD.

Key difference from Auto-PGD (DeepUnrolledNetwork):
    - PGD-Net: manually chosen fixed depth, per-layer learned scalar step sizes
    - Auto-PGD: depth and initial step-size found by AutoGluon HPO, per-layer
                             step sizes also learned afterwards
"""
import torch
import torch.nn as nn

from data_generator import (
        compute_sum_rate_gradient,
        generate_6g_miso_dataset,
        project_frobenius_ball,
        zero_forcing_beamforming,
)


class PGDNetBaseline(nn.Module):
    """PGD-Net baseline: unrolled PGD with fixed depth and learned per-layer step sizes.

    Args:
        num_layers : K — fixed number of unrolled PGD iterations (default 10)
        p_max      : transmit power budget  (default 1.0)
        M          : number of BS antennas  (default 8)
        K_users    : number of single-antenna users (default 4)
        noise_var  : AWGN power σ²          (default 0.1)
    """

    def __init__(
        self,
        num_layers: int = 10,
        p_max: float = 1.0,
        M: int = 8,
        K_users: int = 4,
        noise_var: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.p_max = p_max
        self.M = M
        self.K_users = K_users
        self.noise_var = noise_var
        # Step-sizes are learned, but the depth is static/manually chosen.
        # Init at 0.001 — near-optimal for the B-invariant gradient scaling.
        self.step_sizes = nn.Parameter(torch.ones(num_layers) * 0.001)

    def forward(self, w_init: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply K learned PGD steps.

        Args:
            w_init : (B, M, K) complex — initial beamforming matrix
            H      : (B, K, M) complex — channel matrix (batch)

        Returns:
            W : (B, M, K) complex,  ||W||_F^2 <= p_max
        """
        w = w_init
        for k in range(self.num_layers):
            # Iterative PGD update rule with batched Frobenius projection
            grad = compute_sum_rate_gradient(
                w, H, self.noise_var, create_graph=self.training
            )
            w = w - self.step_sizes[k] * grad
            w = project_frobenius_ball(w, self.p_max)
        return w

    def compute_loss(self, H: torch.Tensor) -> torch.Tensor:
        """Negative mean sum-rate — training loss compatible with train_model()."""
        W_init = zero_forcing_beamforming(H, p_max=self.p_max).to(H.device)
        W_out = self.forward(W_init, H)
        h_w = torch.bmm(H, W_out)
        signal       = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2
        interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal
        sinr = signal / (interference + self.noise_var)
        return -torch.log2(1.0 + sinr).sum(dim=1).mean()

    def get_learned_step_sizes(self) -> list:
        """Return learned step sizes per layer — useful for analysis."""
        return [s.item() for s in self.step_sizes]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import (
        generate_channel_dataset, generate_6g_miso_dataset, compute_sum_rate
    )

    torch.manual_seed(0)
    B, M, K = 16, 8, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, _ = generate_channel_dataset(N=B, M=M, K=K, seed=42)
    H = H.to(device)

    model = PGDNetBaseline(num_layers=10, M=M, K_users=K).to(device)

    # Loss + backward (compute_loss generates W_init internally)
    model.train()
    loss = model.compute_loss(H)
    loss.backward()
    print(f"Training loss       : {loss.item():.4f}")
    print(f"Step-size grads     : {'OK' if model.step_sizes.grad is not None else 'FAILED'}")

    # Eval forward
    model.eval()
    _, W_init = generate_6g_miso_dataset(num_samples=B, num_antennas=M, num_users=K)
    W_out = model(W_init.to(device), H)
    frob_norms = torch.linalg.norm(W_out.reshape(B, -1), dim=1)
    rate = compute_sum_rate(W_out.cpu(), H.cpu(), 0.1)
    print(f"W_out shape         : {W_out.shape}")
    print(f"Max Frobenius norm  : {frob_norms.max().item():.4f}  (should be ≤ 1.0)")
    print(f"Sum-rate (untrained): {rate:.4f} bits/s/Hz")
    print(f"Learned step sizes  : {[f'{s:.5f}' for s in model.get_learned_step_sizes()]}")
    print("PGDNet.py  OK")