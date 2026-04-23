import torch
import torch.nn as nn
"""
lista.py
========
LISTA (Learned Iterative Soft-Thresholding Algorithm) adapted for MISO
downlink beamforming sum-rate maximisation.

Architecture (per layer k):
    x_{k+1} = soft_thresh( W1_k * h + W2_k * x_k, λ_k )
where h = [Re(H); Im(H)] is the 2*K*M-dim real vectorisation of the channel.
A trainable output layer maps the final hidden state to a complex (B,M,K)
beamforming matrix, then projected onto the Frobenius power ball.

Training objective (same as all other models): negative mean sum-rate.
"""

from data_generator import (
    compute_sum_rate_gradient,
    generate_6g_miso_dataset,
    project_frobenius_ball,
)


class LISTABeamformer(nn.Module):
    """LISTA-based beamformer for MISO downlink.

    Args:
        M          : number of BS antennas   (default 8)
        K_users    : number of single-antenna users (default 4)
        hidden_dim : LISTA hidden-state width  (default 256)
        num_layers : number of LISTA iterations  (default 8)
        p_max      : transmit power budget        (default 1.0)
        noise_var  : AWGN power σ²                (default 0.1)
    """

    def __init__(
        self,
        M: int = 8,
        K_users: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        p_max: float = 1.0,
        noise_var: float = 0.1,
    ):
        super().__init__()
        self.M = M
        self.K_users = K_users
        self.p_max = p_max
        self.noise_var = noise_var
        self.num_layers = num_layers

        input_dim = 2 * K_users * M   # Re/Im of flattened H: (B, 2*K*M)
        out_dim   = 2 * M * K_users   # Re/Im of flattened W: (B, 2*M*K)

        # LISTA learns specific weight matrices for each iteration
        self.W1 = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])
        self.W2 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers)
        ])
        self.thresholds = nn.Parameter(torch.ones(num_layers, hidden_dim) * 0.1)

        # Output projection: hidden state → beamforming space
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    # ------------------------------------------------------------------
    def soft_threshold(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - lam)

    def _vectorize_H(self, H: torch.Tensor) -> torch.Tensor:
        """Flatten complex (B, K, M) channel to real (B, 2*K*M)."""
        B = H.shape[0]
        return torch.cat(
            [H.real.reshape(B, -1), H.imag.reshape(B, -1)], dim=1
        ).to(torch.float32)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Map channel H → power-feasible beamforming W.

        Args:
            H : (B, K, M) complex — channel batch

        Returns:
            W : (B, M, K) complex,  ||W||_F^2 <= p_max
        """
        B = H.shape[0]
        y = self._vectorize_H(H)  # (B, 2*K*M)

        # Initial estimate x_0
        x = self.soft_threshold(self.W1[0](y), self.thresholds[0])

        for k in range(1, self.num_layers):
            # LISTA update: x_{k+1} = soft_thresh( W1_k*h + W2_k*x_k, λ_k )
            z = self.W1[k](y) + self.W2[k](x)
            x = self.soft_threshold(z, self.thresholds[k])
        B = H.shape[0]
        y = self._vectorize_H(H)   # (B, 2*K*M)

        # Start from x=0 so W2[0] participates in the graph (avoids None grads)
        x = torch.zeros(B, self.W2[0].out_features, device=H.device)
        for k in range(self.num_layers):
            # LISTA update: x_{k+1} = soft_thresh( W1_k*h + W2_k*x_k, λ_k )
            z = self.W1[k](y) + self.W2[k](x)
            x = self.soft_threshold(z, self.thresholds[k])

        # Map hidden state → complex W → Frobenius ball
        out = self.output_layer(x)      # (B, 2*M*K)
        half = self.M * self.K_users
        W_real = out[:, :half].reshape(B, self.M, self.K_users)
        W_imag = out[:, half:].reshape(B, self.M, self.K_users)
        W = torch.complex(W_real, W_imag)
        return project_frobenius_ball(W, self.p_max)

    def compute_loss(self, H: torch.Tensor) -> torch.Tensor:
        """Negative mean sum-rate — training loss compatible with train_model()."""
        W = self.forward(H)
        h_w = torch.bmm(H, W)
        signal       = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2
        interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal
        sinr = signal / (interference + self.noise_var)
        return -torch.log2(1.0 + sinr).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import generate_channel_dataset, compute_sum_rate

    torch.manual_seed(0)
    B, M, K = 16, 8, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, _ = generate_channel_dataset(N=B, M=M, K=K, seed=42)
    H = H.to(device)

    model = LISTABeamformer(M=M, K_users=K).to(device)

    # Forward pass
    W_out = model(H)
    frob_norms = torch.linalg.norm(W_out.reshape(B, -1), dim=1)
    print(f"W_out shape         : {W_out.shape}")
    print(f"Max Frobenius norm  : {frob_norms.max().item():.4f}  (should be ≤ 1.0)")

    # Loss + backward
    model.train()
    loss = model.compute_loss(H)
    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters())
    print(f"Training loss       : {loss.item():.4f}")
    print(f"Param grad check    : {'OK (no None grads)' if grad_ok else 'FAILED'}")

    rate_init  = compute_sum_rate(W_out, H.cpu(), 0.1)
    model.eval()
    W_final = model(H)
    rate_final = compute_sum_rate(W_final, H.cpu(), 0.1)
    print(f"Sum-rate (untrained): {rate_init:.4f} bits/s/Hz")
    print("lista.py  OK")