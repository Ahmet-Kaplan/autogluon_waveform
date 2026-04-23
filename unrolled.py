"""
unrolled.py
===========
Auto-Unrolled Proximal Gradient Descent network for MISO beamforming.

Each layer corresponds to exactly one PGD iteration:
    W^{k+1} = proj_{P_max}( W^k - eta^k * ∇f(W^k) )

where ∇f is the gradient of the negative sum-rate and proj_{P_max} is the
Frobenius ball projection enforcing the power constraint.

The step-sizes {eta^k} are learnable nn.Parameters — AutoGluon's HPO
optimises their initialisation and the network depth K.
"""

import torch
import torch.nn as nn

from data_generator import (
    compute_sum_rate,
    compute_sum_rate_gradient,
    generate_6g_miso_dataset,
    project_frobenius_ball,
    zero_forcing_beamforming,
)


# ---------------------------------------------------------------------------
# Single PGD layer
# ---------------------------------------------------------------------------

class UnrolledPGDLayer(nn.Module):
    """One unrolled PGD iteration with a learnable step-size eta.

    Args:
        init_eta : initial value for the learnable step-size
        p_max    : transmit power budget (non-trainable)
    """

    def __init__(self, init_eta: float = 0.01, p_max: float = 1.0):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(float(init_eta)))
        # Store p_max as a non-trainable buffer so it moves with .to(device)
        self.register_buffer("p_max", torch.tensor(float(p_max)))

    # Bug fix: replaced L1 soft-thresholding with Frobenius ball projection
    def _project(self, Z: torch.Tensor) -> torch.Tensor:
        """Project Z onto the power ball ||W||_F^2 <= p_max."""
        return project_frobenius_ball(Z, self.p_max.item())

    def forward(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        noise_var: float = 0.1,
    ) -> torch.Tensor:
        """One PGD step.

        Args:
            W        : (B, M, K) complex — current beamforming iterate
            H        : (B, K, M) complex — channel matrix (batch)
            noise_var: AWGN power σ²

        Returns:
            W_new : (B, M, K) complex — updated, power-feasible W
        """
        # create_graph only during training so eta gradients flow correctly;
        # False during eval so inference works even inside torch.no_grad().
        grad = compute_sum_rate_gradient(W, H, noise_var, create_graph=self.training)
        Z = W - self.eta * grad
        return self._project(Z)


# ---------------------------------------------------------------------------
# Full unrolled network
# ---------------------------------------------------------------------------

class DeepUnrolledNetwork(nn.Module):
    """K-layer unrolled PGD network for MISO downlink beamforming.

    Args:
        num_layers : K  — number of unrolled PGD iterations (network depth)
        init_eta   : initial value for ALL layer step-sizes (each layer gets
                     its own independent nn.Parameter)
        p_max      : transmit power budget
        noise_var  : AWGN power σ²
        M          : number of BS antennas
        K_users    : number of single-antenna users
    """

    def __init__(
        self,
        num_layers: int = 10,
        init_eta: float = 0.01,
        p_max: float = 1.0,
        noise_var: float = 0.1,
        M: int = 8,
        K_users: int = 4,
        layer_type: str = "pgd",
        activation: str = "identity",
    ):
        super().__init__()
        self.noise_var = noise_var
        self.p_max = p_max
        self.M = M
        self.K_users = K_users
        self.layer_type = layer_type
        self.activation = activation

        def get_layer():
            if layer_type == "pgd":
                return UnrolledPGDLayer(init_eta=init_eta, p_max=p_max)
            else:
                # Hybrid layer: PGD step + learnable linear transformation on gradient + power projection
                class HybridPGDLayer(nn.Module):
                    def __init__(self, M, K, init_eta, p_max, activation):
                        super().__init__()
                        self.eta = nn.Parameter(torch.tensor(float(init_eta)))
                        self.register_buffer("p_max", torch.tensor(float(p_max)))
                        self.linear = nn.Linear(M*K, M*K)
                        self.activation = activation
                    def forward(self, W, H, noise_var=0.1):
                        grad = compute_sum_rate_gradient(W, H, noise_var, create_graph=self.training)
                        grad_flat = grad.view(grad.shape[0], -1)
                        grad_trans = self.linear(grad_flat)
                        grad_trans = grad_trans.view(grad.shape)
                        Z = W - self.eta * grad_trans
                        if self.activation == "relu":
                            Z = torch.relu(Z)
                        elif self.activation == "tanh":
                            Z = torch.tanh(Z)
                        # identity: no activation
                        return project_frobenius_ball(Z, self.p_max.item())
                return HybridPGDLayer(M, K_users, init_eta, p_max, activation)

        self.layers = nn.ModuleList(
            [get_layer() for _ in range(num_layers)]
        )

    def forward(self, W_init: torch.Tensor, H: torch.Tensor, log_metrics=None) -> torch.Tensor:
        """Run all K unrolled PGD/custom iterations. Optionally logs layer-wise metrics."""
        W = W_init
        layer_outputs = []
        step_sizes = []
        grad_transforms = []
        activations = []
        sum_rates = []
        for idx, layer in enumerate(self.layers):
            prev_W = W.detach().cpu().numpy()
            if hasattr(layer, 'eta'):
                step_sizes.append(layer.eta.item())
            # Compute gradient transformation for hybrid layers
            if hasattr(layer, 'linear'):
                grad = compute_sum_rate_gradient(W, H, self.noise_var, create_graph=self.training)
                grad_flat = grad.view(grad.shape[0], -1)
                grad_trans = layer.linear(grad_flat)
                grad_transforms.append(grad_trans.detach().cpu().numpy())
            else:
                grad_transforms.append(None)
            # Activation impact
                if hasattr(layer, 'activation'):
                    activations.append(layer.activation)
                else:
                    activations.append('none')
                # Forward
                W = layer(W, H, self.noise_var)
                layer_outputs.append(W.detach().cpu().numpy())
                # Per-layer sum-rate
                sr = compute_sum_rate(W, H, self.noise_var)
                sum_rates.append(sr)
        if log_metrics is not None:
            log_metrics['layer_outputs'] = layer_outputs
            log_metrics['step_sizes'] = step_sizes
            log_metrics['grad_transforms'] = grad_transforms
            log_metrics['activations'] = activations
            log_metrics['sum_rates'] = sum_rates
        return W

    def compute_loss(self, H: torch.Tensor) -> torch.Tensor:
        """Compute training loss (negative mean sum-rate) for a channel batch.

        Handles custom layers by projecting output onto power ball before sum-rate.

        Args:
            H : (B, K, M) complex — channel batch (on correct device)

        Returns:
            loss : scalar tensor (differentiable w.r.t. self.parameters())
        """
        B = H.shape[0]
        W_init = zero_forcing_beamforming(H, p_max=self.p_max)
        W_init = W_init.to(H.device)

        W_out = self.forward(W_init, H)
        # For custom layers, project onto power ball
        if self.layer_type == "custom":
            W_out = project_frobenius_ball(W_out, self.p_max)

        h_w = torch.bmm(H, W_out)
        signal = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2
        interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal
        sinr = signal / (interference + self.noise_var)
        loss = -torch.log2(1.0 + sinr).sum(dim=1).mean()
        return loss

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def get_learned_step_sizes(self) -> list[float]:
        """Return the learned eta for each layer — used for Figure 2."""
        return [layer.eta.item() for layer in self.layers]

    def get_intermediate_W(
        self, W_init: torch.Tensor, H: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return W at every layer including the initial W_init — used for Figure 3.

        Returns:
            List of K+1 tensors, each (B, M, K) complex.
        """
        with torch.enable_grad():
            trajectory = [W_init.detach().clone()]
            W = W_init
            for layer in self.layers:
                W = layer(W, H, self.noise_var)
                trajectory.append(W.detach().clone())
        return trajectory


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    B, M, K = 16, 8, 4
    H, W_init = generate_6g_miso_dataset(num_samples=B, num_antennas=M, num_users=K)

    model = DeepUnrolledNetwork(num_layers=5, init_eta=0.01, M=M, K_users=K)

    # Forward pass
    W_out = model(W_init, H)
    fro = torch.norm(W_out.reshape(B, -1), dim=1)
    assert (fro <= 1.0 + 1e-5).all(), "Power constraint violated in W_out"
    print(f"W_out shape         : {W_out.shape}")
    print(f"Max Frobenius norm  : {fro.max().item():.4f}  (should be ≤ 1.0)")

    # Loss
    loss = model.compute_loss(H)
    print(f"Training loss       : {loss.item():.4f}")

    # Gradients flow back
    loss.backward()
    etas = model.get_learned_step_sizes()
    grads = [layer.eta.grad for layer in model.layers]
    assert all(g is not None for g in grads), "eta gradients are None"
    print(f"Learned step-sizes  : {[f'{e:.5f}' for e in etas]}")
    print(f"Step-size gradients : {[f'{g.item():.5e}' for g in grads]}")

    # Trajectory
    traj = model.get_intermediate_W(W_init, H)
    print(f"Trajectory length   : {len(traj)}  (should be num_layers+1 = 6)")

    # Sum-rate improvement
    sr_init = compute_sum_rate(W_init, H)
    sr_out  = compute_sum_rate(W_out.detach(), H)
    print(f"Sum-rate init → out : {sr_init:.4f} → {sr_out:.4f} bits/s/Hz")
    print("unrolled.py  OK")
