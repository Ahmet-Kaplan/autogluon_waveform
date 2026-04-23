"""
Auto-MLP ablation: HPO over MLP depth at N=100 and N=1000.
Isolates whether physics structure (vs. HPO alone) drives Auto-PGD's advantage.
"""
import json, time, numpy as np, torch, torch.optim as optim, torch.nn as nn
from itertools import product
from train_evaluate import generate_channel_dataset
from data_generator import compute_sum_rate

def compute_sum_rate_loss(W, H, noise_var=0.1):
    """Differentiable sum-rate loss for backprop."""
    h_w = torch.bmm(H, W)
    signal = torch.abs(torch.diagonal(h_w, dim1=1, dim2=2)) ** 2
    interference = torch.sum(torch.abs(h_w) ** 2, dim=2) - signal
    sinr = signal / (interference + noise_var)
    return -torch.log2(1.0 + sinr).sum()

SEEDS = [42, 678, 888, 123, 456]
M, K_users = 8, 4
p_max, noise_var = 1.0, 0.1
N_test = 5000
HPO_TRIALS = 50

device = "cpu"

H_test, _ = generate_channel_dataset(N=N_test, M=M, K=K_users,
                                      p_max=p_max, noise_var=noise_var, seed=999)

# ── Auto-MLP model ────────────────────────────────────────────────────────────
class AutoMLP(nn.Module):
    def __init__(self, num_layers: int, hidden: int = 256):
        super().__init__()
        in_dim = 2 * M * K_users
        out_dim = 2 * M * K_users
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self.p_max = p_max
        self.M = M
        self.K = K_users

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B = H.shape[0]
        x = torch.cat([H.real.reshape(B, -1), H.imag.reshape(B, -1)], dim=-1)
        out = self.net(x)
        W = torch.complex(out[:, :M*K_users], out[:, M*K_users:]).reshape(B, M, K_users)
        # Power projection
        nrm = torch.sqrt((W.abs()**2).sum(dim=(1,2), keepdim=True))
        scale = torch.sqrt(torch.tensor(p_max)) / torch.clamp(nrm, min=1e-8)
        W = W * torch.clamp(scale, max=1.0)
        return W


def train_mlp(model, H_train, epochs, lr, batch_size, scheduler_type):
    model = model.to(device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    if scheduler_type == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = optim.lr_scheduler.StepLR(opt, step_size=epochs//3, gamma=0.5)
    H = H_train.to(device)
    Ntrain = H.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(Ntrain)
        for start in range(0, Ntrain, batch_size):
            hb = H[perm[start:start+batch_size]]
            opt.zero_grad()
            W = model(hb)
            loss = compute_sum_rate_loss(W, hb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()


def evaluate_mlp(model, H):
    model.eval()
    with torch.no_grad():
        W = model(H.to(device))
    return float(compute_sum_rate(W.cpu(), H.cpu(), noise_var))


# ── HPO search ────────────────────────────────────────────────────────────────
import random

def run_hpo(N, H_train_seed0, epochs=500, batch_size=128):
    """50-trial random search over MLP depth and lr (same budget as Auto-PGD)."""
    # 10% validation split from training data
    n_val = max(10, N // 10)
    H_val = H_train_seed0[:n_val]
    H_tr  = H_train_seed0[n_val:]

    rng = random.Random(0)
    best_val, best_cfg = -1e9, None

    for trial in range(HPO_TRIALS):
        num_layers = rng.randint(3, 8)   # MLP: 3–8 layers (deeper = more params)
        lr_exp = rng.uniform(-4, -1)
        lr = 10 ** lr_exp
        sched = rng.choice(["cosine", "step"])

        torch.manual_seed(trial * 13 + 7)
        model = AutoMLP(num_layers=num_layers)
        train_mlp(model, H_tr, epochs=min(epochs, 200), lr=lr,
                  batch_size=batch_size, scheduler_type=sched)
        val_rate = evaluate_mlp(model, H_val)

        if val_rate > best_val:
            best_val = val_rate
            best_cfg = {"num_layers": num_layers, "lr": lr, "scheduler": sched}
        print(f"  Trial {trial+1:2d}/{HPO_TRIALS}: layers={num_layers} lr={lr:.2e} "
              f"sched={sched} → val={val_rate:.4f}  (best={best_val:.4f})")

    print(f"  Best config: {best_cfg}  val={best_val:.4f}")
    return best_cfg


# ── Main ──────────────────────────────────────────────────────────────────────
results_auto_mlp = {}

for N, epochs in [(100, 500), (1000, 500)]:
    print(f"\n{'='*60}")
    print(f"Auto-MLP HPO  N={N}")
    H_seed0, _ = generate_channel_dataset(N=N, M=M, K=K_users,
                                           p_max=p_max, noise_var=noise_var, seed=42)
    best_cfg = run_hpo(N, H_seed0, epochs=epochs)

    # Multi-seed evaluation with best config
    rates = []
    for seed in SEEDS:
        H_train, _ = generate_channel_dataset(N=N, M=M, K=K_users,
                                               p_max=p_max, noise_var=noise_var, seed=seed)
        torch.manual_seed(seed * 10 + 4)
        model = AutoMLP(num_layers=best_cfg["num_layers"])
        train_mlp(model, H_train, epochs=epochs,
                  lr=best_cfg["lr"], batch_size=128,
                  scheduler_type=best_cfg["scheduler"])
        r = evaluate_mlp(model, H_test)
        rates.append(r)
        print(f"  seed={seed}: {r:.4f}")

    arr = np.array(rates)
    print(f"  Auto-MLP N={N}: {arr.mean():.4f}±{arr.std():.4f}")
    results_auto_mlp[str(N)] = {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "per_seed": rates, "best_cfg": best_cfg
    }

print("\n=== FINAL RESULTS ===")
for N, d in results_auto_mlp.items():
    pct = d["mean"] / 14.8133 * 100
    print(f"Auto-MLP N={N}: {d['mean']:.4f}±{d['std']:.4f} ({pct:.1f}% of PGD)")

# Save
with open("results/auto_mlp_ablation.json", "w") as f:
    json.dump(results_auto_mlp, f, indent=2)
print("Saved to results/auto_mlp_ablation.json")
