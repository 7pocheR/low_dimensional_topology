# resnet_disk_annulus_width2_POLAR_CMAP.py
# Linearly separate A = {(x,y): x^2+y^2 < 1} and B = {(x,y): 2 < x^2+y^2 < 4}
# with a width‑2 ReLU ResNet (depth 3–5). Deterministic hinge training with
# margin‑aware early stopping. NO SVM/Stage‑B.
#
# Plots use a PERSISTENT POLAR COLOR ENCODING:
#   • Angle (θ) -> cyclic hue from a cyclic colormap (default: matplotlib 'twilight')
#   • Radius (r) -> lightness (by blending toward white)
# Colors are computed once from the ORIGINAL inputs used for plotting and reused
# identically at every layer and in block‑step plots, so you can track point motion
# without class markers. (Classes are implicitly separable by the radial lightness.)
#
# Also included: interactive Plotly HTML (if available), radial warp plot,
# dense sign check, CPU‑safe piecewise‑affine extraction, activation tiling,
# and LaTeX/JSON parameter export (optional PDF if pdflatex available).
import colorsys
import matplotlib.colors as mcolors
import os, sys, random, json, math, shutil, subprocess
from datetime import datetime
# ---- Determinism (set before importing torch) ----
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Headless‑safe PNG backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Optional: interactive HTML
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False


# =========================
# Utilities
# =========================
def seed_everything(seed: int):
    import torch.backends.cudnn as cudnn
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False; cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def ts(): return datetime.now().strftime("%Y%m%d-%H%M%S")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA {torch.version.cuda}")


# =========================
# Data: Disk vs Annulus
# =========================
class DiskAnnulusDataset:
    """Class 0: disk r<1; Class 1: annulus sqrt(2)<r<2 (2<r^2<4)."""
    def __init__(self, n_A=5000, n_B=5000, noise=0.0, seed=0):
        self.n_A, self.n_B, self.noise, self.seed = n_A, n_B, noise, seed

    def sample(self):
        rs = np.random.RandomState(self.seed)
        # Disk: area‑uniform => r = sqrt(U)
        U = rs.rand(self.n_A); r_A = np.sqrt(U); th_A = 2*np.pi*rs.rand(self.n_A)
        x_A, y_A = r_A*np.cos(th_A), r_A*np.sin(th_A)
        # Annulus: r^2 uniform in (2,4) => r = sqrt(2+2V)
        V = rs.rand(self.n_B); r_B = np.sqrt(2.0 + 2.0*V); th_B = 2*np.pi*rs.rand(self.n_B)
        x_B, y_B = r_B*np.cos(th_B), r_B*np.sin(th_B)
        X0 = np.stack([x_A, y_A], 1); X1 = np.stack([x_B, y_B], 1)
        if self.noise > 0:
            X0 += rs.normal(0, self.noise, X0.shape); X1 += rs.normal(0, self.noise, X1.shape)
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.hstack([np.zeros(self.n_A, np.int64), np.ones(self.n_B, np.int64)])
        return X, y


# =========================
# Model: width‑2 ResNet
# =========================
class ResidualBlock2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, 2)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(2, 2)
        self.bias = nn.Parameter(torch.zeros(2))  # residual bias

    def forward(self, x):
        return x + self.lin2(self.act(self.lin1(x))) + self.bias

class Width2ResNet(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        assert depth >= 1
        self.blocks = nn.ModuleList([ResidualBlock2D() for _ in range(depth)])
        self.head   = nn.Linear(2, 2)  # binary logits

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    @torch.no_grad()
    def forward_with_intermediates(self, x):
        outs = [x]
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs  # [x0..xD]


# =========================
# Persistent polar colors
# =========================
def make_persistent_polar_colors(
    X,
    angle_cmap="twilight",   # accepted for API compatibility; used for phase shift only
    l_range=(0.24, 0.62),    # lightness range (darker center → lighter outer ring)
    s=0.90,                  # HLS saturation (0..1)
    hue_offset=None          # optional manual hue rotation in turns (0..1)
):
    """
    Angle (theta) -> hue (cyclic, continuous at 0↔2π).
    Radius r∈[0,2] -> lightness via L = Lmin + (Lmax-Lmin)*(r/2).

    `angle_cmap` is accepted to keep older call sites working. We don’t sample the
    colormap; we rotate hue instead for cleaner control:
      - "twilight"          → no rotation
      - "twilight_shifted"  → 0.5 turn rotation (phase shift)
    You can override with `hue_offset` in turns (e.g., 0.25 = 90°).
    """
    x = X[:, 0].astype(float)
    y = X[:, 1].astype(float)

    # base angle in [0,1)
    theta = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
    h = theta / (2*np.pi)

    # optional hue rotation from argument
    if hue_offset is None:
        if angle_cmap == "twilight_shifted":
            hue_offset = 0.5
        else:
            hue_offset = 0.0
    h = (h + hue_offset) % 1.0

    # radius → lightness in [0,1]
    r = np.sqrt(x*x + y*y)
    r_norm = np.clip(r / 2.0, 0.0, 1.0)  # radius ∈ [0,2]
    Lmin, Lmax = l_range
    L = Lmin + (Lmax - Lmin) * r_norm

    # HLS → RGB
    rgb = np.array([colorsys.hls_to_rgb(h[i], L[i], s) for i in range(len(h))], dtype=float)
    rgb = np.clip(rgb, 0, 1)
    rgba = np.concatenate([rgb, 0.95*np.ones((len(rgb),1))], axis=1)
    hexs = np.array([mcolors.to_hex(c) for c in rgb], dtype=object)
    return rgba, hexs


# =========================
# Metrics (hinge)
# =========================
@torch.no_grad()
def eval_margin_metrics(model, loader):
    """acc (%), min_margin, mean_margin, mean_hinge_slack (margin=1)."""
    model.eval()
    total = 0; correct = 0
    diffs = []; ys = []
    for xb, yb in loader:
        logits = model(xb)
        pred = logits.argmax(dim=1)
        total += yb.size(0); correct += (pred == yb).sum().item()
        diffs.append((logits[:, 1] - logits[:, 0]).detach().cpu())
        ys.append(yb.detach().cpu())
    acc = 100.0 * correct / max(1, total)
    diffs = torch.cat(diffs); ys = torch.cat(ys).float()
    ypm1 = 2*ys - 1
    u = (model.head.weight[1] - model.head.weight[0]).detach().cpu()
    u_norm = u.norm().item() + 1e-12
    signed = ypm1 * (diffs / u_norm)
    min_margin = float(signed.min().item())
    mean_margin = float(signed.mean().item())
    hinge_slack = float(torch.clamp(1 - signed, min=0).mean().item())
    return acc, min_margin, mean_margin, hinge_slack


# =========================
# Training (hinge, margin‑aware early stop)
# =========================
def train_stage_a(model, train_loader, val_loader, epochs=400, lr=1e-3, patience=100, log_every=100):
    criterion = nn.MultiMarginLoss()  # binary hinge (2‑class special case)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_key = (float("inf"), -float("inf"))  # (slack, -min_margin)
    best_state = None; no_improve = 0

    for ep in range(epochs):
        model.train(); running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()

        acc, min_m, mean_m, slack = eval_margin_metrics(model, val_loader)
        key = (slack, -min_m)
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (ep+1) % log_every == 0 or ep == 0:
            print(f"  Ep {ep+1:4d} | loss={running/max(1,len(train_loader)):.4f} | "
                  f"acc={acc:6.2f}% | slack={slack:.5f} | min_margin={min_m:.5f}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {ep+1} (no margin improvement for {patience} epochs)")
            break

        if torch.cuda.is_available() and (ep % 100 == 0):
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)


# =========================
# Plotting with persistent colors
# =========================
@torch.no_grad()
def plot_layerwise_2d(model, X, colors_rgba, colors_hex, out_dir, title_prefix=""):
    """Scatter *all points* with persistent polar colors (no class markers)."""
    os.makedirs(out_dir, exist_ok=True)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    feats = model.forward_with_intermediates(X_t)

    for li, F in enumerate(feats):
        P = F.detach().cpu().numpy()

        # PNG
        fig = plt.figure(figsize=(6,6))
        plt.scatter(P[:,0], P[:,1], c=colors_rgba, s=3, linewidths=0)
        plt.axis('equal'); plt.grid(True, alpha=0.2)
        plt.title(f"{title_prefix} Layer {li} (0=input)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"layer_{li:02d}.png"),
                                        dpi=220, bbox_inches='tight'); plt.close(fig)

        # HTML
        if HAVE_PLOTLY:
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=P[:,0], y=P[:,1], mode='markers', name='points',
                    marker=dict(size=3, opacity=0.95, color=colors_hex.tolist())
                ))
                fig.update_layout(title=f"{title_prefix} Layer {li} (0=input)",
                                  xaxis_title="f1", yaxis_title="f2",
                                  width=700, height=700)
                fig.write_html(os.path.join(out_dir, f"layer_{li:02d}.html"),
                               include_plotlyjs="cdn", auto_open=False)
            except Exception as e:
                print(f"    (plotly HTML skipped for layer {li}: {e})")

    # Final layer decision line
    P = feats[-1].detach().cpu().numpy()
    u = (model.head.weight[1] - model.head.weight[0]).detach().cpu().numpy()
    b = (model.head.bias[1] - model.head.bias[0]).detach().cpu().numpy()
    xx = np.linspace(P[:,0].min(), P[:,0].max(), 200)
    yy = -(u[0]*xx + b)/u[1] if abs(u[1]) > 1e-8 else np.full_like(xx, np.nan)

    fig = plt.figure(figsize=(6,6))
    plt.scatter(P[:,0], P[:,1], c=colors_rgba, s=3, linewidths=0)
    plt.plot(xx, yy, 'k-', linewidth=2)
    plt.axis('equal'); plt.grid(True, alpha=0.2)
    plt.title(f"{title_prefix} Final layer with decision line")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "final_with_boundary.png"),
                                    dpi=220, bbox_inches='tight'); plt.close(fig)

    print(f"  Saved layerwise plots to: {os.path.abspath(out_dir)}")


@torch.no_grad()
def plot_blockwise_transforms(model, X, colors_rgba, colors_hex, out_dir, title_prefix="", quiver_max=3000):
    """
    For each block: step0=input, step1=W1x+b1, step2=ReLU, step3=W2·+b2, step4=residual add.
    All steps use the SAME persistent colors from the original X.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    Z = X_t
    for bi, blk in enumerate(model.blocks, start=1):
        Z0 = Z
        Z1 = blk.lin1(Z0)
        Z2 = blk.act(Z1)
        Z3 = blk.lin2(Z2)
        Z4 = Z0 + Z3 + blk.bias

        steps = [(Z0, f"block{bi}_step0_input"),
                 (Z1, f"block{bi}_step1_lin1"),
                 (Z2, f"block{bi}_step2_relu"),
                 (Z3, f"block{bi}_step3_lin2"),
                 (Z4, f"block{bi}_step4_resout")]

        for tensor, tag in steps:
            P = tensor.detach().cpu().numpy()
            # PNG
            fig = plt.figure(figsize=(6,6))
            plt.scatter(P[:,0], P[:,1], c=colors_rgba, s=3, linewidths=0)
            plt.axis('equal'); plt.grid(True, alpha=0.2)
            plt.title(f"{title_prefix} {tag}")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{tag}.png"),
                                            dpi=220, bbox_inches='tight'); plt.close(fig)
            # HTML
            if HAVE_PLOTLY:
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=P[:,0], y=P[:,1], mode='markers', name=tag,
                        marker=dict(size=3, opacity=0.95, color=colors_hex.tolist())
                    ))
                    fig.update_layout(title=f"{title_prefix} {tag}",
                                      xaxis_title="f1", yaxis_title="f2",
                                      width=700, height=700)
                    fig.write_html(os.path.join(out_dir, f"{tag}.html"),
                                   include_plotlyjs="cdn", auto_open=False)
                except Exception as e:
                    print(f"    (plotly HTML skipped for {tag}: {e})")

        # overlay & quiver (subset)
        Pin, Pout = Z0.detach().cpu().numpy(), Z4.detach().cpu().numpy()
        fig = plt.figure(figsize=(6,6))
        plt.scatter(Pin[:,0],  Pin[:,1],  c=colors_rgba, s=2, alpha=0.25, linewidths=0)
        plt.scatter(Pout[:,0], Pout[:,1], c=colors_rgba, s=3, alpha=0.95, linewidths=0)
        plt.axis('equal'); plt.grid(True, alpha=0.2)
        plt.title(f"{title_prefix} block{bi}_overlay_in_vs_out")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"block{bi}_overlay_in_vs_out.png"),
                                        dpi=220, bbox_inches='tight'); plt.close(fig)

        n = Pin.shape[0]
        idx = np.linspace(0, n-1, num=min(quiver_max, n), dtype=int)
        U, V = Pout[idx,0]-Pin[idx,0], Pout[idx,1]-Pin[idx,1]
        fig = plt.figure(figsize=(6,6))
        plt.quiver(Pin[idx,0], Pin[idx,1], U, V, angles='xy', scale_units='xy', scale=1)
        plt.axis('equal'); plt.grid(True, alpha=0.2)
        plt.title(f"{title_prefix} block{bi}_residual_quiver")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"block{bi}_residual_quiver.png"),
                                        dpi=220, bbox_inches='tight'); plt.close(fig)

        Z = Z4

    print(f"  Saved block‑step plots to: {os.path.abspath(out_dir)}")


# =========================
# Radial warp & dense sign check
# =========================
@torch.no_grad()
def radial_reverse_engineer(model, R_samples=256, TH_samples=360, out_dir=None, title="Radial warp"):
    os.makedirs(out_dir, exist_ok=True)
    u = (model.head.weight[1] - model.head.weight[0]).detach()
    b = (model.head.bias[1] - model.head.bias[0]).detach()
    u_norm = u.norm() + 1e-12
    rs = np.linspace(0, 2.0, R_samples)
    thetas = np.linspace(0, 2*np.pi, TH_samples, endpoint=False)
    vals = []
    for r in rs:
        pts = np.stack([r*np.cos(thetas), r*np.sin(thetas)], 1).astype(np.float32)
        Z = torch.tensor(pts, device=device)
        for blk in model.blocks: Z = blk(Z)
        score = (Z @ u + b)/u_norm
        vals.append(score.detach().cpu().numpy())
    vals = np.stack(vals, 0)
    mean_val = vals.mean(1); low_q = np.quantile(vals, 0.1, 1); hi_q = np.quantile(vals, 0.9, 1)
    fig = plt.figure(figsize=(7,4))
    plt.plot(rs, mean_val, label="mean")
    plt.fill_between(rs, low_q, hi_q, alpha=0.2, label="10–90% over angles")
    plt.axvspan(0,1, color='green', alpha=0.1, label="A: r<1")
    plt.axvspan(math.sqrt(2),2, color='orange', alpha=0.1, label="B: sqrt(2)<r<2")
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel("radius r"); plt.ylabel("normalized decision")
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "radial_warp.png"),
                                    dpi=220, bbox_inches='tight'); plt.close(fig)
    if HAVE_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rs, y=mean_val, mode='lines', name='mean'))
        fig.add_trace(go.Scatter(x=np.concatenate([rs, rs[::-1]]),
                                 y=np.concatenate([low_q, hi_q[::-1]]),
                                 fill='toself', mode='lines', name='10–90%'))
        fig.update_layout(title=title, xaxis_title='r', yaxis_title='normalized decision')
        fig.write_html(os.path.join(out_dir, "radial_warp.html"),
                       include_plotlyjs="cdn", auto_open=False)
    print(f"  Saved radial warp plots to: {os.path.abspath(out_dir)}")

@torch.no_grad()
def dense_sign_check(model, r_ranges=((0,1),(math.sqrt(2),2)), R=1024, TH=720):
    u = (model.head.weight[1] - model.head.weight[0]).detach()
    b = (model.head.bias[1] - model.head.bias[0]).detach()
    u_norm = u.norm() + 1e-12
    out = {}
    for name, (r_lo, r_hi) in zip(['A','B'], r_ranges):
        rs = np.linspace(r_lo, r_hi, R)
        thetas = np.linspace(0, 2*np.pi, TH, endpoint=False)
        mins = []
        for r in rs:
            pts = np.stack([r*np.cos(thetas), r*np.sin(thetas)], 1).astype(np.float32)
            Z = torch.tensor(pts, device=device)
            for blk in model.blocks: Z = blk(Z)
            score = (Z @ u + b)/u_norm
            mins.append(score.abs().min().item())
        out[name] = float(np.min(mins))
    return out


# =========================
# Piecewise‑affine extraction (CPU‑safe) & activation tiling
# =========================
@torch.no_grad()
def block_affine_from_mask(block: ResidualBlock2D, mask: torch.Tensor):
    W1 = block.lin1.weight.detach().cpu(); b1 = block.lin1.bias.detach().cpu()
    W2 = block.lin2.weight.detach().cpu(); b2 = block.lin2.bias.detach().cpu()
    rB = block.bias.detach().cpu()
    D = torch.diag(mask.float()); I = torch.eye(2)
    A = I + W2 @ D @ W1
    c = (W2 @ (D @ b1)) + b2 + rB
    return A, c

@torch.no_grad()
def mask_at_point_cpu(block: ResidualBlock2D, z_cpu: torch.Tensor):
    W1 = block.lin1.weight.detach().cpu(); b1 = block.lin1.bias.detach().cpu()
    pre = W1 @ z_cpu + b1
    return (pre > 0).to(torch.int64)

@torch.no_grad()
def composed_affine_up_to_each_layer(model: Width2ResNet, x_cpu: torch.Tensor):
    A_cum = torch.eye(2); c_cum = torch.zeros(2)
    As, cs, masks = [], [], []; z = x_cpu.clone()
    for blk in model.blocks:
        m = mask_at_point_cpu(blk, z)
        A, c = block_affine_from_mask(blk, m)
        A_cum = A @ A_cum; c_cum = (A @ c_cum) + c
        z = A @ z + c
        As.append(A_cum.clone()); cs.append(c_cum.clone()); masks.append(m.clone())
    return As, cs, masks

@torch.no_grad()
def verify_local_affinity(model: Width2ResNet, x_cpu: torch.Tensor):
    As, cs, _ = composed_affine_up_to_each_layer(model, x_cpu)
    z_dev = x_cpu.to(device); errs = []
    for l, blk in enumerate(model.blocks):
        z_dev = blk(z_dev); z_hat = As[l] @ x_cpu + cs[l]
        errs.append(torch.norm(z_dev.detach().cpu() - z_hat).item())
    return max(errs) if errs else 0.0

@torch.no_grad()
def dump_affine_report_for_points(model: Width2ResNet, points_np: np.ndarray, out_json_path: str):
    rep = {"depth": len(model.blocks), "points": []}
    for i in range(points_np.shape[0]):
        x_cpu = torch.tensor(points_np[i], dtype=torch.float32)
        As, cs, masks = composed_affine_up_to_each_layer(model, x_cpu)
        entry = {"x": points_np[i].tolist(), "layers": []}
        for l in range(len(As)):
            entry["layers"].append({
                "mask": masks[l].tolist(),
                "A": As[l].numpy().tolist(),
                "c": cs[l].numpy().tolist()
            })
        entry["recon_error_max"] = verify_local_affinity(model, x_cpu)
        rep["points"].append(entry)
    with open(out_json_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[extract] wrote affine report for {points_np.shape[0]} points -> {os.path.abspath(out_json_path)}")

@torch.no_grad()
def activation_tiling(model: Width2ResNet, extent=2.2, N=400, out_png=None):
    xs = np.linspace(-extent, extent, N); ys = np.linspace(-extent, extent, N)
    X, Y = np.meshgrid(xs, ys); pts = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)
    codes = []
    for p in pts:
        z = torch.tensor(p, dtype=torch.float32)  # CPU path
        code_bits = []
        for blk in model.blocks:
            m = mask_at_point_cpu(blk, z)
            code_bits += [int(m[0].item()), int(m[1].item())]
            A, c = block_affine_from_mask(blk, m)
            z = A @ z + c
        code = 0
        for b in code_bits: code = (code << 1) | b
        codes.append(code)
    codes = np.array(codes).reshape(N, N)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(codes, origin='lower', extent=[-extent, extent, -extent, extent], cmap='tab20')
    plt.colorbar(shrink=0.8, label='activation pattern id')
    for r in [1.0, math.sqrt(2.0), 2.0]:
        plt.gca().add_patch(plt.Circle((0,0), r, color='k', fill=False, linestyle='--', alpha=0.6))
    plt.title("Activation tiling"); plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal')
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200, bbox_inches='tight')
        print(f"[extract] saved activation tiling -> {os.path.abspath(out_png)}")
    plt.close(fig)


# =========================
# Parameters: print, JSON, LaTeX (+optional PDF)
# =========================
def _np(a): return a.detach().cpu().numpy()

def print_parameter_summary(model: Width2ResNet):
    print("\n=== Parameter Summary ===")
    for i, blk in enumerate(model.blocks, start=1):
        W1 = _np(blk.lin1.weight); b1 = _np(blk.lin1.bias)
        W2 = _np(blk.lin2.weight); b2 = _np(blk.lin2.bias)
        rB = _np(blk.bias)
        print(f"\nBlock {i}:")
        print("  W1 =\n", np.array2string(W1, formatter={'float_kind':lambda x: f"{x: .6f}"}))
        print("  b1 = ", np.array2string(b1, formatter={'float_kind':lambda x: f"{x: .6f}"}))
        print("  W2 =\n", np.array2string(W2, formatter={'float_kind':lambda x: f"{x: .6f}"}))
        print("  b2 = ", np.array2string(b2, formatter={'float_kind':lambda x: f"{x: .6f}"}))
        print("  rBias = ", np.array2string(rB, formatter={'float_kind':lambda x: f"{x: .6f}"}))
    HW = _np(model.head.weight); Hb = _np(model.head.bias)
    print("\nHead:")
    print("  W_head =\n", np.array2string(HW, formatter={'float_kind':lambda x: f"{x: .6f}"}))
    print("  b_head = ", np.array2string(Hb, formatter={'float_kind':lambda x: f"{x: .6f}"}))

def save_parameters_json(model: Width2ResNet, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    data = {"blocks": [], "head": {}}
    for i, blk in enumerate(model.blocks, start=1):
        data["blocks"].append({
            "W1": _np(blk.lin1.weight).tolist(),
            "b1": _np(blk.lin1.bias).tolist(),
            "W2": _np(blk.lin2.weight).tolist(),
            "b2": _np(blk.lin2.bias).tolist(),
            "rBias": _np(blk.bias).tolist()
        })
    data["head"] = {"W": _np(model.head.weight).tolist(), "b": _np(model.head.bias).tolist()}
    path = os.path.join(out_dir, "weights.json")
    with open(path, "w") as f: json.dump(data, f, indent=2)
    print(f"[params] wrote JSON -> {os.path.abspath(path)}")

def _latex_matrix(M: np.ndarray, fmt: str = "{:.6f}") -> str:
    M = np.asarray(M, float)
    rows = [" & ".join(fmt.format(float(x)) for x in r) for r in M]
    return "\\begin{bmatrix}\n" + " \\\\ \n".join(rows) + "\n\\end{bmatrix}"

def write_latex_params(model: Width2ResNet, out_dir: str, depth: int, seed: int, title: str = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, f"weights_depth{depth}_seed{seed}.tex")
    title = title or f"Width-2 ResNet Parameters (depth={depth}, seed={seed})"
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{amsmath,amssymb,mathtools}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\begin{document}",
        rf"\section*{{{title}}}",
        r"All matrices are shown with 6 decimal places.\\",
    ]
    for i, blk in enumerate(model.blocks, start=1):
        W1 = _np(blk.lin1.weight); b1 = _np(blk.lin1.bias)
        W2 = _np(blk.lin2.weight); b2 = _np(blk.lin2.bias)
        rB = _np(blk.bias).reshape(1,-1)
        lines += [
            rf"\subsection*{{Block {i}}}",
            r"$W_{1,"+str(i)+r"} = " + _latex_matrix(W1) + r"$,\\",
            r"$b_{1,"+str(i)+r"} = " + _latex_matrix(b1.reshape(1,-1)) + r"$,\\",
            r"$W_{2,"+str(i)+r"} = " + _latex_matrix(W2) + r"$,\\",
            r"$b_{2,"+str(i)+r"} = " + _latex_matrix(b2.reshape(1,-1)) + r"$,\\",
            r"$b_{\text{res},"+str(i)+r"} = " + _latex_matrix(rB) + r"$. ",
        ]
    HW = _np(model.head.weight); Hb = _np(model.head.bias).reshape(1,-1)
    lines += [
        r"\subsection*{Head}",
        r"$W_{\text{head}} = " + _latex_matrix(HW) + r"$,\\",
        r"$b_{\text{head}} = " + _latex_matrix(Hb) + r"$. ",
        r"\end{document}"
    ]
    with open(tex_path, "w") as f: f.write("\n".join(lines))
    print(f"[params] wrote LaTeX -> {os.path.abspath(tex_path)}")
    return tex_path

def try_compile_pdf(tex_path: str):
    workdir = os.path.dirname(os.path.abspath(tex_path)); tex_file = os.path.basename(tex_path)
    if shutil.which("pdflatex") is None:
        print("[params] pdflatex not found; skipping PDF. Compile manually if needed.")
        return None
    try:
        res = subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",tex_file],
                             cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(os.path.join(workdir, tex_file.replace(".tex",".log")), "w") as f: f.write(res.stdout)
        pdf_path = os.path.join(workdir, tex_file.replace(".tex",".pdf"))
        print(f"[params] compiled PDF -> {pdf_path}" if os.path.exists(pdf_path) else "[params] PDF missing; check log")
        return pdf_path if os.path.exists(pdf_path) else None
    except Exception as e:
        print("[params] pdflatex error:", e); return None

def dump_parameter_artifacts(model: Width2ResNet, out_dir: str, depth: int, seed: int):
    print_parameter_summary(model)
    save_parameters_json(model, out_dir)
    tex_path = write_latex_params(model, out_dir, depth, seed)
    try_compile_pdf(tex_path)


# =========================
# Experiment loop
# =========================
def run_experiments(
    depths=(3,4,5),
    seeds=(101,102,103,104,105),
    train_per_class=5000,
    val_per_class=5000,
    plot_resample_factor=5,  # denser plotting set
    noise=0.00,
    epochs=400,
    patience=100,
    batch_size=2048,
    angle_cmap="twilight"
):
    run_id = ts(); base_out = f"results/{run_id}"
    os.makedirs(base_out, exist_ok=True)
    print(f"RUN {run_id}: depths={depths} seeds={seeds}")

    total = len(depths)*len(seeds); k=0
    for d in depths:
        print(f"\n=== Depth {d} ===")
        for s in seeds:
            k += 1; print(f"[{k}/{total}] depth={d} seed={s}")
            seed_everything(s)

            # Train/val
            train_set = DiskAnnulusDataset(train_per_class, train_per_class, noise=noise, seed=s)
            val_set   = DiskAnnulusDataset(val_per_class,   val_per_class,   noise=noise, seed=s+100_000)
            Xtr, ytr = train_set.sample(); Xva, yva = val_set.sample()

            # Denser plotting set (colors computed from ORIGINAL inputs of this set)
            if plot_resample_factor and plot_resample_factor > 1:
                plot_set = DiskAnnulusDataset(val_per_class*plot_resample_factor,
                                              val_per_class*plot_resample_factor,
                                              noise=noise, seed=s+200_000)
                Xplot, yplot = plot_set.sample()
            else:
                Xplot, yplot = Xva, yva
            colors_rgba, colors_hex = make_persistent_polar_colors(Xplot, angle_cmap=angle_cmap)

            # Tensors/loaders
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
            ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
            Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
            yva_t = torch.tensor(yva, dtype=torch.long, device=device)
            g = torch.Generator(device="cpu").manual_seed(s)
            train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size,
                                      shuffle=True, generator=g, num_workers=0)
            val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=batch_size,
                                      shuffle=False, num_workers=0)

            # Train
            model = Width2ResNet(depth=d).to(device)
            train_stage_a(model, train_loader, val_loader, epochs=epochs, lr=1e-3,
                          patience=patience, log_every=100)
            acc, min_m, mean_m, slack = eval_margin_metrics(model, val_loader)
            print(f"  Final: acc={acc:.2f}% | slack={slack:.6f} | min_margin={min_m:.6f}")

            # Output dir for this run
            out_dir = f"{base_out}/{ts()}_depth{d}_seed{s}"
            os.makedirs(out_dir, exist_ok=True)

            # Plots
            plot_layerwise_2d(model, Xplot, colors_rgba, colors_hex,
                              os.path.join(out_dir, "layers"), title_prefix=f"Depth {d}, seed {s}")
            plot_blockwise_transforms(model, Xplot, colors_rgba, colors_hex,
                                      os.path.join(out_dir, "blocksteps"),
                                      title_prefix=f"Depth {d}, seed {s}")

            # Radial warp + dense sign check
            radial_reverse_engineer(model, 256, 360, os.path.join(out_dir, "radial"),
                                    title=f"Depth {d}, seed {s}: radial warp")
            sign_report = dense_sign_check(model, R=1024, TH=720)
            with open(os.path.join(out_dir, "sign_check.json"), "w") as f:
                json.dump(sign_report, f, indent=2)
            print("  Dense sign check (min |decision| over each band):", sign_report)

            # Piecewise‑affine diagnostics
            probes = np.array([
                [0.0, 0.0],
                [0.7, 0.0],
                [1.2, 0.0],
                [1.6, 0.0],
                [1.9/np.sqrt(2), 1.9/np.sqrt(2)],
            ], dtype=np.float32)
            dump_affine_report_for_points(model, probes, os.path.join(out_dir, "affine_report.json"))

            # Activation tiling
            activation_tiling(model, extent=2.2, N=400, out_png=os.path.join(out_dir, "activation_tiling.png"))

            # Parameters
            params_dir = os.path.join(out_dir, "params"); os.makedirs(params_dir, exist_ok=True)
            dump_parameter_artifacts(model, params_dir, depth=d, seed=s)

            if torch.cuda.is_available():
                del model; torch.cuda.empty_cache()

    print(f"\nSaved all runs under: {os.path.abspath(base_out)}")


def main():
    run_experiments(
        depths=(3,4,5),
        seeds=(103,104,105),
        train_per_class=5000,
        val_per_class=5000,
        plot_resample_factor=5,   # 5× denser plotting than val
        noise=0.00,
        epochs=400,
        patience=100,
        batch_size=2048,
        angle_cmap="twilight"     # try "twilight_shifted" as an alternative
    )

if __name__ == "__main__":
    main()
