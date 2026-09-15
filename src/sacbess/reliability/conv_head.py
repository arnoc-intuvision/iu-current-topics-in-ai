"""Temporal (1D causal CNN) reliability head: per-sensor corruption probability
from a causal window of the stage-1 feature stream.

The pointwise MLP head sees one feature snapshot per channel and therefore cannot
represent inherently temporal faults (zero-stuck runs, drift, coherent index
shift). The conv head stacks causal 1D convolutions over the last WINDOW steps of
the same stage-1 features (shared trunk across the 8 sensor channels plus channel
embeddings), emitting a per-channel logit at every step: output at t depends only
on features at steps <= t (left zero-padding at episode start, matching the
deployed rolling buffer).

Interface matches learned_head.FrozenHead: r_learned(feats (8,6)) -> r (8,), with
an internal per-episode buffer that ReliabilityWrapper resets between episodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FEAT_DIM, N_SENSOR

WINDOW = 96  # 24 h of 15-min context


def build_conv_head(width: int = 64, emb_dim: int = 4, kernel: int = 5, layers: int = 3):
    import torch
    from torch import nn

    class CausalConv(nn.Conv1d):
        def __init__(self, c_in, c_out):
            super().__init__(c_in, c_out, kernel, padding=0)
            self.left_pad = kernel - 1

        def forward(self, x):
            x = nn.functional.pad(x, (self.left_pad, 0))
            return super().forward(x)

    class ConvHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(N_SENSOR, emb_dim)
            chans = [FEAT_DIM + emb_dim] + [width] * layers
            self.trunk = nn.Sequential(*[CausalConv(ci, co) for ci, co in zip(chans[:-1], chans[1:])])
            self.out = nn.Linear(width, 1)

        def forward(self, feats, ch):
            """feats (B, C, T, FEAT_DIM); ch (C,) long. Returns logits (B, C, T).
            Channels share the trunk: fold C into the batch dim for conv1d."""
            b, c, t, _ = feats.shape
            emb = self.embed(ch)[None, :, None, :].expand(b, c, t, emb_dim)
            x = torch.cat([feats, emb], dim=-1).permute(0, 1, 3, 2)   # (B, C, F+e, T)
            x = x.reshape(b * c, x.shape[2], t)                       # (B*C, F+e, T)
            h = self.trunk(x)                                         # (B*C, width, T)
            h = h.permute(0, 2, 1).reshape(b, c, t, -1)               # (B, C, T, width)
            return self.out(h).squeeze(-1)

    return ConvHead()


def train_conv_head(
    feats: np.ndarray,          # (N, T, 8, 6)
    mask: np.ndarray,           # (N, T, 8)
    out_path: str | Path,
    epochs: int = 12,
    batch_eps: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    width: int = 64,
    val_frac: float = 0.1,
    device: str = "auto",
) -> dict:
    import torch
    from torch import nn

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = len(feats)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    head = build_conv_head(width=width).to(dev)
    # class weight: negatives/positives over the whole train set
    m_tr = mask[tr_idx].astype(np.float32)
    pos_w = torch.tensor([min((1 - m_tr).mean() / max(m_tr.mean(), 1e-6), 4.0)],
                         dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    ch = torch.arange(N_SENSOR, device=dev)

    def batch(sel):
        f = torch.tensor(feats[sel], dtype=torch.float32, device=dev).permute(0, 2, 1, 3)
        m = torch.tensor(mask[sel], dtype=torch.float32, device=dev).permute(0, 2, 1)
        return f, m

    f_va, m_va = batch(val_idx)
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(tr_idx))
        losses = []
        for lo in range(0, len(order), batch_eps):
            sel = tr_idx[order[lo:lo + batch_eps]]
            f, m = batch(sel)
            opt.zero_grad()
            logits = head(f, ch)
            loss = lossf(logits, m)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            val = float(nn.functional.binary_cross_entropy_with_logits(
                head(f_va, ch), m_va
            ))
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        print(f"  conv epoch {epoch + 1}: train {np.mean(losses):.4f} val {val:.4f}")
    head.load_state_dict(best_state)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"kind": "conv", "state_dict": head.state_dict(), "width": width,
         "window": WINDOW, "val_bce": best_val},
        out_path,
    )
    return {"val_bce": best_val, "n_train": int(len(tr_idx) * feats.shape[1] * N_SENSOR),
            "path": str(out_path), "width": width, "device": str(dev)}


class FrozenConvHead:
    """Drop-in per-step reliability scorer with an internal causal buffer."""

    def __init__(self, path: str | Path):
        import torch

        self.torch = torch
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt.get("kind", "conv") == "conv", f"not a conv head checkpoint: {ckpt.get('kind')}"
        self.window = int(ckpt.get("window", WINDOW))
        self.head = build_conv_head(width=int(ckpt.get("width", 64)))
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        self._buf: list[np.ndarray] = []
        self._ch = torch.arange(N_SENSOR)

    def reset(self) -> None:
        self._buf = []

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        """feats (T,8,6) -> p(corrupted) (T,8), causal over the buffer."""
        with self.torch.no_grad():
            f = self.torch.tensor(feats[None], dtype=self.torch.float32).permute(0, 2, 1, 3)
            p = self.torch.sigmoid(self.head(f, self._ch))[0].numpy().T  # (C,T)->(T,C)
        return p

    def r_learned(self, feats: np.ndarray) -> np.ndarray:
        """feats (8,6) -> r (8,); or (T,8,6) -> r (T,8) for the appended block."""
        single = feats.ndim == 2
        f = feats[None] if single else feats
        t_in = f.shape[0]
        self._buf.extend(list(f))
        if len(self._buf) > self.window:
            self._buf = self._buf[-self.window:]
        p = self._forward(np.stack(self._buf))
        r = 1.0 - p
        return r[-1] if single else r[-t_in:]

    def p_seq(self, feats: np.ndarray) -> np.ndarray:
        """Offline (T,8,6) -> p(corrupted) (T,8) in one causal pass."""
        return self._forward(feats)
