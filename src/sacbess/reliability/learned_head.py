"""S3 learned reliability head: shared MLP over the stage-1 feature block + channel embedding.

Supervised from the corruption wrapper's ground-truth mask with BCE and class weighting;
pretrained offline and frozen (gradient-detached) during RL training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FEAT_DIM, N_SENSOR


def build_head(hidden: int = 512, emb_dim: int = 4):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(N_SENSOR, emb_dim)
            self.net = nn.Sequential(
                nn.Linear(FEAT_DIM + emb_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, max(hidden // 2, 16)),
                nn.ReLU(),
                nn.Linear(max(hidden // 2, 16), 1),
            )

        def forward(self, feats, ch):
            h = torch.cat([feats, self.embed(ch)], dim=-1)
            return self.net(h).squeeze(-1)

    return Head()


def train_head(
    feats: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    epochs: int = 12,
    batch: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    val_frac: float = 0.1,
    hidden: int = 512,
) -> dict:
    """feats: (N, 8, 6); mask: (N, 8) booleans. Samples are (features, channel) pairs."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(feats)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    head = build_head(hidden=hidden)

    def torchify(sel):
        f = torch.tensor(feats[sel].reshape(-1, FEAT_DIM), dtype=torch.float32)
        m = torch.tensor(mask[sel].reshape(-1), dtype=torch.float32)
        c = torch.tensor(np.tile(np.arange(N_SENSOR), len(m) // N_SENSOR), dtype=torch.long)
        return f, m, c

    f_tr, m_tr, c_tr = torchify(tr_idx)
    f_va, m_va, c_va = torchify(val_idx)
    pos = m_tr.sum()
    neg = len(m_tr) - pos
    ratio = float(neg / pos) if pos > 0 else 1.0
    pos_weight = torch.tensor([min(ratio, 4.0)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(f_tr))
        losses = []
        for lo in range(0, len(order), batch):
            sel = order[lo : lo + batch]
            opt.zero_grad()
            logits = head(f_tr[sel], c_tr[sel])
            loss = lossf(logits, m_tr[sel])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.binary_cross_entropy_with_logits(
                head(f_va, c_va), m_va
            ))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "hidden": hidden}, out_path)
    return {"val_bce": best_val, "n_train": len(f_tr), "path": str(out_path), "hidden": hidden}


class FrozenHead:
    def __init__(self, path: str | Path):
        import torch

        self.torch = torch
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            hidden = int(ckpt.get("hidden", 512))
            state = ckpt["state_dict"]
        else:  # legacy plain state-dict checkpoint (32-wide head)
            hidden = 32
            state = ckpt
        self.head = build_head(hidden=hidden)
        self.head.load_state_dict(state)
        self.head.eval()

    def r_learned(self, feats: np.ndarray) -> np.ndarray:
        """feats: (8,6) or (T,8,6) -> reliability r = 1 - p(corrupted), same leading shape."""
        single = feats.ndim == 2
        f = feats[None] if single else feats
        t = self.torch.tensor(f.reshape(-1, FEAT_DIM), dtype=self.torch.float32)
        c = self.torch.tile(self.torch.arange(N_SENSOR), (f.shape[0],))
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.head(t, c)).numpy()
        r = 1.0 - p
        r = r.reshape(f.shape[0], f.shape[1])
        return r[0] if single else r
