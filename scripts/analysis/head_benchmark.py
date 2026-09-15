"""Detection benchmark for reliability heads: deterministic rule vs pointwise MLP
vs temporal CNN, scored per fault x severity against the corruption wrapper's
ground-truth masks (trained on MIX_TRAIN = F1-F4; F5/F6a/F6b held out).

  python scripts/analysis/head_benchmark.py [--episodes 24] [--eval-eps 2] [--width 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml


def collect(config, priors, canonical, fault, severity, n_eps, starts, seed, split="train"):
    """Collect (feats (N,T,8,6), masks (N,T,8)) episodes for one fault condition."""
    import gymnasium

    from sacbess.corruption.wrapper import CorruptionWrapper
    from sacbess.env.microgrid_env import MicrogridEnv
    from sacbess.reliability.features import N_SENSOR, ReliabilityFeatures

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(starts), size=min(n_eps, len(starts)), replace=False)
    feats_all, mask_all = [], []
    for i in sel:
        env = gymnasium.wrappers.TimeLimit(
            CorruptionWrapper(
                MicrogridEnv(canonical, config, split),
                fault=fault if fault != "none" else "none",
                severity=severity,
            ),
            max_episode_steps=config.episode_steps,
        )
        obs, info = env.reset(seed=int(rng.integers(1 << 31)),
                              options={"start_idx": int(starts[i])})
        stream = ReliabilityFeatures(priors)
        stream.reset()
        ep_f, ep_m = [], []
        done = False
        while not done:
            f, _ = stream.step(obs, info["obs_pos"])
            ep_f.append(f)
            ep_m.append(info["corruption_mask"][:N_SENSOR])
            obs, _, term, trunc, info = env.step(np.zeros(1))
            done = term or trunc
        feats_all.append(np.stack(ep_f))
        mask_all.append(np.stack(ep_m))
    return np.stack(feats_all), np.stack(mask_all).astype(np.float32)


def auroc(y, s):
    """Rank-based AUROC with tie handling (Mann-Whitney U / Pearson)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_at(y, s, thr=0.5):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(s) >= thr
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else float("nan")
    rec = tp / (tp + fn) if tp + fn > 0 else float("nan")
    return prec, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--eval-eps", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from sacbess.data.splits import build_splits, row_split_mask
    from sacbess.env.config import EnvConfig
    from sacbess.env.microgrid_env import MicrogridEnv
    from sacbess.reliability.conv_head import FrozenConvHead, train_conv_head
    from sacbess.reliability.deterministic import r_det_from_features
    from sacbess.reliability.features import SitePriors
    from sacbess.reliability.learned_head import build_head, train_head

    with open(ROOT / "configs" / "env.yaml") as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))
    canonical = ROOT / "data" / "processed" / "canonical.parquet"
    core = MicrogridEnv(canonical, config, "train")
    weeks = build_splits(core.frame, embargo_steps=96, seed=0)
    train_mask = row_split_mask(core.frame, weeks, "train")
    priors = SitePriors.fit(core.frame, train_mask)
    del core

    out_dir = ROOT / "runs" / "head_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- training data: identical MIX_TRAIN/none episode mix for both heads ----
    print(f"collecting {args.episodes} training episodes (MIX_TRAIN mix)...")
    rng = np.random.default_rng(args.seed)
    tr_core = MicrogridEnv(canonical, config, "train")
    starts = tr_core.valid_starts["train"]
    del tr_core
    f_mix, m_mix = collect(config, priors, canonical, "MIX_TRAIN", 0,
                           args.episodes, starts, args.seed + 1)
    f_cl, m_cl = collect(config, priors, canonical, "none", 0,
                         max(args.episodes // 2, 4), starts, args.seed + 2)
    feats_tr = np.concatenate([f_mix, f_cl])
    masks_tr = np.concatenate([m_mix, np.zeros_like(m_cl, dtype=np.float32)])
    print(f"train set: {len(feats_tr)} episodes, mask_rate {masks_tr.mean():.3f}")

    print("\ntraining pointwise MLP head (512)...")
    if not (out_dir / "mlp_head.pt").exists():
        stats_mlp = train_head(feats_tr, masks_tr, out_dir / "mlp_head.pt",
                               epochs=args.epochs, seed=args.seed, hidden=512)
        print(f"  {stats_mlp}")
    else:
        print("  (reuse existing mlp_head.pt)")
    print("training temporal CNN head (width %d, %s)..." % (
        args.width, "mps" if torch.backends.mps.is_available() else "cpu"))
    if not (out_dir / "conv_head.pt").exists():
        stats_cnn = train_conv_head(feats_tr, masks_tr, out_dir / "conv_head.pt",
                                    epochs=args.epochs, seed=args.seed, width=args.width)
        print(f"  {stats_cnn}")
    else:
        print("  (reuse existing conv_head.pt)")

    # frozen scorers
    import torch as th

    class FrozenMLP:
        def __init__(self, path):
            ckpt = th.load(path, map_location="cpu")
            hidden = int(ckpt.get("hidden", 512))
            self.head = build_head(hidden=hidden)
            self.head.load_state_dict(ckpt["state_dict"])
            self.head.eval()
            self.ch = th.arange(8)

        def p_seq(self, feats):  # (T,8,6) -> (T,8)
            from sacbess.reliability.features import FEAT_DIM

            with th.no_grad():
                f = th.tensor(feats.reshape(-1, FEAT_DIM), dtype=th.float32)
                c = th.tile(th.arange(8), (len(feats),))
                return th.sigmoid(self.head(f, c)).numpy().reshape(len(feats), 8)

    mlp = FrozenMLP(out_dir / "mlp_head.pt")
    cnn = FrozenConvHead(out_dir / "conv_head.pt")

    # ---- evaluation: every fault x severity + clean, on VAL-split episodes ----
    print(f"\nbenchmarking detection (eval episodes per condition: {args.eval_eps})...")
    val_starts = MicrogridEnv(canonical, config, "val").valid_starts["val"]
    header = f"{'cond':>6} {'mask%':>6} | {'rule':>22} | {'MLP':>22} | {'CNN':>22}"
    print(header)
    print(f"{'':>6} {'':>6} | {'AUROC':>7} {'P':>6} {'R':>6} | {'AUROC':>7} {'P':>6} {'R':>6} | {'AUROC':>7} {'P':>6} {'R':>6}")
    print("-" * len(header))
    rows = []
    for fault, sev in ([("none", 0)] + [(f, s) for f in
                                        ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b"]
                                        for s in [1, 2, 3]]):
        f_e, m_e = collect(config, priors, canonical,
                           fault if fault != "none" else "none",
                           sev,
                           args.eval_eps, val_starts, args.seed + 100 + sev * 10
                           + (ord(fault[0]) if fault != "none" else 0),
                           split="val")
        y = m_e.reshape(-1)
        mask_pct = 100 * y.mean()
        # rule: deterministic r from features (single-step API) -> p = 1 - r
        rule_r = np.stack([r_det_from_features(f) for f in f_e.reshape(-1, 8, 6)])
        p_rule = (1.0 - rule_r).reshape(-1)
        p_mlp = mlp.p_seq(f_e.reshape(-1, 8, 6)).reshape(-1)
        p_cnn = cnn.p_seq(f_e.reshape(-1, 8, 6)).reshape(-1)
        cells = []
        for p in (p_rule, p_mlp, p_cnn):
            if fault == "none":
                cells.append((float("nan"), *pr_at(y, p, thr=0.5)))
            else:
                cells.append((auroc(y, p), *pr_at(y, p, thr=0.5)))
        label = "clean" if fault == "none" else f"{fault}{sev}"
        print(f"{label:>6} {mask_pct:>5.1f}% | " + " | ".join(
            f"{a:>7.3f} {p:>6.3f} {r:>6.3f}" for a, p, r in cells))
        rows.append({"cond": label, "mask_pct": mask_pct,
                     "rule_auroc": cells[0][0], "rule_p": cells[0][1], "rule_r": cells[0][2],
                     "mlp_auroc": cells[1][0], "mlp_p": cells[1][1], "mlp_r": cells[1][2],
                     "cnn_auroc": cells[2][0], "cnn_p": cells[2][1], "cnn_r": cells[2][2]})

    import pandas as pd

    out_csv = out_dir / "benchmark.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("clean row: P = false-positive rate (no true corruptions), AUROC n/a")


if __name__ == "__main__":
    main()
