"""Channel-matched detection benchmark.

Pooled AUROC over all eight sensor channels rewards any scorer that merely ranks the
channels a fault happens to target above the rest. F4 only ever hits IRR1/IRR2, F5 only
PM0_A/PM0_B, F6b only SAT_GTI, and F6a masks every channel. A head that has learned only
a channel prior therefore scores near 1.0 on F4 and below chance on F5 without detecting
anything at all.

This script reports, per fault x severity:
  * pooled AUROC (what head_benchmark.py reports), and
  * AUROC computed WITHIN each affected channel (channel-matched),
plus each scorer's clean-episode mean p(corrupted) per channel, which is the prior that
inflates the pooled figure.

The head forward pass is reimplemented in numpy (_numpy_torch_load.py) so this runs
without torch.

  python scripts/analysis/head_benchmark_channel_matched.py
  HEAD_PATH=runs/protocol/reliability_head.pt python scripts/analysis/head_benchmark_channel_matched.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

CB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CB / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _numpy_torch_load as ptload  # noqa: E402
import gymnasium  # noqa: E402

from sacbess.corruption.wrapper import CorruptionWrapper  # noqa: E402
from sacbess.data.splits import build_splits, row_split_mask  # noqa: E402
from sacbess.env.config import EnvConfig  # noqa: E402
from sacbess.env.microgrid_env import MicrogridEnv  # noqa: E402
from sacbess.reliability.deterministic import r_det_from_features  # noqa: E402
from sacbess.reliability.features import N_SENSOR, ReliabilityFeatures, SitePriors  # noqa: E402

CH = ["PM1", "PM0_A", "PM0_B", "IRR1", "IRR2", "IRR1_T", "IRR2_T", "SAT_GTI"]


class NPHead:
    """numpy forward pass of learned_head.build_head (Linear-ReLU-Linear-ReLU-Linear)."""

    def __init__(self, path):
        sd = ptload.load(path)["state_dict"]
        self.E = sd["embed.weight"]
        self.W = [sd["net.0.weight"], sd["net.2.weight"], sd["net.4.weight"]]
        self.b = [sd["net.0.bias"], sd["net.2.bias"], sd["net.4.bias"]]

    def p(self, feats):                       # (N,8,6) -> (N,8)
        n = feats.shape[0]
        x = np.concatenate([feats.reshape(-1, feats.shape[-1]), np.tile(self.E, (n, 1))], -1)
        h = np.maximum(x @ self.W[0].T + self.b[0], 0)
        h = np.maximum(h @ self.W[1].T + self.b[1], 0)
        z = (h @ self.W[2].T + self.b[2]).ravel()
        return (1 / (1 + np.exp(-z))).reshape(n, N_SENSOR)


def auroc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    sr = s[o]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def collect(config, priors, canonical, fault, severity, n_eps, starts, seed, split="val"):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(starts), size=min(n_eps, len(starts)), replace=False)
    F, M = [], []
    for i in sel:
        env = gymnasium.wrappers.TimeLimit(
            CorruptionWrapper(MicrogridEnv(canonical, config, split),
                              fault=fault if fault != "none" else "none", severity=severity),
            max_episode_steps=config.episode_steps)
        obs, info = env.reset(seed=int(rng.integers(1 << 31)),
                              options={"start_idx": int(starts[i])})
        stream = ReliabilityFeatures(priors)
        stream.reset()
        ef, em = [], []
        done = False
        while not done:
            f, _ = stream.step(obs, info["obs_pos"])
            ef.append(f)
            em.append(info["corruption_mask"][:N_SENSOR])
            obs, _, t, tr, info = env.step(np.zeros(1))
            done = t or tr
        F.append(np.stack(ef))
        M.append(np.stack(em))
    return np.stack(F), np.stack(M).astype(np.float32)


def main():
    head_path = os.environ.get("HEAD_PATH") or str(CB / "runs" / "head_bench" / "mlp_head.pt")
    eval_eps = int(os.environ.get("EVAL_EPS", "2"))
    with open(CB / "configs" / "env.yaml") as f:
        config = EnvConfig.from_dict(yaml.safe_load(f))
    canonical = CB / "data" / "processed" / "canonical.parquet"
    core = MicrogridEnv(canonical, config, "train")
    weeks = build_splits(core.frame, embargo_steps=96, seed=0)
    priors = SitePriors.fit(core.frame, row_split_mask(core.frame, weeks, "train"))
    del core
    val_starts = MicrogridEnv(canonical, config, "val").valid_starts["val"]
    head = NPHead(head_path)
    print(f"head: {head_path}  ({eval_eps} val episodes per condition)")

    fc, _ = collect(config, priors, canonical, "none", 0, eval_eps, val_starts, 100)
    pc = head.p(fc.reshape(-1, 8, 6))
    rc = np.stack([r_det_from_features(x) for x in fc.reshape(-1, 8, 6)])
    print("\n== clean-episode mean p(corrupted) per channel (the channel prior) ==")
    print("  head:", "  ".join(f"{c}={pc[:, i].mean():.3f}" for i, c in enumerate(CH)))
    print("  rule:", "  ".join(f"{c}={1 - rc[:, i].mean():.3f}" for i, c in enumerate(CH)))

    print("\n== pooled vs channel-matched AUROC ==")
    print(f"{'cond':>6} {'mask%':>6} | {'rule pool':>9} {'head pool':>9} | channel-matched (head / rule)")
    rows = []
    for fault in ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b"]:
        for sev in [1, 2, 3]:
            f_e, m_e = collect(config, priors, canonical, fault, sev, eval_eps, val_starts,
                               100 + sev * 10 + ord(fault[0]))
            f2, m2 = f_e.reshape(-1, 8, 6), m_e.reshape(-1, 8)
            ph = head.p(f2)
            pr = 1 - np.stack([r_det_from_features(x) for x in f2])
            cm = [(CH[c], auroc(m2[:, c], ph[:, c]), auroc(m2[:, c], pr[:, c]))
                  for c in range(8) if not np.isnan(auroc(m2[:, c], ph[:, c]))]
            print(f"{fault + str(sev):>6} {100 * m2.mean():>5.1f}% | "
                  f"{auroc(m2.ravel(), pr.ravel()):>9.3f} {auroc(m2.ravel(), ph.ravel()):>9.3f} | "
                  + ("  ".join(f"{n}:{a:.2f}/{b:.2f}" for n, a, b in cm) or
                     "undefined - fault masks whole channels, no within-channel negatives"))
            rows.append(dict(cond=f"{fault}{sev}", mask_pct=100 * m2.mean(),
                             rule_pooled=auroc(m2.ravel(), pr.ravel()),
                             head_pooled=auroc(m2.ravel(), ph.ravel()),
                             head_cm=float(np.mean([a for _, a, _ in cm])) if cm else float("nan"),
                             rule_cm=float(np.mean([b for _, _, b in cm])) if cm else float("nan")))
    out = CB / "runs" / "head_bench" / ("channel_matched_" + Path(head_path).stem + ".csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
