"""Add the two validation-split winter weeks (2026-06-01 = start 29184, 2026-08-10 =
start 35904) to the seasonal evaluation, drop the near-duplicate test-June week
(start 30530 overlaps 630/672 steps with 30488), and write the 9-week parquet."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import eval_seasonal as es

WEEKS = [(29184, "jun-val-winter"), (35904, "aug-val-winter")]


def main() -> None:
    tasks = []
    for start, month in WEEKS:
        tasks.append(("idle", None, None, None, start, month))
        tasks.append(("B0", "B0", None, None, start, month))
        for arm in ["S0", "S1", "S2", "S3"]:
            rel = {"S2": "S2", "S3": "S3"}.get(arm)
            head = (str(ROOT / "runs" / "protocol" / "reliability_head.pt")
                    if arm == "S3" else None)
            for seed in range(10):
                tasks.append((arm, str(ROOT / "runs" / "protocol" /
                                       f"sac_{arm}_seed{seed}.zip"), rel, head,
                              start, month))
    print(f"{len(tasks)} tasks x 22 conditions")

    rows = []
    with ProcessPoolExecutor(max_workers=8, initializer=es._init_worker) as ex:
        for i, r in enumerate(ex.map(es.one_task, tasks), 1):
            rows.extend(r)
            if i % 40 == 0:
                print(f"  {i}/{len(tasks)}")
    new = pd.DataFrame(rows)
    old = pd.read_parquet(ROOT / "runs" / "protocol" / "results_seasonal.parquet")
    all9 = pd.concat([old, new], ignore_index=True)
    all9 = all9[all9["start"] != 30530]  # drop near-duplicate test-June week
    out = ROOT / "runs" / "protocol" / "results_seasonal9.parquet"
    all9.to_parquet(out, index=False)
    print(f"wrote {out} ({len(all9)} rows, {all9.start.nunique()} weeks)")


if __name__ == "__main__":
    main()
