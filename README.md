# sac-bess: SAC battery dispatch under naturalistic sensor degradation

Code for the empirical study *Reliability-Aware State Encoding for SAC Battery
Dispatch in Solar PV Microgrids under Naturalistic Sensor Degradation*: one
grid-connected C&I site in South Africa (13 months of 15-minute VCOM metering,
38,016 intervals), a simulated 2,000 kWh / 1,000 kW battery behind the meter,
economic dispatch against a TOU energy tariff and a monthly maximum-demand
charge, and the question of how the observation pre-processing stage determines
how well a fixed SAC learner survives sensor faults.

## Arms and faults

| Arm | Observation seen by SAC |
|---|---|
| B0 | rule-based floor (peak-window discharge, exposure cap, off-peak charge) |
| S0 | clean observation |
| S1 | raw corrupted observation (robustness by accident of training) |
| S2 | corrupted observation + deterministic gate-and-flag encoding (obs 22 → 46) |
| S3 | corrupted observation + learned reliability head, pretrained on the corruption mask |

Six fault types × 3 severities operate on the 8 raw sensor channels: F1 dropout,
F2 zero-stuck, F3 spikes/sentinels, F4 slowly-drifting bias (trained on, via a
Markov health process fitted to the site's own data-quality audit); F5 metering
drift and F6a/F6b temporal misalignment (held out). Severity 0 is a bitwise
passthrough. Fault parameters live in `src/sacbess/corruption/faults.py`;
ground truth never enters the observation, only `info["corruption_mask"]`.

Env chain: `TruncationBootstrap(TimeLimit(ScaleObservation(ReliabilityWrapper(CorruptionWrapper(MicrogridEnv))))`
(+ dispatch features). Reward and dynamics always consume the true series;
corruption perturbs observations only.

## Layout

```
src/sacbess/data/         VCOM export -> canonical parquet, splits, scaler
src/sacbess/env/          microgrid Gymnasium env, battery, B0 rule, dispatch features
src/sacbess/corruption/   F1-F6 fault processes + Markov health wrapper
src/sacbess/reliability/  stage-1 features, S2 rule, S3 heads (MLP + causal CNN)
src/sacbess/rl/           SAC-from-demonstrations replay buffer
scripts/                  training, protocol, and analysis entry points
configs/env.yaml          battery + tariff + reward flags used by every run
data/processed/           committed: canonical.parquet, splits, scaler, episode starts
runs/                     committed results (CSV/parquet/logs); NOT the 64 MB checkpoints
tests/                    28 tests: env contract, corruption invariants, reliability behaviour
```

The raw VCOM export (`data/vcom_data_export.csv`) is real client metering data
and is not published; `data/processed/canonical.parquet` is committed, so every
training and evaluation step below runs out of the box.

## Reproduction chain

```bash
pip install -e .[dev]                # python >= 3.11
pytest tests/                        # 28 tests

# study protocol (defaults = the study recipe: 300k steps, 10 seeds, 1024x1024,
# B0 demonstrations with linear decay, dispatch features). Training skips models
# that already exist - delete runs/protocol/sac_*.zip to retrain (CPU, ~8 h/arm).
python scripts/train_head.py                             # S3 head (run_protocol does this too)
python scripts/run_protocol.py train --workers 8
python scripts/run_protocol.py eval
python scripts/run_protocol.py analyze                   # IQM + paired bootstrap

# detection benchmarks (Appendix F + the channel-matched AUROC table)
python scripts/analysis/head_benchmark.py
python scripts/analysis/head_benchmark_channel_matched.py

# seasonally stratified 9-week evaluation: run BOTH, in this order
python scripts/analysis/eval_seasonal.py                 # 8 test weeks
python scripts/analysis/eval_seasonal_add.py             # +2 winter weeks, -> results_seasonal9.parquet

# checkpoint ladder / extra arms through the same 22-condition matrix
python scripts/analysis/eval_ladder.py --dir <ckpt_dir> --tag <tag>
python scripts/analysis/eval_arm.py --dir <model_dir> --label <label>

# perfect-foresight LP bound and reward audit
python scripts/analysis/oracle_bound.py
python scripts/analysis/reward_audit.py
```

`scripts/build_canonical.py` rebuilds `data/processed/` from the raw export
when available; the committed artifacts make it optional.

## Observation vector (22 dims)

`pm1_kw, pm0_a_kw, pm0_b_kw, irr1_wm2, irr2_wm2, irr1_temp_c, irr2_temp_c,
sat_gti_wm2, fcst_pv_t1h_kw, fcst_pv_t3h_kw, fcst_pv_t6h_kw, tariff_zar_kwh,
demand_exposure_kw, soc, charge_headroom_kw, discharge_headroom_kw,
sin/cos(hour), sin/cos(dow), sin/cos(doy)`

F1-F5 corrupt sensor channels 0-7 only; time encodings are never corrupted
except under F6a's coherent index shift. The scaler is fitted on training rows
only (`data/processed/scaler.npz`).

## Two reward-side flags worth knowing

`configs/env.yaml` enables both; both default off in `EnvConfig` for exact
replication of the raw dynamics.

- `charge_guard` — clamps charging so a 30-min window cannot import above the
  running demand peak (the LP oracle never benefits from exceeding it on this
  site; see `scripts/analysis/oracle_bound.py`).
- `shaping_lambda_zar_per_kwh` — potential-based shaping on stored-energy
  value, `F = gamma*phi(s') - phi(s)`; policy-invariant, evaluation economics
  in `info` are unchanged.
