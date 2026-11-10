# SuperMimic Quickstart

SuperMimic runs in a separate Python environment from the official PPO tasks. The
extension pins Python 3.12, mjlab commit `8bb2bc9e187797689ec82c78c35019afd2fd0376`,
MuJoCo/Warp 3.16, and Torch cu130. It does not change the root `setup.py` dependencies.
Run all commands from the repository root unless noted otherwise.

## Compatibility matrix

| Component | Official PPO environment | Isolated SuperMimic environment | Result |
|---|---|---|---|
| Python | 3.11.16 | 3.12.16 | Keep separate |
| mjlab | 1.2.0 | 1.6.0, commit `8bb2bc9e187797689ec82c78c35019afd2fd0376` | Keep separate |
| MuJoCo / MuJoCo Warp | 3.5.0 / 3.5.0 | 3.16.0 / 3.16.0 | Keep separate |
| Torch | 2.14.0, cu130 build | 2.14.1+cu130 | Keep separate |
| G1 model | Root `src/assets/robots/unitree_g1/xmls/g1.xml` | Same root asset, verified 36/35 state dimensions and 29-joint order | Compatible |
| Control rate | 0.005 s x 4 = 50 Hz | 0.005 s x 4 = 50 Hz | Compatible |

The two dependency sets must not be installed into one environment. The extension
uses the root G1 MJCF but keeps its World Model and BPTT trainer independent of
RSL-RL PPO.

## Environment and motion

Install `uv`, then create the isolated environment:

```bash
cd extensions/supermimic
uv sync --group dev
```

Convert the included 30 Hz CSV into the 50 Hz format used by the simulator:

```bash
uv run python scripts/data/csv_to_npz_batch.py --input-dir ../../src/assets/motions/g1 --output-dir data/npz/g1 --input-fps 30 --output-fps 50 --workers 1
```

The converter validates 36 columns: root position in metres, root quaternion in
CSV `xyzw` order, and 29 joint angles in radians in MJCF order. It writes MuJoCo
quaternions as `wxyz` and computes link poses and velocities from the repository's
G1 MJCF. The simulation step is 0.005 s with decimation 4, giving the 50 Hz policy
rate expected by the converted motion.

## Train

This short run is intended for a smoke check on a CUDA-enabled RTX 4090, not for
converged policy training:

```bash
cd extensions/supermimic
uv run python ../../scripts/train_supermimic.py --run-name g1-dance-smoke --data.npz-dir data/npz/g1 --data.clips dance1_subject2 --env.num-envs 64 --wm-trainer.batch-size 64 --wm-trainer.batches-per-update 1 --policy-trainer.batch-size 64 --schedule.total-iterations 80 --schedule.warmup-iterations 40
```

## Play

```bash
cd extensions/supermimic
uv run python ../../scripts/play_supermimic.py --ckpt logs/g1-dance-smoke/ckpt_latest.pt --npz-dir data/npz/g1 --clips dance1_subject2 --num-envs 1 --device cuda:0
```

## Evaluate

```bash
cd extensions/supermimic
uv run python ../../scripts/evaluate_supermimic.py --ckpt logs/g1-dance-smoke/ckpt_latest.pt --npz-dir data/npz/g1 --clips dance1_subject2 --num-envs 1 --device cuda:0
```

The extension includes the upstream numerical tests under `extensions/supermimic/tests`.
Run them with `uv run pytest tests -q` from `extensions/supermimic`. GPU-dependent
simulation tests require an available NVIDIA driver and CUDA device. This module only
trains and evaluates in simulation; it adds no real-robot or deployment connection.
