# SuperMimic Quickstart

SuperMimic 是 `unitree_rl_mjlab` 的插件任务，与官方 PPO 任务并列训练。完整说明见
[`extensions/supermimic/README.md`](../extensions/supermimic/README.md)，本文只给最短上手路径。

## 环境

SuperMimic 固定在扩展内的独立环境运行，与官方 PPO 环境隔离：

| 组件 | 官方 PPO 环境 | SuperMimic 环境 |
|---|---|---|
| Python | 3.11.16 | 3.12.16 |
| mjlab | 1.2.0 | 上游 commit `8bb2bc9e` |
| MuJoCo / MuJoCo Warp | 3.5.0 / 3.5.0 | 3.16.0 / 3.16.0 |
| Torch | 2.14.0 cu130 | 2.14.1 cu130 |

两套依赖不能装进同一环境。`scripts/train_supermimic.py` 会在检测到当前解释器缺少
SuperMimic 依赖时自动 `exec` 进 `extensions/supermimic/.venv`，所以两者用同一条命令即可。

首次准备环境：

```bash
cd extensions/supermimic
uv sync --group dev
```

## 列任务

```bash
cd /home/unitree/workwpace/zjy_ws/unitree_rl_mjlab
python scripts/train_supermimic.py --list-tasks
```

## 训练

默认动作是已验证的 `dance1_subject2`：

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2
```

指定 GPU、环境数、种子、迭代数和输出目录：

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 \
  --gpu-ids 0 \
  --env.num-envs 256 \
  --seed 0 \
  --schedule.total-iterations 320000 \
  --log-root logs
```

换动作只需改 `--data.clips`（值为 `extensions/supermimic/data/npz/lafan1/` 下的文件名）：

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 --data.clips walk1_subject1
```

从 checkpoint 恢复：

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 \
  --resume logs/Supermimic-G1-Dance1-Subject2/ckpt_latest.pt \
  --schedule.total-iterations 400000
```

## 评估与回放

```bash
python scripts/evaluate_supermimic.py --ckpt logs/Supermimic-G1-Dance1-Subject2/ckpt_latest.pt --clips dance1_subject2
python scripts/play_supermimic.py     --ckpt logs/Supermimic-G1-Dance1-Subject2/ckpt_latest.pt --clips dance1_subject2
```

## 日志

```bash
ls logs/Supermimic-G1-Dance1-Subject2/
tensorboard --logdir logs/Supermimic-G1-Dance1-Subject2/tb --port 6006
```

## 动作数据

仓库内置 `extensions/supermimic/data/npz/lafan1/`（40 个 LAFAN1 clip，50 Hz）。
转换自己的 CSV：

```bash
cd extensions/supermimic
uv run python scripts/data/csv_to_npz_batch.py --input-dir ../../src/assets/motions/g1 --output-dir data/npz/g1 --input-fps 30 --output-fps 50 --workers 1
```

转换结果 36 列：root 位置（米）、root 四元数（CSV 为 `xyzw` 序）、29 个关节角（弧度，MJCF 序）。
输出为 MuJoCo 的 `wxyz` 四元数，并按仓库 G1 MJCF 计算 link 位姿与速度。
仿真步长 0.005 s、decimation 4，即 50 Hz 策略频率。
