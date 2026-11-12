# SuperMimic

SuperMimic 是 `unitree_rl_mjlab` 的一个**插件任务**，实现
[SuperMimic (arXiv 2610.28480)](https://arxiv.org/abs/2610.28480) 的 G1 全身动作模仿：
可微 World Model + BPTT 训练一个 29-DoF 人形机器人的 tracking policy。

它不是独立研究工程。训练、评估、Play 都由仓库统一的入口启动，与 PPO 任务并列。

```
unitree_rl_mjlab/
├── scripts/
│   ├── train.py                    # PPO 训练
│   ├── train_supermimic.py         # ← SuperMimic 训练（与 PPO 相同的 CLI 形态）
│   ├── play_supermimic.py          # ← SuperMimic 可视化回放
│   └── evaluate_supermimic.py      # ← SuperMimic 批量评估
└── extensions/supermimic/
    ├── src/supermimic/
    │   ├── tasks/registry.py       # ← 任务注册表（动作选择 + 验证过的默认值）
    │   ├── config.py               # TrainConfig（tyro 命令行根）
    │   ├── training/               # 训练循环、调度、分布式、环境快照
    │   ├── world_model/            # World Model、动力学、heads
    │   ├── imagination/            # BPTT rollout、策略目标函数、policy trainer
    │   ├── policy/                 # 网络结构、部署运行时
    │   ├── envs/                   # mjlab 环境、动作、观测、奖励、终止
    │   ├── data/                   # 动作转换、replay buffer、recorder
    │   ├── kinematics/             # G1 正运动学、旋转工具
    │   ├── eval/                   # 论文 Appendix F 协议评估
    │   ├── validation/             # Simulation validation 与 CPU/physics worker
    │   ├── dds/                    # DDS 传输（C++ .so + Python 绑定）
    │   ├── sim_adapter.py          # Sim2Sim 适配层
    │   ├── real_adapter.py         # Sim2Real 实机适配层
    │   ├── real_ownership.py       # 实机控制权租约
    │   ├── s10a.py / s10b*.py      # 单动作接管、姿态恢复、原生控制器
    │   └── dds_*.py                # DDS 部署与运行时
    ├── scripts/                    # 评估与回放脚本、数据转换
    ├── tests/                      # 单元与集成测试
    └── data/npz/lafan1/            # LAFAN1 动作数据（40 clips）
```

## 快速开始

所有命令都从**仓库根目录**执行。

```bash
cd /home/unitree/workwpace/zjy_ws/unitree_rl_mjlab
```

### 1. 查看可用训练任务

```bash
python scripts/train_supermimic.py --list-tasks
```

### 2. 从零训练（默认动作 `dance1_subject2`）

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2
```

### 3. 指定动作文件

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 \
  --data.clips dance1_subject2 \
  --env.num-envs 512 \
  --seed 0 \
  --gpu-ids 0 \
  --schedule.total-iterations 320000 \
  --log-root logs
```

更换动作只需替换 `--data.clips`：

```bash
# 单动作
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 --data.clips walk1_subject1
# 多动作联合训练
python scripts/train_supermimic.py Supermimic-G1-Lafan1-All
```

`--data.clips` 的值是 `extensions/supermimic/data/npz/lafan1/` 下的文件名（不含 `.npz`）。
用 `--data.npz-dir` 指向其他动作目录即可训练自己的数据。

### 4. 从 checkpoint 恢复

```bash
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 \
  --resume logs/Supermimic-G1-Dance1-Subject2/ckpt_latest.pt \
  --schedule.total-iterations 400000
```

恢复会还原权重、优化器、replay buffer、RNG、课程进度和环境状态（格式版本 2 的完整恢复）。
`--schedule.total-iterations` 必须大于 checkpoint 的 iteration。

### 5. 评估与 Play

```bash
# 批量评估（论文 Appendix F 协议，输出 JSON + 逐 clip 结果）
python scripts/evaluate_supermimic.py \
  --ckpt logs/Supermimic-G1-Dance1-Subject2/ckpt_latest.pt \
  --clips dance1_subject2 --num-envs 1

# 可视化回放（有显示器用原生 viewer，否则自动起 viser）
python scripts/play_supermimic.py \
  --ckpt logs/Supermimic-G1-Dance1-Subject2/ckpt_latest.pt \
  --clips dance1_subject2
```

### 6. 日志、TensorBoard、模型位置

```bash
# 训练输出目录
ls logs/Supermimic-G1-Dance1-Subject2/

# TensorBoard
tensorboard --logdir logs/Supermimic-G1-Dance1-Subject2/tb --port 6006
```

一个 run 目录包含：

| 文件 | 内容 |
|---|---|
| `config.json` | 本次运行的完整配置（含 seed、clips、路径） |
| `tb/events.out.tfevents.*` | TensorBoard 事件 |
| `ckpt_latest.pt` | 最近一次 checkpoint |
| `ckpt_<iteration>.pt` | 按 `--schedule.checkpoint-interval` 保存的周期性 checkpoint |

## 运行环境

SuperMimic 固定使用**扩展内的独立环境**，与官方 PPO 环境隔离：

| 组件 | PPO 环境 | SuperMimic 环境 |
|---|---|---|
| Python | 3.11 | 3.12 |
| mjlab | 1.2.0 | 上游 commit `8bb2bc9` |
| MuJoCo / Warp | 3.5.0 | 3.16.0 |

两套依赖不能装进同一个环境。`scripts/train_supermimic.py` 会自动切换解释器：
用 conda 的 `python` 启动时会自动 `exec` 进 `extensions/supermimic/.venv`，因此你不需要记住用哪个 venv。

直接用扩展解释器同样可以（等价）：

```bash
extensions/supermimic/.venv/bin/python scripts/train_supermimic.py --list-tasks
```

## 常用参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--gpu-ids` | GPU 索引，如 `0` 或 `0,1` | `0` |
| `--env.num-envs` | 并行环境数 | 任务默认 256 |
| `--seed` | 随机种子 | `0` |
| `--schedule.total-iterations` | 训练长度（环境步） | 320000 |
| `--log-root` | 输出根目录 | 扩展内 `logs/` |
| `--run-name` | run 目录名 | 任务 id |
| `--resume` | 恢复用的 checkpoint 路径 | 无 |
| `--data.npz-dir` | 动作目录 | `extensions/supermimic/data/npz/lafan1` |
| `--data.clips` | 动作列表 | 任务默认 |
| `--env.anchor-obs / --no-env.anchor-obs` | 是否使用 Anchor 观测 | 任务默认 |

World Model 与 Policy 的超参同样可覆盖，例如 `--wm-trainer.lr`、`--wm-trainer.batch-size`、
`--policy.latent-dim`、`--schedule.updates-per-env-step`。完整列表用 `--help`。

## Smoke test

在改动代码后验证训练链路（约 1 分钟，不产生正式模型）：

```bash
cd /home/unitree/workwpace/zjy_ws/unitree_rl_mjlab
python scripts/train_supermimic.py Supermimic-G1-Dance1-Subject2 \
  --run-name smoke --log-root /tmp/sm_smoke \
  --env.num-envs 8 --schedule.total-iterations 60 \
  --schedule.warmup-iterations 10 --schedule.stop-iteration 60 \
  --schedule.log-interval 10 --schedule.checkpoint-interval 30 \
  --wm-trainer.batch-size 64 --wm-trainer.batches-per-update 1 \
  --wm-trainer.compile None \
  --policy-trainer.batch-size 64 --policy-trainer.compile None \
  --policy-trainer.critic-epochs 1 --policy-trainer.critic-minibatches 1
```

判定标准：日志出现非零 `updates`、`wm nse1` 下降、`/tmp/sm_smoke/smoke/ckpt_latest.pt` 生成。

## 测试

```bash
cd extensions/supermimic
env -u PYTHONPATH .venv/bin/python -m pytest tests -q
```

`env -u PYTHONPATH` 是必需的：系统 ROS 的 `PYTHONPATH` 会向 pytest 注入
`/opt/ros/humble` 下的插件，导致收集阶段报 `No module named 'lark'`。

## 与 PPO 的关系

- SuperMimic 不改动 `scripts/train.py`、`src/tasks/` 或 RSL-RL 的任何代码路径。
- 它复用 mjlab 的 `ManagerBasedRlEnv`、奖励/观测/终止管理器、`RslRlVecEnvWrapper` 和 viewer。
- 训练算法是论文的 World Model + BPTT，因此**不**经由 `mjlab.tasks.registry`（那里的条目是 PPO 的 env+runner 配置对），
  而是由 `supermimic/tasks/registry.py` 提供同等体验的任务注册。

## 当前状态

- 单动作训练已验证：`dance1_subject2` 全长回放 3/3 通过。
- 多个动作（dance1/dance2/jumps1/fight1 等）有训练与仿真验证结果。
- **Sim2Sim 控制器交接仍未完成，真机部署未验证。** 历史仿真结果不等于真机安全。
