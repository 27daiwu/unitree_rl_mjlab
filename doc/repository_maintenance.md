# 仓库维护

正式 PPO 训练与回放入口为 `scripts/train.py` 和 `scripts/play.py`；
使用 `scripts/list_envs.py` 查看可用任务。

SuperMimic 使用独立环境，入口为 `scripts/train_supermimic.py`、
`scripts/play_supermimic.py` 和 `scripts/evaluate_supermimic.py`；
安装与使用见 `docs/supermimic_quickstart.md`。

保留 README、安装说明、上游来源说明、许可证及现行使用文档。
历史阶段报告、一次性分析与审计脚本、阶段专用仿真入口及报告附件已删除。
后续运行输出写入 `logs/`，不要将阶段产物混入源码或文档目录。
