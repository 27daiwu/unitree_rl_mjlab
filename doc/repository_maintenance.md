# 仓库维护

正式训练与回放入口为 `scripts/train.py` 和 `scripts/play.py`。

Phase2C0～Phase2C8J 历史实验任务及其专用配置、评估脚本、奖励和指标已移除，
不再支持 `UNITREE_ENABLE_LEGACY_TASKS`。保留常规 Flat/Rough/Walk/Stairs/Tracking
任务及楼梯基线、消融任务。`scripts/list_envs.py` 可查看实际可用任务。
训练时必须指定任务，例如 `python scripts/train.py Unitree-G1-Flat`。
历史 checkpoint、ONNX、训练参数和运行日志保留；复现已移除的实验需使用清理前的代码版本。

一次性研究报告、评估 JSON 和脚本应放在仓库外；临时输出写入
`doc/tmp/` 或 `logs/`，不要将运行结果与源码放在一起。
保留 README、安装说明和许可证。
`simulate/*_AUDIT.md` 为历史阶段审计报告，`simulate/tests/` 包含本地测试、
审计脚本及结果；均保留在磁盘但由 Git 忽略，报告不再纳入版本控制。
第三方 MuJoCo 的 `testspeed` 和 CMake 使用的手柄诊断程序 `jstest`
属于随附工具，继续保留在版本控制中。

2026-09-18 清理保留了全部 checkpoint、ONNX 和训练参数，现有仿真日志
因仍被审计/测试读取也予以保留。历史研究报告、Phase2C8 一次性脚本、
旧 archive/tmp、TensorBoard/控制台日志及 W&B 产物已移出仓库。
恢复时按备份 manifest.json 的相对路径，从 removed/ 复制回仓库。

本次外部备份：`/home/hebe/unitree_workspace/cleanup_backups/unitree_rl_mjlab_20260918_113334`。
