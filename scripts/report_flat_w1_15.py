"""Render W1.15 endpoint acceptance and staged evidence from saved analysis."""
from pathlib import Path
import json

ROOT=Path('logs/flat_w1_15_fixed_lr_optimizer_20261104_retry01')
NAMES={'forward_020':'前进 +0.20','forward_040':'前进 +0.40','backward_015':'后退 -0.15','left_020':'左移 +0.20','right_020':'右移 -0.20','turn_left_050':'左转 +0.50','turn_right_050':'右转 -0.50'}


def main():
    data=json.loads((ROOT/'comparison_final.json').read_text())
    audit=json.loads((ROOT/'post_training_integrity_audit.json').read_text())
    target=Path('doc/flat_w1_15_acceptance_report.md')
    if target.exists(): raise FileExistsError(target)
    gates=data['gates'];extra=data['supplementary_posture_gates']
    passed=sum(g['passed'] for g in gates.values())
    lines=['# W1.15 固定 LR 完整 Adam 恢复验收报告','',
      '日期：2026-11-04。正式运行：`logs/flat_w1_15_fixed_lr_optimizer_20261104_retry01/`。', '',
      f'**最终结论：{"PASS" if data["passed"] else "FAIL"}，W1.15 书面门槛 {passed}/{len(gates)} 项通过。仅以两组 `model_399.pt` 作正式验收；中间 checkpoint 不替代终点。**','',
      '## 运行与数据完整性','',
      '两组均已经完成 400 次 PPO update，无需再次训练。两份 LR ledger 各 400 条，update 序号连续为 1–400，iteration 为 0–399；algorithm scalar LR 与所有 optimizer param-group 的 update 前后 LR 均严格等于 `0.0002562890625`，两组逐行一致、违规数 0。`schedule="fixed"` 使 PPO 不进入 adaptive-KL LR 修改分支。','',
      '两组 actor/critic/normalizer 初始化与 W1.3 一致，reward 与其他 PPO 参数匹配；Control fresh Adam，Experiment 恢复完整 moments 和 step。iteration 与环境计数归零，未恢复 simulator/RNG/curriculum 状态。终点 Control/Experiment Adam step 为 8000/16000，均有 17 个 state entries。所有阶段的 actor/critic、optimizer 张量均有限且可加载。', '',
      '| 终点 | SHA-256 | 文件字节数 |','|---|---|---:|']
    for label in ('control','experiment'):
      c=audit[label]['checkpoints'][-1]
      lines.append(f'| {label} model_399.pt | `{c["sha256"]}` | {c["size_bytes"]} |')
    lines+=['',
      '训练前审计、源 checkpoint 副本、参数 YAML、TensorBoard、LR ledger 和历史尝试均保留。旧根目录中的 26-update 部分 Control 不纳入比较。完整性审计：`post_training_integrity_audit.json`。','',
      '## 评估协议','',
      '五阶段（50/100/200/300/399）各使用 Control 与 Experiment 的相应 checkpoint，Flat、CPU MuJoCo Warp、seed 42、4 副本、七固定命令、每命令 600 步；稳态统计窗口为 `time > 7 s`。RMSE、bias、std 先按副本计算，再取四副本平均，沿用 W1.13 聚合方法。直线单位 mm/s，转向单位 mrad/s；步长和 toe 均为 mm，姿态 RMS 为 rad。','',
      'W1.15 手工保存的中间 checkpoint 省略了可选 `infos` 元数据，第一次标准 runner 加载因 `KeyError: infos` 停止，未生成轨迹。失败日志与 manifest 保留。随后在诊断入口显式启用 `--native-actor-only-load`，仅加载 actor（含 normalizer），逐张量严格核验，不加载环境计数、critic 或 optimizer；十个 checkpoint 一致采用此路径，未修改 checkpoint。gait phase 由 episode length 决定；评估 curriculum 为空且每命令重置。','',
      '成功评估在 `stage_{50,100,200,300,399}_actor_diagnostics/`；各目录包含 manifest、原始 NPZ、trials 和聚合指标。`model_N` 沿用 RSL-RL 迭代编号，对应 N+1 次 update；399 即 400 次 update。','',
      '## 正式终点：七命令速度误差','',
      '| 命令 | W1.3 RMSE | Control RMSE / bias / std | Experiment RMSE / bias / std | Experiment RMSE 相对 Control |',
      '|---|---:|---:|---:|---:|']
    for scenario,name in NAMES.items():
      rows=data['rows'][scenario]['policies'];c,e,s=(rows[k] for k in ('control','experiment','w1_3'))
      def triple(r):return ' / '.join(f'{1000*r[k]:+.1f}' if k.startswith('bias') else f'{1000*r[k]:.1f}' for k in ('rmse_m_s_or_rad_s','bias_m_s_or_rad_s','std_m_s_or_rad_s'))
      delta=100*(e['rmse_m_s_or_rad_s']/c['rmse_m_s_or_rad_s']-1)
      lines.append(f'| {name} | {s["rmse_m_s_or_rad_s"]*1000:.1f} | {triple(c)} | {triple(e)} | {delta:+.1f}% |')
    lines+=['', '正 bias 在后退与右移负命令下表示目标轴速度偏向正方向（通常为速度幅值不足）；bias/std 分开报告，不能只凭 RMSE 判定改善。','',
      '## 正式终点：步长、toe 与稳定性','',
      '| 命令 | 组别 | 步长 L/R | 计划摆动中段 toe L/R | roll/pitch RMS | 完成 / 提前终止 |','|---|---|---:|---:|---:|---:|']
    for scenario,name in NAMES.items():
      for label in ('control','experiment'):
        row=data['rows'][scenario]['policies'][label]
        def feet(key):return '/'.join(f'{row[key][f]:.2f}' for f in ('left','right'))
        pose='/'.join(f'{v:.5f}' for v in row['roll_pitch_rms_rad'])
        lines.append(f'| {name} | {label} | {feet("step_length_mm")} | {feet("toe_mid_mean_mm")} | {pose} | {row["completed_trials"]}/4；{row["terminated_trials"]} |')
    lines+=['', 'toe 为 MJCF 碰撞胶囊包络最低点，计划中段使用既有 0.66–0.90 相位窗口。没有新增 toe 奖励，也没有变更 gait。此处“完成/提前终止”依据诊断原始 alive/termination 记录，不以轨迹长度单独推断。','',
      '## 预注册门槛逐项 PASS/FAIL','',
      'W1.15 书面条款合计 36 项原子检查。W1.13 程序额外检查其他五命令的 roll/pitch，共另 10 项，完整保留为补充检查。两套计数均公开；未放宽数值门槛。后退步长精确下限为 W1.3 原始左右脚均值的 90%：66.8191934229 / 65.4637909167 mm。','',
      '| 检查 | 实测值 | 判定关系 | 阈值 | 结果 |','|---|---:|:---:|---:|---|']
    for name,g in gates.items():
      lines.append(f'| `{name}` | {g["value"]:.9g} | {">=" if g["relation"]=="min" else "<="} | {g["limit"]:.9g} | {"PASS" if g["passed"] else "FAIL"} |')
    lines+=['', '门槛 JSON 保留原生单位：速度为 m/s 或 rad/s，`right_rmse_gain_vs_control` 为 mm/s，stride/toe 为 mm；本表与 JSON 完全一致。','',
      f'额外姿态检查：{sum(g["passed"] for g in extra.values())}/10 通过。若把旧程序额外项一并计入，则共 {passed+sum(g["passed"] for g in extra.values())}/46 通过。','',
      '| 额外检查 | 实测 rad | 上限 rad | 结果 |','|---|---:|---:|---|']
    for name,g in extra.items():lines.append(f'| `{name}` | {g["value"]:.6f} | {g["limit"]:.6f} | {"PASS" if g["passed"] else "FAIL"} |')
    lines+=['', '## 中间 checkpoint：仅作过程分析','',
      '| Iteration | 命令 | Control RMSE / bias / std | Experiment RMSE / bias / std | Control 步长 L/R | Experiment 步长 L/R |','|---:|---|---:|---:|---:|---:|']
    for stage,rows in data['stages'].items():
      for scenario in ('backward_015','right_020'):
        c,e=(rows[scenario]['policies'][k] for k in ('control','experiment'))
        def tri(r):return '/'.join(f'{1000*r[k]:.1f}' for k in ('rmse_m_s_or_rad_s','bias_m_s_or_rad_s','std_m_s_or_rad_s'))
        def strides(r):return '/'.join(f'{r["step_length_mm"][f]:.1f}' for f in ('left','right'))
        lines.append(f'| {stage} | {NAMES[scenario]} | {tri(c)} | {tri(e)} | {strides(c)} | {strides(e)} |')
    lines+=['', '全部七命令五阶段数据及逐副本数据见 `comparison_final.json` 的 `stages`；没有用任何中间最优点替换终点。','',
      '## 采用决定及边界','']
    if not data['passed']:
      lines+=['**不采用完整 Adam 状态恢复方案，不替换 W1.3 当前研究候选。** W1.15 正式终点未达到预注册的联合验收要求。按本次用户安排，停止当前 optimizer 方向的重复尝试；下一步应结合 W1.12–W1.15 的已保存数据分析多方向能力干扰，不追加 optimizer 系数、LR 或训练长度搜索。']
    else:
      lines+=['正式终点通过数值验收，可进入用户下一步安排的正常步行视觉验收；本轮未自动启动该阶段，需进一步观察拖地、碎步和对称性。']
    lines+=['', '本次对照在固定 LR 下比较 fresh Adam 与完整 Adam state 恢复的整体效果，不能单独归因 moments 或 step。单 seed 的两条训练路径及同策略四副本评估不能证明普遍因果规律；真实训练中的轨迹与命令采样可能随策略变化而分化。固定命令无摔倒只代表本组评估，不能认定达到真机部署条件。未启动 Navigation adaptation、视觉任务或真机部署。','']
    target.write_text('\n'.join(lines))
    print(target)


if __name__=='__main__': main()
