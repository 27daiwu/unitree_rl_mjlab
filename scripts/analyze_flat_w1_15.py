"""W1.15 read-only staged analysis; final acceptance uses only model_399.pt."""
from pathlib import Path
import json
import numpy as np
from analyze_flat_w1_13 import summarize, W13, W11
from analyze_flat_w1_6 import capsule_points
from analyze_flat_w1_8 import trace_metrics
from diagnose_flat_gait import COMMANDS

ROOT = Path('logs/flat_w1_15_fixed_lr_optimizer_20261104_retry01')
STAGES = (50, 100, 200, 300, 399)
FEET = ('left', 'right')


def acceptance(rows):
    gates, supplementary = {}, {}
    def gate(name, value, limit, relation='max', extra=False):
        passed = value <= limit + 1e-12 if relation == 'max' else value >= limit - 1e-12
        target = supplementary if extra else gates
        target[name] = dict(value=value, limit=limit, relation=relation, passed=bool(passed))
    b = rows['backward_015']['policies']
    c, e, s = (b[k] for k in ('control', 'experiment', 'w1_3'))
    gate('backward_rmse_gain_20pct', c['rmse_m_s_or_rad_s']-e['rmse_m_s_or_rad_s'], .2*c['rmse_m_s_or_rad_s'], 'min')
    gate('backward_rmse_absolute_vs_w13', e['rmse_m_s_or_rad_s'], 1.25*s['rmse_m_s_or_rad_s'])
    for foot in FEET:
        gate(f'backward_{foot}_stride_vs_w13', e['step_length_mm'][foot], .9*s['step_length_mm'][foot], 'min')
    r = rows['right_020']['policies']
    c, e, s = (r[k] for k in ('control', 'experiment', 'w1_11_experiment'))
    gate('right_rmse_gain_vs_control', 1000*(c['rmse_m_s_or_rad_s']-e['rmse_m_s_or_rad_s']), 15., 'min')
    gate('right_rmse_absolute_vs_w11', e['rmse_m_s_or_rad_s'], 1.1*s['rmse_m_s_or_rad_s'])
    gate('right_abs_bias_vs_control', abs(e['bias_m_s_or_rad_s'])-abs(c['bias_m_s_or_rad_s']), .01)
    for scenario in ('forward_020','forward_040','left_020','turn_left_050','turn_right_050'):
        c,e = (rows[scenario]['policies'][k] for k in ('control','experiment'))
        gate(f'{scenario}_rmse_nonregression_10pct', e['rmse_m_s_or_rad_s'], 1.1*c['rmse_m_s_or_rad_s'])
    c,e = (rows['left_020']['policies'][k] for k in ('control','experiment'))
    for foot in FEET:
        gate(f'left_{foot}_stride_nonregression_10pct', e['step_length_mm'][foot], .9*c['step_length_mm'][foot], 'min')
    for scenario in COMMANDS:
        c,e = (rows[scenario]['policies'][k] for k in ('control','experiment'))
        for label,row in (('control',c),('experiment',e)):
            gate(f'{scenario}_completion_{label}', row['completed_trials'], 4, 'min')
        for i,name in enumerate(('roll','pitch')):
            gate(f'{scenario}_{name}_rms', e['roll_pitch_rms_rad'][i], c['roll_pitch_rms_rad'][i]+.01,
                 extra=scenario not in ('backward_015','right_020'))
    for scenario in ('backward_015','right_020'):
        c,e = (rows[scenario]['policies'][k] for k in ('control','experiment'))
        for foot in FEET:
            gate(f'{scenario}_{foot}_toe', e['toe_mid_mean_mm'][foot], c['toe_mid_mean_mm'][foot]-5., 'min')
    assert len(gates)==36 and len(supplementary)==10
    return gates,supplementary


def main():
    output = ROOT/'comparison_final.json'
    if output.exists():
        raise FileExistsError(output)
    geometry,radius = capsule_points()
    history = {}
    for scenario,command in COMMANDS.items():
        history[scenario] = {}
        for name,directory,label in (('w1_3',W13,'w1_3'),('w1_11_experiment',W11,'experiment')):
            history[scenario][name] = summarize(directory,label,scenario,command,geometry,radius)
    stages = {}
    for stage in STAGES:
        directory = ROOT/f'stage_{stage}_actor_diagnostics'
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['config']['device']=='cpu'
        assert manifest['config']['seed']==42 and manifest['config']['steps']==600 and manifest['config']['num_envs']==4
        assert manifest['config']['task']=='Unitree-G1-Flat'
        metrics = json.loads((directory/'metrics.json').read_text())
        rows = {}
        for scenario,command in COMMANDS.items():
            rows[scenario] = dict(command=list(command),policies=dict(history[scenario]))
            for label in ('control','experiment'):
                assert manifest['checkpoints'][label]==str(ROOT/label/f'model_{stage}.pt')
                row = summarize(directory,label,scenario,command,geometry,radius)
                row['replicas'] = [trace_metrics(directory/f'{label}_{scenario}_{replica}.npz',command,geometry,radius) for replica in range(4)]
                row['terminated_trials'] = sum(metrics['trials'][f'{label}/{scenario}/replica_{i}']['terminated_early'] for i in range(4))
                row['swing_excursion_mm'] = {f:1000*metrics['comparison'][scenario][label]['feet'][f]['swing_excursion_mean_m'] for f in FEET}
                rows[scenario]['policies'][label] = row
        stages[str(stage)] = rows
    gates,supplementary = acceptance(stages['399'])
    result = dict(acceptance_checkpoint='model_399.pt', rows=stages['399'],stages=stages,gates=gates,
                  supplementary_posture_gates=supplementary,passed=all(g['passed'] for g in gates.values()),
                  all_46_passed=all(g['passed'] for g in (*gates.values(),*supplementary.values())),
                  aggregation='mean of four per-replica RMSE/bias/population std, identical to W1.13',
                  gate_scope='36 W1.15 written gates; 10 additional W1.13 analyzer posture checks reported separately')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),passed=result['passed'],passed_gates=sum(g['passed'] for g in gates.values()),
                         total_gates=len(gates),failures={k:v for k,v in gates.items() if not v['passed']}),indent=2))


if __name__=='__main__':
    main()
