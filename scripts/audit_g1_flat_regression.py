"""Read-only shared-path regression and checkpoint-state inventory."""
import ast,hashlib,json,subprocess
from pathlib import Path
import torch

ROOT=Path('logs/navigation_phase_a4')
def main():
  path='src/tasks/velocity/mdp/rewards.py'
  old=ast.parse(subprocess.check_output(['git','show','HEAD:'+path],text=True));new=ast.parse(Path(path).read_text())
  nodes=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  a,b=nodes(old),nodes(new)
  active=['track_linear_velocity','track_angular_velocity','body_orientation_l2','body_angular_velocity_penalty','angular_momentum_penalty','feet_gait','feet_clearance','feet_slip','soft_landing','stand_still','variable_posture']
  out=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    local_reward_active_ast_matches_head={k:a.get(k)==b.get(k) for k in active},
    removed_reward_definitions=sorted(a.keys()-b.keys()),checkpoints=[])
  run=sorted(Path('logs/rsl_rl/g1_velocity').glob('*phase_a4_flat_control'))[-1]
  for p in sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem[6:])):
    d=torch.load(p,map_location='cpu',weights_only=False)
    std=d['actor_state_dict']['distribution.std_param']
    out['checkpoints'].append(dict(path=str(p),iteration=d['iter'],std_min=std.min().item(),std_mean=std.mean().item(),std_max=std.max().item(),optimizer_lr=[g['lr'] for g in d['optimizer_state_dict']['param_groups']],actor_finite=all(torch.isfinite(x).all().item() for x in d['actor_state_dict'].values())))
  (ROOT/'shared_path_audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
