"""Runtime action mapping and strict Flat-to-Navigation actor schema audit."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from audit_g1_flat_control import encode,ROOT
import src.tasks


def main():
  run=sorted(Path('logs/rsl_rl/g1_velocity').glob('*phase_a4_flat_control'))[-1]
  checkpoint=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem[6:]))[-1]
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  results={}
  for task in ['Unitree-G1-Flat','Unitree-G1-Navigation']:
    cfg=load_env_cfg(task);cfg.scene.num_envs=1;cfg.seed=42
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
      rl=load_rl_cfg(task);runner=load_runner_cls(task)(RslRlVecEnvWrapper(env,clip_actions=rl.clip_actions),asdict(rl),device='cuda:0')
      initial_std={k:encode(v) for k,v in runner.alg.actor.state_dict().items() if 'std_param' in k}
      runner.alg.actor.load_state_dict(state['actor_state_dict'],strict=True)
      action=env.action_manager.get_term('joint_pos')
      # CPU MuJoCo model is the compiled baseline before startup device DR.
      model=env.sim.mj_model
      results[task]=dict(strict_actor_load=True,initial_std=initial_std,
        resolved_action_scale=encode(action.scale[0]),resolved_action_offset=encode(action.offset[0]),
        joint_order=list(env.scene['robot'].joint_names),actuator_order=list(env.scene['robot'].actuator_names),
        gainprm=model.actuator_gainprm.tolist(),biasprm=model.actuator_biasprm.tolist(),
        force_range=model.actuator_forcerange.tolist(),dof_armature=model.dof_armature.tolist(),
        actor_order=encode(env.observation_manager.active_terms['actor']),
        actor_dimensions=encode(env.observation_manager.group_obs_term_dim['actor']))
    finally:env.close()
  (ROOT/'actor_compatibility.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),tasks=results),indent=2)+'\n')


if __name__=='__main__':main()
