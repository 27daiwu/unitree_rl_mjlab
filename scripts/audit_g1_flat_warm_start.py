"""Read-only structure/provenance audit; never warm-starts or trains a policy."""
from pathlib import Path
from dataclasses import asdict
import json
import torch
import yaml
import onnx
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
import src.tasks  # noqa: F401


class DataOnlyLoader(yaml.SafeLoader):
  pass


def unknown(loader,node):
  if isinstance(node,yaml.MappingNode):return loader.construct_mapping(node,deep=True)
  if isinstance(node,yaml.SequenceNode):return loader.construct_sequence(node,deep=True)
  return loader.construct_scalar(node)
DataOnlyLoader.add_constructor(None,unknown)


def main():
  task='Unitree-G1-Navigation';cfg=load_env_cfg(task);cfg.scene.num_envs=1
  env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
  try:
    rl=load_rl_cfg(task);vec=RslRlVecEnvWrapper(env,clip_actions=rl.clip_actions)
    runner=load_runner_cls(task)(vec,asdict(rl),device='cuda:0')
    names=env.observation_manager.active_terms['actor']
    dims=env.observation_manager.group_obs_term_dim['actor']
    robot=env.scene['robot']
    flat_cfg=load_env_cfg('Unitree-G1-Flat')
    schema=dict(actor_terms=list(zip(names,dims)),actor_dim=env.observation_manager.group_obs_dim['actor'],action_dim=env.action_manager.total_action_dim,joint_order=list(robot.joint_names),actuator_order=list(robot.actuator_names),actor_architecture=asdict(rl.actor),
      current_flat_actor_term_order=list(flat_cfg.observations['actor'].terms),
      current_flat_robot_config_equal=asdict(flat_cfg.scene.entities['robot'])==asdict(cfg.scene.entities['robot']),
      current_flat_action_config_equal=asdict(flat_cfg.actions['joint_pos'])==asdict(cfg.actions['joint_pos']))
    candidates=[]
    for p in sorted(Path('logs/rsl_rl/g1_velocity').glob('*/model*.pt')):
      d=torch.load(p,map_location='cpu',weights_only=False)
      saved=yaml.load((p.parent/'params/env.yaml').read_text(),Loader=DataOnlyLoader)
      saved_actor=list(saved['observations']['actor']['terms'])
      compatible=True;error=None
      try:runner.alg.actor.load_state_dict(d['actor_state_dict'],strict=True)
      except RuntimeError as exc:compatible=False;error=str(exc)
      candidates.append(dict(path=str(p),iteration=d['iter'],terrain=saved['scene']['terrain']['terrain_type'],
        saved_actor_order=saved_actor,observation_order_matches=saved_actor==names,
        saved_action_config_matches=saved['actions']['joint_pos']==yaml.load(Path('logs/rsl_rl/g1_navigation/2026-09-28_18-26-35_phase_a1_pilot/params/env.yaml').read_text(),Loader=DataOnlyLoader)['actions']['joint_pos'],
        strict_actor_load=compatible,load_error=error,
        normalizer_keys=[k for k in d['actor_state_dict'] if 'obs_normalizer' in k],
        credible_locomotion_checkpoint=False,reason='Only iteration 0 is present; no learned locomotion validation or later checkpoint.'))
    exports=[]
    for p in sorted(Path('deploy/robots/g1/config/policy').glob('*/v0/exported/policy.onnx')):
      m=onnx.load(str(p));exports.append(dict(path=str(p),inputs={x.name:[d.dim_value for d in x.type.tensor_type.shape.dim] for x in m.graph.input},metadata={v.key:v.value for v in m.metadata_props},warm_start_available=False,reason='Deployment ONNX artifact: no matching native training state and no verified Flat task/checkpoint provenance.'))
    out=dict(navigation=schema,flat_candidates=candidates,exports=exports,FLAT_CHECKPOINT_FOUND=bool(candidates),ACTOR_STRUCTURALLY_COMPATIBLE=all(x['strict_actor_load'] and x['observation_order_matches'] for x in candidates),WARM_START_AVAILABLE=False,FLAT_WARM_START_COMPATIBLE=False,
      semantic_caveat='Flat phase observation zeros below norm .1; Navigation only zeros at exact-zero. Command heading/DR/noise also differ. Shape compatibility is not functional readiness. For a future credible actor-only warm start: load actor including normalizer, start fresh optimizer/critic, do not transplant old episode/history state.')
    Path('logs/navigation_phase_a2/warm_start_audit.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
  finally:env.close()


if __name__=='__main__':main()
