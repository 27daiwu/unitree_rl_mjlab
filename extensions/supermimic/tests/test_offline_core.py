"""CPU-only contracts for the local G1 assets and differentiable model components."""

from pathlib import Path
import re

import numpy as np
import torch
import mujoco

from supermimic.data.motion_convert import (
    G1_JOINT_NAMES,
    convert_csv_to_npz,
    load_g1_model,
)
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.heads import (
    OBS_DIM,
    AnalyticHeads,
    ReferenceFrame,
    analytic_observation,
    flatten_history_term_major,
)
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_local_csv_conversion_has_expected_g1_contract(tmp_path):
    model = load_g1_model()
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    )
    assert model.nq == 36 and model.nv == 35
    assert names == G1_JOINT_NAMES

    source = REPO_ROOT / "src/assets/motions/g1/dance1_subject2.csv"
    output = tmp_path / "dance1_subject2.npz"
    result = convert_csv_to_npz(source, output, input_fps=30.0, output_fps=50.0, model=model)
    assert result["input_frames"] == 3945
    with np.load(output) as motion:
        assert motion["joint_pos"].shape == (6574, 29)
        assert motion["body_pos_w"].shape == (6574, 30, 3)
        assert motion["body_quat_w"].shape == (6574, 30, 4)
        assert tuple(motion["joint_names"]) == G1_JOINT_NAMES
        np.testing.assert_allclose(motion["fps"], [50.0])
        np.testing.assert_allclose(
            np.linalg.norm(motion["body_quat_w"], axis=-1), 1.0, atol=2e-6
        )
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            assert np.isfinite(motion[key]).all(), key


def test_observation_action_and_state_dimensions():
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=["unused.npz"], num_envs=1))
    assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
    assert cfg.actions["joint_pos"].clip is None

    joint_names = G1_JOINT_NAMES
    scale_patterns = cfg.actions["joint_pos"].scale
    scales = []
    for name in joint_names:
        matches = [v for pattern, v in scale_patterns.items() if re.fullmatch(pattern, name)]
        assert len(matches) == 1, (name, matches)
        scales.append(matches[0])
    scales = torch.tensor(scales)
    assert scales.shape == (29,) and torch.isfinite(scales).all() and (scales > 0).all()

    state = torch.zeros(2, 71)
    state[:, 3] = 1.0
    actions = torch.zeros(2, 29)
    xi = torch.zeros(2, 33)
    obs = analytic_observation(state, actions, xi, torch.zeros(29))
    assert obs.shape == (2, OBS_DIM) and OBS_DIM == 93
    history = flatten_history_term_major(obs.unsqueeze(0).expand(10, -1, -1))
    assert history.shape == (2, 930)

    policy_cfg = PolicyCfg(encoder_hidden=(32,), decoder_hidden=(32,))
    policy = Policy(policy_cfg)
    action = policy(history, torch.zeros(2, 680))
    assert action.shape == (2, 29) and torch.isfinite(action).all()

    spec = StateSpec(StateSpecCfg(), "cpu")
    canonical = spec.canonical(state)
    reconstructed = spec.integrate(state, torch.zeros(2, 70))
    assert canonical.shape == (2, 71) and reconstructed.shape == (2, 71)
    torch.testing.assert_close(reconstructed, state)


def test_world_model_prediction_gradient_and_checkpoint(tmp_path):
    model = load_g1_model()
    fk = G1Kinematics(model, "cpu")
    heads = AnalyticHeads(
        fk,
        ("pelvis", "torso_link"),
        "torso_link",
        "pelvis",
        soft_limits=torch.ones(29),
        default_joint_pos=torch.zeros(29),
    )
    wm = WorldModel(
        WorldModelCfg(hidden=(64, 64), term_hidden=(16, 16)), heads, "cpu"
    )
    states = torch.zeros(3, 4, 71)
    states[..., 3] = 1.0
    states[1, :, 0] = 0.01
    states[2, :, 0] = 0.02
    actions = torch.zeros(2, 4, 29)
    actions[0] = 0.1
    actions[1] = -0.1
    xi = torch.zeros(4, 33)
    done = torch.zeros(2, 4)
    refs = [
        ReferenceFrame(None, None, None, None, states[i + 1, :, :3], states[i + 1, :, 3:7])
        for i in range(2)
    ]

    loss, metrics = wm.window_loss(states, actions, xi, done, refs)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters())
    assert torch.isfinite(metrics["state_loss"])

    with torch.no_grad():
        wm.dynamics.mlp.net[-1].weight.normal_(std=1e-3)
    action = torch.zeros(4, 29, requires_grad=True)
    prediction = wm.step(states[0], action, xi)
    prediction.square().mean().backward()
    assert prediction.shape == (4, 71) and torch.isfinite(prediction).all()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0

    checkpoint = tmp_path / "world_model.pt"
    torch.save(wm.state_dict(), checkpoint)
    restored = WorldModel(
        WorldModelCfg(hidden=(64, 64), term_hidden=(16, 16)), heads, "cpu"
    )
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.testing.assert_close(restored.step(states[0], action.detach(), xi), prediction.detach())
