"""
GPU-parallel FrankaEnv for RL training with Genesis.

Key changes vs. env_franka.py (single-env):
  - scene.build(n_envs=num_envs)          → N envs simulated in parallel on GPU
  - all state: numpy (dim,) → torch (N, dim) on gs.device
  - step(actions) accepts (N, action_dim) tensors
  - reset(envs_idx) allows selective per-env reset
  - controller fully vectorised with torch.linalg.solve (batched Jacobian)
  - camera removed (not needed for training)
"""

import math
from pathlib import Path

import torch
import numpy as np
from tensordict import TensorDict

import genesis as gs

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Torch-native helpers (Genesis's quat_to_rotvec is numpy-only)
# ---------------------------------------------------------------------------

def _tc_quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Angle-axis (rotvec) from quaternion (w, x, y, z). Supports any batch shape."""
    q_w = quat[..., :1]           # (..., 1)
    q_vec = quat[..., 1:]         # (..., 3)
    s2 = q_vec.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s2, q_w.abs())
    inv_sinc = angle / s2.clamp(min=1e-8)
    sign = torch.where(q_w < 0.0, torch.full_like(q_w, -1.0), torch.ones_like(q_w))
    return sign * inv_sinc * q_vec


def _tc_inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse) of a unit quaternion."""
    inv = quat.clone()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def _tc_quat_mul(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hamilton product  u ⊗ v  for unit quaternions (w, x, y, z)."""
    uw, ux, uy, uz = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    vw, vx, vy, vz = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    quat = torch.stack([
        uw * vw - ux * vx - uy * vy - uz * vz,
        uw * vx + ux * vw + uy * vz - uz * vy,
        uw * vy - ux * vz + uy * vw + uz * vx,
        uw * vz + ux * vy - uy * vx + uz * vw,
    ], dim=-1)
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Parallel Franka environment
# ---------------------------------------------------------------------------

class FrankaEnvParallel:
    """
    Vectorised Franka environment.

    Observations: flat tensor (N, OBS_DIM=15)
        [0]    ee_pos_z
        [1]    ee_vel_z
        [2]    fingertip_distance
        [3]    target_z_vel
        [4]    target_z_acc
        [5:8]  left_force  (3,)
        [8:11] right_force (3,)
        [11]   cuboid_rel_z
        [12]   cuboid_rel_x
        [13]   cuboid_rel_y
        [14]   desired_rel_z

    Actions (per env, shape (N, action_dim=3)):
        [0]   target_z_vel
        [1:3] gripper_pos (left, right finger)
    """

    # Observation layout constants
    OBS_DIM            = 15
    OBS_EE_POS_Z       = 0
    OBS_EE_VEL_Z       = 1
    OBS_FINGERTIP_DIST = 2
    OBS_TARGET_Z_VEL   = 3
    OBS_TARGET_Z_ACC   = 4
    OBS_LEFT_FORCE     = slice(5, 8)
    OBS_RIGHT_FORCE    = slice(8, 11)
    OBS_CUBOID_REL_Z   = 11
    OBS_CUBOID_REL_X   = 12
    OBS_CUBOID_REL_Y   = 13
    OBS_DESIRED_REL_Z  = 14

    # Action scaling constants
    Z_VEL_MAX       = 0.85
    Z_ACC_MAX       = 60.0   # m/s² — hard limit on target-velocity rate of change
    Z_ACC_PENALTY_THRESHOLD = 13.0
    Z_ACC_PENALTY_WEIGHT    = 0.25
    EE_Z_TARGET     = 0.7
    GRIPPER_CLOSED  = 0.000251
    GRIPPER_OPEN    = 0.0124
    # Tracking normalization: value of 20*exp(-30*z_error) at z_error=0.001 m
    TRACKING_PERFECT = 20.0 * math.exp(-30.0 * 0.001)  # ≈ 19.41

    def __init__(
        self,
        num_envs: int = 1,
        *,
        vis: bool = False,
        record: bool = False,
        dt: float = 0.001,
        target_dt: float = 0.02,
        gripper_pos_min: float = 0.000251,
        gripper_pos_max: float = 0.0124,
        pos_gain: float = 8.0,
        rot_gain: float = 4.0,
        jacobian_damping: float = 1e-4,
    ):
        self.num_envs = num_envs
        self.device = gs.device
        self.dt = dt
        self.target_dt = target_dt
        self.target_update_every = max(1, int(round(target_dt / dt)))
        self.target_period = self.target_update_every * dt
        self.num_actions = 3
        self.max_episode_length = 120
        self.extras: dict = {}
        self.cfg = {
            "num_envs": num_envs,
            "num_actions": 3,
            "obs_dim": self.OBS_DIM,
            "max_episode_length": self.max_episode_length,
            "dt": dt,
            "target_dt": target_dt,
            "z_vel_max": self.Z_VEL_MAX,
            "gripper_closed": self.GRIPPER_CLOSED,
            "gripper_open": self.GRIPPER_OPEN,
        }

        self.gripper_pos_min = torch.tensor([gripper_pos_min, gripper_pos_min], device=self.device)
        self.gripper_pos_max = torch.tensor([gripper_pos_max, gripper_pos_max], device=self.device)

        # Controller gains (scalar – same for all envs)
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        # Jacobian regulariser: (6, 6), broadcast over batch in _control_once
        reg = jacobian_damping * torch.eye(6, device=self.device)
        self.jacobian_regularizer = reg  # (6, 6)

        # ------------------------------------------------------------------ #
        # Build scene with N parallel envs                                   #
        # ------------------------------------------------------------------ #
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=1),
            rigid_options=gs.options.RigidOptions(),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=vis,
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=str(REPO_ROOT / "genesis/assets/xml/franka_emika_panda/panda.xml")),
        )
        self.cuboid = self.scene.add_entity(
            gs.morphs.MJCF(file=str(REPO_ROOT / "genesis/assets/xml/franka_emika_panda/box.xml")),
            surface=gs.surfaces.Plastic(color=(0.18, 0.42, 0.82)),
        )

        # Optional offscreen recording camera — must be added BEFORE scene.build()
        self.cam = None
        if record:
            self.cam = self.scene.add_camera(
                res=(1280, 960),
                pos=(3.5, 0.0, 2.5),
                lookat=(0.0, 0.0, 0.5),
                fov=30,
                GUI=False,
            )

        # One shared spacing so envs do not overlap visually
        self.scene.build(n_envs=num_envs, env_spacing=(1.5, 1.5))

        # ------------------------------------------------------------------ #
        # DOF / link indices                                                  #
        # ------------------------------------------------------------------ #
        self.motors_dof = torch.arange(7, device=self.device)
        self.fingers_dof = torch.arange(7, 9, device=self.device)
        self.q_home = torch.tensor(
            [0.0, -0.82, 0.0, -2.180, 0.0, 2.9, 0.78, 0.01090, 0.01090],
            device=self.device,
        )  # (9,)

        self.left_finger = self.franka.get_link("left_finger")
        self.right_finger = self.franka.get_link("right_finger")
        self.ee_link = self.franka.get_link("hand")

        # Fingertip offsets in local finger frame
        self.fingertip_local = torch.tensor([0.0, 0.0055, 0.0445], device=self.device)

        self._set_franka_gains()
        self.reset()

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, envs_idx: torch.Tensor | None = None, warmup_steps: int = 100):
        """
        Reset selected environments (all if envs_idx is None).

        Returns obs tensor (num_envs, obs_dim) or (len(envs_idx), obs_dim).
        """
        N = self.num_envs
        if envs_idx is None:
            envs_idx = torch.arange(N, device=self.device)

        # ---- robot pose ----
        q_home_batch = self.q_home.unsqueeze(0).expand(len(envs_idx), -1)  # (|idx|, 9)
        self.franka.set_qpos(q_home_batch, envs_idx=envs_idx)
        self.franka.control_dofs_position(q_home_batch, envs_idx=envs_idx)

        # ---- cuboid ----
        self._reset_cuboid_home_pose(envs_idx)

        # ---- controller state (only for selected envs) ----
        if not hasattr(self, "target_center"):
            # First call: allocate full buffers
            self.target_center = torch.zeros(N, 3, device=self.device)
            self.target_quat = torch.zeros(N, 4, device=self.device)
            self.target_z = torch.zeros(N, device=self.device)
            self.target_z_vel = torch.zeros(N, device=self.device)
            self.target_z_acc = torch.zeros(N, device=self.device)
            self.prev_target_z_vel = torch.zeros(N, device=self.device)
            self.desired_rel_z = torch.zeros(N, device=self.device)
            self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
            self.obs_buf = torch.zeros(N, self.OBS_DIM, device=self.device)
            self.rew_buf = torch.zeros(N, device=self.device)
            self.reset_buf = torch.zeros(N, dtype=torch.bool, device=self.device)
            self.initial_z_error = torch.ones(N, device=self.device)
            self.initial_tracking = torch.zeros(N, device=self.device)
            self.direction_change_count = torch.zeros(N, dtype=torch.long, device=self.device)
            # Cubic-hermite segment state per env
            self._seg_start = None   # (N, 3): (z, z_vel, z_acc)
            self._seg_end = None     # (N, 3)
            self._seg_t0 = torch.zeros(N, device=self.device)  # wall-time at segment start

        # Warmup first so ee_link.get_pos() is valid
        for _ in range(warmup_steps):
            q_home_batch_all = self.q_home.unsqueeze(0).expand(N, -1)
            self.franka.control_dofs_position(q_home_batch_all)
            self.scene.step()

        # Read ee state after warmup
        ee_pos = self.ee_link.get_pos()    # (N, 3)
        ee_quat = self.ee_link.get_quat()  # (N, 4)

        self.target_center[envs_idx] = ee_pos[envs_idx].clone()
        self.target_quat[envs_idx] = ee_quat[envs_idx].clone()
        self.target_z[envs_idx] = ee_pos[envs_idx, 2].clone()
        self.target_z_vel[envs_idx] = 0.0
        self.target_z_acc[envs_idx] = 0.0
        self.prev_target_z_vel[envs_idx] = 0.0
        self.direction_change_count[envs_idx] = 0
        _n = len(envs_idx)
        # _sign = torch.where(torch.rand(_n, device=self.device) < 0.5,
        #                     torch.ones(_n, device=self.device),
        #                     -torch.ones(_n, device=self.device))
        _sign = torch.ones(_n, device=self.device)
        self.desired_rel_z[envs_idx] = _sign * (0.025 + torch.rand(_n, device=self.device) * 0.02)
        self.episode_length_buf[envs_idx] = 0

        self._seg_start = None
        self._seg_end = None
        self._seg_t0 = torch.zeros(N, device=self.device)

        self.sim_step = 0

        self._store_initial_z_error(envs_idx)
        self._update_obs_buf()
        return self.get_observations()

    # ------------------------------------------------------------------ #
    # Step                                                                #
    # ------------------------------------------------------------------ #

    def step(self, actions: torch.Tensor):
        """
        actions: (num_envs, 3)  —  [target_z_vel, finger_l, finger_r]

        Runs target_update_every sim steps and returns obs.
        """
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"actions must be ({self.num_envs}, {self.num_actions}), got {actions.shape}")

        new_z_vel = actions[:, 0].clamp(-1.0, 1.0) * self.Z_VEL_MAX  # (N,)
        # Limit acceleration: |Δv| ≤ Z_ACC_MAX * target_period
        max_dv = self.Z_ACC_MAX * self.target_period
        new_z_vel = new_z_vel.clamp(self.target_z_vel - max_dv, self.target_z_vel + max_dv)
        gripper_raw = actions[:, 1:].clamp(-1.0, 1.0)                 # (N, 2)
        gripper_pos = self.gripper_pos_min + (gripper_raw + 1.0) * 0.5 * (self.gripper_pos_max - self.gripper_pos_min)  # (N, 2)

        # Trapezoid integration for z
        new_z_acc = (new_z_vel - self.target_z_vel) / self.target_period
        new_z = self.target_z + 0.5 * (self.target_z_vel + new_z_vel) * self.target_period

        # Update segment for cubic-hermite interpolation
        old_sample = torch.stack([self.target_z, self.target_z_vel, self.target_z_acc], dim=-1)  # (N,3)
        new_sample = torch.stack([new_z, new_z_vel, new_z_acc], dim=-1)
        self._seg_start = old_sample
        self._seg_end = new_sample
        self._seg_t0 = torch.full((self.num_envs,), self.sim_step * self.dt, device=self.device)

        self.prev_target_z_vel = self.target_z_vel.clone()
        self.target_z = new_z
        self.target_z_vel = new_z_vel
        self.target_z_acc = new_z_acc

        # Control gripper (same pos for all substeps within this target period)
        self.franka.control_dofs_position(gripper_pos, dofs_idx_local=self.fingers_dof)

        for local_step in range(self.target_update_every):
            t = (self.sim_step + local_step) * self.dt
            tz, tz_vel = self._sample_target_z(t)   # (N,), (N,)
            self._control_once(tz, tz_vel)
            self.scene.step()

        self.sim_step += self.target_update_every
        self.episode_length_buf += 1
        done, reward, timeout = self._compute_done_and_reward()
        done_idx = done.nonzero(as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            self._reset_idx(done_idx)
        self.rew_buf = reward
        self.reset_buf = done
        self.extras["time_outs"] = timeout.float()
        self._update_obs_buf()
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------ #
    # Observations                                                        #
    # ------------------------------------------------------------------ #

    def _update_obs_buf(self):
        """Compute observations and write into self.obs_buf."""
        ee_pos = self.ee_link.get_pos()     # (N, 3)
        ee_vel = self.ee_link.get_vel()     # (N, 3)
        cuboid_pos = self.cuboid.get_pos()  # (N, 3)
        left_ft = self._fingertip_pos(self.left_finger)    # (N, 3)
        right_ft = self._fingertip_pos(self.right_finger)  # (N, 3)
        fingertip_dist = (left_ft - right_ft).norm(dim=-1)  # (N,)

        link_forces = self.franka.get_links_net_contact_force()  # (N, n_links, 3)
        left_force = link_forces[:, self.left_finger.idx_local, :]   # (N, 3)
        right_force = link_forces[:, self.right_finger.idx_local, :]  # (N, 3)

        finger_mid = (left_ft + right_ft) / 2.0  # (N, 3)
        self.obs_buf = torch.cat([
            ee_pos[:, 2:3],                                       # [0]    ee_pos_z
            ee_vel[:, 2:3],                                       # [1]    ee_vel_z
            fingertip_dist.unsqueeze(-1),                         # [2]    fingertip_distance
            self.target_z_vel.unsqueeze(-1),                      # [3]    target_z_vel
            self.target_z_acc.unsqueeze(-1),                      # [4]    target_z_acc
            left_force,                                           # [5:8]  left_force
            right_force,                                          # [8:11] right_force
            (cuboid_pos[:, 2] - finger_mid[:, 2]).unsqueeze(-1),  # [11]   cuboid_rel_z
            (cuboid_pos[:, 0] - finger_mid[:, 0]).unsqueeze(-1),  # [12]   cuboid_rel_x
            (cuboid_pos[:, 1] - finger_mid[:, 1]).unsqueeze(-1),  # [13]   cuboid_rel_y
            self.desired_rel_z.unsqueeze(-1),                     # [14]   desired_rel_z
        ], dim=-1)  # (N, 15)

    def get_observation(self) -> torch.Tensor:
        """Returns flat obs tensor (num_envs, OBS_DIM). Convenience for non-rsl_rl usage."""
        self._update_obs_buf()
        return self.obs_buf

    def get_observations(self) -> TensorDict:
        """Returns TensorDict({"policy": obs_buf}) for rsl_rl compatibility."""
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------ #
    # Done detection and partial reset                                    #
    # ------------------------------------------------------------------ #

    def _compute_done_and_reward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (done, reward, timeout) each (num_envs,). done is bool, reward/timeout are float32."""
        cuboid_pos = self.cuboid.get_pos()                          # (N, 3)
        ee_pos = self.ee_link.get_pos()                             # (N, 3)
        ee_vel = self.ee_link.get_vel()                             # (N, 3)
        left_ft = self._fingertip_pos(self.left_finger)             # (N, 3)
        right_ft = self._fingertip_pos(self.right_finger)           # (N, 3)
        finger_mid = (left_ft + right_ft) / 2.0                    # (N, 3)
        fingertip_dist = (left_ft - right_ft).norm(dim=-1)          # (N,)

        cuboid_rel_z = cuboid_pos[:, 2] - finger_mid[:, 2]          # (N,)
        cuboid_rel_x = cuboid_pos[:, 0] - finger_mid[:, 0]          # (N,)
        cuboid_rel_y = cuboid_pos[:, 1] - finger_mid[:, 1]          # (N,)
        ee_z = ee_pos[:, 2]                                         # (N,)
        ee_vel_z = ee_vel[:, 2]                                     # (N,)

        # ---- done conditions ----
        timeout = self.episode_length_buf >= self.max_episode_length  # (N,)
        success = (~timeout) & (
            ((cuboid_rel_z - self.desired_rel_z).abs() <= 0.01)
            & (ee_vel_z.abs() < 0.01)
            # & (ee_z < 0.86)
        )
        fail = (
            (cuboid_rel_x.abs() > 0.015)                            # relaxed: 15 mm
            | (cuboid_rel_y.abs() > 0.015)                          # relaxed: 15 mm
            | (fingertip_dist < 0.02)
            | (cuboid_rel_z.abs() > 0.15)
            | (ee_z < 0.76)                                         # ee too low
            | (ee_z > 0.96)                                         # ee too high
        )
        done = timeout | success | fail

        # ---- dense tracking: normalized linearly from 0 (episode start) to 1 (z_error=0.001) ----
        z_error = (cuboid_rel_z - self.desired_rel_z).abs()         # (N,)
        tracking_raw = 20.0 * torch.exp(-30.0 * z_error)            # (0, 20]
        denom = (self.TRACKING_PERFECT - self.initial_tracking).clamp(min=1e-6)  # (N,)
        tracking = ((tracking_raw - self.initial_tracking) / denom).clamp(0.0, 1.0)  # [0, 1]
        # print(f"z_error={z_error.mean().item():.4f}  tracking={tracking.mean().item():.2f}")

        # ---- jerk penalty: quadratic, normalized by Z_VEL_MAX ----
        jerk = (self.target_z_vel - self.prev_target_z_vel) / self.Z_VEL_MAX  # (N,)
        jerk_penalty = -0.2 * jerk.pow(2)                           # [-1.2, 0]

        # ---- commanded z acceleration penalty: linear above threshold ----
        commanded_z_acc = (self.target_z_vel - self.prev_target_z_vel) / self.target_period  # (N,)
        z_acc_excess = (commanded_z_acc.abs() - self.Z_ACC_PENALTY_THRESHOLD).clamp(min=0.0)
        z_acc_penalty = -self.Z_ACC_PENALTY_WEIGHT * z_acc_excess
        # direction_changed = (
        #     ((self.prev_target_z_vel > 0.0) & (self.target_z_vel < 0.0))
        #     | ((self.prev_target_z_vel < 0.0) & (self.target_z_vel > 0.0))
        # )
        # self.direction_change_count += direction_changed.long()
        # direction_change_penalty = -10.0 * direction_changed.float()

        # ---- grip quality: reward keeping fingertips around box ----
        grip = torch.exp(-200.0 * (fingertip_dist - 0.03).clamp(min=0.0))  # (0, 1]

        # ---- terminal / alive base reward ----
        ep = self.episode_length_buf.float()                        # (N,)
        base_reward = torch.where(success,
                                  +500.0 - ep * 0.1,
                                  torch.where(fail | timeout,
                                              -250.0,
                                              torch.full_like(ee_z, -0.15)))

        # ---- ee_z height penalty: push arm toward target height from both sides ----
        ee_z_penalty = -0.35 * (ee_z - self.EE_Z_TARGET).abs()

        reward = (
            base_reward
            + 3.0 * tracking
            + jerk_penalty
            + z_acc_penalty
            + 0.5 * grip
            + ee_z_penalty
        )
        return done, reward, timeout

    def _store_initial_z_error(self, envs_idx: torch.Tensor):
        """Snapshot |cuboid_rel_z - desired_rel_z| at episode start for normalization."""
        cuboid_pos = self.cuboid.get_pos()
        left_ft  = self._fingertip_pos(self.left_finger)
        right_ft = self._fingertip_pos(self.right_finger)
        finger_mid = (left_ft + right_ft) / 2.0
        cuboid_rel_z = cuboid_pos[:, 2] - finger_mid[:, 2]
        err = (cuboid_rel_z[envs_idx] - self.desired_rel_z[envs_idx]).abs()
        self.initial_z_error[envs_idx] = err.clamp(min=1e-3)
        self.initial_tracking[envs_idx] = 20.0 * torch.exp(-30.0 * self.initial_z_error[envs_idx])

    def _reset_idx(self, envs_idx: torch.Tensor):
        """Reset a subset of envs in-place; sim_step continues uninterrupted."""
        q = self.q_home.unsqueeze(0).expand(len(envs_idx), -1)
        self.franka.set_qpos(q, envs_idx=envs_idx, zero_velocity=True)
        self.franka.control_dofs_position(q, envs_idx=envs_idx)
        self._reset_cuboid_home_pose(envs_idx)

        ee_pos = self.ee_link.get_pos()    # FK updated by set_qpos
        ee_quat = self.ee_link.get_quat()
        self.target_center[envs_idx] = ee_pos[envs_idx].clone()
        self.target_quat[envs_idx] = ee_quat[envs_idx].clone()
        self.target_z[envs_idx] = ee_pos[envs_idx, 2].clone()
        self.target_z_vel[envs_idx] = 0.0
        self.target_z_acc[envs_idx] = 0.0
        self.prev_target_z_vel[envs_idx] = 0.0
        self.direction_change_count[envs_idx] = 0
        _n = len(envs_idx)
        # _sign = torch.where(torch.rand(_n, device=self.device) < 0.5,
        #                     torch.ones(_n, device=self.device),
        #                     -torch.ones(_n, device=self.device))
        _sign = torch.ones(_n, device=self.device)
        self.desired_rel_z[envs_idx] = _sign * (0.025 + torch.rand(_n, device=self.device) * 0.02)
        self.episode_length_buf[envs_idx] = 0
        self._store_initial_z_error(envs_idx)

    # ------------------------------------------------------------------ #
    # Internal: controller                                                #
    # ------------------------------------------------------------------ #

    def _control_once(self, target_z: torch.Tensor, target_z_vel: torch.Tensor):
        """
        One Jacobian-based velocity-IK step for all N envs simultaneously.

        target_z, target_z_vel: (N,)
        """
        N = self.num_envs

        # Build Cartesian target pos/vel (only z changes)
        target_pos = self.target_center.clone()          # (N, 3)
        target_pos[:, 2] = target_z
        target_vel = torch.zeros(N, 3, device=self.device)
        target_vel[:, 2] = target_z_vel

        # EE state
        ee_pos = self.ee_link.get_pos()    # (N, 3)
        ee_quat = self.ee_link.get_quat()  # (N, 4)

        # Cartesian error
        error_pos = target_pos - ee_pos    # (N, 3)
        rel_quat = _tc_quat_mul(self.target_quat, _tc_inv_quat(ee_quat))  # (N, 4)
        error_rotvec = _tc_quat_to_rotvec(rel_quat)  # (N, 3)

        # ee_velocity_cmd: (N, 6)
        ee_vel_cmd = torch.cat([
            target_vel + self.pos_gain * error_pos,   # (N, 3)
            self.rot_gain * error_rotvec,              # (N, 3)
        ], dim=-1)

        # Jacobian: (N, 6, n_dof_total) → slice motor dofs → (N, 6, 7)
        J_full = self.franka.get_jacobian(link=self.ee_link)  # (N, 6, n_dof)
        J = J_full[:, :, self.motors_dof]  # (N, 6, 7)

        # Damped-least-squares solve: (J J^T + λI) x = ee_vel_cmd
        JJT = J @ J.transpose(-1, -2)  # (N, 6, 6)
        JJT_reg = JJT + self.jacobian_regularizer.unsqueeze(0)  # (N, 6, 6)

        # x: (N, 6)
        x = torch.linalg.solve(JJT_reg, ee_vel_cmd.unsqueeze(-1)).squeeze(-1)

        # qvel = J^T x  →  (N, 7)
        qvel = (J.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)

        self.franka.control_dofs_velocity(qvel, dofs_idx_local=self.motors_dof)

    # ------------------------------------------------------------------ #
    # Internal: cubic-Hermite z reference                                 #
    # ------------------------------------------------------------------ #

    def _sample_target_z(self, t: float):
        """
        Returns z, z_vel tensors (N,) at global sim time t,
        interpolated along the current cubic-Hermite segment.
        """
        if self._seg_end is None:
            return self.target_z.clone(), self.target_z_vel.clone()

        local_t = torch.clamp(
            torch.full((self.num_envs,), t, device=self.device) - self._seg_t0,
            0.0, self.target_period,
        )
        s = (local_t / self.target_period).clamp(0.0, 1.0)  # (N,)
        s2, s3 = s * s, s * s * s

        z0, zv0 = self._seg_start[:, 0], self._seg_start[:, 1]
        z1, zv1 = self._seg_end[:, 0],   self._seg_end[:, 1]
        T = self.target_period

        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2

        z = h00 * z0 + h10 * T * zv0 + h01 * z1 + h11 * T * zv1

        dh00 = 6 * s2 - 6 * s
        dh10 = 3 * s2 - 4 * s + 1
        dh01 = -6 * s2 + 6 * s
        dh11 = 3 * s2 - 2 * s
        z_vel = (dh00 * z0 + dh10 * T * zv0 + dh01 * z1 + dh11 * T * zv1) / T

        return z, z_vel

    # ------------------------------------------------------------------ #
    # Internal: utilities                                                 #
    # ------------------------------------------------------------------ #

    def _fingertip_pos(self, finger_link) -> torch.Tensor:
        """Compute fingertip world position (N, 3) from link pose."""
        pos = finger_link.get_pos()    # (N, 3)
        quat = finger_link.get_quat()  # (N, 4)
        # Rotate local offset by link orientation, then add to link pos
        # transform_by_quat supports torch batched inputs
        from genesis.utils.geom import transform_by_quat as _tbq
        offset = self.fingertip_local.unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)
        return pos + _tbq(offset, quat)

    def _reset_cuboid_home_pose(self, envs_idx: torch.Tensor):
        from genesis.utils.geom import transform_by_quat as _tbq, transform_quat_by_quat as _tqbq

        hand_pos = self.ee_link.get_pos()[envs_idx]    # (|idx|, 3)
        hand_quat = self.ee_link.get_quat()[envs_idx]  # (|idx|, 4)

        local_offset = torch.tensor([0.0, 0.0, 0.1029], device=self.device)
        local_quat_np = np.array([0.00187891, -0.71790805, -0.00193768, -0.69613270])
        local_quat_np /= np.linalg.norm(local_quat_np)
        local_quat = torch.tensor(local_quat_np, dtype=hand_quat.dtype, device=self.device)

        M = len(envs_idx)
        offset_batch = local_offset.unsqueeze(0).expand(M, -1)
        lq_batch = local_quat.unsqueeze(0).expand(M, -1)

        cuboid_pos = hand_pos + _tbq(offset_batch, hand_quat)
        cuboid_quat = _tqbq(lq_batch, hand_quat)

        self.cuboid.set_pos(cuboid_pos, zero_velocity=True, envs_idx=envs_idx)
        self.cuboid.set_quat(cuboid_quat, zero_velocity=True, envs_idx=envs_idx)

    def _set_franka_gains(self):
        kp_motors = torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000],
                                  dtype=torch.float32, device=self.device)
        kv_motors = torch.tensor([450, 450, 350, 350, 200, 200, 200],
                                  dtype=torch.float32, device=self.device)
        f_lo = torch.tensor([-87, -87, -87, -87, -12, -12, -12],
                             dtype=torch.float32, device=self.device)
        f_hi = torch.tensor([87, 87, 87, 87, 12, 12, 12],
                              dtype=torch.float32, device=self.device)

        self.franka.set_dofs_kp(kp_motors, self.motors_dof)
        self.franka.set_dofs_kv(kv_motors, self.motors_dof)
        self.franka.set_dofs_force_range(f_lo, f_hi, self.motors_dof)

        self.franka.set_dofs_kp(torch.tensor([100.0, 100.0], device=self.device), self.fingers_dof)
        self.franka.set_dofs_kv(torch.tensor([10.0, 10.0], device=self.device), self.fingers_dof)
        self.franka.set_dofs_force_range(
            torch.tensor([-100.0, -100.0], device=self.device),
            torch.tensor([100.0, 100.0], device=self.device),
            self.fingers_dof,
        )


# ---------------------------------------------------------------------------
# Minimal smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--num_envs", type=int, default=16)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    env = FrankaEnvParallel(num_envs=args.num_envs, vis=args.vis)
    obs_td = env.reset()
    print("obs shape:", obs_td["policy"].shape)
    print("obs_dim:", FrankaEnvParallel.OBS_DIM)

    for i in range(args.steps):
        actions = torch.zeros(args.num_envs, 3, device=gs.device)
        obs_td, rew_buf, reset_buf, extras = env.step(actions)

    print("Done. ee_pos_z mean:", obs_td["policy"][:, FrankaEnvParallel.OBS_EE_POS_Z].mean().item())
