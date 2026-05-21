import math
from pathlib import Path

import numpy as np

import genesis as gs
import genesis.utils.geom as gu

try:
    from .controller_franka import FrankaVelocityController
except ImportError:
    from controller_franka import FrankaVelocityController

REPO_ROOT = Path(__file__).resolve().parents[2]


class FrankaEnv:
    # Keep in sync with FrankaEnvParallel
    OBS_DIM            = 15
    Z_VEL_MAX          = 0.85
    Z_ACC_MAX          = 60.0   # m/s² — hard limit on target-velocity rate of change
    Z_ACC_PENALTY_THRESHOLD = 13.0
    Z_ACC_PENALTY_WEIGHT    = 0.2
    EE_Z_TARGET        = 0.7
    GRIPPER_CLOSED     = 0.000251
    GRIPPER_OPEN       = 0.0124
    TRACKING_PERFECT   = 20.0 * math.exp(-30.0 * 0.001)  # ≈ 19.41
    MAX_EPISODE_LENGTH = 120

    def __init__(
        self,
        *,
        vis=False,
        cpu=False,
        dt=0.001,
        target_dt=0.02,
        render_fps=60,
        playback_speed=1.0,
        gripper_pos_min=0.000251,
        gripper_pos_max=0.0124,
    ):
        if playback_speed <= 0.0:
            raise ValueError("playback_speed must be greater than 0")

        self.vis = vis
        self.dt = dt
        self.target_dt = target_dt
        self.target_update_every = max(1, int(round(target_dt / dt)))
        self.target_period = self.target_update_every * dt
        self.render_fps = render_fps
        self.render_every = max(1, int(round(1.0 / (render_fps * dt))))
        self.playback_speed = playback_speed
        self.action_dim = 3
        self.z_vel_max = 0.85
        self.gripper_pos_min = np.broadcast_to(np.asarray(gripper_pos_min, dtype=float), (2,)).copy()
        self.gripper_pos_max = np.broadcast_to(np.asarray(gripper_pos_max, dtype=float), (2,)).copy()
        self.closed_gripper_pos = np.array([-1.0, -1.0])
        self.open_gripper_pos = np.array([1.0, 1.0])

        gs.init(backend=gs.cpu if cpu else gs.gpu)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                # constraint_solver=gs.constraint_solver.Newton,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=render_fps,
            ),
            show_viewer=vis,
        )

        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=str(REPO_ROOT / "genesis/assets/xml/franka_emika_panda/panda.xml")),
            visualize_contact=True,
        )
        self.cuboid = self.scene.add_entity(
            gs.morphs.MJCF(file=str(REPO_ROOT / "genesis/assets/xml/franka_emika_panda/box.xml")),
            surface=gs.surfaces.Plastic(color=(0.18, 0.42, 0.82)),
        )

        self.cam_0 = self.scene.add_camera(
            res=(1280, 960),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=True,
        )

        self.scene.build()

        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.q_home = np.array([0.0, -0.82, 0.0, -2.180, 0.0, 2.9, 0.78, 0.01090, 0.01090])
        self.fingertip_local_pos = np.array([0.0, 0.0055, 0.0445])

        self.left_finger = self.franka.get_link("left_finger")
        self.right_finger = self.franka.get_link("right_finger")
        self.ee_link = self.franka.get_link("hand")

        self._set_franka_gains()
        self.last_done_reason = None
        self.reset()

    def reset(self, warmup_steps=100):
        self.franka.set_qpos(self.q_home)
        self.franka.control_dofs_position(self.q_home)
        self._reset_cuboid_home_pose()

        self.target_center = self.ee_link.get_pos().cpu().numpy()
        self.target_quat = self.ee_link.get_quat().cpu().numpy()
        self.target_z = float(self.target_center[2])
        self.target_z_vel = 0.0
        self.target_z_acc = 0.0
        self.prev_target_z_vel = 0.0

        pos_gain = 8.0
        rot_gain = 4.0
        damping = 1e-4
        jacobian_regularizer = damping * np.eye(6)
        self.controller = FrankaVelocityController(
            franka=self.franka,
            scene=self.scene,
            ee_link=self.ee_link,
            motors_dof=self.motors_dof,
            target_center=self.target_center,
            target_quat=self.target_quat,
            pos_gain=pos_gain,
            rot_gain=rot_gain,
            jacobian_regularizer=jacobian_regularizer,
            dt=self.dt,
            target_period=self.target_period,
            steps_per_target=self.target_update_every,
            render_every=self.render_every,
            fingers_dof=self.fingers_dof,
            playback_speed=self.playback_speed if self.vis else None,
        )
        self.controller.reset((self.target_z, self.target_z_vel, self.target_z_acc))

        self.sim_step = 0
        self.episode_step = 0
        # Match parallel env: always lift, range [0.04, 0.06]
        self.desired_rel_z = (0.025 + float(np.random.rand()) * 0.02)
        self.prev_actual_z_vel = None
        self.prev_torque = None
        self.direction_change_count = 0

        for i in range(warmup_steps):
            self.franka.control_dofs_position(self.q_home)
            self.scene.step(update_visualizer=i % self.render_every == 0)

        self.controller.reset_playback_clock()
        self._store_initial_z_error()
        return self.get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {action.shape}")

        target_z_vel = float(np.clip(action[0], -1.0, 1.0)) * self.z_vel_max
        # Limit acceleration: |Δv| ≤ Z_ACC_MAX * target_period
        max_dv = self.Z_ACC_MAX * self.target_period
        target_z_vel = float(np.clip(target_z_vel, self.target_z_vel - max_dv, self.target_z_vel + max_dv))
        gripper_raw = np.clip(action[1:], -1.0, 1.0)
        gripper_pos = self.gripper_pos_min + (gripper_raw + 1.0) * 0.5 * (self.gripper_pos_max - self.gripper_pos_min)
        target_z_acc = (target_z_vel - self.target_z_vel) / self.target_period
        target_z = self.target_z + 0.5 * (self.target_z_vel + target_z_vel) * self.target_period

        self.prev_target_z_vel = self.target_z_vel
        self.target_z = float(target_z)
        self.target_z_vel = target_z_vel
        self.target_z_acc = target_z_acc
        records = self.controller.step(
            target_z=self.target_z,
            target_z_vel=self.target_z_vel,
            target_z_acc=self.target_z_acc,
            start_sim_step=self.sim_step,
            gripper_pos=gripper_pos,
        )
        self.sim_step += len(records)
        self.episode_step += 1

        done, reward = self._compute_done_and_reward()
        if done:
            self.reset(warmup_steps=0)
        return records, reward, done

    def get_observation(self):
        left_force, right_force = self.get_finger_net_contact_forces()
        ee_pos = self.ee_link.get_pos().cpu().numpy()
        cuboid_pos = self.cuboid.get_pos().cpu().numpy()
        left_ft = self.get_fingertip_pos(self.left_finger)
        right_ft = self.get_fingertip_pos(self.right_finger)
        finger_mid = (left_ft + right_ft) / 2.0
        return {
            "ee_pos": float(ee_pos[2]),
            "ee_vel": float(self.ee_link.get_vel().cpu().numpy()[2]),
            "fingertip_distance": float(np.linalg.norm(left_ft - right_ft)),
            "target_z_vel": self.target_z_vel,
            "target_z_acc": self.target_z_acc,
            "link_forces": np.stack([left_force, right_force]),
            "cuboid_rel_z": float(cuboid_pos[2] - finger_mid[2]),
            "cuboid_rel_x": float(cuboid_pos[0] - finger_mid[0]),
            "cuboid_rel_y": float(cuboid_pos[1] - finger_mid[1]),
            "desired_rel_z": self.desired_rel_z,
        }

    def get_fingertip_pos(self, finger_link):
        finger_pos = finger_link.get_pos().cpu().numpy().flatten()
        finger_quat = finger_link.get_quat().cpu().numpy().flatten()
        return finger_pos + gu.transform_by_quat(self.fingertip_local_pos, finger_quat)

    def get_fingertip_distance(self):
        left_fingertip_pos = self.get_fingertip_pos(self.left_finger)
        right_fingertip_pos = self.get_fingertip_pos(self.right_finger)
        return np.linalg.norm(left_fingertip_pos - right_fingertip_pos)

    def get_finger_net_contact_forces(self):
        link_forces = self.franka.get_links_net_contact_force().cpu().numpy()
        return link_forces[self.left_finger.idx_local], link_forces[self.right_finger.idx_local]

    def _store_initial_z_error(self):
        """Snapshot tracking baseline at episode start (matches parallel env)."""
        cuboid_pos = self.cuboid.get_pos().cpu().numpy().flatten()
        left_ft  = self.get_fingertip_pos(self.left_finger)
        right_ft = self.get_fingertip_pos(self.right_finger)
        finger_mid = (left_ft + right_ft) / 2.0
        err = abs(cuboid_pos[2] - finger_mid[2] - self.desired_rel_z)
        self.initial_z_error   = max(err, 1e-3)
        self.initial_tracking  = 20.0 * math.exp(-30.0 * self.initial_z_error)

    def get_obs_flat(self) -> np.ndarray:
        """Returns flat (OBS_DIM=15,) float32 array in the same order as FrankaEnvParallel."""
        left_force, right_force = self.get_finger_net_contact_forces()
        ee_pos   = self.ee_link.get_pos().cpu().numpy().flatten()
        ee_vel_z = float(self.ee_link.get_vel().cpu().numpy()[2])
        cuboid_pos = self.cuboid.get_pos().cpu().numpy().flatten()
        left_ft    = self.get_fingertip_pos(self.left_finger)
        right_ft   = self.get_fingertip_pos(self.right_finger)
        finger_mid = (left_ft + right_ft) / 2.0
        ft_dist    = float(np.linalg.norm(left_ft - right_ft))
        return np.array([
            ee_pos[2],                                         # [0]  ee_pos_z
            ee_vel_z,                                         # [1]  ee_vel_z
            ft_dist,                                          # [2]  fingertip_distance
            self.target_z_vel,                                # [3]  target_z_vel
            self.target_z_acc,                                # [4]  target_z_acc
            left_force[0], left_force[1], left_force[2],      # [5:8] left_force
            right_force[0], right_force[1], right_force[2],   # [8:11] right_force
            cuboid_pos[2] - finger_mid[2],                    # [11] cuboid_rel_z
            cuboid_pos[0] - finger_mid[0],                    # [12] cuboid_rel_x
            cuboid_pos[1] - finger_mid[1],                    # [13] cuboid_rel_y
            self.desired_rel_z,                               # [14] desired_rel_z
        ], dtype=np.float32)

    def _compute_done_and_reward(self) -> tuple[bool, float]:
        """Reward function matching FrankaEnvParallel exactly."""
        cuboid_pos = self.cuboid.get_pos().cpu().numpy().flatten()
        ee_pos = self.ee_link.get_pos().cpu().numpy().flatten()
        ee_vel = self.ee_link.get_vel().cpu().numpy().flatten()
        left_ft = self.get_fingertip_pos(self.left_finger)
        right_ft = self.get_fingertip_pos(self.right_finger)
        finger_mid = (left_ft + right_ft) / 2.0
        cuboid_rel_z  = float(cuboid_pos[2] - finger_mid[2])
        cuboid_rel_x  = float(cuboid_pos[0] - finger_mid[0])
        cuboid_rel_y  = float(cuboid_pos[1] - finger_mid[1])
        fingertip_dist = float(np.linalg.norm(left_ft - right_ft))
        ee_z    = float(ee_pos[2])
        ee_vel_z = float(ee_vel[2])

        # ---- done conditions (match parallel env) ----
        timeout = self.episode_step >= self.MAX_EPISODE_LENGTH
        success = (not timeout) and (
            abs(cuboid_rel_z - self.desired_rel_z) <= 0.005
            and abs(ee_vel_z) < 0.01
            and ee_z < 0.86
        )
        fail = (
            abs(cuboid_rel_x) > 0.015
            or abs(cuboid_rel_y) > 0.015
            or fingertip_dist < 0.02
            or abs(cuboid_rel_z) > 0.15
            or ee_z < 0.6
            or ee_z > 0.96
        )
        done = timeout or success or fail
        fail_reasons = []
        if abs(cuboid_rel_x) > 0.015:
            fail_reasons.append(f"cuboid_rel_x={cuboid_rel_x:+.4f}")
        if abs(cuboid_rel_y) > 0.015:
            fail_reasons.append(f"cuboid_rel_y={cuboid_rel_y:+.4f}")
        if fingertip_dist < 0.02:
            fail_reasons.append(f"fingertip_dist={fingertip_dist:.4f}")
        if abs(cuboid_rel_z) > 0.15:
            fail_reasons.append(f"cuboid_rel_z={cuboid_rel_z:+.4f}")
        if ee_z < 0.6:
            fail_reasons.append(f"ee_z_low={ee_z:.4f}")
        if ee_z > 0.96:
            fail_reasons.append(f"ee_z_high={ee_z:.4f}")

        if success:
            self.last_done_reason = "success"
        elif fail_reasons:
            self.last_done_reason = ", ".join(fail_reasons)
        elif timeout:
            self.last_done_reason = "timeout"
        else:
            self.last_done_reason = None

        # ---- dense tracking: normalized [0, 1] ----
        z_error = abs(cuboid_rel_z - self.desired_rel_z)
        tracking_raw = 20.0 * math.exp(-30.0 * z_error)
        denom = max(self.TRACKING_PERFECT - self.initial_tracking, 1e-6)
        tracking = max(0.0, min(1.0, (tracking_raw - self.initial_tracking) / denom))
        
        # ---- jerk penalty ----
        jerk = (self.target_z_vel - self.prev_target_z_vel) / self.Z_VEL_MAX
        jerk_penalty = -0.3 * jerk ** 2

        # ---- commanded z acceleration penalty: linear above threshold ----
        commanded_z_acc = (self.target_z_vel - self.prev_target_z_vel) / self.target_period
        z_acc_excess = max(0.0, abs(commanded_z_acc) - self.Z_ACC_PENALTY_THRESHOLD)
        z_acc_penalty = -self.Z_ACC_PENALTY_WEIGHT * z_acc_excess
        direction_changed = (
            (self.prev_target_z_vel > 0.0 and self.target_z_vel < 0.0)
            or (self.prev_target_z_vel < 0.0 and self.target_z_vel > 0.0)
        )
        if direction_changed:
            self.direction_change_count += 1
        direction_change_penalty = -100.0 if direction_changed else 0.0

        # ---- grip quality ----
        grip = math.exp(-200.0 * max(0.0, fingertip_dist - 0.03))

        # ---- terminal / alive base reward ----
        ep = float(self.episode_step)
        if success:
            base_reward = 200.0 - ep * 0.1
        elif fail:
            base_reward = -300.0
        elif timeout:
            base_reward = -100.0
        else:
            base_reward = -0.75

        # ---- ee_z height penalty ----
        ee_z_penalty = -2.0 * abs(ee_z - self.EE_Z_TARGET)

        reward = (
            base_reward
            + 3.0 * tracking
            + jerk_penalty
            + z_acc_penalty
            + 0.5 * grip
            + ee_z_penalty
        )

        print(f"z_error={z_error:.4f}  tracking={tracking:.2f} reward={reward:.1f}  ")

        return done, reward

    def print_finger_status(self, label):
        fingertip_distance = self.get_fingertip_distance()
        left_force, right_force = self.get_finger_net_contact_forces()
        print(f"{label}: fingertip distance = {fingertip_distance:.6f} m")
        print("left finger force:", left_force, "norm:", np.linalg.norm(left_force))
        print("right finger force:", right_force, "norm:", np.linalg.norm(right_force))
        cuboid_pos = self.cuboid.get_pos().cpu().numpy().flatten()
        finger_mid = (self.get_fingertip_pos(self.left_finger) + self.get_fingertip_pos(self.right_finger)) / 2.0
        print("relative z pos of cuboid and finger_mid:", cuboid_pos[2] - finger_mid[2])
        print("relative x pos of cuboid and finger_mid:", cuboid_pos[0] - finger_mid[0])
        print("relative y pos of cuboid and finger_mid:", cuboid_pos[1] - finger_mid[1])

    def _set_franka_gains(self):
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
            self.motors_dof,
        )

        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200]),
            self.motors_dof,
        )

        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12]),
            np.array([87, 87, 87, 87, 12, 12, 12]),
            self.motors_dof,
        )

        self.franka.set_dofs_kp(np.array([100, 100]), self.fingers_dof)
        self.franka.set_dofs_kv(np.array([10, 10]), self.fingers_dof)
        self.franka.set_dofs_force_range(
            np.array([-100, -100]),
            np.array([100, 100]),
            self.fingers_dof,
        )

    def _reset_cuboid_home_pose(self):
        hand_pos = self.ee_link.get_pos().cpu().numpy().flatten()
        hand_quat = self.ee_link.get_quat().cpu().numpy().flatten()
        local_offset = np.array([0.0, 0.0, 0.1029])
        local_quat = np.array([0.00187891, -0.71790805, -0.00193768, -0.69613270])
        local_quat /= np.linalg.norm(local_quat)
        cuboid_home_pos = hand_pos + gu.transform_by_quat(local_offset, hand_quat)
        cuboid_home_quat = gu.transform_quat_by_quat(local_quat, hand_quat)
        self.cuboid.set_pos(cuboid_home_pos, zero_velocity=True)
        self.cuboid.set_quat(cuboid_home_quat, zero_velocity=True)
