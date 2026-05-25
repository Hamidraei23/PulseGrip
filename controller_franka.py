import time
from dataclasses import dataclass

import numpy as np

import genesis as gs


@dataclass
class FrankaControlRecord:
    sim_step: int
    time: float
    target_pos: np.ndarray
    target_vel: np.ndarray
    target_z_acc: float
    qvel: np.ndarray
    actual_pos: np.ndarray
    actual_vel: np.ndarray
    tau: np.ndarray


class FrankaVelocityController:
    def __init__(
        self,
        franka,
        scene,
        ee_link,
        motors_dof,
        target_center,
        target_quat,
        pos_gain,
        rot_gain,
        jacobian_regularizer,
        dt,
        target_period,
        steps_per_target,
        render_every,
        fingers_dof=None,
        playback_speed=None,
    ):
        self.franka = franka
        self.scene = scene
        self.ee_link = ee_link
        self.motors_dof = motors_dof
        self.fingers_dof = fingers_dof
        self.target_center = target_center
        self.target_quat = target_quat
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self.jacobian_regularizer = jacobian_regularizer
        self.dt = dt
        self.target_period = target_period
        self.steps_per_target = steps_per_target
        self.render_every = render_every
        self.playback_speed = playback_speed

        self._last_high_level_sample = None
        self._segment_start_sample = None
        self._segment_end_sample = None
        self._segment_playback_start_t = 0.0
        self._playback_wall_start = time.perf_counter()

    def reset(self, initial_sample=None):
        self._last_high_level_sample = initial_sample
        self._segment_start_sample = None
        self._segment_end_sample = None
        self._segment_playback_start_t = 0.0

    def reset_playback_clock(self):
        self._playback_wall_start = time.perf_counter()

    def step(self, target_z, target_z_vel, target_z_acc, start_sim_step, gripper_pos=None):
        high_level_sample = (target_z, target_z_vel, target_z_acc)
        self._update_target_segment(high_level_sample, start_sim_step * self.dt)
        self.control_gripper(gripper_pos)

        records = []
        for local_step in range(self.steps_per_target):
            sim_step = start_sim_step + local_step
            t = sim_step * self.dt
            target_z, target_z_vel, target_z_acc = self._sample_target(t)
            target_pos, target_vel, qvel = self._control_once(target_z, target_z_vel)

            update_visualizer = sim_step % self.render_every == 0
            self._pace_visualizer(update_visualizer, sim_step)
            self.scene.step(update_visualizer=update_visualizer)

            records.append(
                FrankaControlRecord(
                    sim_step=sim_step,
                    time=(sim_step + 1) * self.dt,
                    target_pos=target_pos,
                    target_vel=target_vel,
                    target_z_acc=target_z_acc,
                    qvel=qvel,
                    actual_pos=self.ee_link.get_pos().cpu().numpy(),
                    actual_vel=self.ee_link.get_vel().cpu().numpy(),
                    tau=self.franka.get_dofs_control_force(self.motors_dof).cpu().numpy(),
                )
            )

        return records

    def _pace_visualizer(self, update_visualizer, sim_step):
        if self.playback_speed is None or not update_visualizer:
            return

        t_playback = ((sim_step + 1) * self.dt) / self.playback_speed
        t_wall = time.perf_counter() - self._playback_wall_start
        if t_wall < t_playback:
            time.sleep(t_playback - t_wall)

    def control_gripper(self, gripper_pos):
        if gripper_pos is None:
            return
        if self.fingers_dof is None:
            raise ValueError("fingers_dof must be provided to control the gripper")

        gripper_pos = np.asarray(gripper_pos, dtype=float).reshape(-1)
        if gripper_pos.shape != (2,):
            raise ValueError(f"gripper_pos must have shape (2,), got {gripper_pos.shape}")

        self.franka.control_dofs_position(gripper_pos, self.fingers_dof)

    def _update_target_segment(self, high_level_sample, playback_start_t):
        if self._last_high_level_sample is None:
            self._last_high_level_sample = high_level_sample
            return

        self._segment_start_sample = self._last_high_level_sample
        self._segment_end_sample = high_level_sample
        self._segment_playback_start_t = playback_start_t
        self._last_high_level_sample = high_level_sample

    def _sample_target(self, t):
        if self._segment_end_sample is None:
            return self._last_high_level_sample

        return cubic_hermite_z_reference(
            local_t=t - self._segment_playback_start_t,
            segment_dt=self.target_period,
            start_sample=self._segment_start_sample,
            end_sample=self._segment_end_sample,
        )

    def _control_once(self, target_z, target_z_vel):
        target_pos = self.target_center.copy()
        target_vel = np.zeros(3)
        target_pos[2] = target_z
        target_vel[2] = target_z_vel

        ee_velocity_cmd, jacobian = compute_ee_velocity_command_and_jacobian(
            franka=self.franka,
            ee_link=self.ee_link,
            motors_dof=self.motors_dof,
            target_pos=target_pos,
            target_vel=target_vel,
            target_quat=self.target_quat,
            pos_gain=self.pos_gain,
            rot_gain=self.rot_gain,
        )
        qvel = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + self.jacobian_regularizer,
            ee_velocity_cmd,
        )

        self.franka.control_dofs_velocity(qvel, self.motors_dof)

        return target_pos, target_vel, qvel


def cubic_hermite_z_reference(local_t, segment_dt, start_sample, end_sample):
    z0, z_vel0, _ = start_sample
    z1, z_vel1, _ = end_sample
    s = np.clip(local_t / segment_dt, 0.0, 1.0)

    s2 = s * s
    s3 = s2 * s

    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    z = h00 * z0 + h10 * segment_dt * z_vel0 + h01 * z1 + h11 * segment_dt * z_vel1

    dh00 = 6.0 * s2 - 6.0 * s
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = -6.0 * s2 + 6.0 * s
    dh11 = 3.0 * s2 - 2.0 * s
    z_vel = (dh00 * z0 + dh10 * segment_dt * z_vel0 + dh01 * z1 + dh11 * segment_dt * z_vel1) / segment_dt

    ddh00 = 12.0 * s - 6.0
    ddh10 = 6.0 * s - 4.0
    ddh01 = -12.0 * s + 6.0
    ddh11 = 6.0 * s - 2.0
    z_acc = (ddh00 * z0 + ddh10 * segment_dt * z_vel0 + ddh01 * z1 + ddh11 * segment_dt * z_vel1) / (
        segment_dt**2
    )

    return z, z_vel, z_acc


def compute_ee_velocity_command_and_jacobian(
    franka,
    ee_link,
    motors_dof,
    target_pos,
    target_vel,
    target_quat,
    pos_gain,
    rot_gain,
):
    error_pos = target_pos - ee_link.get_pos().cpu().numpy()

    ee_quat = ee_link.get_quat().cpu().numpy()
    error_quat = gs.transform_quat_by_quat(gs.inv_quat(ee_quat), target_quat)
    error_rotvec = gs.quat_to_rotvec(error_quat)

    ee_velocity_cmd = np.concatenate(
        [
            target_vel + pos_gain * error_pos,
            rot_gain * error_rotvec,
        ]
    )

    jacobian = franka.get_jacobian(link=ee_link).cpu().numpy()[:, motors_dof]
    return ee_velocity_cmd, jacobian
