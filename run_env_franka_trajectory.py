import argparse
import time
from pathlib import Path

import numpy as np

try:
    from .env_franka import FrankaEnv
    from .single_franka_vel_sep import cyclic_z_reference, save_motion_plot, save_torque_plot
except ImportError:
    from env_franka import FrankaEnv
    from single_franka_vel_sep import cyclic_z_reference, save_motion_plot, save_torque_plot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_FILE = REPO_ROOT / "franka_env_velocity_motion.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--plot-file", type=str, default=str(DEFAULT_PLOT_FILE))
    parser.add_argument("--training", action="store_true", default=False,
                        help="Training mode: skip plot generation")
    args = parser.parse_args()

    env = FrankaEnv(vis=args.vis, cpu=args.cpu)

    target_distance = 0.1
    upward_accel = 5.0
    braking_accel = 13.0

    def sample_target_z(t):
        return cyclic_z_reference(
            t=t,
            start_z=env.target_center[2],
            distance=target_distance,
            accel_up=upward_accel,
            decel_mag=braking_accel,
        )

    times = []
    target_z_history = []
    actual_z_history = []
    target_z_vel_history = []
    actual_z_vel_history = []
    target_z_acc_history = []
    actual_z_acc_history = []
    position_error_history = []
    qvel_norm_history = []
    torque_history = []

    prev_actual_z_vel = None
    prev_torque = None
    gripper_target_sample = None
    t_offset = 0.0          # keeps plot time monotonic across episode resets
    high_level_steps = args.steps // env.target_update_every

    env.print_finger_status("before motion start")
    print(
        "Realistic smooth target stream enabled: "
        f"target_dt={env.target_period:.3f}s, sim_dt={env.dt:.3f}s, "
        f"interpolation_delay={env.target_period:.3f}s"
    )

    t_real_start = time.perf_counter()

    try:
        for i in range(high_level_steps):
            t = env.sim_step * env.dt
            high_level_sample = sample_target_z(t)

            if gripper_target_sample is None:
                gripper_target_sample = high_level_sample
            if gripper_target_sample[0] >= 0.93:
                gripper_pos = env.open_gripper_pos
            else:
                gripper_pos = env.closed_gripper_pos
            v_des = 1.2 * high_level_sample[1]
            # high_level_sample[1] = 1.2 * high_level_sample[1]
            action = np.concatenate(([v_des], gripper_pos))
            records, reward, done = env.step(action)
            if done:
                print(f"  [DONE] high-level step={i}  reward={reward:.1f}  episode reset")
                t_offset = times[-1] if times else 0.0
                prev_actual_z_vel = None
                prev_torque = None
                gripper_target_sample = None
                continue   # skip appending crash-state records to the plot
            gripper_target_sample = high_level_sample

            for record in records:
                actual_z_acc = (
                    (record.actual_vel[2] - prev_actual_z_vel) / env.dt
                    if prev_actual_z_vel is not None
                    else 0.0
                )
                prev_actual_z_vel = record.actual_vel[2]

                times.append(t_offset + record.time)
                target_z_history.append(record.target_pos[2])
                actual_z_history.append(record.actual_pos[2])
                target_z_vel_history.append(record.target_vel[2])
                actual_z_vel_history.append(record.actual_vel[2])
                target_z_acc_history.append(record.target_z_acc)
                actual_z_acc_history.append(actual_z_acc)
                position_error_history.append(np.linalg.norm(record.target_pos - record.actual_pos))
                qvel_norm_history.append(np.linalg.norm(record.qvel))

                tau_dot = (record.tau - prev_torque) / env.dt if prev_torque is not None else np.zeros_like(record.tau)
                prev_torque = record.tau
                torque_history.append((record.tau, tau_dot))

                if (record.sim_step + 1) % 1000 == 0:
                    env.print_finger_status(f"step {record.sim_step + 1}")

            t_sim = records[-1].time
            t_wall = time.perf_counter() - t_real_start
            if t_wall < t_sim:
                time.sleep(t_sim - t_wall)

            if not args.training and (i + 1) % 10 == 0:
                obs = env.get_observation()
                dist = abs(obs["cuboid_rel_z"] - obs["desired_rel_z"])
                print(
                    f"  hl_step {i+1:5d}  "
                    f"cuboid_rel_z={obs['cuboid_rel_z']:+.4f}  "
                    f"desired_rel_z={obs['desired_rel_z']:+.4f}  "
                    f"dist={dist:.4f}"
                    f"ee_pos_z={obs['ee_pos']:.4f}  "
                    f"ee_vel_z={obs['ee_vel']:.4f}  "
                )

    except KeyboardInterrupt:
        print("Simulation stopped manually.")

    if not args.training and len(times) > 0:
        save_motion_plot(
            args.plot_file,
            np.array(times),
            np.array(target_z_history),
            np.array(actual_z_history),
            np.array(target_z_vel_history),
            np.array(actual_z_vel_history),
            np.array(target_z_acc_history),
            np.array(actual_z_acc_history),
            np.array(position_error_history),
            np.array(qvel_norm_history),
        )
        torque_plot_file = str(Path(args.plot_file).with_stem(Path(args.plot_file).stem + "_torques"))
        save_torque_plot(
            torque_plot_file,
            np.array(times),
            [t[0] for t in torque_history],
            [t[1] for t in torque_history],
        )


if __name__ == "__main__":
    main()
