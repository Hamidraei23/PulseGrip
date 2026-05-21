"""
Equivalent of run_env_franka_trajectory.py using FrankaEnvParallel with num_envs=1.

Run this to visually verify the parallel env produces the same trajectory as the
single-env version.  The motion plot is saved at target_dt resolution (every 0.02 s)
rather than sim_dt (0.001 s) because the parallel env's step() returns one obs per
high-level action, not per sim step.  The trajectory shape should be identical.

Usage:
    python3 examples/rigid/run_env_franka_trajectory_parallel.py -v
    python3 examples/rigid/run_env_franka_trajectory_parallel.py -v --steps 5000
    python3 examples/rigid/run_env_franka_trajectory_parallel.py --record  # headless MP4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs

try:
    from .env_franka_parallel import FrankaEnvParallel
    from .single_franka_vel_sep import cyclic_z_reference, save_motion_plot
except ImportError:
    from env_franka_parallel import FrankaEnvParallel
    from single_franka_vel_sep import cyclic_z_reference, save_motion_plot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_FILE = REPO_ROOT / "franka_parallel_trajectory.png"
CLOSED_GRIPPER = -1.0
OPEN_GRIPPER = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                        help="Open interactive viewer")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record offscreen video to franka_parallel.mp4")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total sim steps to run")
    parser.add_argument("--plot-file", type=str, default=str(DEFAULT_PLOT_FILE))
    parser.add_argument("--training", action="store_true", default=False,
                        help="Training mode: skip plot generation")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    env = FrankaEnvParallel(
        num_envs=1,
        vis=args.vis or args.record,
        dt=0.001,
        target_dt=0.02,
    )

    # Optionally attach an offscreen camera to record env 0
    cam = None
    if args.record:
        cam = env.scene.add_camera(
            res=(1280, 960),
            pos=(3.5, 0.0, 2.5),
            lookat=(0.0, 0.0, 0.5),
            fov=30,
            GUI=False,
        )
        cam.start_recording()

    # start_z comes from the settled ee position after warmup
    start_z = env.target_center[0, 2].item()

    target_distance = 0.1
    upward_accel = 5.0
    braking_accel = 13.0

    def sample_target_z(t):
        return cyclic_z_reference(
            t=t,
            start_z=start_z,
            distance=target_distance,
            accel_up=upward_accel,
            decel_mag=braking_accel,
        )

    # ------------------------------------------------------------------ #
    # Tracking buffers  (one sample per high-level step = every target_dt)#
    # ------------------------------------------------------------------ #
    times = []
    target_z_history = []
    actual_z_history = []
    target_z_vel_history = []
    actual_z_vel_history = []
    target_z_acc_history = []
    actual_z_acc_history = []
    position_error_history = []

    prev_actual_z_vel = None
    gripper_z = None  # last high-level z used for gripper decision
    episode_t0 = 0.0  # sim_step*dt at the start of the current episode
    t_offset = 0.0    # keeps plot time monotonic across episode resets
    high_level_steps = args.steps // env.target_update_every

    print(
        f"target_dt={env.target_period:.3f}s  sim_dt={env.dt:.3f}s  "
        f"steps_per_target={env.target_update_every}"
    )
    print(f"Running {high_level_steps} high-level steps ({args.steps} sim steps total)...")

    t_real_start = time.perf_counter()

    try:
        for i in range(high_level_steps):
            t = env.sim_step * env.dt - episode_t0
            z_ref, z_vel_ref, z_acc_ref = sample_target_z(t)
            z_vel_ref = z_vel_ref * 1.25

            # Gripper: open above threshold, closed below
            if gripper_z is None:
                gripper_z = z_ref
            gripper_val = OPEN_GRIPPER if gripper_z >= 0.93 else CLOSED_GRIPPER
            gripper_z = z_ref

            # Action tensor shape (1, 3): [target_z_vel, finger_l, finger_r]
            action = torch.tensor(
                [[z_vel_ref, gripper_val, gripper_val]],
                dtype=torch.float32,
                device=gs.device,
            )
            obs_td, reward, done, _ = env.step(action)
            obs = obs_td["policy"]
            n_done = done.sum().item()
            if n_done > 0:
                total_reward = reward[done].sum().item()
                print(f"  [DONE] step={i}  envs_reset={int(n_done)}  total_reward={total_reward:.1f}")
                episode_t0 = env.sim_step * env.dt  # restart reference phase
                t_offset = times[-1] if times else 0.0  # keep plot time monotonic
                prev_actual_z_vel = None
                gripper_z = None
                continue   # skip appending crash-state obs to the plot

            if cam is not None:
                cam.render()

            # --- read scalars from obs (env index 0) ---
            actual_z = obs[0, FrankaEnvParallel.OBS_EE_POS_Z].item()
            actual_z_vel = obs[0, FrankaEnvParallel.OBS_EE_VEL_Z].item()

            actual_z_acc = (
                (actual_z_vel - prev_actual_z_vel) / env.target_period
                if prev_actual_z_vel is not None
                else 0.0
            )
            prev_actual_z_vel = actual_z_vel

            t_after = env.sim_step * env.dt - episode_t0
            times.append(t_offset + t_after)
            target_z_history.append(z_ref)
            actual_z_history.append(actual_z)
            target_z_vel_history.append(z_vel_ref)
            actual_z_vel_history.append(actual_z_vel)
            target_z_acc_history.append(z_acc_ref)
            actual_z_acc_history.append(actual_z_acc)
            position_error_history.append(abs(z_ref - actual_z))

            if not args.training and (i + 1) % 100 == 0:
                ft_dist = obs[0, FrankaEnvParallel.OBS_FINGERTIP_DIST].item()
                lf = obs[0, FrankaEnvParallel.OBS_LEFT_FORCE].cpu().numpy()
                rf = obs[0, FrankaEnvParallel.OBS_RIGHT_FORCE].cpu().numpy()
                cuboid_rel_z = obs[0, FrankaEnvParallel.OBS_CUBOID_REL_Z].item()
                desired_rel_z = obs[0, FrankaEnvParallel.OBS_DESIRED_REL_Z].item()
                dist_to_desired = abs(cuboid_rel_z - desired_rel_z)
                print(
                    f"step {env.sim_step:6d}  z={actual_z:.4f}  "
                    f"err={position_error_history[-1]*1000:.2f}mm  "
                    f"ft_dist={ft_dist:.4f}  "
                    f"|lf|={np.linalg.norm(lf):.3f}  |rf|={np.linalg.norm(rf):.3f}  "
                    f"cuboid_rel_z={cuboid_rel_z:+.4f}  "
                    f"desired_rel_z={desired_rel_z:+.4f}  "
                    f"dist={dist_to_desired:.4f}"
                )

            # Optional: pace playback so viewer is not too fast
            if args.vis:
                t_sim = t_after
                t_wall = time.perf_counter() - t_real_start
                if t_wall < t_sim:
                    time.sleep(t_sim - t_wall)

    except KeyboardInterrupt:
        print("Stopped manually.")

    if cam is not None:
        cam.stop_recording(save_to_filename="franka_parallel.mp4", fps=60)
        print("Saved franka_parallel.mp4")

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
            np.zeros(len(times)),   # qvel_norm not available from obs dict
        )


if __name__ == "__main__":
    main()
