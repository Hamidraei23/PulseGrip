"""
Visualised policy rollout for a trained FrankaEnvParallel PPO policy.

Runs a single env with the interactive viewer (or records an MP4), printing
per-step obs diagnostics every 100 high-level steps — same style as
run_env_franka_trajectory_parallel.py, but driven by the trained policy
instead of a hand-crafted trajectory.

Usage:
    # Interactive viewer, latest checkpoint
    python3 examples/rigid/run_policy_franka_parallel.py -e franka-lift-v1

    # Specific checkpoint iteration
    python3 examples/rigid/run_policy_franka_parallel.py -e franka-lift-v1 --ckpt 500

    # Headless (no viewer)
    python3 examples/rigid/run_policy_franka_parallel.py -e franka-lift-v1 --no-vis

    # Record an MP4
    python3 examples/rigid/run_policy_franka_parallel.py -e franka-lift-v1 --record

Checkpoints are loaded from:
    logs/<exp_name>/model_<iter>.pt
"""

import argparse
import os
import pickle
import time
from importlib import metadata

import numpy as np
import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

try:
    from .env_franka_parallel import FrankaEnvParallel
except ImportError:
    from env_franka_parallel import FrankaEnvParallel


def main():
    parser = argparse.ArgumentParser(description="Visualised PPO policy rollout for Franka")
    parser.add_argument("-e", "--exp_name", type=str, default="franka-lift",
                        help="Experiment name matching the training run")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration (e.g. 500). Defaults to latest.")
    parser.add_argument("--no-vis", dest="vis", action="store_false", default=True,
                        help="Disable interactive viewer (headless)")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record offscreen video to franka_policy.mp4. Forces --no-vis.")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Total high-level steps to run (0 = run until Ctrl-C)")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--target_dt", type=float, default=0.02)
    args = parser.parse_args()

    if args.record:
        args.vis = False

    log_dir = f"logs/{args.exp_name}"

    # ---- resolve checkpoint -----------------------------------------------
    if args.ckpt is not None:
        ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        pts = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not pts:
            raise FileNotFoundError(f"No model_*.pt found in {log_dir}")
        pts.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
        ckpt_path = os.path.join(log_dir, pts[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    # ---- load training config ---------------------------------------------
    with open(os.path.join(log_dir, "train_cfg.pkl"), "rb") as f:
        train_cfg = pickle.load(f)

    # ---- Genesis + env ----------------------------------------------------
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # record=True tells the env to add the camera BEFORE scene.build()
    env = FrankaEnvParallel(
        num_envs=1,
        vis=args.vis,
        record=args.record,
        dt=args.dt,
        target_dt=args.target_dt,
    )

    # ---- load policy ------------------------------------------------------
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    if args.record:
        env.cam.start_recording()

    # ---- run loop ---------------------------------------------------------
    obs_td = env.reset()

    high_level_steps = args.steps if args.steps > 0 else float("inf")

    ep_reward = 0.0
    ep_len = 0
    ep_count = 0
    prev_actual_z_vel = None
    t_real_start = time.perf_counter()

    print(
        f"target_dt={env.target_period:.3f}s  sim_dt={env.dt:.3f}s  "
        f"steps_per_target={env.target_update_every}"
    )
    print(f"Running {'forever' if args.steps == 0 else high_level_steps} high-level steps...")

    i = 0
    try:
        with torch.no_grad():
            while i < high_level_steps:
                actions = policy(obs_td)
                obs_td, rew_buf, reset_buf, _ = env.step(actions)
                obs = obs_td["policy"]  # (1, OBS_DIM)

                ep_reward += rew_buf[0].item()
                ep_len += 1

                if args.record:
                    env.cam.render()

                if reset_buf[0].item():
                    success = ep_reward > 0
                    ep_count += 1
                    print(
                        f"  [EP {ep_count}] len={ep_len}  reward={ep_reward:.1f}  "
                        f"{'SUCCESS' if success else 'fail'}"
                    )
                    ep_reward = 0.0
                    ep_len = 0
                    prev_actual_z_vel = None

                elif i % 100 == 0:
                    actual_z     = obs[0, FrankaEnvParallel.OBS_EE_POS_Z].item()
                    actual_z_vel = obs[0, FrankaEnvParallel.OBS_EE_VEL_Z].item()
                    ft_dist      = obs[0, FrankaEnvParallel.OBS_FINGERTIP_DIST].item()
                    lf           = obs[0, FrankaEnvParallel.OBS_LEFT_FORCE].cpu().numpy()
                    rf           = obs[0, FrankaEnvParallel.OBS_RIGHT_FORCE].cpu().numpy()
                    cuboid_rel_z = obs[0, FrankaEnvParallel.OBS_CUBOID_REL_Z].item()
                    desired_rel_z = obs[0, FrankaEnvParallel.OBS_DESIRED_REL_Z].item()
                    z_acc = (
                        (actual_z_vel - prev_actual_z_vel) / env.target_period
                        if prev_actual_z_vel is not None else 0.0
                    )
                    prev_actual_z_vel = actual_z_vel
                    print(
                        f"step {env.sim_step:6d}  z={actual_z:.4f}  z_vel={actual_z_vel:+.3f}  "
                        f"z_acc={z_acc:+.2f}  ft_dist={ft_dist:.4f}  "
                        f"|lf|={np.linalg.norm(lf):.3f}  |rf|={np.linalg.norm(rf):.3f}  "
                        f"cuboid_rel_z={cuboid_rel_z:+.4f}  desired={desired_rel_z:+.4f}  "
                        f"err={abs(cuboid_rel_z - desired_rel_z)*1000:.2f}mm  "
                        f"ep_rew={ep_reward:.1f}"
                    )

                # Real-time pacing when viewer is open
                if args.vis:
                    t_sim = env.sim_step * env.dt
                    t_wall = time.perf_counter() - t_real_start
                    if t_wall < t_sim:
                        time.sleep(t_sim - t_wall)

                i += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    if args.record and env.cam is not None:
        env.cam.stop_recording(save_to_filename="franka_policy.mp4", fps=60)
        print("Saved franka_policy.mp4")


if __name__ == "__main__":
    main()
