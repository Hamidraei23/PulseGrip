"""
Evaluate a trained FrankaEnvParallel PPO policy.

Usage (inside docker):
    # Run with viewer, load latest checkpoint
    docker exec -it genesis python /workspace/examples/rigid/eval_franka_ppo.py \
        -e franka-lift-v1

    # Specific checkpoint iteration
    docker exec -it genesis python /workspace/examples/rigid/eval_franka_ppo.py \
        -e franka-lift-v1 --ckpt 500

    # Headless stats over 200 episodes with 64 parallel envs
    docker exec genesis python /workspace/examples/rigid/eval_franka_ppo.py \
        -e franka-lift-v1 --ckpt 500 --no-vis --num_envs 64 --episodes 200

    # Record an MP4 (camera is created inside env before scene.build)
    docker exec genesis python /workspace/examples/rigid/eval_franka_ppo.py \
        -e franka-lift-v1 --ckpt 500 --record

Checkpoints are saved every 100 iterations under:
    logs/<exp_name>/model_<iter>.pt
"""

import argparse
import os
import pickle
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
    parser = argparse.ArgumentParser(description="Evaluate a trained Franka PPO policy")
    parser.add_argument("-e", "--exp_name", type=str, default="franka-lift",
                        help="Experiment name matching the training run")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration (e.g. 500). Defaults to latest.")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Parallel envs during eval (default 1 for viewer)")
    parser.add_argument("--no-vis", dest="vis", action="store_false", default=True,
                        help="Disable interactive viewer (for headless / stats)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Collect stats over N completed episodes then exit. 0 = run forever.")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record offscreen video to franka_eval.mp4. Forces --no-vis and num_envs=1.")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--target_dt", type=float, default=0.02)
    args = parser.parse_args()

    if args.record:
        args.vis = False
        args.num_envs = 1

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

    # NOTE: record=True tells the env to add the camera BEFORE scene.build().
    # Adding a camera after build raises "Scene is already built."
    env = FrankaEnvParallel(
        num_envs=args.num_envs,
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

    # ---- evaluation loop --------------------------------------------------
    obs_td = env.reset()

    ep_rewards   = []
    ep_lengths   = []
    ep_successes = []

    cur_ep_reward = torch.zeros(args.num_envs, device=gs.device)
    cur_ep_len    = torch.zeros(args.num_envs, device=gs.device)

    completed = 0
    target_episodes = args.episodes if args.episodes > 0 else float("inf")

    print("Running evaluation. Press Ctrl-C to stop." if args.vis else
          f"Collecting {args.episodes} episodes..." if args.episodes > 0 else
          "Running headless evaluation. Press Ctrl-C to stop.")

    try:
        with torch.no_grad():
            while completed < target_episodes:
                actions = policy(obs_td)
                obs_td, rew_buf, reset_buf, extras = env.step(actions)

                cur_ep_reward += rew_buf
                cur_ep_len    += 1

                if args.record:
                    env.cam.render()

                done_idx = reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if done_idx.numel() > 0:
                    for idx in done_idx.tolist():
                        ep_rewards.append(cur_ep_reward[idx].item())
                        ep_lengths.append(cur_ep_len[idx].item())
                        # success base reward is 1000-ep (>0); fail is -250-ep (<0)
                        ep_successes.append(1 if cur_ep_reward[idx].item() > 0 else 0)
                        cur_ep_reward[idx] = 0.0
                        cur_ep_len[idx]    = 0.0
                        completed += 1

                    if args.episodes > 0 and completed % max(1, args.episodes // 10) == 0:
                        print(
                            f"  [{completed}/{args.episodes}]  "
                            f"success={np.mean(ep_successes):.2%}  "
                            f"avg_reward={np.mean(ep_rewards):.1f}  "
                            f"avg_len={np.mean(ep_lengths):.0f}"
                        )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # ---- summary ----------------------------------------------------------
    if ep_rewards:
        print("\n=== Evaluation Summary ===")
        print(f"  Episodes completed : {len(ep_rewards)}")
        print(f"  Success rate       : {np.mean(ep_successes):.2%}")
        print(f"  Avg episode reward : {np.mean(ep_rewards):.2f}  +/- {np.std(ep_rewards):.2f}")
        print(f"  Avg episode length : {np.mean(ep_lengths):.1f} steps  "
              f"({np.mean(ep_lengths) * args.target_dt:.1f} s)")
        print(f"  Reward range       : [{np.min(ep_rewards):.1f}, {np.max(ep_rewards):.1f}]")

    if args.record and env.cam is not None:
        env.cam.stop_recording(save_to_filename="franka_eval.mp4", fps=60)
        print("Saved franka_eval.mp4")


if __name__ == "__main__":
    main()
