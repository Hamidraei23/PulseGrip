"""
CPU single-env policy rollout for a trained Franka PPO policy.

Based on run_env_franka_trajectory.py but driven by the trained PPO actor
instead of a hand-crafted trajectory.  Runs on CPU with FrankaEnv (single env).

Usage:
    python3 examples/rigid/run_policy_franka_cpu.py -e franka-lift-v1
    python3 examples/rigid/run_policy_franka_cpu.py -e franka-lift-v1 --ckpt 200
    python3 examples/rigid/run_policy_franka_cpu.py -e franka-lift-v1 --vis
    python3 examples/rigid/run_policy_franka_cpu.py -e franka-lift-v1 --record

Checkpoints are loaded from:
    logs/<exp_name>/model_<iter>.pt
"""

import argparse
import os
import pickle
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

try:
    from .env_franka import FrankaEnv
    from .single_franka_vel_sep import save_torque_plot
except ImportError:
    from env_franka import FrankaEnv
    from single_franka_vel_sep import save_torque_plot

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_FILE = REPO_ROOT / "franka_policy_cpu_motion.png"


class _RunnerEnvProxy:
    """
    Minimal env shim used only to initialise OnPolicyRunner.
    No scene is built — it just satisfies the interface so the runner
    can reconstruct the actor/critic network architectures.
    """
    num_envs    = 1
    num_actions = 3
    extras      = {}
    cfg         = {}   # OnPolicyRunner passes this to Logger; empty dict is fine

    def _obs_td(self):
        obs = torch.zeros(1, FrankaEnv.OBS_DIM)
        return TensorDict({"policy": obs}, batch_size=[1])

    def get_observations(self):
        return self._obs_td()

    def reset(self):
        return self._obs_td()

    def step(self, _actions):
        return self._obs_td(), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}


def obs_to_tensordict(obs_flat: np.ndarray, device) -> TensorDict:
    """Convert (OBS_DIM,) numpy array to rsl_rl-compatible TensorDict."""
    t = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 15)
    return TensorDict({"policy": t}, batch_size=[1])


def save_policy_motion_plot(
    plot_file,
    times,
    target_z,
    actual_z,
    target_z_vel,
    actual_z_vel,
    target_z_acc,
    actual_z_acc,
    z_error,
    position_error,
    qvel_norm,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_file).expanduser()
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

    axes[0].plot(times, target_z, label="target z", linewidth=1.8)
    axes[0].plot(times, actual_z, label="actual z", linewidth=1.2)
    axes[0].set_ylabel("z position (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, target_z_vel, label="target vz", linewidth=1.8)
    axes[1].plot(times, actual_z_vel, label="actual vz", linewidth=1.2)
    axes[1].set_ylabel("z velocity (m/s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, target_z_acc, label="target az", linewidth=1.8)
    axes[2].plot(times, actual_z_acc, label="actual az", linewidth=1.2)
    axes[2].set_ylabel("z acceleration (m/s^2)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, z_error, color="tab:purple", linewidth=1.2)
    axes[3].set_ylabel("z error (m)")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(times, position_error, color="tab:red", linewidth=1.2)
    axes[4].set_ylabel("position error (m)")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(times, qvel_norm, color="tab:green", linewidth=1.2)
    axes[5].set_ylabel("||qvel||")
    axes[5].set_xlabel("time (s)")
    axes[5].grid(True, alpha=0.3)

    fig.suptitle("Franka PPO Policy Rollout Motion")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Saved motion plot to {plot_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="CPU single-env PPO policy rollout for Franka")
    parser.add_argument("-e", "--exp_name", type=str, default="franka-lift-v1",
                        help="Experiment name matching the training run")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration (e.g. 200). Defaults to latest.")
    parser.add_argument("--vis", action="store_true", default=False,
                        help="Open interactive viewer")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Record offscreen video to franka_policy_cpu.mp4. Forces --no-vis.")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Max high-level steps for the single episode (0 = no cap)")
    parser.add_argument("--plot-file", type=str, default=str(DEFAULT_PLOT_FILE),
                        help="Path for the motion plot. Torque plot uses the same stem plus '_torques'.")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--target_dt", type=float, default=0.02)
    parser.add_argument("--playback-speed", type=float, default=0.5,
                        help="Viewer playback speed relative to real time. 0.5 = half speed.")
    args = parser.parse_args()

    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be greater than 0")

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

    # ---- Genesis on CPU ---------------------------------------------------
    # FrankaEnv calls gs.init() internally (cpu=True).
    env = FrankaEnv(
        vis=args.vis,
        cpu=True,
        dt=args.dt,
        target_dt=args.target_dt,
        playback_speed=args.playback_speed,
    )

    # Use a lightweight proxy (no scene) to initialise the runner so that
    # no second Genesis scene is built (which would crash the viewer).
    runner = OnPolicyRunner(_RunnerEnvProxy(), train_cfg, log_dir, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    # ---- run loop ---------------------------------------------------------
    obs_flat = env.reset()  # warms up the env
    obs_flat = env.get_obs_flat()
    obs_td   = obs_to_tensordict(obs_flat, gs.device)

    high_level_steps = args.steps if args.steps > 0 else float("inf")

    ep_reward = 0.0
    ep_len    = 0
    ep_count  = 0
    prev_actual_z_vel = None
    prev_record_actual_z_vel = None
    prev_torque = None

    times = []
    target_z_history = []
    actual_z_history = []
    target_z_vel_history = []
    actual_z_vel_history = []
    target_z_acc_history = []
    actual_z_acc_history = []
    z_error_history = []
    position_error_history = []
    qvel_norm_history = []
    torque_history = []

    print(
        f"target_dt={env.target_period:.3f}s  sim_dt={env.dt:.3f}s  "
        f"steps_per_target={env.target_update_every}  playback_speed={args.playback_speed:.2f}x"
    )
    print(
        f"Running one episode "
        f"({'no step cap' if args.steps == 0 else f'max {int(high_level_steps)} high-level steps'}) "
        f"(checkpoint: {os.path.basename(ckpt_path)})"
    )

    i = 0
    try:
        with torch.no_grad():
            while i < high_level_steps:
                # Policy inference: TensorDict → action tensor (1, 3)
                action_td = policy(obs_td)
                # rsl_rl v5 returns a tensor directly from get_inference_policy
                action_np = action_td.cpu().numpy().flatten()  # (3,)

                # Step the single-env (returns records, reward, done)
                records, reward, done = env.step(action_np)

                for record in records:
                    actual_z_acc = (
                        (record.actual_vel[2] - prev_record_actual_z_vel) / env.dt
                        if prev_record_actual_z_vel is not None
                        else 0.0
                    )
                    prev_record_actual_z_vel = record.actual_vel[2]

                    times.append(record.time)
                    target_z_history.append(record.target_pos[2])
                    actual_z_history.append(record.actual_pos[2])
                    target_z_vel_history.append(record.target_vel[2])
                    actual_z_vel_history.append(record.actual_vel[2])
                    target_z_acc_history.append(record.target_z_acc)
                    actual_z_acc_history.append(actual_z_acc)
                    z_error_history.append(record.target_pos[2] - record.actual_pos[2])
                    position_error_history.append(np.linalg.norm(record.target_pos - record.actual_pos))
                    qvel_norm_history.append(np.linalg.norm(record.qvel))

                    tau_dot = (
                        (record.tau - prev_torque) / env.dt
                        if prev_torque is not None
                        else np.zeros_like(record.tau)
                    )
                    prev_torque = record.tau
                    torque_history.append((record.tau, tau_dot))

                ep_reward += reward
                ep_len    += 1

                if done:
                    success = env.last_done_reason == "success"
                    ep_count += 1
                    result = "SUCCESS" if success else f"fail reason={env.last_done_reason or 'unknown'}"
                    print(
                        f"  [EP {ep_count}] len={ep_len}  reward={ep_reward:.1f}  "
                        f"{result}"
                    )
                    break

                obs_flat = env.get_obs_flat()
                obs_td   = obs_to_tensordict(obs_flat, gs.device)

                if i % 100 == 0:
                    actual_z      = float(obs_flat[0])   # ee_pos_z
                    actual_z_vel  = float(obs_flat[1])   # ee_vel_z
                    ft_dist       = float(obs_flat[2])   # fingertip_dist
                    lf            = obs_flat[5:8]
                    rf            = obs_flat[8:11]
                    cuboid_rel_z  = float(obs_flat[11])
                    desired_rel_z = float(obs_flat[14])
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

                i += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    if i >= high_level_steps and ep_count == 0:
        print(f"Stopped after step cap before episode ended: {int(high_level_steps)} high-level steps.")

    if len(times) > 0:
        save_policy_motion_plot(
            args.plot_file,
            np.array(times),
            np.array(target_z_history),
            np.array(actual_z_history),
            np.array(target_z_vel_history),
            np.array(actual_z_vel_history),
            np.array(target_z_acc_history),
            np.array(actual_z_acc_history),
            np.array(z_error_history),
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
