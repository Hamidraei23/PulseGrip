"""
PPO training for FrankaEnvParallel with rsl_rl v5+.

Usage (from workspace root):
    python examples/rigid/train_franka_ppo.py
    python examples/rigid/train_franka_ppo.py -B 512 --max_iterations 1000
    python examples/rigid/train_franka_ppo.py --resume logs/franka-lift/model_100.pt

Inside docker (genesis container):
    docker exec genesis python /workspace/examples/rigid/train_franka_ppo.py -B 512
"""

import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

try:
    from .env_franka_parallel import FrankaEnvParallel
except ImportError:
    from env_franka_parallel import FrankaEnvParallel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_train_cfg(exp_name: str) -> dict:
    """rsl_rl v5+ OnPolicyRunner config dict."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
        # TensorDict key groups fed to actor / critic
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 32,   # rollout steps collected per env per iteration
        "save_interval": 20,        # checkpoint every N iterations
        "run_name": exp_name,
        "logger": "tensorboard",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO training for FrankaEnvParallel")
    parser.add_argument("-e", "--exp_name", type=str, default="franka-lift",
                        help="Experiment name; also used as log subdirectory")
    parser.add_argument("-B", "--num_envs", type=int, default=1024,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Total PPO update iterations")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume training from")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Physics sim timestep (seconds)")
    parser.add_argument("--target_dt", type=float, default=0.02,
                        help="High-level action period (seconds); must be a multiple of dt")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    train_cfg = get_train_cfg(args.exp_name)

    # Fresh run: wipe old logs; resume: keep them
    if args.resume is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/train_cfg.pkl", "wb") as f:
        pickle.dump(train_cfg, f)

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    env = FrankaEnvParallel(
        num_envs=args.num_envs,
        vis=False,
        dt=args.dt,
        target_dt=args.target_dt,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume is not None:
        runner.load(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
