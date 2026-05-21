# Franka Lift — PPO Training with Genesis

Reinforcement learning pipeline for training a Franka Panda robot to grasp and lift a cuboid using PPO. Built on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics simulator with [rsl-rl-lib](https://github.com/leggedrobotics/rsl_rl) v5+.

---

## Task

The robot must lift a cuboid (placed in its gripper at reset) to a target height above the fingertip midpoint. The target offset `desired_rel_z` is sampled uniformly from `[0.025, 0.045] m` at each episode reset.

---

## File Overview

| File | Description |
|---|---|
| `env_franka_parallel.py` | GPU-vectorised env for PPO training (`FrankaEnvParallel`) |
| `env_franka.py` | Single-env CPU env for evaluation (`FrankaEnv`) |
| `train_franka_ppo.py` | PPO training entry point (rsl_rl `OnPolicyRunner`) |
| `eval_franka_ppo.py` | Headless / GPU-viewer evaluation with stats |
| `run_policy_franka_cpu.py` | CPU single-env visualisation of a trained policy |
| `run_policy_franka_parallel.py` | GPU parallel visualisation of a trained policy |
| `run_env_franka_trajectory.py` | Manual trajectory test (no policy) |
| `run_env_franka_trajectory_parallel.py` | Parallel manual trajectory test |
| `assets/xml/franka_emika_panda/` | MJCF robot and object descriptions |

---

## Dependencies

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # or cpu / metal

pip install rsl-rl-lib>=5.0.0
```

Genesis must be installed from the parent workspace (`genesis-world`).

---

## Quick Start

### Train

```bash
# From workspace root — single GPU
python franka_lift/train_franka_ppo.py -e franka-lift-v1 -B 512 --max_iterations 1000

# Inside docker (genesis container)
docker exec genesis python /workspace/franka_lift/train_franka_ppo.py \
    -e franka-lift-v1 -B 512 --max_iterations 1000
```

Checkpoints are saved every **20 iterations** under `logs/<exp_name>/model_<iter>.pt`.

### Resume training

```bash
python franka_lift/train_franka_ppo.py -e franka-lift-v1 --resume logs/franka-lift-v1/model_500.pt
```

### Evaluate (GPU, parallel envs, with viewer)

```bash
docker exec -it genesis python /workspace/franka_lift/eval_franka_ppo.py \
    -e franka-lift-v1 --ckpt 500

# Headless stats over 200 episodes with 64 envs
docker exec genesis python /workspace/franka_lift/eval_franka_ppo.py \
    -e franka-lift-v1 --ckpt 500 --no-vis --num_envs 64 --episodes 200
```

### Visualise on CPU (single env)

```bash
docker exec genesis python /workspace/franka_lift/run_policy_franka_cpu.py \
    -e franka-lift-v1 --ckpt 520 --vis

# Record a video
docker exec genesis python /workspace/franka_lift/run_policy_franka_cpu.py \
    -e franka-lift-v1 --ckpt 520 --record

# Run N steps headless
docker exec genesis python /workspace/franka_lift/run_policy_franka_cpu.py \
    -e franka-lift-v1 --ckpt 520 --steps 500
```

---

## Environment Details

### Observation Space — `OBS_DIM = 15`

| Index | Name | Description |
|---|---|---|
| 0 | `ee_pos_z` | End-effector height (m) |
| 1 | `ee_vel_z` | End-effector vertical velocity (m/s) |
| 2 | `fingertip_dist` | Distance between fingertips (m) |
| 3 | `target_z_vel` | Current commanded Z velocity (m/s) |
| 4 | `target_z_acc` | Current commanded Z acceleration (m/s²) |
| 5–7 | `left_force` | Net contact force on left finger (N, xyz) |
| 8–10 | `right_force` | Net contact force on right finger (N, xyz) |
| 11 | `cuboid_rel_z` | Cuboid Z above fingertip midpoint (m) |
| 12 | `cuboid_rel_x` | Cuboid X offset from fingertip midpoint (m) |
| 13 | `cuboid_rel_y` | Cuboid Y offset from fingertip midpoint (m) |
| 14 | `desired_rel_z` | Target lift height above fingertips (m) |

### Action Space — `action_dim = 3`

| Index | Range | Description |
|---|---|---|
| 0 | `[-1, 1]` | Target Z velocity command → scaled to `±Z_VEL_MAX = 0.85 m/s` |
| 1 | `[-1, 1]` | Left finger position → mapped to `[GRIPPER_CLOSED, GRIPPER_OPEN]` |
| 2 | `[-1, 1]` | Right finger position → mapped to `[GRIPPER_CLOSED, GRIPPER_OPEN]` |

**Acceleration hard cap:** velocity commands are clamped so the commanded acceleration never exceeds `Z_ACC_MAX = 60 m/s²` (with a soft penalty applied above `13 m/s²`).

### Control

- **Sim dt:** 1 ms (`dt = 0.001 s`)
- **Policy frequency:** 50 Hz (`target_dt = 0.02 s`, 20 sim steps per policy step)
- **Controller:** Jacobian-based velocity IK with damped least-squares, cubic-Hermite interpolation between policy steps
- **Gripper:** Direct position control

### Episode / Done Conditions

| Condition | Trigger |
|---|---|
| **Success** | `\|cuboid_rel_z − desired_rel_z\| ≤ 0.01 m` AND `\|ee_vel_z\| < 0.01 m/s` |
| **Fail** | Cuboid drifts > 15 mm in X or Y, fingertip distance < 20 mm, cuboid Z offset > 150 mm, EE below 0.76 m or above 0.96 m |
| **Timeout** | 120 steps (2.4 s of sim time) |

### Reward Function

```
reward = base_reward
       + 3.0 × tracking          # normalised lift progress [0, 1]
       + jerk_penalty             # −0.2 × (Δv / Z_VEL_MAX)²
       + z_acc_penalty            # −0.25 × max(|acc| − 13, 0)
       + 0.5 × grip               # exp(-200 × max(fingertip_dist − 0.03, 0))
       + ee_z_penalty             # −0.35 × |ee_z − 0.70|
```

**Base reward:**
- Success: `+500 − episode_step × 0.1`
- Fail / Timeout: `−250`
- Alive (per step): `−0.15`

### PPO Hyperparameters

| Parameter | Value |
|---|---|
| Network | MLP 256→128→64, ELU |
| Rollout steps per env | 32 |
| Mini-batches | 4 |
| Learning epochs | 5 |
| Learning rate | 3 × 10⁻⁴ (adaptive KL) |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Entropy coefficient | 0.005 |
| Clip ε | 0.2 |

---

## Logs and Checkpoints

```
logs/
└── franka-lift-v1/
    ├── model_20.pt
    ├── model_40.pt
    ├── ...
    └── config.pkl      ← env + train config snapshot
```

TensorBoard logs are written alongside checkpoints:

```bash
tensorboard --logdir logs/
```
