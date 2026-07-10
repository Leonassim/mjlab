![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab — RHPS1 fork

> **Fork by [Léo Moussafir](https://github.com/Leonassim) (AIST)**
> Adds support for the **RHPS1 humanoid robot** on top of the official [mujocolab/mjlab](https://github.com/mujocolab/mjlab).

[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).

## What's added in this fork

### RHPS1 robot support

- **`src/mjlab/asset_zoo/robots/RHPS1/`** — RHPS1 MuJoCo XML model, mesh assets, robot constants and action scales
- **`src/mjlab/tasks/velocity/config/rhps1/`** — Velocity training config (`env_cfgs.py`, `rl_cfg.py`) with:
  - Split-foot contact rewards (8-slot foot sensor)
  - Foot height curriculum (min 8 cm, target 15 cm)
  - Air-time curriculum (5 → 50 weight over training)
  - Velocity curriculum up to 0.3 m/s
  - Impact velocity penalty, flat foot landing reward
- **`src/mjlab/actuator/finite_difference_pd_actuator.py`** — `FiniteDifferencePdActuator`: PD actuator with finite-difference velocity estimation and joint velocity damper, matching mc-rtc QP deployment constraints

### New MDP functions

- **`tasks/velocity/mdp/rewards.py`** — `split_feet_air_time`, `swing_foot_height`, `feet_distance_penalty`, `no_double_flight_penalty`, `standing_single_support_penalty`, `flat_touchdown_penalty`, `flat_support_penalty`, `stance_action_acc_l2`, `impact_velocity`, `split_feet_swing_height`
- **`tasks/velocity/mdp/curriculums.py`** — `air_time_curriculum`, `standing_envs_curriculum`, `reward_weight`, `velocity_damper_progress`
- **`tasks/velocity/mdp/constraints.py`** — Constraint terms for CaT-style training: `cstr_joint_torque`, `cstr_joint_position`, `cstr_impact_vel`
- **`envs/mdp/rewards.py`** — `joint_effort_l2`, `joint_torque_rate_l2`, `joint_torque_limit_margin_penalty`, `joints_action_acc_l2`

### Utilities

- **`src/mjlab/utils/torque_recorder.py`** — Record actuator torques during play for deployment analysis
- **`src/mjlab/scripts/plot_torques.py`** — Plot recorded torque data

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install this fork from source:**

```bash
git clone git@github.com:Leonassim/mjlab.git && cd mjlab
uv sync
```

**Install upstream from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train the **RHPS1** humanoid:

```bash
uv run train Mjlab-Velocity-Flat-RHPS1 --env.scene.num-envs 4096
```

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
