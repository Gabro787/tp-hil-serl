# TP: Interactive Robot Learning with HIL-SERL

A 4-hour lab on **human-in-the-loop reinforcement learning**. Students train a simulated Franka Panda arm to pick up a cube with SAC, first on its own, then while they coach it by taking over with a gamepad or keyboard, and measure how much their interventions speed up learning.

Built on [LeRobot](https://github.com/huggingface/lerobot)'s HIL-SERL implementation and the [`gym_hil`](https://github.com/huggingface/gym-hil) MuJoCo environment. No real robot needed.

**Students: the lab handout with all the exercises is [`TP.md`](TP.md). Write your report in [`answers.md`](answers.md).**

| Part | Duration | Content | Script |
| --- | --- | --- | --- |
| Setup | 15 min | Check the machine, controls, W&B | `check_setup.py` |
| 1 | 30 min | Explore the environment, teleoperate | `inspect_env.py`, `teleop.py` |
| 2 | 20 min | Record 10 demonstrations | `record.py` |
| 3 | 35 min | RL **without** interventions (baseline) | `train.py --run noHIL` |
| 4 | 40 min | HIL-SERL **with** interventions | `train.py --run HIL`, `plot_results.py` |
| 5 | 45 min | One controlled experiment per group | `train.py --run exp --exp X` |
| 6 | 30 min | Class comparison and report | `plot_results.py --class` |

## Requirements

Everything runs **on the machine the student is sitting at**: the simulator opens a 3D window and reads the local keyboard or gamepad.

- Linux with a desktop session using **X11** (not Wayland: keyboard input is ignored under Wayland)
- An **NVIDIA GPU** with a recent driver (`nvidia-smi` works)
- [Miniconda](https://docs.anaconda.com/miniconda/) and git
- A gamepad per group (recommended; Logitech F310 or Xbox-type) or the keyboard
- A free [Weights & Biases](https://wandb.ai) account per student (recommended; the TP also works offline with the observer logs)

It does **not** work on Google Colab, a remote JupyterHub or over SSH: there is no screen for the simulator and your keyboard is not the server's. A remote desktop (NoMachine, VNC, X2Go) into a GPU machine does work.

## Install (once per machine)

```bash
git clone https://github.com/<you>/tp-hil-serl.git
cd tp-hil-serl
bash install.sh          # conda env "tp-hil" with LeRobot 0.6.1 + HIL-SERL extras
conda activate tp-hil
python prefetch.py       # downloads the 30 reference demos + vision encoder
python check_setup.py    # every line should be PASS
```

## Quick start

In a terminal **opened inside the desktop session**, from the repository root:

```bash
conda activate tp-hil
export TP_GROUP=group07             # your group name, in every terminal
python scripts/inspect_env.py       # Part 1.1
python scripts/teleop.py            # Part 1.2 (add --keyboard if no gamepad)
python scripts/record.py            # Part 2
python scripts/train.py --run noHIL # Part 3: prints the learner/actor commands to run
```

Then follow [`TP.md`](TP.md). Everything you produce goes to `runs/` (ignored by git).

## Repository layout

```
tp-hil-serl/
├── TP.md                  lab handout: background, instructions, exercises
├── answers.md             report template to fill in
├── configs/               reference gym_hil configs from LeRobot (never modified)
├── scripts/
│   ├── inspect_env.py     Part 1.1: spaces, camera views, state ranges
│   ├── teleop.py          Part 1.2: drive the robot
│   ├── record.py          Part 2: record demonstrations + summary
│   ├── train.py           Parts 3-5: write a run config, print the learner/actor commands
│   ├── log_progress.py    observer log (first success, successes/10, interventions)
│   ├── plot_results.py    Parts 4-6: learning curves, group and class comparison
│   └── common.py          shared paths and config helpers
├── install.sh             creates the conda env
├── prefetch.py            caches the dataset and encoder from the Hugging Face Hub
├── check_setup.py         machine check: GPU, display, rendering, controller, cache, W&B
└── instructor/            preparation checklist, pitfalls, grading, answer key
```

## Controls

| Action | Gamepad | Keyboard |
| --- | --- | --- |
| Move in x–y plane | Left stick | Arrow keys |
| Move up / down (z) | Right stick, vertical | Right Shift / Left Shift |
| Close gripper | LT | Left Ctrl |
| Open gripper | RT | Right Ctrl |
| **Take over from the policy** | **Hold RB** | **Space** (toggle) |
| End episode: success | Y / Triangle | Enter |
| End episode: failure | A / Cross | Esc |
| Re-record episode | X / Square | — |

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `GLFWError: X11: The DISPLAY environment variable is missing`, then `FatalError: an OpenGL platform library has not been loaded` | Running without a screen (SSH, Colab, JupyterHub) | Run the scripts from a terminal inside the desktop session |
| `CUDA available: False`, torch version ends in `+cpu` | CPU-only PyTorch, or no GPU/driver | Check `nvidia-smi`; reinstall a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) |
| Keys do nothing | Wayland session | Log out, pick "Ubuntu on Xorg" (or your distro's X11 session) |
| Gamepad buttons do the wrong thing | Unknown controller model | Add a mapping to gym-hil's `controller_config.json` ([instructions](https://github.com/huggingface/gym-hil#controller-configuration)) |
| Actor cannot connect / "address already in use" | A previous learner still runs on port 50051 | Ctrl+C in its terminal, or `pkill -f lerobot.rl` |
| Learner takes minutes to start | Torch compilation on first run | Wait, or set `"use_torch_compile": false` in `configs/train_config.json` |
| W&B plots empty | Metric names differ in your LeRobot version | `python scripts/plot_results.py --list-metrics`, then `--reward-key` / `--intervention-key` |
| `Set your group name first` | `TP_GROUP` not set in this terminal | `export TP_GROUP=group07` |

## References

- Luo, Xu, Wu & Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*. [arXiv:2410.21845](https://arxiv.org/abs/2410.21845), 2024.
- Kelly et al. *HG-DAgger: Interactive Imitation Learning with Human Experts*. [arXiv:1810.02890](https://arxiv.org/abs/1810.02890), 2019.
- LeRobot docs: [Train RL in Simulation](https://huggingface.co/docs/lerobot/en/hilserl_sim) · [HIL-SERL guide](https://huggingface.co/docs/lerobot/en/hilserl)

The files in `configs/` are copied from LeRobot's [`config_examples`](https://huggingface.co/datasets/lerobot/config_examples) dataset. LeRobot and gym-hil are Apache-2.0 licensed.
