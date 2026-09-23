"""Shared helpers for the TP scripts: paths, group name, config generation."""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
RUNS = REPO / "runs"                 # everything students produce (ignored by git)
RUN_CONFIGS = RUNS / "configs"
LOGS = RUNS / "logs"
DATA = RUNS / "data"
OUTPUTS = RUNS / "outputs"
PLOTS = RUNS / "plots"
for _d in (RUN_CONFIGS, LOGS, DATA, OUTPUTS, PLOTS):
    _d.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT = os.environ.get("TP_WANDB_PROJECT", "tp_hilserl")
REFERENCE_DEMOS = "lilkm/pick_cube_franka_panda_30"
LEARNER_PORT = 50051

# Part 5 experiments: one factor changed with respect to the HIL run of Part 4.
EXPERIMENTS = {
    "A": ("Intervention style: long takeovers (no config change)", {}),
    "B": ("Exploration: temperature_init 0.01 -> 0.1", {"algorithm.temperature_init": 0.1}),
    "C": ("Demo source: your own 10 demos from Part 2", None),  # filled in per group, see train_changes()
    "D": ("Offline data: online_ratio 0.5 -> 1.0 (no demos in batches)", {"online_ratio": 1.0}),
    "E": ("Interface: the other controller (keyboard <-> gamepad)", None),
    "F": ("Weight sync: push weights every 50 s instead of 4 s",
          {"policy.actor_learner_config.policy_parameters_push_frequency": 50}),
}


# --------------------------------------------------------------------------- group
def add_common_args(parser):
    parser.add_argument("--group", default=os.environ.get("TP_GROUP"),
                        help="Group name, e.g. group07 (default: $TP_GROUP)")
    parser.add_argument("--keyboard", action="store_true",
                        help="Use the keyboard instead of the gamepad")
    return parser


def require_group(args):
    if not args.group:
        sys.exit("Set your group name first:  export TP_GROUP=group07   (or pass --group group07)")
    return args.group


# --------------------------------------------------------------------------- configs
def load_reference(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text())


def set_path(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_path(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[k]
    return d


def make_config(base: dict, changes: dict, name: str) -> Path:
    """Copy a reference config, apply {'dotted.path': value} changes, save to runs/configs/<name>.json."""
    cfg = copy.deepcopy(base)
    for k, v in changes.items():
        set_path(cfg, k, v)
    path = RUN_CONFIGS / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2))
    print(f"Config written: {path.relative_to(REPO)}")
    for k, v in changes.items():
        print(f"  {k} = {v!r}")
    return path


def control_changes(keyboard: bool) -> dict:
    if keyboard:
        return {"env.task": "PandaPickCubeKeyboard-v0", "env.processor.control_mode": "keyboard"}
    return {"env.task": "PandaPickCubeGamepad-v0", "env.processor.control_mode": "gamepad"}


def demos_repo(group: str) -> str:
    return f"tp/pick_cube_{group}"


def demos_root(group: str) -> Path:
    return DATA / f"pick_cube_{group}"


def train_changes(group: str, tag: str, keyboard: bool, experiment: str | None = None) -> dict:
    changes = {
        **control_changes(keyboard),
        "wandb.enable": True,
        "wandb.project": WANDB_PROJECT,
        "dataset.repo_id": REFERENCE_DEMOS,
        "policy.actor_learner_config.policy_parameters_push_frequency": 4,
        "resume": False,
        "output_dir": str(OUTPUTS / tag),
        "job_name": tag,
    }
    if experiment == "C":
        if not demos_root(group).exists():
            sys.exit(f"Experiment C needs your Part 2 demos in {demos_root(group)}: run scripts/record.py first.")
        changes.update({"dataset.repo_id": demos_repo(group), "dataset.root": str(demos_root(group))})
    elif experiment == "E":
        changes.update(control_changes(not keyboard))
    elif experiment:
        changes.update(EXPERIMENTS[experiment][1])
    return changes


# --------------------------------------------------------------------------- processes
def run_module(module: str, cfg_path: Path):
    cmd = [sys.executable, "-m", module, "--config_path", str(cfg_path)]
    print("\n$ " + " ".join(cmd) + "\n")
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


def port_in_use(port: int = LEARNER_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_display():
    if not os.environ.get("DISPLAY"):
        sys.exit("No DISPLAY: run this from a terminal inside the desktop session of the lab machine "
                 "(not over SSH, not on Colab).")
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        print("WARNING: Wayland session, the keyboard controls will not work. Log out and pick an X11 session.")
