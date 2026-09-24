"""Shared helpers for the TP scripts: paths, group name, config generation."""

from __future__ import annotations

import copy
import json
import os
import platform
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


def detect_device() -> str:
    """Best training/inference device on this machine: NVIDIA CUDA > Apple MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_compile_ok(device: str) -> bool:
    """torch.compile is only well-tested for this TP on CUDA; keep it off on mps/cpu."""
    return device == "cuda"


def control_changes(keyboard: bool) -> dict:
    if keyboard:
        return {"env.task": "PandaPickCubeKeyboard-v0", "env.processor.control_mode": "keyboard"}
    return {"env.task": "PandaPickCubeGamepad-v0", "env.processor.control_mode": "gamepad"}


def demos_repo(group: str) -> str:
    return f"tp/pick_cube_{group}"


def demos_root(group: str) -> Path:
    return DATA / f"pick_cube_{group}"


def train_changes(group: str, tag: str, keyboard: bool, experiment: str | None = None) -> dict:
    device = detect_device()
    changes = {
        **control_changes(keyboard),
        "wandb.enable": True,
        "wandb.project": WANDB_PROJECT,
        "dataset.repo_id": REFERENCE_DEMOS,
        "policy.actor_learner_config.policy_parameters_push_frequency": 4,
        "policy.device": device,
        "algorithm.use_torch_compile": torch_compile_ok(device),
        "resume": False,
        "output_dir": str(OUTPUTS / tag),
        "job_name": tag,
    }
    if device != "cuda":
        print(f"NOTE: no CUDA GPU detected, training on '{device}'. This will be much slower than "
              "the GPU lab machines: expect Parts 3-5 to need longer than the suggested duration to "
              "show the same learning curves. Record your device in answers.md so Part 6 class "
              "comparisons can account for it.")
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
def viewer_python() -> str:
    """Interpreter to use for anything that opens the MuJoCo viewer window.

    On macOS the interactive viewer (`launch_passive`) refuses to run under plain `python`
    and requires the `mjpython` launcher that ships with the mujoco package. Elsewhere the
    current interpreter is fine.
    """
    if platform.system() != "Darwin":
        return sys.executable
    mjpython = Path(sys.executable).with_name("mjpython")
    if mjpython.exists():
        return str(mjpython)
    print("WARNING: mjpython not found next to this interpreter. On macOS the simulator window "
          "needs it; if the run fails with 'requires that the Python script be run under mjpython', "
          "reinstall mujoco in this environment.")
    return sys.executable


def run_module(module: str, cfg_path: Path):
    cmd = [viewer_python(), "-m", module, "--config_path", str(cfg_path)]
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
    """The simulator needs a real screen. DISPLAY/Wayland only mean anything on Linux (X11);
    macOS and Windows render through their own native windowing and don't set DISPLAY at all."""
    system = platform.system()
    is_wsl = system == "Linux" and "microsoft" in platform.uname().release.lower()
    if is_wsl:
        # WSLg (Windows 11, on by default) or a separate X server handles the window; there is no
        # cheap way to check either is actually working, so just point at the requirement.
        print("NOTE: running inside WSL. This needs WSLg (Windows 11, on by default) or an X server "
              "(e.g. VcXsrv) running on the Windows side for the simulator window to appear.")
    elif system == "Linux":
        if not os.environ.get("DISPLAY"):
            sys.exit("No DISPLAY: run this from a terminal inside the desktop session of the lab machine "
                     "(not over SSH, not on Colab). Over SSH, use a remote-desktop session (NoMachine, VNC, "
                     "X2Go) instead.")
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            print("WARNING: Wayland session, the keyboard controls will not work. Log out and pick an X11 session.")
    # macOS and native Windows render through their own windowing system; nothing to check.
