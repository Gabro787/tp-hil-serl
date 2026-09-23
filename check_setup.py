"""Check that this machine can run the HIL-SERL TP.

    python check_setup.py

Run it on every lab machine the week before the session (instructor), and first thing
in the session (students). It exits with a non-zero code if a blocking check fails.
"""

import json
import os
import platform
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from common import detect_device  # noqa: E402

EXPECTED_LEROBOT = "0.6.1"
REPO = Path(__file__).resolve().parent
results = []


def report(status, name, detail=""):
    results.append(status)
    mark = {"PASS": "\033[32mPASS\033[0m", "WARN": "\033[33mWARN\033[0m", "FAIL": "\033[31mFAIL\033[0m"}[status]
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))


def check(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        report("FAIL", name, f"{type(e).__name__}: {e}")


# 1. Python and packages ----------------------------------------------------------
report("PASS", "Python", sys.version.split()[0])


def c_lerobot():
    import lerobot
    v = getattr(lerobot, "__version__", "unknown")
    report("PASS" if v == EXPECTED_LEROBOT else "WARN", "LeRobot",
           v if v == EXPECTED_LEROBOT else f"{v} (TP tested with {EXPECTED_LEROBOT})")
    import lerobot.rl.actor, lerobot.rl.learner, lerobot.rl.gym_manipulator  # noqa: F401,E401
    report("PASS", "LeRobot RL modules", "actor, learner, gym_manipulator import fine")


def c_gymhil():
    import gym_hil  # noqa: F401
    report("PASS", "gym_hil", "imported")


check("LeRobot", c_lerobot)
check("gym_hil", c_gymhil)


# 2. Compute device --------------------------------------------------------------------
def c_gpu():
    import torch
    device = detect_device()
    if device == "cuda":
        report("PASS", "GPU", f"{torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    elif device == "mps":
        report("PASS", "GPU", f"Apple Silicon (MPS, torch {torch.__version__}). "
               "Training works but is not as fast as an NVIDIA GPU; torch.compile is disabled on this device.")
    else:
        report("WARN", "GPU", f"No CUDA or MPS device found, training will run on CPU (torch {torch.__version__}). "
               "This is much slower: expect Parts 3-5 to take longer than the suggested duration. "
               "If you expected a GPU here, check `nvidia-smi` and reinstall a CUDA build of PyTorch.")


check("GPU", c_gpu)

# 3. Display and session ----------------------------------------------------------
system = platform.system()
is_wsl = system == "Linux" and "microsoft" in platform.uname().release.lower()
if is_wsl:
    report("WARN", "Display", "Running under WSL: needs WSLg (Windows 11, on by default) or an X server "
           "(e.g. VcXsrv) on the Windows side for the simulator window to appear.")
elif system == "Linux":
    display = os.environ.get("DISPLAY")
    session = os.environ.get("XDG_SESSION_TYPE", "unknown")
    if not display:
        report("FAIL", "Display", "DISPLAY is not set. Run this from a terminal inside the desktop session "
               "(not over SSH, not on Colab/JupyterHub; use a remote desktop such as NoMachine/VNC/X2Go instead).")
    else:
        report("PASS", "Display", f"DISPLAY={display}")
    if session == "wayland":
        report("FAIL", "Session type", "Wayland: keyboard controls will not work. Log out and choose an X11 / Xorg session.")
    else:
        report("PASS", "Session type", session)
else:
    report("PASS", "Display", f"{system}: renders through the native windowing system, no DISPLAY needed.")
    if system == "Darwin":
        report("WARN", "Input capture", "macOS may ask you to grant your terminal app Accessibility / Input "
               "Monitoring permission the first time (System Settings > Privacy & Security) for the keyboard "
               "controls to be captured.")


# 4. Simulation rendering ------------------------------------------------------------
def c_render():
    import gymnasium
    import gym_hil  # noqa: F401
    env = gymnasium.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
    obs, _ = env.reset(seed=0)
    shapes = {k: tuple(v.shape) for k, v in obs["pixels"].items()}
    obs, rew, *_ = env.step(env.action_space.sample())
    env.close()
    report("PASS", "MuJoCo rendering", f"cameras {shapes}")


check("MuJoCo rendering", c_render)


# 5. Input devices -----------------------------------------------------------------
def c_keyboard():
    from pynput import keyboard  # noqa: F401
    report("PASS", "Keyboard listener", "pynput loaded")


def c_gamepad():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    if n == 0:
        report("WARN", "Gamepad", "none detected (fine if this group uses the keyboard)")
    else:
        names = [pygame.joystick.Joystick(i).get_name() for i in range(n)]
        report("PASS", "Gamepad", ", ".join(names))
    pygame.quit()


check("Keyboard listener", c_keyboard)
check("Gamepad", c_gamepad)


# 6. Files, cache, accounts, network --------------------------------------------
def c_configs():
    for name in ("env_config.json", "train_config.json"):
        json.loads((REPO / "configs" / name).read_text())
    report("PASS", "Config files", "configs/env_config.json, configs/train_config.json")


def c_cache():
    from huggingface_hub import snapshot_download
    missing = []
    for repo_id, repo_type in (("lilkm/pick_cube_franka_panda_30", "dataset"), ("lerobot/resnet10", "model")):
        try:
            snapshot_download(repo_id, repo_type=repo_type, local_files_only=True)
        except Exception:  # noqa: BLE001
            missing.append(repo_id)
    if missing:
        report("WARN", "Hub cache", f"not cached: {', '.join(missing)}. Run `python prefetch.py`.")
    else:
        report("PASS", "Hub cache", "reference demos and vision encoder are cached")


def c_wandb():
    netrc = Path.home() / ".netrc"
    ok = bool(os.environ.get("WANDB_API_KEY")) or (netrc.exists() and "api.wandb.ai" in netrc.read_text())
    if ok:
        report("PASS", "Weights & Biases", "logged in")
    else:
        report("WARN", "Weights & Biases", "not logged in. Run `wandb login` (or set WANDB_MODE=offline).")


def c_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = s.connect_ex(("127.0.0.1", 50051)) == 0
    if in_use:
        report("WARN", "gRPC port 50051", "already in use: a learner is probably still running. Stop it first.")
    else:
        report("PASS", "gRPC port 50051", "free")


check("Config files", c_configs)
check("Hub cache", c_cache)
check("Weights & Biases", c_wandb)
check("gRPC port", c_port)

# Summary -------------------------------------------------------------------------
n_fail, n_warn = results.count("FAIL"), results.count("WARN")
print()
if n_fail:
    print(f"{n_fail} blocking problem(s), {n_warn} warning(s). See README.md > Troubleshooting.")
    sys.exit(1)
print(f"Ready for the TP ({n_warn} warning(s)).")
