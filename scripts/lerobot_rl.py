"""Start a LeRobot RL process with a one-line compatibility fix.

    python scripts/lerobot_rl.py learner --config_path runs/configs/<run>.json
    python scripts/lerobot_rl.py actor   --config_path runs/configs/<run>.json

LeRobot 0.6.1 types `ResetConfig.fixed_reset_joint_positions` as `Any | None`, and the
config serialiser (draccus) cannot encode that: the learner crashes at start-up with
"typing.Any cannot be used with isinstance()". We give the field its real type, then run
the normal LeRobot module unchanged. `train.py` prints the right command for you.

LeRobot also refuses to start a process whose output directory already exists, and the learner
creates it before the actor starts, so the actor gets its own sibling directory (`<dir>_actor`).

Two more fixes for teleoperation: LeRobot's INFO messages ("Episode ended after ... steps") are
switched back on (an import silently sets logging to WARNING first), and a SUCCESS/FAILURE key
pressed during the reset pause no longer ends the next episode on its first step.
"""

import dataclasses
import json
import logging
import runpy
import sys

MODULES = {"learner": "lerobot.rl.learner", "actor": "lerobot.rl.actor",
           "gym_manipulator": "lerobot.rl.gym_manipulator"}


def patch_reset_config():
    from lerobot.envs.configs import ResetConfig

    name = "fixed_reset_joint_positions"
    ResetConfig.__annotations__[name] = list[float] | None
    for f in dataclasses.fields(ResetConfig):
        if f.name == name:
            f.type = list[float] | None


def patch_input_controllers():
    """Forget an episode-end key (Enter/Esc/Y/A/X) on reset instead of applying it to the next episode."""
    from gym_hil.wrappers import intervention_utils as iu

    for cls in (iu.KeyboardController, iu.GamepadController, iu.GamepadControllerHID):
        def reset(self, _orig=cls.reset):
            _orig(self)
            self.episode_end_status = None
        cls.reset = reset


def actor_output_args(argv):
    """`--output_dir=<run dir>_actor` for the actor, unless the caller already set one."""
    if any(a.startswith("--output_dir") for a in argv):
        return []
    for i, a in enumerate(argv):
        path = argv[i + 1] if a == "--config_path" and i + 1 < len(argv) else (
            a.split("=", 1)[1] if a.startswith("--config_path=") else None)
        if path:
            out = json.load(open(path)).get("output_dir")
            return [f"--output_dir={out}_actor"] if out else []
    return []


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        sys.exit(f"usage: python scripts/lerobot_rl.py {{{'|'.join(MODULES)}}} --config_path <file>")
    module = MODULES[sys.argv[1]]
    patch_reset_config()
    patch_input_controllers()
    logging.getLogger().setLevel(logging.INFO)
    extra = actor_output_args(sys.argv[2:]) if sys.argv[1] == "actor" else []
    sys.argv = [module] + sys.argv[2:] + extra
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
