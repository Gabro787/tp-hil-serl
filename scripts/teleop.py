"""Part 1.2: drive the robot yourself (nothing is recorded).

    python scripts/teleop.py              # gamepad
    python scripts/teleop.py --keyboard   # keyboard

End each attempt with SUCCESS (Y / Enter) or FAILURE (A / Esc). Close the window or press
Ctrl+C in the terminal to quit.
"""

import argparse

from common import add_common_args, check_display, control_changes, load_reference, make_config, run_module


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    check_display()
    name = f"{args.group or 'anon'}_play"
    cfg = make_config(load_reference("env_config.json"), {**control_changes(args.keyboard), "mode": None}, name)
    run_module("lerobot.rl.gym_manipulator", cfg)


if __name__ == "__main__":
    main()
