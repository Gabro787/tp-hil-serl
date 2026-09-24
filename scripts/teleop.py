"""Part 1.2: drive the robot yourself (nothing is recorded).

    python scripts/teleop.py

Press Space to take control at the start of each attempt, then end it with SUCCESS (Enter)
or FAILURE (Esc). Close the window or press
Ctrl+C in the terminal to quit.
"""

import argparse

from common import add_common_args, check_display, detect_device, load_reference, make_config, run_module


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    check_display()
    name = f"{args.group or 'anon'}_play"
    cfg = make_config(load_reference("env_config.json"),
                       {"mode": None, "device": detect_device()}, name)
    run_module("gym_manipulator", cfg)


if __name__ == "__main__":
    main()
