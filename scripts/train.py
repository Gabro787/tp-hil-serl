"""Parts 3-5: prepare a HIL-SERL training run and print the commands to launch it.

    python scripts/train.py --run noHIL          # Part 3: RL without interventions
    python scripts/train.py --run HIL            # Part 4: RL with your interventions
    python scripts/train.py --run exp --exp B    # Part 5: your assigned experiment (A-F)
    python scripts/train.py --list-experiments

The script writes the run's config to runs/configs/<group>_<run>.json, then prints the two
commands to paste into two terminals: the LEARNER first, then the ACTOR. Stop both with
Ctrl+C (actor first) when the time is up.

Add --launch to start the learner in the background and the actor in this terminal instead.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import (EXPERIMENTS, LOGS, REPO, add_common_args, check_display, load_reference, make_config,
                    port_in_use, require_group, train_changes, viewer_python)

DURATIONS = {"noHIL": 25, "HIL": 25, "exp": 20}


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", choices=["noHIL", "HIL", "exp"])
    p.add_argument("--exp", choices=sorted(EXPERIMENTS), help="Experiment letter for --run exp")
    p.add_argument("--launch", action="store_true", help="Start learner (background) and actor (foreground) here")
    p.add_argument("--list-experiments", action="store_true")
    args = p.parse_args()

    if args.list_experiments:
        for k, (desc, _) in EXPERIMENTS.items():
            print(f"  {k}: {desc}")
        return
    if not args.run:
        p.error("--run is required (noHIL, HIL or exp)")
    if args.run == "exp" and not args.exp:
        p.error("--run exp needs --exp A..F (see --list-experiments)")

    group = require_group(args)
    tag = f"{group}_{args.run}" if args.run != "exp" else f"{group}_exp{args.exp}"
    changes = train_changes(group, tag, args.keyboard, args.exp if args.run == "exp" else None)
    cfg = make_config(load_reference("train_config.json"), changes, tag)
    rel = cfg.relative_to(REPO)

    if port_in_use():
        print("\nWARNING: port 50051 is busy, a learner is probably still running. Stop it first "
              "(Ctrl+C in its terminal, or: pkill -f lerobot_rl).")

    # macOS opens the MuJoCo viewer only under mjpython; the learner has no window, so plain python.
    actor_py = "mjpython" if Path(viewer_python()).name == "mjpython" else "python"
    minutes = DURATIONS[args.run]
    hands = ("DO NOT intervene (except to end an episode where the robot is clearly stuck)."
             if args.run == "noHIL" else
             "Intervene following the protocol in TP.md (hold RB / toggle Space).")
    if args.run == "exp" and args.exp == "A":
        hands = "Experiment A: take over for LONG periods, driving the robot to success each time it hesitates."

    print(f"""
Run '{tag}' is ready. Let it run for {minutes} minutes.
{hands}

  Terminal 1 (learner, start it first and give it ~20 s before starting the actor):
    python scripts/lerobot_rl.py learner --config_path {rel}

  Terminal 2 (actor, opens the simulator window):
    {actor_py} scripts/lerobot_rl.py actor --config_path {rel}

  Terminal 3 (observer):
    python scripts/log_progress.py --run {args.run if args.run != 'exp' else 'exp' + args.exp}

Stop: Ctrl+C in the actor terminal, then in the learner terminal.
""")

    if args.launch:
        check_display()
        log = open(LOGS / f"{tag}_learner.log", "w")
        learner = subprocess.Popen([sys.executable, str(REPO / "scripts" / "lerobot_rl.py"), "learner", "--config_path", str(cfg)],
                                   stdout=log, stderr=subprocess.STDOUT)
        print(f"Learner started in the background (log: {log.name}). Waiting 20 s before starting the actor...")
        time.sleep(20)
        if learner.poll() is not None:
            sys.exit(f"The learner stopped, see {log.name}")
        try:
            subprocess.run([viewer_python(), str(REPO / "scripts" / "lerobot_rl.py"), "actor", "--config_path", str(cfg)], check=False)
        except KeyboardInterrupt:
            pass
        finally:
            learner.terminate()
            learner.wait(timeout=30)
            print("Learner stopped.")


if __name__ == "__main__":
    main()
