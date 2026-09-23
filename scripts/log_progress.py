"""Observer log for a training run (Parts 3-5). Start it at the same time as the actor.

    python scripts/log_progress.py --run noHIL
    python scripts/log_progress.py --run HIL
    python scripts/log_progress.py --run expB

Commands (type then Enter):
    f                    first success happened now
    s <n>                n successes among the last 10 episodes (do this every 5 minutes)
    i <sec> [reason]     one intervention of <sec> seconds, e.g.  i 4 moving away from cube
    n <text>             free note
    q                    quit

Everything is appended to runs/logs/<group>_<run>_observer.csv with the time since start.
plot_results.py reads these files.
"""

import argparse
import csv
import time
from datetime import datetime

from common import LOGS, REPO, add_common_args, require_group

FIELDS = ["wall_time", "minutes", "kind", "value", "note"]


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", required=True, help="noHIL, HIL or expX")
    args = p.parse_args()
    group = require_group(args)
    path = LOGS / f"{group}_{args.run}_observer.csv"
    new = not path.exists()

    t0 = time.time()
    n_int, t_int = 0, 0.0
    print(__doc__.split("Commands")[1].split("Everything")[0])
    print(f"Logging to {path.relative_to(REPO)}. Clock started at {datetime.now():%H:%M:%S}.\n")

    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"wall_time": datetime.now().isoformat(timespec="seconds"), "minutes": 0.0,
                    "kind": "start", "value": "", "note": ""})
        fh.flush()
        while True:
            try:
                line = input(f"[{(time.time() - t0) / 60:5.1f} min] > ").strip()
            except (EOFError, KeyboardInterrupt):
                line = "q"
            if not line:
                continue
            cmd, _, rest = line.partition(" ")
            row = {"wall_time": datetime.now().isoformat(timespec="seconds"),
                   "minutes": round((time.time() - t0) / 60, 2), "value": "", "note": ""}
            try:
                if cmd == "q":
                    row["kind"] = "stop"
                    w.writerow(row)
                    break
                elif cmd == "f":
                    row["kind"] = "first_success"
                elif cmd == "s":
                    n = int(rest)
                    assert 0 <= n <= 10
                    row.update(kind="successes_last10", value=n)
                elif cmd == "i":
                    sec, _, reason = rest.partition(" ")
                    row.update(kind="intervention", value=float(sec), note=reason)
                    n_int += 1
                    t_int += float(sec)
                elif cmd == "n":
                    row.update(kind="note", note=rest)
                else:
                    print("  unknown command (f, s <n>, i <sec> [reason], n <text>, q)")
                    continue
            except (ValueError, AssertionError):
                print("  could not read that, try again (e.g.  s 7   or   i 3 hovering above cube)")
                continue
            w.writerow(row)
            fh.flush()
            if cmd == "i":
                print(f"  interventions so far: {n_int}, total {t_int:.0f} s")

    print(f"\nSaved {path.relative_to(REPO)}  ({n_int} interventions, {t_int:.0f} s of human effort this session)")


if __name__ == "__main__":
    main()
