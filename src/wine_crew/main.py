"""Entry point: `python -m wine_crew.main` runs WineFlow end to end.

Exit code 0 when every gate passed (artifacts/ updated), 1 when the run stopped
(artifacts/ unchanged; details in logs/run_<ts>.log and logs/run_<ts>_summary.json).

Demonstrating a controlled failure, without touching artifacts/:
    python -m wine_crew.main --raw /tmp/missing.csv        # preflight fails at once, no LLM
    python -m wine_crew.main --demo-fail contract          # Crew 1 runs, gate 1 fails
    python -m wine_crew.main --demo-fail features          # both crews run, gate 2 fails
    python -m wine_crew.main --demo-fail model             # both crews run, gate 3 fails
"""

from __future__ import annotations

import argparse
import sys

from wine_crew.config import load_config
from wine_crew.flow import DEMO_FAILURES, run_flow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m wine_crew.main", description="Run WineFlow.")
    parser.add_argument("--demo-fail", choices=DEMO_FAILURES, default=None,
                        help="break one staged file so this gate fails (for demos); artifacts/ is not touched")
    parser.add_argument("--raw", default=None,
                        help="raw CSV to use instead of paths.raw_csv in config.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    if args.raw:
        config["paths"] = {**config["paths"], "raw_csv": args.raw}
    return run_flow(config=config, demo_fail=args.demo_fail)


if __name__ == "__main__":
    sys.exit(main())
