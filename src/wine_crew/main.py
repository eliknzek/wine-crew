"""Entry point: `python -m wine_crew.main` runs WineFlow end to end.

Exit code 0 when every gate passed, 1 when the run stopped (details in
artifacts/run_summary.json and logs/run_<timestamp>.log).
"""

import sys

from wine_crew.flow import run_flow


def main() -> int:
    return run_flow()


if __name__ == "__main__":
    sys.exit(main())
