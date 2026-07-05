from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N evaluation cases.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="Path to evaluation CSV. Defaults to records/evaluation_set.csv.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path for per-case outputs. Defaults to records/results.csv.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path for aggregate metrics. Defaults to records/summary.csv.",
    )
    args = parser.parse_args()
    main(
        limit=args.limit,
        evaluation_set_path=args.eval_set,
        results_path=args.results,
        summary_path=args.summary,
    )
