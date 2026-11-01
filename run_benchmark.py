#!/usr/bin/env python3
"""Run one AeroMirage model and append its raw answers to a prediction JSONL.

    python run_benchmark.py --config aeromirage/configs/qwen35_9b_vllm.yaml
    python run_benchmark.py --config ... runtime.limit=5 output.overwrite=true
    python run_benchmark.py --config ... --validate-only

Any config field can be overridden with key=value. Score the predictions
afterwards with ``python -m evaluation.evaluate``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aeromirage.config import load_config
from aeromirage.inference.runner import BenchmarkRunner, validate


def main(argv: list[str] | None = None) -> int:
    # Progress lines reach a redirected log as they are printed, not in 8 KB blocks.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run an AeroMirage model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="check the QA files and videos and show example prompts; no model is loaded",
    )
    parser.add_argument("overrides", nargs="*", help="config overrides such as runtime.limit=5")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    if args.validate_only:
        return validate(config)
    return BenchmarkRunner(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
