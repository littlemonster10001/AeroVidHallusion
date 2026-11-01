#!/usr/bin/env python3
"""Run AeroVera on AeroMirage and append one result line per question.

    python run_aerovera.py --config aerovera/configs/qwen25_vl_7b.yaml
    python run_aerovera.py --config ... runtime.question_ids=[0002q000000] output.predictions=...

Any config field can be overridden with key=value. Score the final answers
with ``python -m evaluation.evaluate``; export an intermediate answer
(probing_response, positive_response, ...) for scoring with
``python -m aerovera.export``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The Probing attribution and the 48-image prefill run close to a 48 GB card's
# limit; growable segments keep fragmentation from turning that into OOM.
# Set before anything initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from aerovera.config import load_config
from aerovera.runner import AeroVeraRunner


def main(argv: list[str] | None = None) -> int:
    # Progress lines reach a redirected log as they are printed, not in 8 KB blocks.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run AeroVera.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="config overrides such as runtime.limit=5")
    args = parser.parse_args(argv)
    return AeroVeraRunner(load_config(args.config, args.overrides)).run()


if __name__ == "__main__":
    raise SystemExit(main())
