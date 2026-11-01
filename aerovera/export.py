"""Export one intermediate answer field of an AeroVera result file as a standard prediction file.

    python -m aerovera.export outputs/predictions/aerovera_qwen25_vl_7b.jsonl --field probing_response
    #   -> outputs/predictions/aerovera_qwen25_vl_7b.probing_response.jsonl

The exported file has the same meta line (plus its source) and the standard
fields, with ``raw_response`` replaced by the chosen field, so that
``evaluation.judge`` and ``evaluation.evaluate`` score it unchanged.
``positive_response`` is the w/o Negative Construction ablation (Tab. 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeromirage.inference.predictions import read_predictions

FIELDS = ("probing_response", "positive_response", "negative_response")
STANDARD = ("type", "question_type", "question_id", "video_id", "status", "error", "prompt", "elapsed", "run_timestamp")


def export(path: str | Path, field: str, output: str | Path | None = None) -> Path:
    path = Path(path)
    meta, records = read_predictions(path)
    target = Path(output) if output else path.with_name(f"{path.stem}.{field}.jsonl")
    lines = [json.dumps({**meta, "source": {"file": str(path), "field": field}}, ensure_ascii=False)] if meta else []
    for record in records.values():
        exported = {key: record[key] for key in STANDARD if key in record}
        exported["raw_response"] = record.get(field, "") if record.get("status") == "ok" else ""
        lines.append(json.dumps(exported, ensure_ascii=False))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export an AeroVera answer field as a prediction file.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--field", choices=FIELDS, required=True)
    parser.add_argument("--output", type=Path, help="default: <predictions>.<field>.jsonl")
    args = parser.parse_args(argv)
    print(export(args.predictions, args.field, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
