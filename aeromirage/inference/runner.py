"""Inference loop: answer every pending question and append it to the prediction file."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from aeromirage.benchmark.loader import load_samples, qa_sha256
from aeromirage.benchmark.parser import verified_options
from aeromirage.benchmark.prompts import (
    SYSTEM_PROMPT,
    VERIFICATION_STATEMENT,
    build_task_prompt,
    ordinary_options,
    verification_prompt,
)
from aeromirage.config import BenchmarkConfig, DatasetConfig
from aeromirage.inference.predictions import PredictionStore
from aeromirage.models import create_model_adapter
from aeromirage.models.base import BaseModelAdapter
from aeromirage.types import BenchmarkSample


def prompt_sha256(dataset: DatasetConfig) -> str:
    """Hash of the rendered prompts of every question (all tasks), not of source code."""
    digest = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8"))
    every_task = DatasetConfig(root=dataset.root, video_dir=dataset.video_dir, task="all")
    for sample in load_samples(every_task, check_videos=False):
        digest.update(b"\0" + build_task_prompt(sample).encode("utf-8"))
    return digest.hexdigest()


def fingerprint(config: BenchmarkConfig, adapter: BaseModelAdapter) -> dict[str, str]:
    """Everything that changes a model's answers; resuming requires an exact match."""
    result = {
        "model_family": config.model.family,
        "model_path": config.model.path,
        "model_options": json.dumps(config.model.options, sort_keys=True),
        "video": json.dumps(adapter.video, sort_keys=True),
        "generation_limits": json.dumps(config.generation.limits, sort_keys=True),
        "video_dir": config.dataset.video_dir,
        "qa_sha256": qa_sha256(config.dataset.root),
        "prompt_sha256": prompt_sha256(config.dataset),
    }
    # Absent for the whole-question protocol, so files written before the switch still resume.
    if config.generation.multiple_choice == "per_option":
        statement = hashlib.sha256(VERIFICATION_STATEMENT.encode("utf-8")).hexdigest()
        result["multiple_choice"] = f"per_option {statement}"
    return result


def validate(config: BenchmarkConfig) -> int:
    """Check the QA files and videos, and show one rendered prompt per task."""
    samples = load_samples(
        config.dataset, limit=config.runtime.limit, limit_per_task=config.runtime.limit_per_task
    )
    print(f"{len(samples)} questions on {len({s.video_id for s in samples})} videos; all videos found")
    for task, count in Counter(s.question_type for s in samples).items():
        print(f"  {task}: {count}")
    shown: set[str] = set()
    for sample in samples:
        if sample.question_type not in shown:
            shown.add(sample.question_type)
            if sample.question_type == "multiple_choice" and config.generation.multiple_choice == "per_option":
                label, text = ordinary_options(sample)[0]
                prompt = f"(one per ordinary option; option {label})\n{verification_prompt(sample, text)}"
            else:
                prompt = build_task_prompt(sample)
            print(f"\n----- {sample.question_type} {sample.question_id} -----\n{prompt}")
    return 0


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        config.validate()
        self.config = config

    def run(self) -> int:
        config = self.config
        samples = load_samples(
            config.dataset, limit=config.runtime.limit, limit_per_task=config.runtime.limit_per_task
        )
        # Questions on the same video run back to back, so each video is decoded once.
        samples.sort(key=lambda sample: sample.video_id)
        adapter = create_model_adapter(config.model, config.video)
        store = PredictionStore(
            config.output.predictions,
            {
                "dataset_root": str(config.dataset.root),
                "task": config.dataset.task,
                "fingerprint": fingerprint(config, adapter),
            },
            resume=config.output.resume,
            overwrite=config.output.overwrite,
        )
        pending = store.pending(samples, retry_errors=config.output.retry_errors)
        print(
            f"Predictions: {config.output.predictions}\n"
            f"Questions: {len(samples)}; done: {len(samples) - len(pending)}; pending: {len(pending)}"
        )
        if not pending:
            return 0

        adapter.set_seed(config.runtime.seed)
        adapter.load()
        try:
            for index, sample in enumerate(pending, 1):
                if not self._answer(adapter, store, sample, f"[{index}/{len(pending)}]"):
                    return 1
        except KeyboardInterrupt:
            print("Interrupted; finished answers are already saved.", file=sys.stderr)
            return 130
        finally:
            adapter.close()
        print(f"Done. Predictions saved to {config.output.predictions}")
        return 0

    def _answer(self, adapter: BaseModelAdapter, store: PredictionStore, sample: BenchmarkSample, tag: str) -> bool:
        """Answer one question; False means the run must stop."""
        started = time.perf_counter()
        extra, prompt = None, None
        try:
            if sample.question_type == "multiple_choice" and self.config.generation.multiple_choice == "per_option":
                # Each ordinary option as a yes/no question; the video is decoded once.
                responses, prompts = {}, []
                for label, text in ordinary_options(sample):
                    responses[label] = adapter.generate(
                        sample.video_path,
                        SYSTEM_PROMPT,
                        verification_prompt(sample, text),
                        self.config.generation.limit_for("binary_verification"),
                    )
                    prompts.append(adapter.last_prompt)
                raw = ",".join(verified_options(responses))
                extra, prompt = {"option_responses": responses}, "\n\n".join(prompts)
            else:
                raw = adapter.generate(
                    sample.video_path,
                    SYSTEM_PROMPT,
                    build_task_prompt(sample),
                    self.config.generation.limit_for(sample.question_type),
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            store.append(sample, error=error, prompt=adapter.last_prompt, elapsed=time.perf_counter() - started)
            print(f"{tag} {sample.question_type}/{sample.question_id} ERROR: {error}", file=sys.stderr)
            # A device-side assert leaves the CUDA context unusable; nothing
            # after it can succeed in this process.
            if "device-side assert" in error.lower():
                return False
            adapter.clear_cache()
            return not self.config.runtime.fail_fast
        elapsed = time.perf_counter() - started
        store.append(sample, raw_response=raw, prompt=prompt or adapter.last_prompt, elapsed=elapsed, extra=extra)
        preview = raw.rsplit("</think>", 1)[-1].strip().replace("\n", " ")[:60]
        print(f"{tag} {sample.question_type}/{sample.question_id} {elapsed:.1f}s -> {preview}")
        return True
