"""Batch inference: read questions, group by video, resume, and write one prediction line per question."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from aeromirage.benchmark import SYSTEM_PROMPT
from aeromirage.benchmark.loader import load_samples, qa_sha256
from aeromirage.inference.predictions import PredictionStore
from aeromirage.inference.runner import prompt_sha256
from aeromirage.types import BenchmarkSample
from aerovera.config import AeroVeraConfig
from aerovera.pipeline import Models, VideoContext, answer, prepare_video
from aerovera.prompts import TARGET_SYSTEM_PROMPT, VERIFICATION_STATEMENT

# Config sections that change the answers; devices do not.
ANSWER_SECTIONS = ("model", "probing", "targets", "dino", "motion", "selection", "resolution", "negative",
                   "contrastive", "generation")
# Fields added after the first result files. At these values (the behaviour
# before them) they are left out, so that those files still resume.
NEUTRAL = {"model": {"family": "qwen25_vl"}, "resolution": {"hr_scale": 0.0},
           "contrastive": {"option_rule": "sign", "min_confidence": 0.0}, "generation": {"multiple_choice": "whole"}}


def fingerprint(config: AeroVeraConfig) -> dict[str, str]:
    """Everything that changes AeroVera's answers; resuming requires an exact match."""
    sections = {}
    for name in ANSWER_SECTIONS:
        value = getattr(config, name)
        values = dict(value) if isinstance(value, dict) else asdict(value)
        neutral = NEUTRAL.get(name, {})
        sections[name] = json.dumps(
            {k: v for k, v in values.items() if k != "device" and not (k in neutral and neutral[k] == v)},
            sort_keys=True, default=str)
    own_prompts = "\0".join((SYSTEM_PROMPT, TARGET_SYSTEM_PROMPT, VERIFICATION_STATEMENT))
    return {
        "method": "aerovera",
        **sections,
        "video_dir": config.dataset.video_dir,
        "qa_sha256": qa_sha256(config.dataset.root),
        "prompt_sha256": prompt_sha256(config.dataset),
        "aerovera_prompt_sha256": hashlib.sha256(own_prompts.encode("utf-8")).hexdigest(),
    }


def select_samples(config: AeroVeraConfig) -> list[BenchmarkSample]:
    """Questions to answer, grouped by video (file order within a video).

    ``runtime.question_ids`` entries are ``question_id`` (every type with that
    id) or ``question_type/question_id``.
    """
    runtime = config.runtime
    samples = load_samples(config.dataset, limit=runtime.limit, limit_per_task=runtime.limit_per_task)
    if runtime.question_ids:
        wanted = set(runtime.question_ids)
        samples = [s for s in samples if s.question_id in wanted or f"{s.question_type}/{s.question_id}" in wanted]
        found = {s.question_id for s in samples} | {f"{s.question_type}/{s.question_id}" for s in samples}
        if wanted - found:
            raise ValueError(f"Questions not found: {sorted(wanted - found)}")
    samples.sort(key=lambda sample: sample.video_id)
    return samples


class AeroVeraRunner:
    def __init__(self, config: AeroVeraConfig):
        self.config = config
        self._stream = None  # the background thread's CUDA stream on the evidence GPU

    def _prepare(self, models: Models, path) -> VideoContext:
        """``prepare_video`` on a CUDA stream of its own on the evidence GPU.

        On the default stream, the main thread's evidence work for the current
        question (DINO scoring, R_t) would queue behind the next video's DINO
        encoding there (measured: 0.5 s became 34 s). Each preparation stage
        ends with a device-wide synchronize, so the context is complete when
        it is handed over; the main thread copies its results to the CPU, so
        the context's memory is idle by the time it is freed.
        """
        import torch

        device = torch.device(self.config.dino.device)
        if device.type != "cuda":
            return prepare_video(models, self.config, path)
        if self._stream is None:
            self._stream = torch.cuda.Stream(device)
        with torch.cuda.stream(self._stream):
            return prepare_video(models, self.config, path)

    def run(self) -> int:
        config = self.config
        samples = select_samples(config)
        store = PredictionStore(
            config.output.predictions,
            {"dataset_root": str(config.dataset.root), "task": config.dataset.task, "fingerprint": fingerprint(config)},
            resume=config.output.resume,
            overwrite=config.output.overwrite,
        )
        pending = store.pending(samples, retry_errors=config.output.retry_errors)
        print(f"Predictions: {config.output.predictions}\n"
              f"Questions: {len(samples)}; done: {len(samples) - len(pending)}; pending: {len(pending)}")
        if not pending:
            return 0

        models = Models.load(config)
        groups = [list(group) for _, group in itertools.groupby(pending, key=lambda sample: sample.video_id)]
        started, count = time.perf_counter(), 0
        # One background thread prepares the next video (CPU and the evidence
        # GPU) while the backbone answers the questions of the current one.
        # Nothing may call torch.cuda.empty_cache() meanwhile: it releases
        # memory on every GPU, and with expandable segments that crashes the
        # other thread's running kernels (illegal memory access, reproduced).
        # The allocator still frees its own cache, one GPU at a time, when an
        # allocation would otherwise fail.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prepare_video")
        upcoming = pool.submit(self._prepare, models, groups[0][0].video_path)
        try:
            for index, group in enumerate(groups):
                current = upcoming
                if index + 1 < len(groups):
                    upcoming = pool.submit(self._prepare, models, groups[index + 1][0].video_path)
                video_id, waiting = group[0].video_id, time.perf_counter()
                try:
                    context = current.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"video {video_id} ERROR: {error}", file=sys.stderr)
                    for sample in group:
                        store.append(sample, error=error)
                    count += len(group)
                    if config.runtime.fail_fast or "device-side assert" in error.lower():
                        return 1
                    continue
                stages = ", ".join(f"{name} {seconds:.1f}" for name, seconds in context.timings.items())
                print(f"video {video_id}: prepared in {context.seconds:.1f}s ({stages}); "
                      f"waited {time.perf_counter() - waiting:.1f}s; {len(group)} questions")
                for sample in group:
                    count += 1
                    if not self._answer(models, context, store, sample, f"[{count}/{len(pending)}]"):
                        return 1
                del context
        except KeyboardInterrupt:
            print("Interrupted; finished answers are already saved.", file=sys.stderr)
            return 130
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        print(f"Done in {(time.perf_counter() - started) / 60:.1f} min. Predictions saved to {config.output.predictions}")
        return 0

    def _answer(self, models: Models, context: VideoContext, store: PredictionStore,
                sample: BenchmarkSample, tag: str) -> bool:
        """Answer one question; False means the run must stop."""
        started = time.perf_counter()
        try:
            result = answer(models, self.config, context, sample)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            store.append(sample, error=error, elapsed=time.perf_counter() - started)
            print(f"{tag} {sample.question_type}/{sample.question_id} ERROR: {error}", file=sys.stderr)
            # A device-side assert leaves the CUDA context unusable.
            if "device-side assert" in error.lower():
                return False
            return not self.config.runtime.fail_fast
        elapsed = time.perf_counter() - started
        raw = result.pop("raw_response")
        store.append(sample, raw_response=raw, prompt=result.pop("prompt"), elapsed=elapsed, extra=result)
        print(f"{tag} {sample.question_type}/{sample.question_id} {elapsed:.1f}s -> {raw[:40]!r} "
              f"(probing {result['probing_response'][:20]!r}, positive {result['positive_response'][:20]!r})")
        return True
