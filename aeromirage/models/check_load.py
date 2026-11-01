"""Load one configured model, optionally read a video, and release it.

Run from the AeroVidHallusion directory inside the model's Conda environment:

    python -m aeromirage.models.check_load --config aeromirage/configs/qwen25_vl_3b_vllm.yaml
    python -m aeromirage.models.check_load --config ... --video /path/to/clip.mp4
    python -m aeromirage.models.check_load --config ... model.options.tensor_parallel_size=2
    python -m aeromirage.models.check_load --config ... --video clip.mp4 --prompt
    python -m aeromirage.models.check_load --config ... --video clip.mp4 --generate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from aeromirage.benchmark import SYSTEM_PROMPT, build_task_prompt
from aeromirage.config import load_config
from aeromirage.models import create_model_adapter
from aeromirage.types import BenchmarkSample


def demo_samples() -> list[BenchmarkSample]:
    """Real questions covering both prompt layouts, for checking build_inputs().

    Single choice shares the binary layout, so it is left out.
    """
    common = dict(
        video_id="0000", reference_answer=None, video_path=Path("0000.mp4"),
        source_file="demo", source_line=1,
    )
    return [
        BenchmarkSample(
            question_id="0002q000000", question_type="binary_verification",
            question="视频显示有人骑自行车经过建筑工地。", options=[], **common,
        ),
        BenchmarkSample(
            question_id="0000q000000", question_type="multiple_choice",
            question="在视频中可以看到哪些物体？",
            options=["A. 蓝色汽车在道路上", "B. 深色的结构延伸从海滩进入水面",
                     "C. 一排深色的立柱在浅水区", "D. 蓝色游泳池漂浮在近海",
                     "E. 蓝色矩形游泳池在陆地上", "F. 以上都不是", "G. 以上全部"],
            **common,
        ),
        BenchmarkSample(
            question_id="0000q000000_e01", question_type="open_ended",
            question="水面上有三个小物体：一个是黄色浮标，另一个是一艘载有人员的____小船和黄色小船。",
            options=[], clue="color",
            prompt_question="视频的水域中可以看到多少个有颜色的漂浮目标？", **common,
        ),
    ]


def describe(value: Any, depth: int = 0) -> str:
    """One-line summary of prepare_video() output: types, lengths, shapes."""
    if hasattr(value, "shape"):
        return f"{type(value).__name__}{tuple(value.shape)}"
    if hasattr(value, "size") and hasattr(value, "mode"):
        return f"PIL{value.size}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {describe(v, depth + 1)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        head = describe(value[0], depth + 1) if depth < 3 else "..."
        return f"[{len(value)} x {head}]"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that an AeroMirage model loads.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--video", type=Path, nargs="+", help="also run prepare_video() on these files"
    )
    parser.add_argument(
        "--prompt", action="store_true",
        help="with --video, print build_inputs() for demo questions on the first video"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="with --video, answer the demo questions on the first video"
    )
    parser.add_argument("overrides", nargs="*", help="config overrides such as runtime.seed=1")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    adapter = create_model_adapter(config.model, config.video)
    started = time.perf_counter()
    adapter.load()
    try:
        elapsed = time.perf_counter() - started
        print(f"OK: {config.model.family} loaded from {config.model.path} in {elapsed:.1f}s")
        print(f"video options: {adapter.video}")
        if args.video:
            for path in reversed(args.video):  # the first video is prepared last and kept
                started = time.perf_counter()
                visual = adapter.prepare_video(path)
                elapsed = time.perf_counter() - started
                print(f"OK: prepare_video({path.name}) in {elapsed:.1f}s -> {describe(visual)}")
            video = args.video[0]
            if args.prompt:
                before = describe(visual)
                for sample in demo_samples():
                    inputs = adapter.build_inputs(visual, SYSTEM_PROMPT, build_task_prompt(sample))
                    rest = {key: value for key, value in inputs.items() if key != "prompt"}
                    print(f"----- {sample.question_type} prompt -----\n{inputs['prompt']}<<END>>")
                    print(f"other inputs: {describe(rest)}")
                if describe(visual) != before:
                    raise RuntimeError("build_inputs() modified the prepared video")
                print("OK: build_inputs left the prepared video unchanged")
            if args.generate:
                reads = []
                prepare_video = adapter.prepare_video
                adapter.prepare_video = lambda path: reads.append(path) or prepare_video(path)
                for sample in demo_samples():
                    started = time.perf_counter()
                    answer = adapter.generate(
                        video,
                        SYSTEM_PROMPT,
                        build_task_prompt(sample),
                        config.generation.limit_for(sample.question_type),
                    )
                    elapsed = time.perf_counter() - started
                    print(f"----- {sample.question_type} answer ({elapsed:.1f}s) -----\n{answer}<<END>>")
                print(f"OK: video read {len(reads)} time(s) for {len(demo_samples())} questions")
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
