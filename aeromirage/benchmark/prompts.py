"""AeroMirage prompts (Chinese), shared by every model.

Only the wording lives here. Each model's ``build_inputs()`` puts these two
plain strings into its own chat format (chat template, ``<image>``
placeholders, frame prefixes, ...). Output-format instructions sit in the
task prompt rather than the system prompt because some models have no system
role or follow it loosely.
"""

from __future__ import annotations

import dataclasses
import re

from aeromirage.types import BenchmarkSample


SYSTEM_PROMPT = (
    "你将回答关于一段无人机视频的问题。请依据所给画面中可见的证据作答，"
    "不要编造画面无法支持的物体、动作、数量或细节。"
)

# Every valid open-ended clue and its name in the prompt.
CLUE_NAMES = {
    "object": "物体",
    "color": "颜色",
    "attribute": "属性",
    "action": "动作",
    "direction": "方向",
    "count": "数量",
    "spatial": "空间位置",
    "temporal": "时间",
    "text": "文字",
}

# "最终回答" scopes the format to the answer, leaving a thinking model's
# <think> section free; parsing drops everything before </think>.
SINGLE_CHOICE_INSTRUCTION = (
    "请将每个选项与视频逐一对照，最终回答只输出一个正确选项的字母，不要附加解释。"
)
MULTIPLE_CHOICE_INSTRUCTION = (
    "请将每个选项与视频逐一对照，最终回答只输出所有正确选项的字母，"
    "用英文逗号分隔（如 A,C），不要附加解释。"
)
OPEN_ENDED_INSTRUCTION = (
    "最终回答只输出缺失处应填的最短内容，不要复述句子或附加解释；"
    "若无法从视频判断，只输出：无法判断。"
)


def option_letters(options: list[str]) -> list[str]:
    """Label of each option: its own ``A.`` prefix, else its position."""
    letters: list[str] = []
    for index, option in enumerate(options):
        match = re.match(r"\s*([A-Z])(?:\s*[.)]|\s+)", option, flags=re.IGNORECASE)
        letters.append(match.group(1).upper() if match else chr(ord("A") + index))
    return letters


def option_text(label: str, text: str) -> str:
    """Drop the dataset's ``A.`` prefix; the prompt writes ``A: text``."""
    return re.sub(
        rf"^\s*{re.escape(label)}\s*[.):：、]\s*", "", text, flags=re.IGNORECASE
    ).strip()


# Multiple-choice options implied by the others rather than seen in the video.
# The data has exactly these two texts, always last, and never in a reference
# answer; an ordinary option such as "全部三个汽车停止在交叉处。" must stay, so
# the whole text is compared.
META_OPTIONS = ("以上都不是", "以上全部")

# Per-option protocol of multiple choice (generation.multiple_choice):
# every ordinary option is asked as a yes/no question (A = 是, B = 否) about
# this statement, and the options answered 是 form the answer.
VERIFICATION_STATEMENT = "针对问题“{question}”，“{option}”是正确答案之一。"


def ordinary_options(sample: BenchmarkSample) -> list[tuple[str, str]]:
    """(letter, text) of the options of a multiple-choice question other than the meta options."""
    return [
        (label, option_text(label, option))
        for label, option in zip(option_letters(sample.options), sample.options)
        if option_text(label, option) not in META_OPTIONS
    ]


def verification_prompt(sample: BenchmarkSample, option: str) -> str:
    """Task prompt asking whether ``option`` (text without its letter) is one of the answers."""
    statement = VERIFICATION_STATEMENT.format(question=sample.question.strip(), option=option)
    return build_task_prompt(dataclasses.replace(
        sample, question_type="binary_verification", question=statement, options=[]))


def build_task_prompt(sample: BenchmarkSample) -> str:
    if sample.question_type == "open_ended":
        return _open_ended_prompt(sample)
    if sample.question_type in {"binary_verification", "single_choice", "multiple_choice"}:
        return _choice_prompt(sample)
    raise ValueError(f"unsupported question type: {sample.question_type}")


def _choice_prompt(sample: BenchmarkSample) -> str:
    # A binary statement is asked as a single-choice question with A=是, B=否.
    if sample.question_type == "binary_verification":
        options = [("A", "是"), ("B", "否")]
    else:
        options = [
            (label, option_text(label, text))
            for label, text in zip(option_letters(sample.options), sample.options)
        ]
    multiple = sample.question_type == "multiple_choice"
    lines = [
        f"题型：{'多选题' if multiple else '单选题'}",
        f"问题：{sample.question.strip()}",
        "选项：",
    ]
    lines.extend(f"{label}: {text}" for label, text in options)
    lines.append(MULTIPLE_CHOICE_INSTRUCTION if multiple else SINGLE_CHOICE_INSTRUCTION)
    return "\n".join(lines)


def _open_ended_prompt(sample: BenchmarkSample) -> str:
    return "\n".join([
        "请仔细分析视频画面，补全给定句子的指定缺失内容。",
        f"问：{sample.prompt_question.strip()} 答：{sample.question.strip()}",
        f"填空类别：{CLUE_NAMES[sample.clue]}",
        OPEN_ENDED_INSTRUCTION,
    ])
