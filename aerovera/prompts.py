"""AeroVera's own prompts: meta-option removal and target extraction (per-option verification is the benchmark's)."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from typing import Any

from aeromirage.benchmark import (  # noqa: F401  (re-exported: the per-option protocol is the benchmark's)
    META_OPTIONS,
    VERIFICATION_STATEMENT,
    option_letters,
    option_text,
    ordinary_options,
    verification_prompt,
)
from aeromirage.types import BenchmarkSample


def without_meta_options(sample: BenchmarkSample) -> BenchmarkSample:
    """The sample with "以上都不是" / "以上全部" removed from a multiple-choice question.

    The remaining options keep their own letters (A-D); the reference answer
    never includes a meta option, so it is unchanged.
    """
    if sample.question_type != "multiple_choice":
        return sample
    kept = [
        option
        for label, option in zip(option_letters(sample.options), sample.options)
        if option_text(label, option) not in META_OPTIONS
    ]
    return dataclasses.replace(sample, options=kept)


# Per-option verification (P5): META_OPTIONS, ordinary_options,
# VERIFICATION_STATEMENT and verification_prompt are the benchmark's own
# (aeromirage.benchmark.prompts), so the refine pass and the per-option
# Probing / baseline ask exactly the same question.


# --- Target extraction (P4) -------------------------------------------------
# The backbone reads only the question text and extracts the objects it
# mentions, for DINOv3 to find ("a photo of a {target}."), so targets must be
# English noun phrases.

TARGET_SYSTEM_PROMPT = """\
You extract the physical objects that a question about a drone video mentions, so that they can be found in the video frames. You see only the text, not the video. Do not answer the question.

Rules:
- List every physical object or place that the question or the options name: vehicles, people, animals, buildings, roads, water, plants, signs and the like. Keep the colour or other visual attributes the text gives, e.g. "red car", "white boat".
- The drone and its camera are not targets.
- The question may ask about motion, direction, time, counts or camera movement; still list the objects it names. Return an empty list only when the text names nothing besides the drone or its camera.
- A target is always an object, never an action, motion, direction, position, time, count, number, event or the answer: write "red car", not "red car turning".
- With options, read the question together with each option. If the option names its own object, use that object; otherwise (for example when the option is a time, a number or an action) use the objects of the question. A number, including 0, still refers to the object being counted.
- Give 1 to 3 targets per option (or for the whole question when there are no options), the most important first, without repeats.
- Write every target in English; translate Chinese nouns (骑行者 -> cyclist). Never output Chinese.
- Return JSON only, with no explanation.

Example 1
Question: 画面中有几辆红色汽车？
Options:
A: 0
B: 两辆蓝色公交车
C: 2
Output: {"A": ["red car"], "B": ["blue bus"], "C": ["red car"]}

Example 2
Question: 红色汽车接下来做了什么？
Options:
A: 左转
B: 一辆黄色公交车出现
C: 停下
Output: {"A": ["red car"], "B": ["yellow bus"], "C": ["red car"]}

Example 3
Question: 视频中是否能看到一名骑行者经过白色货车？
Output: ["cyclist", "white van"]

Example 4
Question: 停车场里那辆车是什么颜色？
Sentence with a blank: 停车场入口旁停着一辆____的轿车。
Blank type: color
Output: ["sedan"]

Example 5
Question: 无人机经过水塔后做了什么？
Sentence with a blank: 无人机飞过水塔后，____并朝停车场下降。
Blank type: action
Output: ["water tower", "parking lot"]

Example 6
Question: 无人机相对于红白相间的灯塔如何移动？
Sentence with a blank: 无人机沿____绕灯塔飞行，并逐渐靠近它。
Blank type: action
Output: ["red and white lighthouse"]

Example 7
Question: 视频过程中，桥上的路灯是否从熄灭变为亮起？
Output: ["street lamp", "bridge"]

Example 8
Question: 镜头下移后，教堂后方出现了什么？
Sentence with a blank: 一条两旁有路灯的小路，通向____。
Blank type: object
Output: ["church", "path", "street lamp"]

Example 9
Question: 白色货车什么时候开始移动？
Options:
A: 第2秒
B: 第5秒
C: 一直没有移动
Output: {"A": ["white van"], "B": ["white van"], "C": ["white van"]}"""

MAX_TARGETS = 3
# Motion words the model sometimes appends to an object ("red car turning").
_ACTION_SUFFIX = re.compile(
    r"\s+(?:turning|stopping|moving|appearing|disappearing|driving|walking|running|"
    r"flying|sailing|accelerating|decelerating|landing)$",
    re.IGNORECASE,
)
_CHINESE = re.compile(r"[㐀-鿿]")
# A bare number is an answer (e.g. a count), not an object to look for.
_NUMBER = re.compile(
    r"^(?:\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)$", re.IGNORECASE
)


@dataclass
class TargetRequest:
    """What the backbone is asked for one question, and how its answer must look."""

    user: str
    labels: list[str] | None  # keys of the JSON object for a choice question; None: a JSON list
    meta_labels: list[str]  # meta options: not shown, their targets are []


class InvalidTargets(ValueError):
    """An extraction output to retry; the message is the reason given in the retry prompt."""


def target_request(sample: BenchmarkSample) -> TargetRequest:
    """The user prompt of target extraction (P4): only what is needed to find objects."""
    if sample.question_type in ("single_choice", "multiple_choice"):
        labels, meta, lines = [], [], [f"Question: {sample.question.strip()}", "Options:"]
        for label, option in zip(option_letters(sample.options), sample.options):
            text = option_text(label, option)
            if text in META_OPTIONS:
                meta.append(label)
            else:
                labels.append(label)
                lines.append(f"{label}: {text}")
        lines.append(f"Output a JSON object with exactly the keys {', '.join(labels)}.")
        return TargetRequest("\n".join(lines), labels, meta)
    if sample.question_type == "binary_verification":
        return TargetRequest(f"Question: {sample.question.strip()}\nOutput a JSON list.", None, [])
    if sample.question_type == "open_ended":
        lines = [
            f"Question: {sample.prompt_question.strip()}",
            f"Sentence with a blank: {sample.question.strip()}",
            f"Blank type: {sample.clue}",
            "Output a JSON list.",
        ]
        return TargetRequest("\n".join(lines), None, [])
    raise ValueError(f"unsupported question type: {sample.question_type}")


def retry_prompt(user: str, reason: str) -> str:
    """The same request in a new conversation, with why the previous output was rejected."""
    return f"{user}\nYour previous output was invalid: {reason}. Follow the rules and return JSON only."


def parse_targets(raw: str, request: TargetRequest, last_attempt: bool = False) -> dict[str, list[str]] | list[str]:
    """Targets from an extraction output, or ``InvalidTargets`` with the reason to retry.

    Retried: no JSON (or cut off), wrong type or keys, Chinese targets, every
    list empty. Fixed in place: action words at the end, bare numbers,
    repeats, more than three targets. On the last attempt Chinese targets are
    dropped instead.
    Choice questions return every option letter, meta options with [].
    """
    data = _first_json(raw)
    if data is None:
        raise InvalidTargets("not valid JSON (possibly cut off)")
    if request.labels is None:
        if not isinstance(data, list):
            raise InvalidTargets("the output must be a JSON list")
        groups: dict[str, Any] = {"": data}
    else:
        keys = ", ".join(request.labels)
        if not isinstance(data, dict):
            raise InvalidTargets(f"the output must be a JSON object with the keys {keys}")
        groups = {str(key).strip().upper(): value for key, value in data.items()}
        if set(groups) != set(request.labels) or len(groups) != len(data):
            raise InvalidTargets(f"the keys must be exactly {keys}")

    cleaned, chinese = {}, []
    for key, value in groups.items():
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise InvalidTargets("every target list must be a list of strings")
        phrases: list[str] = []
        for item in value:
            phrase = _ACTION_SUFFIX.sub("", " ".join(item.split()))
            if _CHINESE.search(phrase):
                chinese.append(key)
                continue
            if _NUMBER.match(phrase):
                continue
            if phrase and phrase.casefold() not in {seen.casefold() for seen in phrases}:
                phrases.append(phrase)
        cleaned[key] = phrases[:MAX_TARGETS]
    if chinese and not last_attempt:
        where = "the list contains" if request.labels is None else f"options {', '.join(sorted(set(chinese)))} contain"
        raise InvalidTargets(f"{where} Chinese targets")
    if not any(cleaned.values()):
        raise InvalidTargets("every target list is empty")
    if request.labels is None:
        return cleaned[""]
    every = {label: cleaned[label] for label in request.labels} | {label: [] for label in request.meta_labels}
    return {label: every[label] for label in sorted(every)}


def _first_json(raw: str) -> Any:
    """The JSON object or list that starts at the first bracket of ``raw`` (it may sit in a ``` block).

    Only the outermost value counts: when a cut-off object fails to parse, a
    list nested in it must not be taken for the answer.
    """
    match = re.search(r"[\[{]", raw)
    if match is None:
        return None
    try:
        return json.JSONDecoder().raw_decode(raw[match.start():])[0]
    except json.JSONDecodeError:
        return None
