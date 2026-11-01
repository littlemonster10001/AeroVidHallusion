"""Turn a raw model response into a normalized answer and score it.

Parsing reads only the final answer: everything up to the last </think> is
dropped. An answer that cannot be read is an empty prediction, scored wrong.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable

from aeromirage.benchmark.prompts import option_letters
from aeromirage.types import BenchmarkSample


REFUSAL = "uncertain"  # normalized form of 无法判断 and its variants
_REFUSALS = {"无法判断", "uncertain", "不确定", "不能判断", "无法确定", "无法从视频判断"}
_SMALL_NUMBERS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def final_answer(raw: str) -> str:
    """Text after a completed thinking section, without a code fence."""
    text = raw.rsplit("</think>", 1)[-1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def _json_answer(text: str) -> Any:
    """``{"answer": ...}`` if the model answered in JSON, else the text itself."""
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return text
        if isinstance(value, dict) and "answer" in value:
            return value["answer"]
    return text


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def extract_letters(value: Any, allowed: Iterable[str]) -> list[str]:
    """Option letters in an answer such as "C", "A, C", "答案是C" or "B: 否"."""
    allowed_set = {letter.upper() for letter in allowed}
    if isinstance(value, list):
        parts = [str(item).strip().upper() for item in value]
    else:
        text = unicodedata.normalize("NFKC", str(value or "")).upper()
        # A letter counts when no other Latin letter touches it, so Chinese
        # neighbours ("选C") are fine but words ("CAR") are not.
        parts = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", text)
    return sorted({part for part in parts if part in allowed_set})


def match_option_text(value: Any, options: list[str]) -> str:
    """Letter of the option whose text the answer repeats exactly."""
    target = normalized_text(value)
    if not target:
        return ""
    for letter, option in zip(option_letters(options), options):
        text = re.sub(r"^\s*[A-Z]\s*[.):：、]?\s*", "", option, flags=re.IGNORECASE)
        if target in {normalized_text(option), normalized_text(text)}:
            return letter
    return ""


def parse_binary(value: Any) -> str:
    """Yes / No from an A(是) / B(否) answer; "" when undecidable."""
    text = normalized_text(value)
    if not text:
        return ""
    labelled = re.fullmatch(r"(?:(?:answer|答案)\s*)?([ab])(?:\s*(?:yes|no|是|否))?", text)
    if labelled:
        return "Yes" if labelled.group(1) == "a" else "No"
    if re.match(r"^(?:no|false)\b", text) or re.fullmatch(r"否|不是|不对|不正确|错误|假的|假", text):
        return "No"
    if re.match(r"^(?:yes|true)\b", text) or re.match(r"^(?:是的|是|对的|对|正确|真的|真)(?:\s|$)", text):
        return "Yes"
    if re.search(r"(?:无法|不能|难以)(?:确定|确认|判断)|不确定|看不清", text):
        return ""
    # Negations first: "没有……展示的是……" contains an unrelated 是.
    if re.search(r"没有|没看到|没看见|未看到|未看见|未发现|未出现|未显示|并无|并非|不存在|不是|不对|不正确|错误|假的", text):
        return "No"
    if re.search(r"\b(?:no|false)\b", text):
        return "No"
    if re.search(r"(?:可以|能)(?:看到|看见|观察到)|看到了|看见了|确实|存在|出现了|显示了|展示了|视频中有|画面中有", text):
        return "Yes"
    if re.search(r"\b(?:yes|true)\b", text):
        return "Yes"
    return ""


def normalize_fill(value: Any, clue: str = "") -> str:
    """Canonical short fill; refusals become ``uncertain``."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"^(?:答案|填空|answer|fill)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    result = normalized_text(text)
    if result in _REFUSALS:
        return REFUSAL
    if clue == "count":
        number = re.fullmatch(r"(\d+|[零〇一二两三四五六七八九十]+)(?:个|辆|人|只|台|艘|架|名|条|棵)?", result)
        if number:
            token = number.group(1)
            if token.isdecimal():
                return str(int(token))
            if token in _SMALL_NUMBERS:
                return str(_SMALL_NUMBERS[token])
            if len(token) == 2 and token[0] == "十" and token[1] in _SMALL_NUMBERS:
                return str(10 + _SMALL_NUMBERS[token[1]])
            if len(token) == 2 and token[1] == "十" and token[0] in _SMALL_NUMBERS:
                return str(10 * _SMALL_NUMBERS[token[0]])
            if len(token) == 3 and token[1] == "十" and token[0] in _SMALL_NUMBERS and token[2] in _SMALL_NUMBERS:
                return str(10 * _SMALL_NUMBERS[token[0]] + _SMALL_NUMBERS[token[2]])
    if clue == "color" and re.fullmatch(r"[㐀-鿿]+色", result):
        return result[:-1]
    if clue in {"direction", "spatial"}:
        direction = re.fullmatch(r"(?:向)?(左上|右上|左下|右下|左|右|上|下|前|后)(?:方|侧|边)?", result)
        if direction:
            return direction.group(1)
    return result


def parse_response(raw: str, sample: BenchmarkSample) -> tuple[str, str]:
    """(model_answer, prediction): the final answer text and its normalized form."""
    answer = _json_answer(final_answer(raw))
    model_answer = ",".join(map(str, answer)) if isinstance(answer, list) else str(answer).strip()
    task = sample.question_type
    if task == "binary_verification":
        return model_answer, parse_binary(answer)
    if task == "single_choice":
        letters = extract_letters(answer, option_letters(sample.options))
        if len(letters) == 1:
            return model_answer, letters[0]
        return model_answer, match_option_text(answer, sample.options)
    if task == "multiple_choice":
        return model_answer, ",".join(extract_letters(answer, option_letters(sample.options)))
    return model_answer, normalize_fill(answer, sample.clue)


def normalize_reference(sample: BenchmarkSample) -> str:
    task = sample.question_type
    if task == "binary_verification":
        return str(sample.reference_answer)  # loader stores Yes / No
    if task == "single_choice":
        return str(sample.reference_answer).upper()
    if task == "multiple_choice":
        return ",".join(sorted(str(item).upper() for item in sample.reference_answer))
    if sample.answerability == "unanswerable":
        return REFUSAL
    return normalize_fill(sample.reference_answer, sample.clue)


def is_correct(sample: BenchmarkSample, prediction: str) -> bool:
    """Exact match after normalization (open-ended: the automatic check; the
    LLM judge gives the reported open-ended score)."""
    if not prediction:
        return False
    if sample.question_type != "open_ended":
        return prediction == normalize_reference(sample)
    if sample.answerability == "unanswerable":
        return prediction == REFUSAL
    accepted = {normalize_reference(sample)}
    accepted.update(normalize_fill(value, sample.clue) for value in sample.accepted_equivalents)
    return prediction != REFUSAL and prediction in accepted
