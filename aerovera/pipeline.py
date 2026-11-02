"""One question end to end: Probing Pass, evidence-guided positive branch, masked negative branch, contrastive decision.

Work that depends only on the video (frames, DINOv3 features, motion, HR/LR
images and their band energy) is done once per video in ``prepare_video``
(A11); ``answer`` then handles one question on that ``VideoContext``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeromirage.benchmark import SYSTEM_PROMPT, build_task_prompt, option_letters
from aeromirage.benchmark.parser import verified_options
from aeromirage.types import BenchmarkSample
from aerovera.backbones import create_backbone
from aerovera.backbones.base import Attribution, Backbone, Video
from aerovera.config import AeroVeraConfig
from aerovera.contrastive import best_label, chosen_options, confident, contrast, top_candidates, top_percent
from aerovera.evidence.candidates import candidate_frame_ids, read_frames
from aerovera.evidence.motion import FrameMotion, MotionChange
from aerovera.evidence.resolution import band_energy, resolution_gains, scaled_size, top_frames
from aerovera.evidence.selection import Pick, select_frames
from aerovera.evidence.target_relevance import FrameFeatures, TargetRelevance
from aerovera.evidence.uniqueness import pair_similarity
from aerovera.prompts import (
    TARGET_SYSTEM_PROMPT,
    InvalidTargets,
    ordinary_options,
    parse_targets,
    retry_prompt,
    target_request,
    verification_prompt,
    without_meta_options,
)


@dataclass
class Probing:
    """The Probing Pass: the backbone's own answer on the uniform frames."""

    response: str  # written as probing_response
    answer_ids: list[int]  # the answer whose evidence the negative branch removes
    inputs: Any  # model inputs, kept for the negative branch
    video: Video
    attribution: Attribution | None  # A_i of every visual token to the answer
    task_prompt: str  # the question as asked (P2), reused by the positive branch
    option_responses: dict[str, str] | None = None  # per-option multiple choice: each option's raw yes/no answer


@dataclass
class Targets:
    """Objects to look for in the frames, extracted from the question text alone (P4)."""

    # Choice questions: targets per option letter (meta options []); otherwise
    # one list. None when both attempts failed: the question then keeps the
    # Probing answer (target_fallback, A8).
    targets: dict[str, list[str]] | list[str] | None
    outputs: list[str]  # raw output of each attempt, for the result file

    @property
    def fallback(self) -> bool:
        return self.targets is None


@dataclass
class Negative:
    """The negative branch: the Probing inputs with the visual tokens that most support the answer masked."""

    mask: list[int]  # visual-token indices (flattened grid) whose Values are zeroed
    token_count: int  # visual tokens of the Probing inputs


class Stopwatch:
    """Seconds spent in each named stage since the previous one."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, name: str, device: str | None = None) -> None:
        """Close stage ``name``; ``device`` waits for its queued GPU work first, so that it counts here."""
        if device is not None and device.startswith("cuda"):
            import torch

            torch.cuda.synchronize(device)
        now = time.perf_counter()
        self.times[name] = round(now - self._last, 3)
        self._last = now


@dataclass
class Models:
    """The three networks of AeroVera, loaded once per run."""

    backbone: Backbone
    relevance: TargetRelevance
    motion: MotionChange

    @classmethod
    def load(cls, config: AeroVeraConfig) -> "Models":
        m, d, f = config.model, config.dino, config.motion
        backbone = create_backbone(m.family, m.path, m.device, m.dtype)
        relevance = TargetRelevance(d.repo_dir, d.weights, d.backbone_weights, d.bpe_path, d.device, d.short_side,
                                    d.window, d.stride, d.batch_size, d.topk, d.feature_topk)
        motion = MotionChange(f.weights, f.device, f.interval, f.max_side, f.num_flow_updates, f.grid_step,
                              f.ransac_threshold, f.min_inlier_ratio, f.percentile, f.gamma, f.speed_scale,
                              f.change_scale, f.batch_size)
        for model in (backbone, relevance, motion):
            model.load()
        return cls(backbone, relevance, motion)


@dataclass
class VideoContext:
    """Everything about one video that does not depend on the question (A11).

    Frames are addressed by their position among the scanned frames: the
    Probing frames and the candidates, in time order.
    """

    video: Video
    scan_ids: list[int]  # source frame ids of the scanned frames
    probing: list[int]  # positions of the Probing frames
    candidates: list[int]  # positions of the candidates
    features: FrameFeatures  # DINOv3 features
    motion: FrameMotion | None  # None when no frames are added
    lr: list[Any]  # each frame at the Probing size (equal to the Probing frames), uint8
    hr: list[Any] | None  # each frame at the HR size; None when no frame gets HR
    energies: Any  # [T, H, W] band energy for R_t; None when no frame gets HR
    timings: dict[str, float]  # seconds per preparation stage

    @property
    def seconds(self) -> float:
        return round(sum(self.timings.values()), 3)

    def time(self, position: int) -> float:
        return self.scan_ids[position] / self.video.fps


def prepare_video(models: Models, config: AeroVeraConfig, path: str | Path) -> VideoContext:
    """Read the video and compute its question-independent evidence once."""
    import torch

    clock = Stopwatch()
    sel, res = config.selection, config.resolution
    backbone = models.backbone
    video = backbone.read_video(path, **config.probing)
    clock.lap("read_video")
    candidates = candidate_frame_ids(video.frame_ids, video.total_frames, sel.num_candidates) if sel.num_added else []
    scan = sorted(video.frame_ids + candidates)
    added = set(candidates)
    frames = read_frames(video.path, scan)
    own = backbone.source_frames(video)
    if own is not None:  # the Probing frames exactly as the backbone read them (D3)
        position = {frame_id: index for index, frame_id in enumerate(scan)}
        for frame_id, frame in zip(video.frame_ids, own):
            frames[position[frame_id]] = frame
    clock.lap("read_frames")
    features = models.relevance.encode_frames(frames)
    clock.lap("dino", config.dino.device)
    motion = models.motion.score(video.path, scan, video.total_frames, video.fps) if candidates else None
    clock.lap("motion", config.motion.device)
    lr = [backbone.resize(frame, video.size) for frame in frames]
    source = frames[0].shape[:2]
    hr_size = scaled_size(video.size, source, res.hr_scale) if res.hr_scale else backbone.hr_size(*source, res.hr_short_side)
    # A source no larger than the Probing size (360p and the like) gains nothing from HR.
    use_hr = res.num_hr > 0 and hr_size[0] > video.size[0] and hr_size[1] > video.size[1]
    hr = [backbone.resize(frame, hr_size) for frame in frames] if use_hr else None
    clock.lap("resize")
    energies = torch.stack([band_energy(h, l, res.device) for h, l in zip(hr, lr)]) if use_hr else None
    clock.lap("energy", res.device)
    return VideoContext(
        video, scan,
        [i for i, frame in enumerate(scan) if frame not in added],
        [i for i, frame in enumerate(scan) if frame in added],
        features, motion, lr, hr, energies, clock.times,
    )


def probing(
    backbone: Backbone, sample: BenchmarkSample, video: Video, max_new_tokens: int, multiple_choice: str = "whole",
) -> Probing:
    """Answer ``sample`` on the uniform frames exactly as the benchmark baseline asks it.

    Multiple choice follows the benchmark's ``generation.multiple_choice``:
    per_option asks each ordinary option's yes/no question on the same video
    and keeps the options answered 是 (A22); whole asks the question without
    its meta options (P2). The attribution comes from the same pass;
    multiple-choice questions skip it, because their negative branch is built
    for each option instead.
    """
    task_prompt = build_task_prompt(without_meta_options(sample))
    if sample.question_type == "multiple_choice" and multiple_choice == "per_option":
        responses = {
            label: backbone.generate(backbone.video_inputs(video, SYSTEM_PROMPT, verification_prompt(sample, text)),
                                     max_new_tokens).text
            for label, text in ordinary_options(sample)
        }
        return Probing(",".join(verified_options(responses)), [], None, video, None, task_prompt, responses)
    inputs = backbone.video_inputs(video, SYSTEM_PROMPT, task_prompt)
    if sample.question_type == "multiple_choice":
        answer, attribution = backbone.generate(inputs, max_new_tokens), None
    else:
        answer, attribution = backbone.attribute(inputs, max_new_tokens=max_new_tokens)
    return Probing(answer.text, answer.token_ids, inputs, video, attribution, task_prompt)


def targets(backbone: Backbone, sample: BenchmarkSample, max_new_tokens: int) -> Targets:
    """Extract the visual targets of ``sample`` with at most one retry (P4).

    The retry is a new conversation: the same request plus the reason the
    first output was rejected, never the rejected output itself.
    """
    request = target_request(sample)
    user, outputs = request.user, []
    for attempt in (1, 2):
        answer = backbone.generate(backbone.text_inputs(TARGET_SYSTEM_PROMPT, user), max_new_tokens)
        outputs.append(answer.text)
        try:
            return Targets(parse_targets(answer.text, request, last_attempt=attempt == 2), outputs)
        except InvalidTargets as error:
            user = retry_prompt(request.user, str(error))
    return Targets(None, outputs)


def negative(probe: Probing, mask_percent: float) -> Negative:
    """Mask the top ``mask_percent``% visual tokens by signed attribution (rho = 20)."""
    if probe.attribution is None:
        raise ValueError("The Probing Pass has no attribution (multiple choice, or an empty answer)")
    scores = probe.attribution.scores.tolist()
    return Negative(top_percent(scores, mask_percent), len(scores))


def evidence_frames(
    models: Models, config: AeroVeraConfig, context: VideoContext, extracted: dict[str, list[str]] | list[str]
) -> tuple[list[int], list[int], list[Pick]]:
    """Positions of the positive branch's frames (Probing frames plus the B
    selected candidates, in time order), the positions among them fed at HR
    (top R_t), and the selection record (Sec. 4.2-4.3)."""
    sel, res = config.selection, config.resolution
    scores = models.relevance.score(context.features, extracted)
    picks: list[Pick] = []
    if context.motion is not None:
        similarity = pair_similarity(context.features.scene, scores.target_features, context.motion.speed,
                                     context.motion.valid, context.candidates)
        picks = select_frames(scores.relevance, context.motion.motion, similarity, context.candidates,
                              context.probing, sel.num_added,
                              (sel.relevance_weight, sel.motion_weight, sel.uniqueness_weight))
    positive = sorted(context.probing + [pick.position for pick in picks])
    hr: list[int] = []
    if context.hr is not None:
        gains = resolution_gains(context.energies, scores.sim, scores.option_targets, positive,
                                 res.lower_quantile, res.upper_quantile, res.floor, res.noise_floor)
        hr = [positive[i] for i in top_frames(gains, res.num_hr)]
    return positive, hr, picks


def answer(models: Models, config: AeroVeraConfig, context: VideoContext, sample: BenchmarkSample) -> dict[str, Any]:
    """One question end to end; the fields of its result line, ``raw_response`` (the final answer) included."""
    backbone, clock = models.backbone, Stopwatch()
    limit = config.generation.limit_for(sample.question_type)
    protocol = config.generation.multiple_choice
    per_option = sample.question_type == "multiple_choice" and protocol == "per_option"
    # Per option each question asked is a yes/no question, with that length limit (as in the benchmark).
    probe = probing(backbone, sample, context.video,
                    config.generation.limit_for("binary_verification") if per_option else limit, protocol)
    clock.lap("probing")
    extracted = targets(backbone, sample, config.targets.max_new_tokens)
    clock.lap("targets")
    record: dict[str, Any] = {
        "prompt": probe.task_prompt,
        "probing_response": probe.response,
        **({"probing_option_responses": probe.option_responses} if probe.option_responses is not None else {}),
        "targets": extracted.targets,
        "target_outputs": extracted.outputs,
        "target_fallback": extracted.fallback,
    }
    if extracted.fallback:
        # A8: no frame selection; every branch keeps the Probing answer.
        return {"raw_response": probe.response, **record, "positive_response": probe.response,
                "negative_response": probe.response, "timings": {**clock.times, "video": context.timings}}

    positive, hr, picks = evidence_frames(models, config, context, extracted.targets)
    frames = [context.hr[p] if p in hr else context.lr[p] for p in positive]
    seconds = [context.time(p) for p in positive]
    record["frames"] = {
        "probing": [context.scan_ids[p] for p in context.probing],
        "added": [{"frame": context.scan_ids[k.position], "relevance": round(k.relevance, 4),
                   "motion": round(k.motion, 4), "uniqueness": round(k.uniqueness, 4)} for k in picks],
        "hr": [context.scan_ids[p] for p in hr],
        "lr_size": list(context.video.size),
        "hr_size": list(context.hr[0].shape[:2]) if context.hr is not None else None,
    }
    clock.lap("evidence")
    if sample.question_type == "multiple_choice":
        decision = _multiple_choice(backbone, config, context, sample, frames, seconds, clock)
    elif sample.question_type == "open_ended":
        decision = _open_ended(backbone, config, probe, frames, seconds, limit, clock)
    else:
        decision = _single_choice(backbone, config, sample, probe, frames, seconds, clock)
    return {"raw_response": decision.pop("raw_response"), **record, **decision,
            "timings": {**clock.times, "video": context.timings}}


def _mask(probe: Probing, config: AeroVeraConfig) -> list[int]:
    """Visual tokens to block in the negative branch; none if the Probing answer was empty."""
    return negative(probe, config.negative.mask_percent).mask if probe.attribution is not None else []


def _single_choice(
    backbone: Backbone, config: AeroVeraConfig, sample: BenchmarkSample, probe: Probing,
    frames: list[Any], seconds: list[float], clock: Stopwatch,
) -> dict[str, Any]:
    """Single choice and yes/no (asked as A = 是 / B = 否): the option letters'
    logits at the first answer position, contrasted. When the contrasted
    answer is not confident enough (``contrastive.min_confidence``, A21) the
    Probing answer stands."""
    labels = ["A", "B"] if sample.question_type == "binary_verification" else option_letters(sample.options)
    ids = backbone.label_ids(labels)
    inputs = backbone.image_inputs(frames, seconds, SYSTEM_PROMPT, probe.task_prompt)
    positive = backbone.first_logits(inputs)[ids].cpu()
    del inputs
    clock.lap("positive")
    negative_logits = backbone.first_logits(probe.inputs, mask=_mask(probe, config))[ids].cpu()
    clock.lap("negative")
    combined = contrast(positive, negative_logits, config.contrastive.alpha)
    kept_probing = not confident(combined, config.contrastive.min_confidence)
    return {
        "raw_response": probe.response if kept_probing else best_label(labels, combined),
        "kept_probing": kept_probing,
        "positive_response": best_label(labels, positive),
        "negative_response": best_label(labels, negative_logits),
        "label_scores": {"labels": labels, "positive": [round(float(v), 4) for v in positive],
                         "negative": [round(float(v), 4) for v in negative_logits]},
    }


def _multiple_choice(
    backbone: Backbone, config: AeroVeraConfig, context: VideoContext, sample: BenchmarkSample,
    frames: list[Any], seconds: list[float], clock: Stopwatch,
) -> dict[str, Any]:
    """Each ordinary option as a yes/no question (P5). The positive branch
    judges every option on the shared images; the judgment is attributed on
    the Probing video with that option's question, and the top tokens masked
    give the option's own negative branch. The options are chosen from their
    yes-minus-no margins by ``contrastive.option_rule`` (A20), each branch's
    answer by the same rule; the meta options are never answered."""
    options = ordinary_options(sample)
    prompts = [verification_prompt(sample, text) for _, text in options]
    yes_no = backbone.label_ids(["A", "B"])
    positive = [logits[yes_no].cpu() for logits in backbone.first_logits_many(frames, seconds, SYSTEM_PROMPT, prompts)]
    clock.lap("positive")
    margins: dict[str, dict[str, float]] = {"raw_response": {}, "positive_response": {}, "negative_response": {}}
    scores = {}
    for (label, _), prompt, judged in zip(options, prompts, positive):
        inputs = backbone.video_inputs(context.video, SYSTEM_PROMPT, prompt)
        answer_id = yes_no[0] if best_label(["A", "B"], judged) == "A" else yes_no[1]
        _, attribution = backbone.attribute(inputs, answer_ids=[answer_id])
        mask = top_percent(attribution.scores.tolist(), config.negative.mask_percent)
        negative_logits = backbone.first_logits(inputs, mask=mask)[yes_no].cpu()
        combined = contrast(judged, negative_logits, config.contrastive.alpha)
        for field, values in (("raw_response", combined), ("positive_response", judged),
                              ("negative_response", negative_logits)):
            margins[field][label] = float(values[0]) - float(values[1])
        scores[label] = {"positive": [round(float(v), 4) for v in judged],
                         "negative": [round(float(v), 4) for v in negative_logits]}
    clock.lap("negative")
    rule = config.contrastive.option_rule
    return {**{field: ",".join(chosen_options(m, rule)) for field, m in margins.items()}, "label_scores": scores}


def _open_ended(
    backbone: Backbone, config: AeroVeraConfig, probe: Probing, frames: list[Any], seconds: list[float], limit: int,
    clock: Stopwatch,
) -> dict[str, Any]:
    """Contrast on the first token only, then the positive branch continues (P6)."""
    mask = _mask(probe, config)
    inputs = backbone.image_inputs(frames, seconds, SYSTEM_PROMPT, probe.task_prompt)
    final, positive, (q_positive, q_negative, q) = backbone.contrastive_generate(
        inputs, probe.inputs, config.contrastive.alpha, limit, mask)
    del inputs
    clock.lap("contrastive")
    negative_answer = backbone.generate(probe.inputs, limit, mask=mask)
    clock.lap("negative")
    return {
        "raw_response": final.text,
        "positive_response": positive.text,
        "negative_response": negative_answer.text,
        "label_scores": top_candidates(q_positive.cpu(), q_negative.cpu(), q.cpu(), backbone.token_text),
    }
