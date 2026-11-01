"""AeroVera configuration: dataclasses and load_config(path, overrides).

Same pattern as ``aeromirage.config``: the dataclasses are an OmegaConf
structured schema, so unknown keys and wrong types fail when the YAML is
merged, and any field can be overridden with ``key=value``.

Note: no ``from __future__ import annotations`` here; OmegaConf reads these
annotations at runtime.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from omegaconf import MISSING, OmegaConf

from aeromirage.config import DatasetConfig, GenerationConfig, OutputConfig


@dataclass
class ModelConfig:
    family: str = "qwen25_vl"  # backbone: qwen25_vl | videollama3
    path: str = MISSING
    device: str = "cuda:0"
    dtype: str = "bfloat16"


@dataclass
class DinoConfig:
    """DINOv3 dino.txt for target relevance Q_t (Appendix C.1); weights under checkpoints/."""

    repo_dir: str = "checkpoints/DINOv3/dinov3"
    weights: str = "checkpoints/DINOv3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
    backbone_weights: str = "checkpoints/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    bpe_path: str = "checkpoints/DINOv3/bpe_simple_vocab_16e6.txt.gz"
    device: str = "cuda:1"
    short_side: int = 1088
    window: int = 384
    stride: int = 192
    batch_size: int = 16  # windows per backbone forward pass; speed only
    topk: int = 4  # P: patches averaged per frame and target
    feature_topk: int = 16  # patches averaged into the target feature h_{t,k} for uniqueness (A17)


@dataclass
class MotionConfig:
    """RAFT-Small motion change M_t (Eq. 10-11, Appendix C.1); weights under checkpoints/."""

    weights: str = "checkpoints/optical_flow/raft_small_C_T_V2-01064c6d.pth"
    device: str = "cuda:1"
    interval: float = 0.3  # Delta: seconds between the frames of a triplet
    max_side: int = 640
    num_flow_updates: int = 12
    grid_step: int = 16  # stride of the camera-fit correspondences, in flow pixels
    ransac_threshold: float = 2.0
    min_inlier_ratio: float = 0.5
    percentile: float = 99.0  # V_t, A_t: this percentile of the per-pixel residual speed / change
    gamma: float = 0.3  # weight of the motion magnitude; the motion change gets 1 - gamma
    speed_scale: float = 0.03  # kappa_V
    change_scale: float = 0.1  # kappa_A
    batch_size: int = 16  # flows per RAFT pass (two per frame); speed only


@dataclass
class SelectionConfig:
    """Evidence-aware frame selection (Sec. 4.2, Appendix C.1)."""

    num_candidates: int = 32  # |C|: new frames sampled in the gaps between the Probing frames
    num_added: int = 16  # B: candidates added to the positive branch
    # lambda_q, lambda_m, lambda_u of Eq. 3: weights of Q_t, M_t and U_t (summing to 1 in the paper)
    relevance_weight: float = 0.6
    motion_weight: float = 0.2
    uniqueness_weight: float = 0.2


@dataclass
class ResolutionConfig:
    """Resolution gain R_t and the high-resolution frames of the positive branch (Appendix C.2)."""

    num_hr: int = 16  # B_hr: frames with the highest R_t fed at high resolution
    hr_short_side: int = 1024
    # > 0: HR is the LR size times this per side instead (VideoLLaMA3, whose context
    # cannot hold short-side-1024 frames), never beyond the source
    hr_scale: float = 0.0
    lower_quantile: float = 0.5  # target maps scaled between these quantiles of the video
    upper_quantile: float = 0.95
    floor: float = 0.1  # mu: map values below it are background
    noise_floor: float = 2 / 255  # epsilon_n: lower bound of the background noise level
    device: str = "cuda:1"


@dataclass
class TargetsConfig:
    """Target extraction from the question text (P4): greedy, one retry."""

    max_new_tokens: int = 256


@dataclass
class NegativeConfig:
    """Attribution-guided negative branch (Appendix C.3)."""

    mask_percent: float = 20.0  # rho: share of the Probing visual tokens whose Values are zeroed


@dataclass
class ContrastiveConfig:
    """Contrastive decision of the two branches (Eq. 2)."""

    alpha: float = 0.5  # q = q+ + alpha (q+ - q-)


@dataclass
class RuntimeConfig:
    limit: int = 0  # first N questions (0: all)
    limit_per_task: int = 0
    question_ids: List[str] = field(default_factory=list)  # only these questions (any type), for checks
    fail_fast: bool = False


@dataclass
class AeroVeraConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    # The backbone's read_video options, exactly the benchmark baseline's ``video`` section
    # (Qwen2.5-VL: qwen-vl-utils' num_frames / min_pixels / max_pixels / video_reader_backend;
    # VideoLLaMA3: load_video's num_frames / fps / max_visual_tokens)
    probing: Dict[str, Any] = field(default_factory=dict)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    negative: NegativeConfig = field(default_factory=NegativeConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_config(path: "str | Path", overrides: Sequence[str] = ()) -> AeroVeraConfig:
    """YAML file merged into the schema, then command-line ``key=value`` overrides."""
    merged = OmegaConf.merge(
        OmegaConf.structured(AeroVeraConfig),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides)),
    )
    return OmegaConf.to_object(merged)
