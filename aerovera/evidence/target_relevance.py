"""Target relevance Q_t from DINOv3 patch-text similarity (Eq. 8-9, Appendix C.1).

Every scanned frame (Probing frames and candidates) is resized to a short
side of 1088 pixels and encoded by DINOv3 ViT-L/16 in overlapping windows; its
backbone patch features are compared with the patch half of the dino.txt text
embedding of each target ("a photo of a {}."). The frame features do not
depend on the question (``encode_frames``, once per video); the targets do
(``score``, once per question).

Raw patch-text cosines share a large component that does not depend on the
target: the maps of different targets on the same frame are highly correlated
and peak in the same places. Each target embedding therefore has the mean
embedding of a generic aerial vocabulary subtracted, so that a patch scores
its cosine with the target minus its mean cosine with these common classes.
The resulting Q_t still varies little between frames of one video, so it is
min-max normalized over the scanned frames.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


TEMPLATE = "a photo of a {}."
# Class names of public aerial datasets: VisDrone, DOTA, then UAVid / LoveDA.
VOCABULARY = (
    "pedestrian", "bicycle", "car", "van", "truck", "tricycle", "bus", "motorcycle",
    "airplane", "ship", "storage tank", "baseball field", "tennis court", "basketball court",
    "running track", "harbor", "bridge", "helicopter", "roundabout", "soccer field", "swimming pool",
    "building", "road", "tree", "low vegetation", "water", "bare land", "forest", "farmland",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FrameFeatures:
    """DINOv3 features of the scanned frames of one video."""

    patch: Any  # [T, h, w, D] L2-normalized backbone patch features, stitched from the windows
    scene: Any  # [T, D] L2-normalized mean of the windows' CLS features


@dataclass
class TargetScores:
    """How strongly each frame shows the targets of one question."""

    targets: list[str]  # distinct targets (K), in first-seen order
    option_targets: dict[str, list[int]] | None  # choice questions: indices into ``targets`` per option
    sim: Any  # [T, K, h, w] every patch's cosine with each target minus its mean cosine with VOCABULARY (Eq. 8)
    target_scores: Any  # [T, K] Q_{t,k}: mean of the top-P patch scores (Eq. 9)
    target_features: Any  # [T, K, D] h_{t,k}: mean patch feature over the top ``feature_topk`` patches, for uniqueness
    relevance: Any  # [T] Q_t, min-max normalized to [0, 1] over the frames


def index_targets(targets: dict[str, list[str]] | list[str]) -> tuple[list[str], dict[str, list[int]] | None]:
    """Distinct targets for DINOv3, and for a choice question which of them each option uses.

    A target shared by several options (any case) is scored once; options
    keep their own lists of indices, which are empty for meta options.
    """
    distinct: list[str] = []
    index: dict[str, int] = {}

    def add(phrase: str) -> int:
        key = phrase.casefold()
        if key not in index:
            index[key] = len(distinct)
            distinct.append(phrase)
        return index[key]

    if isinstance(targets, dict):
        options = {label: list(dict.fromkeys(add(p) for p in phrases)) for label, phrases in targets.items()}
        return distinct, options
    for phrase in targets:
        add(phrase)
    return distinct, None


def combine_targets(target_scores: Any, option_targets: dict[str, list[int]] | None) -> Any:
    """Per-frame score [T] from per-target scores [T, K]: the mean over targets,
    or for a choice question the maximum over options of each option's mean
    (options without targets skipped), so that a frame supporting any
    competing option counts. Q_t and R_t are both aggregated this way."""
    import torch

    if option_targets is None:
        return target_scores.mean(dim=1)
    per_option = [target_scores[:, indices].mean(dim=1) for indices in option_targets.values() if indices]
    return torch.stack(per_option).max(dim=0).values


def frame_relevance(target_scores: Any, option_targets: dict[str, list[int]] | None) -> Any:
    """Q_t from Q_{t,k}: ``combine_targets``, then min-max normalized over the
    frames (all 0 if they score the same)."""
    import torch

    raw = combine_targets(target_scores, option_targets)
    spread = raw.max() - raw.min()
    return (raw - raw.min()) / spread if spread > 0 else torch.zeros_like(raw)


class TargetRelevance:
    """DINOv3 dino.txt: windowed patch features of frames, text features of targets."""

    def __init__(
        self,
        repo_dir: str | Path,
        weights: str | Path,
        backbone_weights: str | Path,
        bpe_path: str | Path,
        device: str = "cuda:1",
        short_side: int = 1088,
        window: int = 384,
        stride: int = 192,
        batch_size: int = 16,
        topk: int = 4,
        feature_topk: int = 16,
        vocabulary: Sequence[str] = VOCABULARY,
    ):
        if not 0 < stride <= window:
            raise ValueError("Need 0 < stride <= window")
        self.paths = {name: Path(path) for name, path in (
            ("repo_dir", repo_dir), ("weights", weights), ("backbone_weights", backbone_weights), ("bpe_path", bpe_path))}
        self.device = device
        self.short_side, self.window, self.stride = short_side, window, stride
        self.batch_size, self.topk = batch_size, topk
        self.feature_topk = feature_topk  # more patches than Q_t: the top 4 jump around between neighbouring frames
        self.vocabulary = list(vocabulary)  # empty: raw cosines
        self.model: Any = None
        self.tokenize: Any = None
        self.patch_size = 0
        self._text: dict[str, Any] = {}  # phrase -> normalized text patch embedding

    def load(self) -> None:
        for name, path in self.paths.items():
            if not (path / "hubconf.py" if name == "repo_dir" else path).exists():
                raise FileNotFoundError(f"DINOv3 {name} not found: {path}")
        repo = str(self.paths["repo_dir"].resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        build = importlib.import_module("dinov3.hub.dinotxt").dinov3_vitl16_dinotxt_tet1280d20h24l
        model, tokenizer = build(
            weights=str(self.paths["weights"].resolve()),
            backbone_weights=str(self.paths["backbone_weights"].resolve()),
            bpe_path_or_url=str(self.paths["bpe_path"].resolve()),
        )
        self.model = model.to(self.device).eval()
        self.tokenize = tokenizer.tokenize
        self.patch_size = int(self.model.visual_model.backbone.patch_size)
        if self.window % self.patch_size or self.stride % self.patch_size:
            raise ValueError(f"window and stride must be multiples of the patch size {self.patch_size}")

    def encode_frames(self, frames: Sequence[Any]) -> FrameFeatures:
        """Windowed DINOv3 features of RGB uint8 frames [H, W, 3] of one video.

        Windows from all frames share the backbone batches; each patch's
        feature is the mean over the windows that cover it, and each frame's
        scene feature the mean of its windows' CLS features.
        """
        import torch
        import torch.nn.functional as F

        images = self._preprocess(frames)
        count, _, height, width = images.shape
        side, stride, patch_size = self.window, self.stride, self.patch_size

        def starts(length: int) -> list[int]:
            if length <= side:
                return [0]
            positions = list(range(0, length - side + 1, stride))
            if positions[-1] != length - side:
                positions.append(length - side)
            return positions

        tiles = [(t, y, x) for t in range(count) for y in starts(height) for x in starts(width)]
        grid = (count, height // patch_size, width // patch_size)
        coverage = torch.zeros((*grid, 1), device=self.device)
        patch_sum = scene_sum = None
        with torch.inference_mode(False), torch.no_grad():
            for begin in range(0, len(tiles), self.batch_size):
                chunk = tiles[begin:begin + self.batch_size]
                batch = torch.stack([images[t, :, y:y + side, x:x + side] for t, y, x in chunk]).to(self.device)
                cls, patch, _ = self.model.visual_model.get_backbone_features(batch)
                rows, cols = batch.shape[-2] // patch_size, batch.shape[-1] // patch_size
                patch = F.normalize(patch.float(), dim=-1).reshape(len(chunk), rows, cols, -1)
                cls = F.normalize(cls.float(), dim=-1)
                if patch_sum is None:
                    patch_sum = torch.zeros((*grid, patch.shape[-1]), device=self.device)
                    scene_sum = torch.zeros((count, cls.shape[-1]), device=self.device)
                for offset, (t, y, x) in enumerate(chunk):
                    y, x = y // patch_size, x // patch_size
                    patch_sum[t, y:y + rows, x:x + cols] += patch[offset]
                    coverage[t, y:y + rows, x:x + cols] += 1
                    scene_sum[t] += cls[offset]
        return FrameFeatures(F.normalize(patch_sum / coverage, dim=-1), F.normalize(scene_sum, dim=-1))

    def score(self, features: FrameFeatures, targets: dict[str, list[str]] | list[str]) -> TargetScores:
        """Eq. 8-9 for the extracted targets of one question; results on the CPU."""
        import torch

        distinct, option_targets = index_targets(targets)
        text = self._encode_text(distinct)  # [K, D]
        if self.vocabulary:
            text = text - self._encode_text(self.vocabulary).mean(dim=0)  # mean of unit vectors, not renormalized
        sim = torch.einsum("thwd,kd->tkhw", features.patch, text)
        patches = sim.shape[2] * sim.shape[3]
        ranked = sim.flatten(2).topk(min(max(self.topk, self.feature_topk), patches), dim=-1).indices  # [T, K, P], best first
        flat = features.patch.flatten(1, 2)  # [T, h*w, D]
        frames = torch.arange(flat.shape[0], device=flat.device)[:, None, None]
        target_scores = sim.flatten(2).gather(-1, ranked[..., :self.topk]).mean(dim=-1)  # [T, K]
        target_features = flat[frames, ranked[..., :self.feature_topk]].mean(dim=2)  # [T, K, D]
        return TargetScores(
            distinct, option_targets, sim.cpu(), target_scores.cpu(), target_features.cpu(),
            frame_relevance(target_scores, option_targets).cpu(),
        )

    def _encode_text(self, phrases: Sequence[str]) -> Any:
        """Normalized patch half of the dino.txt text embedding of each phrase (cached)."""
        import torch
        import torch.nn.functional as F

        new = [p for p in phrases if p not in self._text]
        if new:
            with torch.inference_mode(False), torch.no_grad():
                tokens = self.tokenize([TEMPLATE.format(p) for p in new]).to(self.device)
                embedding = self.model.encode_text(tokens).float()
            _, patch_half = embedding.chunk(2, dim=-1)
            for phrase, vector in zip(new, F.normalize(patch_half, dim=-1)):
                self._text[phrase] = vector
        return torch.stack([self._text[p] for p in phrases])

    def _preprocess(self, frames: Sequence[Any]) -> Any:
        """Short side to ``short_side`` (PIL bicubic), ImageNet normalization, then
        stretched (bicubic) to a multiple of the patch size: [T, 3, H, W] on the CPU."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode

        transform = T.Compose([
            T.Resize(self.short_side, interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        images = torch.stack([transform(Image.fromarray(frame, mode="RGB")) for frame in frames])
        height, width = images.shape[-2:]
        size = (math.ceil(height / self.patch_size) * self.patch_size, math.ceil(width / self.patch_size) * self.patch_size)
        if (height, width) != size:
            images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
        return images
