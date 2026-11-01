# AeroVidHallusion

**Benchmarking and mitigating perceptual hallucinations in UAV video understanding.**

[Paper (coming soon)]() · [Dataset](https://huggingface.co/datasets/littlemonster10001/AeroMirage) · [AeroMirage benchmark](aeromirage/README.md) · [AeroVera method](aerovera/README.md)

## News / TODO

- [x] Benchmark harness: 23 open-source MLLM configurations, scoring and the LLM judge
- [x] AeroVera for Qwen2.5-VL-7B and VideoLLaMA3-7B
- [x] Release the AeroMirage dataset on [Hugging Face](https://huggingface.co/datasets/littlemonster10001/AeroMirage)
- [ ] Add the paper link
- [ ] Add environment files (`environments/`)

## Overview

<!-- Teaser figure: benchmark examples and the AeroVera pipeline. -->

Multimodal large language models (MLLMs) are increasingly used to understand UAV (drone) videos, yet their answers are often not grounded in what the aerial footage actually shows: targets are small and densely packed, and the scene keeps changing with the camera. This repository contains:

- **AeroMirage**, a benchmark of perceptual hallucinations in UAV videos. It has 827 videos and four question formats covering entity, spatial, temporal and ego-motion hallucinations, plus a unified harness that runs 23 open-source MLLM configurations under the same frames, prompts and greedy decoding.
- **AeroVera**, a training-free, plug-and-play mitigation method that contrasts an evidence-preserving view of the video with an evidence-disrupting one. It currently supports Qwen2.5-VL-7B and VideoLLaMA3-7B.

## Data

Download AeroMirage from [Hugging Face](https://huggingface.co/datasets/littlemonster10001/AeroMirage) and arrange it as:

```
/path/to/AeroMirage/
├── QA/
│   ├── binary_verification.jsonl
│   ├── single_choice.jsonl
│   ├── multiple_choice.jsonl
│   └── open_ended.jsonl
└── video/
    └── <video_id>.mp4
```

## Quick start

**1. Evaluate an MLLM on AeroMirage.** Please refer to [aeromirage/README.md](aeromirage/README.md).

**2. Run AeroVera.** Please refer to [aerovera/README.md](aerovera/README.md).

**3. Score the predictions.** Open-ended answers are first judged by an LLM (Qwen3.5-27B); then every file is scored:

```bash
python -m evaluation.judge outputs/predictions/*.jsonl
python -m evaluation.evaluate outputs/predictions/*.jsonl
```

An AeroVera result file also records the backbone's own answer on the same input (`probing_response`) and the answers of each branch alone. Export any of them as a standard prediction file to score it the same way:

```bash
python -m aerovera.export outputs/predictions/aerovera_qwen25_vl_7b.jsonl --field probing_response
```

## Acknowledgements

AeroVera builds on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3), [DINOv3](https://github.com/facebookresearch/dinov3) and [RAFT](https://github.com/princeton-vl/RAFT); the benchmark harness uses [vLLM](https://github.com/vllm-project/vllm) and [Transformers](https://github.com/huggingface/transformers). We thank their authors for releasing them.
