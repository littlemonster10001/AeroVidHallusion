# AeroVera

AeroVera is a training-free, plug-and-play method that reduces perceptual hallucinations of video MLLMs on UAV videos. It contrasts an evidence-preserving view of the video with an evidence-disrupting one. This folder holds its implementation for Qwen2.5-VL-7B and VideoLLaMA3-7B.

## Quick start

### 1. Set up the environment

AeroVera runs in the same environment as its backbone in the benchmark (see [step 1 of the AeroMirage README](../aeromirage/README.md#1-set-up-the-models-environment)). Then add AeroVera's own dependencies:

```bash
pip install omegaconf==2.4.0 decord opencv-python ftfy regex
```

| Backbone | Config (`aerovera/configs/`) | Tested with |
| --- | --- | --- |
| Qwen2.5-VL-7B | `qwen25_vl_7b.yaml` | Python 3.12, PyTorch 2.13, Transformers 5.17 |
| VideoLLaMA3-7B | `videollama3_7b.yaml` | Python 3.10, PyTorch 2.4, Transformers 4.46.3, flash-attn 2.7.3 |

Activate the environment before running anything (`conda activate <env>`): VideoLLaMA3 reads videos through `ffmpeg`, which must be on `PATH`. Run all commands from the repository root.

### 2. Download the data and the weights

- **Data**: download AeroMirage from [Hugging Face](https://huggingface.co/datasets/littlemonster10001/AeroMirage) and arrange it as described in the [main README](../README.md#data).
- **Backbone**: download the checkpoint from its official page (see the AeroMirage README). Pass its location as `model.path=/path/to/checkpoint`, or set `model.path` in the config.
- **DINOv3 and RAFT**: AeroVera uses DINOv3 ViT-L/16 with the dino.txt text head for target relevance, and RAFT-Small for motion. Place them as follows (other locations can be set in the `dino` and `motion` sections of the config):

```
checkpoints/
├── DINOv3/
│   ├── dinov3/                                   # clone of the official DINOv3 repository
│   ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
│   ├── dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
│   └── bpe_simple_vocab_16e6.txt.gz              # BPE vocabulary of the dino.txt text encoder
└── optical_flow/
    └── raft_small_C_T_V2-01064c6d.pth
```

The DINOv3 code, weights and BPE vocabulary come from the [official DINOv3 repository](https://github.com/facebookresearch/dinov3) (the weights require an access request there). The RAFT-Small weights are torchvision's `Raft_Small_Weights.C_T_V2`, available at `https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth`.

### 3. Run AeroVera

AeroVera uses two GPUs: the backbone runs on `cuda:0`, and DINOv3, RAFT and the resolution analysis run on `cuda:1`. While the backbone answers the questions of one video, the next video is prepared in the background on the second GPU. We ran it on 48 GB GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 python run_aerovera.py --config aerovera/configs/qwen25_vl_7b.yaml \
    dataset.root=/path/to/AeroMirage model.path=/path/to/Qwen2.5-VL-7B-Instruct
```

Every parameter is in the backbone's YAML file, each with a comment. Edit the file, or override any field on the command line as `key=value`, e.g. `runtime.limit=20` to answer only the first 20 questions, or `runtime.question_ids=[0002q000000]` to answer chosen questions.

An interrupted run resumes where it stopped: rerunning the same command answers only the missing questions and retries failed ones. A result file written with different settings is not resumed; write to a new file with `output.predictions=...`, or start over with `output.overwrite=true`.

```bash
python -m aerovera.export outputs/predictions/aerovera_qwen25_vl_7b.jsonl --field probing_response
```

`--field` takes `probing_response`, `positive_response` or `negative_response`.

## License

<!-- TODO: code license -->
