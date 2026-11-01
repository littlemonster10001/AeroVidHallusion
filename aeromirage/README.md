# AeroMirage

AeroMirage is a benchmark of perceptual hallucinations in UAV videos. This folder holds the harness that evaluates MLLMs on it: every model sees the same uniformly sampled frames and the same prompts, and decodes greedily.

## Quick start

### 1. Set up the model's environment

Each model runs in its own environment, because the models pin different versions of PyTorch, Transformers and vLLM. Create the environment by following the model's official installation guide (table below), then add the benchmark's own dependencies:

```bash
pip install omegaconf==2.4.0 openpyxl decord
```

| Model | Configs (`aeromirage/configs/`) | Official guide |
| --- | --- | --- |
| Qwen2.5-VL 3B / 7B / 32B | `qwen25_vl_*_vllm.yaml` | [GitHub](https://github.com/QwenLM/Qwen2.5-VL) |
| Qwen3-VL 4B / 8B / 32B | `qwen3_vl_*_vllm.yaml` | [Model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3.5 2B / 4B / 9B / 27B (also with thinking) | `qwen35_*_vllm*.yaml` | [Model card](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Keye-VL-1.5-8B | `keye_vl_1_5_8b_vllm.yaml` | [Model card](https://huggingface.co/Kwai-Keye/Keye-VL-1_5-8B) |
| LLaVA-OneVision-2-8B | `llava_onevision2_8b_transformers.yaml` | [Model card](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct) |
| VideoLLaMA3 2B / 7B | `videollama3_*_transformers.yaml` | [Model card](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B) |
| InternVL3.5 8B / 14B | `internvl35_*_transformers.yaml` | [GitHub](https://github.com/OpenGVLab/InternVL) |
| MiniCPM-V 4 | `minicpm_v4_transformers.yaml` | [GitHub](https://github.com/OpenBMB/MiniCPM-V) |
| Kimi-VL-A3B-Instruct | `kimi_vl_a3b_instruct_transformers.yaml` | [GitHub](https://github.com/MoonshotAI/Kimi-VL) |
| VideoChat-Flash-Qwen2-7B (res448) | `videochat_flash_7b_res448.yaml` | [Model card](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448) |
| LLaVA-NeXT-Video-7B | `llava_next_video_7b.yaml` | [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) (install the repository as the `llava` package) |

Activate the environment before running anything (`conda activate <env>`), rather than calling its Python directly: vLLM and the ffmpeg-based reader need the environment's executables on `PATH`. Run all commands from the repository root.

### 2. Download the data and the model

- **Data**: download AeroMirage from [Hugging Face](https://huggingface.co/datasets/littlemonster10001/AeroMirage) and arrange it as described in the [main README](../README.md#data).
- **Model**: download the checkpoint from the official page above. Pass its location as `model.path=/path/to/checkpoint`, or set `model.path` in the config.

### 3. Check that the model loads

```bash
python -m aeromirage.models.check_load --config aeromirage/configs/qwen25_vl_7b_vllm.yaml \
    model.path=/path/to/Qwen2.5-VL-7B-Instruct \
    --video /path/to/AeroMirage/video/0000.mp4 --generate
```

This loads the model, reads the video exactly as the benchmark does and answers a few demo questions. Use `--prompt` instead of `--generate` to print the model inputs without generating.

### 4. Run the benchmark

```bash
python run_benchmark.py --config aeromirage/configs/qwen25_vl_7b_vllm.yaml \
    dataset.root=/path/to/AeroMirage model.path=/path/to/Qwen2.5-VL-7B-Instruct
```

The answers go to `outputs/predictions/<config name>.jsonl`, one line per question, written as soon as each answer exists.

Every parameter is in the model's YAML file, each with a comment, e.g. [configs/qwen25_vl_7b_vllm.yaml](configs/qwen25_vl_7b_vllm.yaml). Edit the file, or override any field on the command line as `key=value` (e.g. `runtime.limit=20` to answer only the first 20 questions). The command line also takes:

| Option | Effect |
|---|---|
| `--config` | The model's YAML file (required) |
| `--validate-only` | Check the QA files and videos and print one prompt per question format; loads no model |

An interrupted run resumes where it stopped: rerunning the same command answers only the missing questions and retries failed ones.

### 5. Score

Judge the open-ended answers and score the predictions as described in the [main README](../README.md#quick-start).

## License

AeroMirage is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt the dataset, provided that appropriate credit is given and the dataset is not used for commercial purposes. Please refer to the [official license text](https://creativecommons.org/licenses/by-nc/4.0/legalcode) for the complete terms and conditions.
