"""Shared base for every model run with vLLM (Qwen2.5-VL, Qwen3-VL, Qwen3.5, Keye-VL).

Only the engine lives here; each model's video reading (prepare_video) is in
its own file.
"""

from __future__ import annotations

import os
from typing import Any

from aeromirage.models.base import BaseModelAdapter


class VLLMAdapter(BaseModelAdapter):
    """Build a vLLM engine and Hugging Face processor from ``model.options``."""

    model_label = "vLLM model"

    def __init__(self, config, video=None):
        super().__init__(config, video)
        self.llm: Any = None
        self.processor: Any = None
        self.sampling_params_class: Any = None

    def _engine_kwargs(self) -> dict[str, Any]:
        options = self.config.options
        tensor_parallel_size = int(options.get("tensor_parallel_size", 1))
        pipeline_parallel_size = int(options.get("pipeline_parallel_size", 1))
        gpu_memory_utilization = float(options.get("gpu_memory_utilization", 0.9))
        max_num_seqs = int(options.get("max_num_seqs", 1))
        max_model_len = int(options.get("max_model_len", 32768))

        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if pipeline_parallel_size <= 0:
            raise ValueError("pipeline_parallel_size must be positive")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if max_num_seqs <= 0 or max_model_len <= 0:
            raise ValueError("max_num_seqs and max_model_len must be positive")

        kwargs: dict[str, Any] = {
            "model": self.config.path,
            "dtype": str(options.get("dtype", "bfloat16")),
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "limit_mm_per_prompt": {"video": 1, "image": 0},
            "trust_remote_code": self.config.trust_remote_code,
            "enforce_eager": bool(options.get("enforce_eager", False)),
            "enable_prefix_caching": bool(options.get("enable_prefix_caching", False)),
            "disable_log_stats": bool(options.get("disable_log_stats", True)),
            "seed": int(options.get("seed", 42)),
        }

        # Data-parallel vision encoding only applies when the language model is
        # tensor-parallel; omitting it for TP=1 also keeps older vLLM working.
        mm_encoder_tp_mode = options.get("mm_encoder_tp_mode")
        if tensor_parallel_size > 1 and mm_encoder_tp_mode:
            kwargs["mm_encoder_tp_mode"] = str(mm_encoder_tp_mode)

        optional_engine_args = {
            "swap_space": float,
            "cpu_offload_gb": float,
            "max_num_batched_tokens": int,
            "disable_custom_all_reduce": bool,
            "distributed_executor_backend": str,
            "quantization": str,
            "kv_cache_dtype": str,
        }
        for name, converter in optional_engine_args.items():
            if options.get(name) is not None:
                kwargs[name] = converter(options[name])

        extra_engine_kwargs = options.get("engine_kwargs", {})
        if not isinstance(extra_engine_kwargs, dict):
            raise TypeError("model.options.engine_kwargs must be a mapping")
        kwargs.update(extra_engine_kwargs)
        return kwargs

    def _apply_attention_backend(self, engine_kwargs: dict[str, Any]) -> str | None:
        """Translate ``engine_kwargs.attention_backend`` for the installed vLLM."""
        requested = engine_kwargs.pop("attention_backend", None)
        if not requested:
            return None
        backend_name = str(requested).upper()

        from vllm import envs as vllm_envs

        if "VLLM_ATTENTION_BACKEND" in getattr(vllm_envs, "environment_variables", {}):
            # Legacy vLLM releases select the backend through this env var.
            os.environ["VLLM_ATTENTION_BACKEND"] = backend_name
            return backend_name
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
        except ImportError:
            engine_kwargs["attention_backend"] = backend_name
            return backend_name
        if backend_name in AttentionBackendEnum.__members__:
            engine_kwargs["attention_backend"] = backend_name
            return backend_name
        print(
            f"Ignoring unsupported legacy vLLM attention backend {backend_name}; "
            "using automatic backend selection."
        )
        return None

    def load(self) -> None:
        self._check_model_path()

        use_flashinfer_sampler = self.config.options.get("use_flashinfer_sampler")
        if use_flashinfer_sampler is not None:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer_sampler else "0"
        # vLLM's default fork start method cannot inherit an initialized CUDA
        # context, so use spawn before importing vLLM.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        engine_kwargs = self._engine_kwargs()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                f"{self.config.family} requires vLLM; activate the "
                "`dronevidhallubench` Conda environment."
            ) from exc
        from transformers import AutoProcessor

        attention_backend = self._apply_attention_backend(engine_kwargs)
        print(f"Loading {self.model_label} with vLLM: {self.config.path}")
        if attention_backend:
            print(f"vLLM attention backend: {attention_backend}")
        print(
            f"vLLM parallelism: TP={engine_kwargs['tensor_parallel_size']}, "
            f"PP={engine_kwargs['pipeline_parallel_size']}; "
            f"max_model_len={engine_kwargs['max_model_len']}; "
            f"gpu_memory_utilization={engine_kwargs['gpu_memory_utilization']}"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.path,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=False,
        )
        self.llm = LLM(**engine_kwargs)
        self.sampling_params_class = SamplingParams

    def _messages(self, system_prompt: str, task_prompt: str) -> list[dict[str, Any]]:
        """Chat messages with one video placeholder before the question."""
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": task_prompt}],
            },
        ]

    def _chat_template_kwargs(self) -> dict[str, Any]:
        return {}

    def build_inputs(self, visual: Any, system_prompt: str, task_prompt: str) -> dict[str, Any]:
        """A complete vLLM request: prompt text plus the prepared video."""
        self._require_loaded("processor")
        prompt = self.processor.apply_chat_template(
            self._messages(system_prompt, task_prompt),
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(),
        )
        request: dict[str, Any] = {
            "prompt": prompt,
            "multi_modal_data": {"video": visual["video"]},
        }
        if visual.get("mm_processor_kwargs"):
            request["mm_processor_kwargs"] = visual["mm_processor_kwargs"]
        return request

    def _sample(self, request: dict[str, Any], max_tokens: int, **params: Any) -> Any:
        """One greedy vLLM completion; returns its first candidate."""
        self._require_loaded("llm", "sampling_params_class")
        sampling_params = self.sampling_params_class(
            temperature=0.0, repetition_penalty=1.0, max_tokens=max_tokens, **params
        )
        outputs = self.llm.generate([request], sampling_params=sampling_params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no generation candidates")
        return outputs[0].outputs[0]

    def run(self, inputs: dict[str, Any], max_new_tokens: int) -> str:
        return self._sample(inputs, max_new_tokens).text.strip()

    def clear_cache(self) -> None:
        # vLLM owns its worker processes and KV cache; emptying PyTorch's
        # cache in this process frees nothing.
        return None

    def close(self) -> None:
        if self.llm is not None:
            shutdown = getattr(self.llm, "shutdown", None)
            if not callable(shutdown):
                shutdown = getattr(getattr(self.llm, "llm_engine", None), "shutdown", None)
            if callable(shutdown):
                shutdown()
        self.llm = None
        self.processor = None
        self.sampling_params_class = None
        super().close()
