from __future__ import annotations

from .folder_registry import (
    NO_MMPROJ,
    full_mmproj_path,
    full_model_path,
    load_system_prompt,
    mmproj_options,
    model_options,
    system_prompt_options,
)
from .llama_binary import ensure_llama_cli
from .llama_cli import MAX_LLAMA_SEED, MEMORY_MODES, extract_thinking, run_llama_cli


class QwenGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_options(), {
                    "tooltip": "GGUF model loaded from ComfyUI/models/LLM. mmproj files are hidden from this list.",
                }),
                "mmproj": (mmproj_options(), {
                    "default": NO_MMPROJ,
                    "tooltip": "Vision projector GGUF. Required when an image is connected.",
                }),
                "system_prompt": (system_prompt_options(), {
                    "tooltip": "System prompt preset from ComfyUI/models/LLM/prompts, or None.",
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "User prompt sent to the Qwen model.",
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                    "tooltip": "Maximum number of tokens to generate.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature. Lower is more deterministic.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold.",
                }),
                "top_k": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Top-K sampling cutoff.",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Penalty applied to repeated tokens.",
                }),
                "ctx_size": ("INT", {
                    "default": 8192,
                    "min": 512,
                    "max": 131072,
                    "step": 512,
                    "tooltip": "llama.cpp context window size in tokens.",
                }),
                "memory_mode": (list(MEMORY_MODES), {
                    "default": "auto",
                    "tooltip": "auto passes no layer flags; other modes pass GPU layers, CPU MoE layers, or both.",
                    "advanced": True,
                }),
                "n_gpu_layers": ("INT", {
                    "default": 99,
                    "min": -1,
                    "max": 999,
                    "tooltip": "Used only in gpu_layers and gpu_and_cpu_moe_layers modes. Number of layers to offload to GPU.",
                    "advanced": True,
                }),
                "n_cpu_moe_layers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999,
                    "tooltip": "Used only in cpu_moe_layers and gpu_and_cpu_moe_layers modes. Keeps MoE weights of the first N layers on CPU.",
                    "advanced": True,
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": -1,
                    "max": MAX_LLAMA_SEED,
                    "tooltip": "Random seed used by llama.cpp. Use -1 for a random seed.",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use /think and return extracted reasoning in the THINKING output.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 3600,
                    "tooltip": "Maximum time to wait for llama.cpp before failing.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional ComfyUI image input. The first image in the batch is sent to llama.cpp.",
                }),
                "extra_args": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Advanced raw llama.cpp CLI flags appended to the command.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "THINKING", "PERF")
    OUTPUT_TOOLTIPS = (
        "Final model response with Qwen thinking blocks removed.",
        "Extracted <think> reasoning when thinking mode is enabled.",
        "llama.cpp llama_perf_context_print lines.",
    )
    FUNCTION = "generate"
    CATEGORY = "Qwen/GGUF"
    TITLE = "Qwen GGUF (llama.cpp)"

    def generate(
        self,
        model: str,
        mmproj: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        ctx_size: int,
        memory_mode: str,
        n_gpu_layers: int,
        n_cpu_moe_layers: int,
        seed: int,
        enable_thinking: bool,
        timeout_seconds: int,
        image=None,
        extra_args: str = "",
    ):
        model_path = full_model_path(model)
        mmproj_path = full_mmproj_path(mmproj)
        system_prompt_text = load_system_prompt(system_prompt)

        # Binary installation is lazy so ComfyUI startup stays fast and offline
        # workflows do not touch the network until this node actually runs.
        cli_path = ensure_llama_cli()

        raw_output, perf = run_llama_cli(
            cli_path=cli_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
            image=image,
            system_prompt=system_prompt_text,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            ctx_size=ctx_size,
            memory_mode=memory_mode,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe_layers=n_cpu_moe_layers,
            seed=seed,
            enable_thinking=enable_thinking,
            timeout_seconds=timeout_seconds,
            extra_args=extra_args,
        )
        response, thinking = extract_thinking(raw_output, enable_thinking)
        return (response, thinking, perf)


NODE_CLASS_MAPPINGS = {"QwenGGUF": QwenGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenGGUF": "Qwen GGUF (llama.cpp)"}
