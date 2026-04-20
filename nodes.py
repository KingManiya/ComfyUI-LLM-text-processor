from __future__ import annotations

from .folder_registry import (
    NO_MMPROJ,
    full_mmproj_path,
    full_model_path,
    full_system_prompt_path,
    mmproj_options,
    model_options,
    system_prompt_options,
)
from .llama_cli import (
    MAX_LLAMA_SEED,
    build_command,
    run_llama_cli,
    split_extra_args,
)


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
                    "tooltip": "System prompt preset from ComfyUI/models/LLM/prompts, or none.",
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "User prompt sent to the Qwen model.",
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
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
                    "tooltip": "Context window size in tokens.",
                }),
                "memory_mode": ([
                    "auto",
                    "gpu_layers",
                    "cpu_moe_layers",
                    "gpu_and_cpu_moe_layers",
                ], {
                    "default": "auto",
                    "tooltip": "Advanced memory placement mode: auto, gpu_layers, cpu_moe_layers, or gpu_and_cpu_moe_layers.",
                    "advanced": True,
                }),
                "n_gpu_layers": ("INT", {
                    "default": 99,
                    "min": -1,
                    "max": 999,
                    "tooltip": "Used only in gpu_layers and gpu_and_cpu_moe_layers modes. Number of model layers to place on the GPU.",
                    "advanced": True,
                }),
                "n_cpu_moe_layers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999,
                    "tooltip": "Used only in cpu_moe_layers and gpu_and_cpu_moe_layers modes. Number of MoE layers to keep on the CPU.",
                    "advanced": True,
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": -1,
                    "max": MAX_LLAMA_SEED,
                    "tooltip": "Random seed. Use -1 for a random seed.",
                }),
                "timeout_seconds": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 3600,
                    "tooltip": "Maximum time to wait before generation is stopped.",
                }),
                "reasoning": (["auto", "on", "off"], {
                    "default": "off",
                    "tooltip": "Reasoning output mode.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional ComfyUI image input. Uses the first image in the batch.",
                }),
                "extra_args": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional advanced llama.cpp parameters. Leave empty for normal use.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "REASONING", "PERF")
    OUTPUT_TOOLTIPS = (
        "Final model response with reasoning blocks removed.",
        "Extracted reasoning when present in model output.",
        "llama.cpp prompt and generation speed.",
    )
    FUNCTION = "generate"
    CATEGORY = "Qwen GGUF"
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
        timeout_seconds: int,
        reasoning: str,
        image=None,
        extra_args: str = "",
    ):
        model_path = full_model_path(model)
        mmproj_path = full_mmproj_path(mmproj)
        system_prompt_path = full_system_prompt_path(system_prompt)
        parsed_extra_args = split_extra_args(extra_args)

        command, cleanup_paths = build_command(
            model_path=model_path,
            mmproj_path=mmproj_path,
            system_prompt_path=system_prompt_path,
            image=image,
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
            reasoning=reasoning,
            extra_args=parsed_extra_args,
        )
        response, reasoning_text, perf = run_llama_cli(
            command=command,
            timeout_seconds=timeout_seconds,
            cleanup_paths=cleanup_paths,
        )
        return (response, reasoning_text, perf)


NODE_CLASS_MAPPINGS = {"QwenGGUF": QwenGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenGGUF": "Qwen GGUF (llama.cpp)"}
