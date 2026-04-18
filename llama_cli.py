from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
LLAMA_PERF_PREFIX = "llama_perf_context_print:"
MAX_LLAMA_SEED = 2**32 - 1
MEMORY_MODES = (
    "auto",
    "gpu_layers",
    "cpu_moe_layers",
    "gpu_and_cpu_moe_layers",
)


def tensor_to_temp_png(image) -> Path:
    import numpy as np
    from PIL import Image

    # ComfyUI IMAGE is normally B,H,W,C. The node uses the first batch item in v1.
    tensor = image
    if hasattr(tensor, "dim") and tensor.dim() == 4:
        tensor = tensor[0]
    array = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    fd, path = tempfile.mkstemp(prefix="qwen-gguf-", suffix=".png")
    os.close(fd)
    pil_image.save(path, format="PNG")
    return Path(path)


def _write_prompt_file(system_prompt: str, prompt: str, enable_thinking: bool) -> Path:
    # Qwen chat templates understand these control tokens across llama.cpp builds
    # more reliably than version-specific template kwargs.
    think_prefix = "/think" if enable_thinking else "/no_think"
    chunks = []
    if system_prompt:
        chunks.append(system_prompt.strip())
    chunks.append(think_prefix)
    chunks.append(prompt.strip())

    fd, path = tempfile.mkstemp(prefix="qwen-gguf-prompt-", suffix=".txt")
    os.close(fd)
    prompt_path = Path(path)
    prompt_path.write_text("\n\n".join(chunks), encoding="utf-8", newline="\n")
    return prompt_path


def _split_extra_args(extra_args: str) -> list[str]:
    if not extra_args or not extra_args.strip():
        return []
    parts = shlex.split(extra_args, posix=(os.name != "nt"))
    return [part.strip("\"'") for part in parts]


def normalize_seed(seed: int) -> int:
    # llama.cpp uses -1 as "random seed"; otherwise Windows builds parse the
    # value as an unsigned 32-bit integer.
    seed = int(seed)
    if seed == -1:
        return -1
    if not 0 <= seed <= MAX_LLAMA_SEED:
        raise ValueError(f"seed must be -1 or between 0 and {MAX_LLAMA_SEED}")
    return seed


def build_command(
    cli_path: Path,
    model_path: Path,
    mmproj_path: Path | None,
    image_path: Path | None,
    prompt_path: Path,
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
    extra_args: str = "",
) -> list[str]:
    if memory_mode not in MEMORY_MODES:
        raise ValueError(f"Unsupported memory_mode: {memory_mode}")
    if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"} and n_cpu_moe_layers < 1:
        raise ValueError("n_cpu_moe_layers must be at least 1 when CPU MoE layers are enabled.")

    command = [
        str(cli_path),
        "-m", str(model_path),
        "-f", str(prompt_path),
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--repeat-penalty", str(repeat_penalty),
        "-c", str(ctx_size),
        "--seed", str(normalize_seed(seed)),
    ]

    # In auto mode llama.cpp receives neither flag and uses its own placement
    # defaults. Other modes pass only the user-selected memory controls.
    if memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["-ngl", str(n_gpu_layers)])
    if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["--n-cpu-moe", str(n_cpu_moe_layers)])

    if mmproj_path is not None:
        command.extend(["--mmproj", str(mmproj_path)])
    if image_path is not None:
        command.extend(["--image", str(image_path)])

    command.extend(_split_extra_args(extra_args))
    return command


def run_llama_cli(
    cli_path: Path,
    model_path: Path,
    mmproj_path: Path | None,
    image,
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
    extra_args: str = "",
) -> tuple[str, str]:
    image_path = None
    prompt_path = None
    try:
        if image is not None:
            if mmproj_path is None:
                raise ValueError("Image input requires a selected mmproj GGUF file.")
            image_path = tensor_to_temp_png(image)

        prompt_path = _write_prompt_file(system_prompt, prompt, enable_thinking)

        command = build_command(
            cli_path=cli_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
            image_path=image_path,
            prompt_path=prompt_path,
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
            extra_args=extra_args,
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            shell=False,
        )
    finally:
        # Temp prompt/image files can be large in workflows that run many times,
        # so cleanup happens even when llama.cpp exits with an error.
        for path in (image_path, prompt_path):
            if path is not None and path.exists():
                path.unlink()

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"llama.cpp inference failed with exit code {result.returncode}:\n{stderr[-2000:]}"
        )
    perf = extract_perf(result.stderr)
    return result.stdout, perf


def extract_perf(stderr: str) -> str:
    lines = [
        line.strip() for line in str(stderr or "").splitlines()
        if LLAMA_PERF_PREFIX in line
    ]
    return "\n".join(lines)


def extract_thinking(text: str, enable_thinking: bool) -> tuple[str, str]:
    text = str(text or "")
    thinking = ""

    # llama.cpp returns raw stdout; Qwen reasoning may appear as a full block,
    # a closing-only block, or an unfinished block if generation is truncated.
    match = THINK_BLOCK_RE.search(text)
    if match:
        thinking = re.sub(r"</?think[^>]*>", "", match.group(0), flags=re.IGNORECASE).strip()
        text = THINK_BLOCK_RE.sub("", text).strip()
    elif "</think>" in text:
        before, after = text.split("</think>", 1)
        thinking = before.strip()
        text = after.strip()
    elif "<think>" in text:
        before, after = text.split("<think>", 1)
        thinking = after.strip()
        text = before.strip()

    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(token, "")

    response = text.strip()
    thinking = thinking if enable_thinking else ""
    return response, thinking
