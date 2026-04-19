from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import comfy.model_management

from .llama_binary import ensure_llama_cli_paths


PROMPT_ECHO_END = "... (truncated)"
PROMPT_PADDING = " " * 501
PERF_RE = re.compile(r"\[\s*Prompt:\s*[^|\]]+\|\s*Generation:\s*[^\]]+\]")
START_THINKING = "[Start thinking]"
END_THINKING = "[End thinking]"


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


def _write_temp_text_file(prefix: str, text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    os.close(fd)
    text_path = Path(path)
    text_path.write_text(text, encoding="utf-8", newline="\n")
    return text_path


def _write_prompt_file(system_prompt: str, prompt: str) -> Path:
    chunks = []
    if system_prompt:
        chunks.append(system_prompt.strip())
    chunks.append(prompt.strip() + PROMPT_PADDING)

    return _write_temp_text_file("qwen-gguf-prompt-", "\n\n".join(chunks))


def split_extra_args(extra_args: str) -> list[str]:
    if not extra_args or not extra_args.strip():
        return []
    parts = shlex.split(extra_args, posix=(os.name != "nt"))
    return [part.strip("\"'") for part in parts]


def build_command(
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
    reasoning: str,
    extra_args: list[str] | None = None,
) -> tuple[list[str], tuple[Path | None, ...]]:
    cleanup_paths = []
    cli_paths = ensure_llama_cli_paths()
    image_path = None
    if image is not None:
        if mmproj_path is None:
            raise ValueError("Image input requires a selected mmproj GGUF file.")
        image_path = tensor_to_temp_png(image)
        cleanup_paths.append(image_path)

    prompt_path = _write_prompt_file(system_prompt, prompt)
    cleanup_paths.append(prompt_path)

    command = [
        str(cli_paths.cli),
        "-m", str(model_path),
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--repeat-penalty", str(repeat_penalty),
        "-c", str(ctx_size),
        "--seed", str(seed),
        "--single-turn",
        "--reasoning", reasoning,
    ]

    # In auto mode llama.cpp receives neither flag and uses its own placement
    # defaults. Other modes pass only the user-selected memory controls.
    if memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["-ngl", str(n_gpu_layers)])
    if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["--n-cpu-moe", str(n_cpu_moe_layers)])

    command.extend(["-f", str(prompt_path)])

    if image_path:
        command.extend(["--mmproj", str(mmproj_path)])
        command.extend(["--image", str(image_path)])

    if extra_args:
        command.extend(extra_args)
    return command, tuple(cleanup_paths)


def run_llama_cli(
    command: list[str],
    timeout_seconds: int,
    cleanup_paths: tuple[Path | None, ...] = (),
) -> tuple[str, str, str]:
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        stdout, stderr = _communicate_with_interrupt(process, timeout_seconds)
        print(stdout)
        print(stderr)
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        if process is not None:
            _stop_process(process)
        raise
    finally:
        # Temp prompt/image files can be large in workflows that run many times,
        # so cleanup happens even when llama.cpp exits with an error.
        for path in cleanup_paths:
            if path and path.exists():
                path.unlink()

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"llama.cpp inference failed with exit code {result.returncode}:\n{stderr}"
        )
    return _parse_response(result.stdout + "\n" + result.stderr)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _communicate_with_interrupt(process: subprocess.Popen, timeout_seconds: int) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if comfy.model_management.processing_interrupted():
            _stop_process(process)
            # Reset the global interrupt flag and raise the exact exception
            # ComfyUI expects so the UI reports this as an interruption.
            comfy.model_management.throw_exception_if_processing_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(process)
            raise TimeoutError(f"llama.cpp timed out after {timeout_seconds}s")
        try:
            return process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue


def _parse_response(text: str) -> tuple[str, str, str]:
    text = str(text or "")
    if PROMPT_ECHO_END in text:
        text = text.split(PROMPT_ECHO_END, 1)[1]

    perf_match = PERF_RE.search(text)
    perf = perf_match.group(0).strip() if perf_match else ""
    content = text[:perf_match.start()] if perf_match else text
    content = content.strip()
    if not content.startswith(START_THINKING):
        return content, "", perf

    thinking_text = content[len(START_THINKING):]
    if END_THINKING not in thinking_text:
        return "", thinking_text.strip(), perf

    thinking, response = thinking_text.split(END_THINKING, 1)
    return response.strip(), thinking.strip(), perf
