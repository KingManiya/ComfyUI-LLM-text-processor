
from __future__ import annotations

import os
from pathlib import Path

import folder_paths


LLM_FOLDER = "llm_text_processor_models"
PROMPT_FOLDER = "llm_text_processor_prompts"
NO_SYSTEM_PROMPT = "none"
NO_MMPROJ = "none"
NO_MODELS_FOUND = "No GGUF models found"


def llm_root() -> Path:
    return Path(folder_paths.models_dir) / "LLM"


def prompt_root() -> Path:
    return llm_root() / "prompts"


def register_folders() -> None:
    llm_dir = llm_root()
    prompts_dir = prompt_root()

    # ComfyUI's add_model_folder_path cannot set extensions for a new key, so we
    # register the custom keys directly with the same tuple shape ComfyUI uses.
    llm_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Get all directories registered under ComfyUI's existing LLM key,
    # including directories registered by custom node extensions.
    llm_dirs = list(folder_paths.get_folder_paths("LLM"))

    # Ensure the default ComfyUI LLM directory is included.
    default_llm_dir = str(llm_dir)
    if default_llm_dir not in llm_dirs:
        llm_dirs.insert(0, default_llm_dir)

    # Remove duplicate directories while preserving search order.
    unique_llm_dirs = []
    seen = set()
    for path in llm_dirs:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique_llm_dirs.append(path)
    llm_dirs = unique_llm_dirs

    # Look for prompts under each registered LLM directory.
    prompt_dirs = [
        str(Path(path) / "prompts")
        for path in llm_dirs
    ]

    # Ensure the default prompts directory is included.
    default_prompts_dir = str(prompts_dir)
    if default_prompts_dir not in prompt_dirs:
        prompt_dirs.insert(0, default_prompts_dir)

    folder_paths.folder_names_and_paths[LLM_FOLDER] = (llm_dirs, {".gguf"})
    folder_paths.folder_names_and_paths[PROMPT_FOLDER] = (prompt_dirs, {".txt"})


def model_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    models = [name for name in files if "mmproj" not in Path(name).name.lower()]
    return models or [NO_MODELS_FOUND]


def mmproj_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    mmproj = [name for name in files if "mmproj" in Path(name).name.lower()]
    return [NO_MMPROJ] + mmproj


def system_prompt_options() -> list[str]:
    files = folder_paths.get_filename_list(PROMPT_FOLDER)
    top_level_files = [name for name in files if os.sep not in name and "/" not in name]
    return [NO_SYSTEM_PROMPT] + top_level_files


def full_model_path(name: str) -> Path:
    if name == NO_MODELS_FOUND:
        raise FileNotFoundError(
            f"No GGUF model files were found in {llm_root()}. Place a .gguf model there and refresh ComfyUI."
        )
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"GGUF model not found: {name}")
    return Path(path)


def full_mmproj_path(name: str) -> Path | None:
    if name == NO_MMPROJ:
        return None
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"mmproj GGUF file not found: {name}")
    return Path(path)


def full_system_prompt_path(name: str) -> Path | None:
    if name == NO_SYSTEM_PROMPT:
        return None
    path = folder_paths.get_full_path(PROMPT_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"System prompt preset not found: {name}")
    return Path(path)


register_folders()
