from __future__ import annotations

import fnmatch
import json
import os
import platform
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


LLAMA_CPP_RELEASE_TAG = "b10472"
RELEASE_API_URL = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_RELEASE_TAG}"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    cli_executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class LlamaCliPaths:
    cli: Path


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    cli_executable="llama-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-cli.exe",
        "ggml-cuda.dll",
        "cudart64_13.dll",
    ),
)


def _platform_spec() -> PlatformSpec:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return WINDOWS_CUDA_13
    raise RuntimeError(
        "Automatic llama.cpp binary download currently supports Windows x64 CUDA 13 only. "
        "Other platforms are intentionally isolated behind the platform mapping for future support."
    )


def _json_get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM-text-processor"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_size(num_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LLM-text-processor"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size is not None else None
        downloaded = 0
        chunk_size = 1024 * 256
        started_at = time.monotonic()
        last_reported_at = started_at

        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_reported_at < 1.0:
                    continue

                elapsed = max(now - started_at, 0.001)
                speed = downloaded / elapsed
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} / {_format_size(total_size)} "
                        f"({percent:.1f}%) at {_format_size(speed)}/s"
                    )
                else:
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} at {_format_size(speed)}/s"
                    )
                last_reported_at = now

        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = downloaded / elapsed
        if total_size:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} / {_format_size(total_size)} "
                f"(100.0%) at {_format_size(speed)}/s"
            )
        else:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} at {_format_size(speed)}/s"
            )


def _select_assets(release: dict, spec: PlatformSpec) -> list[dict]:
    assets = release.get("assets", [])
    selected = []
    used_names = set()

    # Match explicit release asset names so a future platform can add patterns
    # without changing the download/extract pipeline.
    for pattern in spec.asset_patterns:
        matches = [
            asset for asset in assets
            if fnmatch.fnmatch(asset.get("name", "").lower(), pattern.lower())
        ]
        if not matches:
            raise RuntimeError(f"Could not find llama.cpp release asset matching: {pattern}")
        asset = sorted(matches, key=lambda item: item.get("name", ""))[0]
        if asset["name"] not in used_names:
            selected.append(asset)
            used_names.add(asset["name"])
    return selected


def _find_file(install_dir: Path, name: str) -> Path | None:
    for path in install_dir.rglob(name):
        if path.is_file():
            return path
    return None


def _find_cli_paths(install_dir: Path, spec: PlatformSpec) -> LlamaCliPaths | None:
    cli = _find_file(install_dir, spec.cli_executable)
    if cli is None:
        return None
    return LlamaCliPaths(cli=cli)


def _has_required_files(install_dir: Path, spec: PlatformSpec) -> bool:
    for name in spec.required_files:
        if not any(path.is_file() for path in install_dir.rglob(name)):
            return False
    return True


def _is_complete_install(install_dir: Path, spec: PlatformSpec) -> bool:
    return _find_cli_paths(install_dir, spec) is not None and _has_required_files(install_dir, spec)


def _existing_install(spec: PlatformSpec) -> LlamaCliPaths | None:
    install_dir = VENDOR_ROOT / LLAMA_CPP_RELEASE_TAG / spec.key
    return _find_cli_paths(install_dir, spec) if _is_complete_install(install_dir, spec) else None


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="llm-text-processor-llama-download-") as temp:
        temp_dir = Path(temp)
        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[LLM Text Processor] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(install_dir)


ENV_CONFIG_VARS = (
    "LLAMA_CLI_PATH",
    "LLAMA_INTEGRATION_CLI",
    "LLAMA_SERVER_PATH",
    "LLAMA_BIN_DIR",
    "LLAMA_CPP_PATH",
)


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key and not key.startswith("#"):
                values[key.strip()] = value.strip().strip("\"'")
    except OSError:
        pass
    return values


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if platform.system().lower() == "windows":
        return True
    return os.access(path, os.X_OK)


def _resolve_cli_path(candidate: Path | str | None) -> Path | None:
    if not candidate:
        return None
    candidate_str = os.path.expandvars(os.path.expanduser(str(candidate).strip()))
    if not candidate_str:
        return None
    path = Path(candidate_str)
    try:
        path = path.resolve()
    except OSError:
        pass
    if not path.exists():
        return None

    cli_names = (
        ("llama-cli.exe", "llama-cli")
        if platform.system().lower() == "windows"
        else ("llama-cli", "llama-cli.exe")
    )

    if path.is_dir():
        for name in cli_names:
            target = path / name
            if _is_executable(target):
                return target
            target_bin = path / "bin" / name
            if _is_executable(target_bin):
                return target_bin
        return None

    if path.name.lower() in {"llama-cli", "llama-cli.exe"}:
        return path if _is_executable(path) else None

    # If file is llama-server: inspect sibling for llama-cli. Strictly do NOT return llama-server.
    if path.name.lower() in {"llama-server", "llama-server.exe"}:
        for name in cli_names:
            sibling = path.parent / name
            if _is_executable(sibling):
                return sibling
        return None

    # Any other directly specified executable file (e.g. custom wrapper or renamed cli)
    if _is_executable(path):
        return path

    return None


def _find_custom_cli() -> Path | None:
    env_file_values: dict[str, str] = {}
    for env_path in (PACKAGE_ROOT / ".env", PACKAGE_ROOT / "tests" / ".env"):
        if env_path.is_file():
            env_file_values = _read_env_file(env_path)
            if env_file_values:
                break

    for var in ENV_CONFIG_VARS:
        val = os.environ.get(var) or env_file_values.get(var)
        resolved = _resolve_cli_path(val)
        if resolved is not None:
            return resolved

    cli_which = shutil.which("llama-cli")
    if cli_which:
        resolved = _resolve_cli_path(cli_which)
        if resolved is not None:
            return resolved

    server_which = shutil.which("llama-server")
    if server_which:
        resolved = _resolve_cli_path(server_which)
        if resolved is not None:
            return resolved

    return _resolve_cli_path(PACKAGE_ROOT / "bin")


def ensure_llama_cli_paths() -> LlamaCliPaths:
    custom_cli = _find_custom_cli()
    if custom_cli is not None:
        return LlamaCliPaths(cli=custom_cli)

    try:
        spec = _platform_spec()
    except RuntimeError as err:
        raise RuntimeError(
            "No llama-cli executable found on this system.\n"
            f"{err}\n"
            "To use LLM Text Processor on Linux / macOS:\n"
            "  1. Ensure 'llama-cli' is installed and available on your PATH, or\n"
            "  2. Set LLAMA_CLI_PATH (or LLAMA_SERVER_PATH) in your environment or in a .env file "
            "pointing to your llama-cli binary or directory."
        ) from err

    existing = _existing_install(spec)
    if existing is not None:
        return existing

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or LLAMA_CPP_RELEASE_TAG
    install_dir = VENDOR_ROOT / tag / spec.key

    if _is_complete_install(install_dir, spec):
        paths = _find_cli_paths(install_dir, spec)
        if paths is None:
            raise RuntimeError(f"Completed install has incomplete CLI executables: {install_dir}")
        return paths

    assets = _select_assets(release, spec)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_assets(assets, install_dir)

    paths = _find_cli_paths(install_dir, spec)
    if paths is None:
        raise RuntimeError(
            f"Downloaded llama.cpp assets but could not find CLI executables in {install_dir}"
        )
    if not _has_required_files(install_dir, spec):
        missing = [
            name for name in spec.required_files
            if not any(path.is_file() for path in install_dir.rglob(name))
        ]
        raise RuntimeError(f"Downloaded llama.cpp assets are incomplete; missing: {', '.join(missing)}")

    return paths
