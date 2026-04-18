from __future__ import annotations

import fnmatch
import json
import platform
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


RELEASE_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    executable: str
    asset_patterns: tuple[str, ...]


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    executable="llama-mtmd-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13.1-x64.zip",
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
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Qwen-gguf"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Qwen-gguf"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


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


def _manifest_path(install_dir: Path) -> Path:
    return install_dir / "manifest.json"


def _find_executable(install_dir: Path, spec: PlatformSpec) -> Path | None:
    for path in install_dir.rglob(spec.executable):
        if path.is_file():
            return path
    return None


def _existing_from_manifest(spec: PlatformSpec) -> Path | None:
    if not VENDOR_ROOT.exists():
        return None
    for manifest in VENDOR_ROOT.glob(f"*/{spec.key}/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        exe = Path(data.get("executable", ""))
        if exe.is_file() and exe.name.lower() == spec.executable.lower():
            return exe
    return None


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="qwen-llama-download-") as temp:
        temp_dir = Path(temp)
        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[Qwen GGUF] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(install_dir)


def ensure_llama_cli() -> Path:
    spec = _platform_spec()
    existing = _existing_from_manifest(spec)
    if existing is not None:
        return existing

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or "latest"
    install_dir = VENDOR_ROOT / tag / spec.key
    manifest = _manifest_path(install_dir)

    executable = _find_executable(install_dir, spec)
    if executable is not None:
        return executable

    assets = _select_assets(release, spec)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_assets(assets, install_dir)

    executable = _find_executable(install_dir, spec)
    if executable is None:
        raise RuntimeError(
            f"Downloaded llama.cpp assets but could not find {spec.executable} in {install_dir}"
        )

    manifest.write_text(
        json.dumps(
            {
                "tag": tag,
                "release_url": release.get("html_url", "https://github.com/ggml-org/llama.cpp/releases"),
                "assets": [asset["name"] for asset in assets],
                "executable": str(executable),
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return executable
