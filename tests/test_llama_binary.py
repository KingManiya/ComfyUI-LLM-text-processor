from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import llama_binary


def release_with_assets(*names: str) -> dict:
    return {
        "assets": [
            {"name": name, "browser_download_url": f"https://example.test/{name}"}
            for name in names
        ]
    }


class PlatformSpecTests(unittest.TestCase):
    def test_windows_x64_selects_cuda13(self) -> None:
        with mock.patch("platform.system", return_value="Windows"):
            with mock.patch("platform.machine", return_value="AMD64"):
                self.assertEqual(llama_binary._platform_spec(), llama_binary.WINDOWS_CUDA_13)

    def test_linux_x64_defaults_to_cpu(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch("platform.machine", return_value="x86_64"):
                    self.assertEqual(llama_binary._platform_spec(), llama_binary.UBUNTU_X64_CPU)

    def test_linux_x64_selects_backend_from_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {llama_binary.LLAMA_BACKEND_ENV: "vulkan"},
            clear=True,
        ):
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch("platform.machine", return_value="x86_64"):
                    self.assertEqual(llama_binary._platform_spec(), llama_binary.UBUNTU_X64_VULKAN)

    def test_linux_x64_rejects_unknown_backend(self) -> None:
        with mock.patch.dict(
            os.environ,
            {llama_binary.LLAMA_BACKEND_ENV: "cuda"},
            clear=True,
        ):
            with mock.patch("platform.system", return_value="Linux"):
                with mock.patch("platform.machine", return_value="x86_64"):
                    with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                        llama_binary._platform_spec()


class AssetSelectionTests(unittest.TestCase):
    def test_selects_ubuntu_cpu_tarball(self) -> None:
        release = release_with_assets(
            "llama-b8840-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b8840-bin-ubuntu-x64.tar.gz",
        )

        assets = llama_binary._select_assets(release, llama_binary.UBUNTU_X64_CPU)

        self.assertEqual([asset["name"] for asset in assets], ["llama-b8840-bin-ubuntu-x64.tar.gz"])

    def test_selects_windows_runtime_and_cudart(self) -> None:
        release = release_with_assets(
            "llama-b8840-bin-win-cuda-13.1-x64.zip",
            "cudart-llama-bin-win-cuda-13.1-x64.zip",
        )

        assets = llama_binary._select_assets(release, llama_binary.WINDOWS_CUDA_13)

        self.assertEqual(
            [asset["name"] for asset in assets],
            [
                "llama-b8840-bin-win-cuda-13.1-x64.zip",
                "cudart-llama-bin-win-cuda-13.1-x64.zip",
            ],
        )


class ArchiveExtractionTests(unittest.TestCase):
    def test_extracts_zip_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            archive_path = temp_dir / "llama.zip"
            install_dir = temp_dir / "install"
            install_dir.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("llama-b8840/llama-cli.exe", "binary")

            llama_binary._extract_archive(archive_path, install_dir)

            self.assertEqual((install_dir / "llama-b8840" / "llama-cli.exe").read_text(), "binary")

    def test_extracts_tar_gz_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            archive_path = temp_dir / "llama.tar.gz"
            install_dir = temp_dir / "install"
            install_dir.mkdir()

            payload = b"binary"
            tar_info = tarfile.TarInfo("llama-b8840/llama-cli")
            tar_info.size = len(payload)
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.addfile(tar_info, io.BytesIO(payload))

            llama_binary._extract_archive(archive_path, install_dir)

            self.assertEqual((install_dir / "llama-b8840" / "llama-cli").read_bytes(), payload)

    def test_rejects_unsafe_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)
            archive_path = temp_dir / "llama.tar.gz"
            install_dir = temp_dir / "install"
            install_dir.mkdir()

            payload = b"bad"
            tar_info = tarfile.TarInfo("../escape")
            tar_info.size = len(payload)
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.addfile(tar_info, io.BytesIO(payload))

            with self.assertRaisesRegex(RuntimeError, "unsafe archive member"):
                llama_binary._extract_archive(archive_path, install_dir)


class InstallDiscoveryTests(unittest.TestCase):
    def test_linux_cli_is_marked_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            install_dir = Path(temp)
            cli = install_dir / "llama-cli"
            cli.write_text("binary")
            cli.chmod(0o644)

            with mock.patch.object(llama_binary.os, "name", "posix"):
                with mock.patch.object(Path, "chmod", autospec=True, wraps=Path.chmod) as chmod_mock:
                    paths = llama_binary._find_cli_paths(install_dir, llama_binary.UBUNTU_X64_CPU)

            self.assertEqual(paths, llama_binary.LlamaCliPaths(cli=cli))
            self.assertTrue(any(call.args[0] == cli for call in chmod_mock.call_args_list))


class LinuxCommandTests(unittest.TestCase):
    def test_build_command_uses_linux_cli_path(self) -> None:
        model_management = types.ModuleType("comfy.model_management")
        model_management.processing_interrupted = lambda: False
        model_management.throw_exception_if_processing_interrupted = lambda: None

        comfy = types.ModuleType("comfy")
        comfy.model_management = model_management
        package = types.ModuleType("llm_text_processor_testpkg")
        package.__path__ = [str(PROJECT_ROOT)]

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": comfy,
                "comfy.model_management": model_management,
                "llm_text_processor_testpkg": package,
                "llm_text_processor_testpkg.llama_binary": llama_binary,
            },
        ):
            sys.modules.pop("llm_text_processor_testpkg.llama_cli", None)
            import importlib

            llama_cli = importlib.import_module("llm_text_processor_testpkg.llama_cli")

        with mock.patch.object(
            llama_cli,
            "ensure_llama_cli_paths",
            return_value=llama_binary.LlamaCliPaths(cli=Path("/opt/llama/llama-cli")),
        ):
            command, cleanup_paths = llama_cli.build_command(
                model_path=Path("/models/model.gguf"),
                mmproj_path=None,
                system_prompt_path=None,
                image=None,
                prompt="Hello",
                max_tokens=32,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.0,
                ctx_size=2048,
                memory_mode="auto",
                n_gpu_layers=99,
                n_cpu_moe_layers=1,
                seed=1,
                reasoning="off",
            )

        self.assertEqual(command[0], "/opt/llama/llama-cli")
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "/models/model.gguf")
        for path in cleanup_paths:
            if path and path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
