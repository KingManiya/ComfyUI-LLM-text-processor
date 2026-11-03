from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import llama_binary

class ExistingInstallTests(unittest.TestCase):
    def test_only_current_release_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            vendor_root = Path(temp)
            for tag in ("old-release", llama_binary.LLAMA_CPP_RELEASE_TAG):
                install_dir = vendor_root / tag / llama_binary.WINDOWS_CUDA_13.key
                install_dir.mkdir(parents=True)
                for name in llama_binary.WINDOWS_CUDA_13.required_files:
                    (install_dir / name).touch()

            with mock.patch.object(llama_binary, "VENDOR_ROOT", vendor_root):
                paths = llama_binary._existing_install(llama_binary.WINDOWS_CUDA_13)

            self.assertEqual(
                paths.cli,
                vendor_root
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.WINDOWS_CUDA_13.key
                / llama_binary.WINDOWS_CUDA_13.cli_executable,
            )


class CustomCliResolutionTests(unittest.TestCase):
    def test_resolve_direct_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cli_path = Path(temp) / "llama-cli"
            cli_path.touch(mode=0o755)
            self.assertEqual(llama_binary._resolve_cli_path(cli_path), cli_path)

    def test_resolve_sibling_llama_server_to_llama_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            server_path = Path(temp) / "llama-server"
            cli_path = Path(temp) / "llama-cli"
            server_path.touch(mode=0o755)
            cli_path.touch(mode=0o755)
            self.assertEqual(llama_binary._resolve_cli_path(server_path), cli_path)

    def test_reject_llama_server_without_sibling_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            server_path = Path(temp) / "llama-server"
            server_path.touch(mode=0o755)
            self.assertIsNone(llama_binary._resolve_cli_path(server_path))

    def test_resolve_directory_direct_and_bin(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dir_path = Path(temp)
            cli_path = dir_path / "llama-cli"
            cli_path.touch(mode=0o755)
            self.assertEqual(llama_binary._resolve_cli_path(dir_path), cli_path)

        with tempfile.TemporaryDirectory() as temp:
            dir_path = Path(temp)
            bin_dir = dir_path / "bin"
            bin_dir.mkdir()
            cli_path = bin_dir / "llama-cli"
            cli_path.touch(mode=0o755)
            self.assertEqual(llama_binary._resolve_cli_path(dir_path), cli_path)

    def test_env_var_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            primary = Path(temp) / "primary" / "llama-cli"
            secondary = Path(temp) / "secondary" / "llama-cli"
            primary.parent.mkdir()
            secondary.parent.mkdir()
            primary.touch(mode=0o755)
            secondary.touch(mode=0o755)

            with mock.patch.dict("os.environ", {
                "LLAMA_CLI_PATH": str(primary),
                "LLAMA_CPP_PATH": str(secondary),
            }):
                self.assertEqual(llama_binary._find_custom_cli(), primary)

    def test_read_env_file_without_mutating_os_environ(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            env_file = Path(temp) / ".env"
            env_file.write_text(
                "# Comment\nLLAMA_CLI_PATH=\"/opt/llama.cpp/llama-cli\"\nOTHER='val'\n",
                encoding="utf-8",
            )
            parsed = llama_binary._read_env_file(env_file)
            self.assertEqual(parsed.get("LLAMA_CLI_PATH"), "/opt/llama.cpp/llama-cli")
            self.assertEqual(parsed.get("OTHER"), "val")

    def test_ensure_llama_cli_paths_missing_actionable_error(self) -> None:
        with mock.patch.object(llama_binary, "_find_custom_cli", return_value=None):
            with mock.patch.object(llama_binary.platform, "system", return_value="Linux"):
                with self.assertRaises(RuntimeError) as ctx:
                    llama_binary.ensure_llama_cli_paths()
                self.assertIn("LLAMA_CLI_PATH", str(ctx.exception))
                self.assertIn("llama-cli", str(ctx.exception))


class BuildCommandMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import importlib
        import types

        model_management = types.ModuleType("comfy.model_management")
        model_management.processing_interrupted = lambda: False
        model_management.throw_exception_if_processing_interrupted = lambda: None
        comfy = types.ModuleType("comfy")
        comfy.model_management = model_management
        package = types.ModuleType("llm_text_processor_pkg")
        package.__path__ = [str(PROJECT_ROOT)]

        with mock.patch.dict(sys.modules, {
            "comfy": comfy,
            "comfy.model_management": model_management,
            "llm_text_processor_pkg": package,
            "llm_text_processor_pkg.llama_binary": llama_binary,
        }):
            cls.llama_cli = importlib.import_module("llm_text_processor_pkg.llama_cli")

    def _build(self, memory_mode: str, n_gpu_layers: int = 99, n_cpu_moe_layers: int = 99, extra_args: list[str] | None = None) -> list[str]:
        with mock.patch.object(
            self.llama_cli,
            "ensure_llama_cli_paths",
            return_value=llama_binary.LlamaCliPaths(cli=Path("/mock/llama-cli")),
        ):
            cmd, _ = self.llama_cli.build_command(
                model_path=Path("/mock/model.gguf"),
                mmproj_path=None,
                system_prompt_path=None,
                image=None,
                prompt="hello",
                max_tokens=100,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repeat_penalty=1.0,
                ctx_size=2048,
                memory_mode=memory_mode,
                n_gpu_layers=n_gpu_layers,
                n_cpu_moe_layers=n_cpu_moe_layers,
                seed=42,
                reasoning="off",
                extra_args=extra_args,
            )
            return cmd

    def test_warmup_disabled_by_default(self) -> None:
        cmd = self._build(memory_mode="auto")
        self.assertIn("--no-warmup", cmd)

    def test_cpu_moe_layers_mode_passes_ngl_and_cpu_moe(self) -> None:
        cmd = self._build(memory_mode="cpu_moe_layers", n_gpu_layers=99, n_cpu_moe_layers=99)
        self.assertIn("-ngl", cmd)
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "99")
        self.assertIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)

    def test_cpu_moe_layers_mode_passes_numeric_moe(self) -> None:
        cmd = self._build(memory_mode="cpu_moe_layers", n_gpu_layers=40, n_cpu_moe_layers=16)
        self.assertIn("-ngl", cmd)
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "40")
        self.assertIn("--n-cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "16")
        self.assertNotIn("--cpu-moe", cmd)

    def test_gpu_layers_mode_passes_ngl_only(self) -> None:
        cmd = self._build(memory_mode="gpu_layers", n_gpu_layers=99, n_cpu_moe_layers=99)
        self.assertIn("-ngl", cmd)
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)

    def test_auto_mode_passes_neither(self) -> None:
        cmd = self._build(memory_mode="auto", n_gpu_layers=99, n_cpu_moe_layers=99)
        self.assertNotIn("-ngl", cmd)
        self.assertNotIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)

    def test_extra_args_appended_after_defaults(self) -> None:
        cmd = self._build(memory_mode="auto", extra_args=["--warmup"])
        self.assertIn("--warmup", cmd)
        self.assertGreater(cmd.index("--warmup"), cmd.index("--no-warmup"))


if __name__ == "__main__":
    unittest.main()
