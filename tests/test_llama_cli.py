from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import llama_binary


def import_llama_cli():
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
        return importlib.import_module("llm_text_processor_testpkg.llama_cli")


class ParseResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.llama_cli = import_llama_cli()

    def test_ignores_backend_logs_from_stderr(self) -> None:
        stdout = "A concise generated prompt."
        stderr = "\n".join(
            [
                "load_backend: loaded Vulkan backend from /tmp/libggml-vulkan.so",
                "load_backend: loaded CPU backend from /tmp/libggml-cpu-zen4.so",
                "llama_memory_breakdown_print: | memory breakdown [MiB] |",
                "[ Prompt: 12.34 toks/s | Generation: 56.78 toks/s ]",
            ]
        )

        response, reasoning, perf = self.llama_cli._parse_response(stdout, stderr)

        self.assertEqual(response, "A concise generated prompt.")
        self.assertEqual(reasoning, "")
        self.assertEqual(perf, "[ Prompt: 12.34 toks/s | Generation: 56.78 toks/s ]")

    def test_keeps_thinking_split_when_logs_are_in_stderr(self) -> None:
        stdout = "[Start thinking]scratchpad[End thinking]final answer"
        stderr = "load_backend: loaded Vulkan backend from /tmp/libggml-vulkan.so"

        response, reasoning, perf = self.llama_cli._parse_response(stdout, stderr)

        self.assertEqual(response, "final answer")
        self.assertEqual(reasoning, "scratchpad")
        self.assertEqual(perf, "")

    def test_removes_perf_when_it_is_printed_to_stdout(self) -> None:
        stdout = "final answer\n[ Prompt: 10.00 toks/s | Generation: 20.00 toks/s ]"

        response, reasoning, perf = self.llama_cli._parse_response(stdout)

        self.assertEqual(response, "final answer")
        self.assertEqual(reasoning, "")
        self.assertEqual(perf, "[ Prompt: 10.00 toks/s | Generation: 20.00 toks/s ]")


if __name__ == "__main__":
    unittest.main()
