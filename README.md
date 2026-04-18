# ComfyUI Qwen GGUF

Run Qwen GGUF models inside ComfyUI through official llama.cpp release binaries.

This node is intentionally small: it discovers local `.gguf` models from
`ComfyUI/models/LLM`, downloads the llama.cpp CLI binary on first use, and calls
`llama-mtmd-cli` as a subprocess.

## Features

- Qwen text generation with GGUF models
- Optional image input through llama.cpp multimodal support
- Separate `RESPONSE` and `THINKING` outputs
- `/think` and `/no_think` control for Qwen reasoning models
- System prompt presets from text files
- Recursive model discovery from `ComfyUI/models/LLM`
- Automatic llama.cpp binary download for Windows x64 CUDA 13
- Advanced escape hatch for extra llama.cpp CLI flags

## Installation

Clone or copy this folder into your ComfyUI custom nodes directory:

```bash
ComfyUI/custom_nodes/ComfyUI-Qwen-gguf
```

Restart ComfyUI. The node appears under:

```text
Qwen/GGUF -> Qwen GGUF (llama.cpp)
```

No Python package install is required for the basic node. ComfyUI already ships
with the runtime pieces used here. Pillow and NumPy are used when an image input
is connected.

## llama.cpp Auto Download

On the first node execution, the extension downloads official llama.cpp release
assets from:

```text
https://github.com/ggml-org/llama.cpp/releases
```

For v1, automatic download supports:

```text
Windows x64 + CUDA 13
```

The node downloads both release archives:

```text
llama-*-bin-win-cuda-13*-x64.zip
cudart-llama-bin-win-cuda-13.1-x64.zip
```

Recent llama.cpp releases may name the main Windows CUDA 13 binary with the
exact CUDA runtime version, for example `llama-b8838-bin-win-cuda-13.1-x64.zip`.

They are extracted into:

```text
ComfyUI/custom_nodes/ComfyUI-Qwen-gguf/vendor/llama.cpp/<release-tag>/win-x64-cuda13
```

A `manifest.json` is written next to the extracted files, so later runs reuse the
same binary instead of downloading again.

Linux, macOS, CPU-only, and other CUDA builds are not auto-downloaded yet. The
platform layer is kept isolated so those targets can be added cleanly later.

## Model Placement

Put your Qwen GGUF files anywhere under:

```text
ComfyUI/models/LLM
```

Example:

```text
ComfyUI/models/LLM/My-Qwen-Model/model-q4_k_m.gguf
ComfyUI/models/LLM/My-Qwen-Model/mmproj-BF16.gguf
```

The `model` dropdown shows recursive `.gguf` files, excluding files with
`mmproj` in the filename. The `mmproj` dropdown shows `none` plus recursive
`.gguf` files with `mmproj` in the filename.

The node downloads llama.cpp binaries only. It does not download Qwen model
weights.

## System Prompt Presets

Create text files in:

```text
ComfyUI/models/LLM/prompts
```

Example:

```text
ComfyUI/models/LLM/prompts/captioner.txt
```

Each top-level `.txt` file appears in the `system_prompt` dropdown. Choose
`None` to run without a system prompt.

## Node Inputs

| Input | Description |
| --- | --- |
| `model` | GGUF model file from `ComfyUI/models/LLM`. |
| `mmproj` | Vision projector GGUF. Required when using image input. |
| `system_prompt` | Prompt preset from `models/LLM/prompts`, or `None`. |
| `prompt` | User prompt sent to the model. |
| `max_tokens` | Maximum generated tokens. |
| `temperature` | Sampling temperature. Lower values are more deterministic. |
| `top_p` | Nucleus sampling threshold. |
| `top_k` | Top-K sampling cutoff. |
| `repeat_penalty` | Penalty for repeated tokens. |
| `ctx_size` | llama.cpp context size. |
| `n_gpu_layers` | Layers offloaded to GPU. Use `0` for CPU. |
| `seed` | Random seed. |
| `enable_thinking` | Uses `/think` when enabled and `/no_think` when disabled. |
| `timeout_seconds` | Maximum runtime before the subprocess is stopped. |
| `image` | Optional ComfyUI image input. Uses the first image in a batch. |
| `extra_args` | Advanced llama.cpp CLI flags appended to the command. |

Every input includes an in-node tooltip.

## Node Outputs

| Output | Description |
| --- | --- |
| `RESPONSE` | Final model response with Qwen thinking blocks removed. |
| `THINKING` | Extracted `<think>...</think>` reasoning when thinking is enabled. |

Both outputs include ComfyUI output tooltips.

## Image Input and mmproj

For image workflows, connect an `IMAGE` input and choose the matching `mmproj`
file. The node saves the first image in the batch as a temporary PNG and passes
it to llama.cpp with `--image`.

If no image is connected, `mmproj` can stay set to `none`. You may still select a
projector manually if you want llama.cpp to receive `--mmproj`.

## Troubleshooting

### No models appear

Place at least one `.gguf` model under:

```text
ComfyUI/models/LLM
```

Then refresh or restart ComfyUI.

### Image input fails

Make sure `mmproj` is not set to `none` and that the projector belongs to the
same model family as the selected GGUF model.

### Download fails

Check your internet connection and GitHub access. You can delete the incomplete
folder under `vendor/llama.cpp` and run the node again.

### Unsupported platform

Automatic binary download currently supports Windows x64 CUDA 13 only. Other
platforms need a future platform mapping entry.

### llama.cpp errors

Use `extra_args` only for flags supported by your downloaded llama.cpp build.
If generation times out, increase `timeout_seconds` or lower `max_tokens`.

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Qwen](https://github.com/QwenLM)
