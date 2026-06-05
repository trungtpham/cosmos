# Cosmos3 Generator Transfer Examples

Cosmos3-Nano video **transfer** examples on the native PyTorch (Cosmos Framework) path.
Sample assets under [`assets/`](./assets) cover spatial control signals paired with
`prompt.json` files:

- **Edge (Canny)** — edge map control plus caption.
- **Blur** — blurred-reference control plus caption.
- **Depth** — depth map control plus caption.
- **Segmentation** — segmentation map control plus caption.
- **World scenario (WSM)** — world-scenario map control plus caption.

vLLM-Omni does not expose transfer controls today.

Environment setup is centralized in the shared
[Cosmos3 cookbooks environment setup](../../README.md) guide.

## Transfer Definition

Video transfer generates a target clip from a `prompt.json` caption and a precomputed
control video on the hint block (`control_path`). Inference uses `model_mode` `video2video`;
there is no `vision_path` or source RGB video at run time. Frame count, fps, and resolution
come from the input spec and the control video geometry. All examples share
`assets/negative_prompt.json` for the negative caption.

| Control | Asset folder | Inference input | Generation duration |
| --- | --- | --- | --- |
| Edge (Canny) | `assets/edge/` | `control_edge.mp4` + `prompt.json` | 121 frames @ 24 FPS |
| Blur | `assets/blur/` | `control_blur.mp4` + `prompt.json` | 121 frames @ 24 FPS |
| Depth | `assets/depth/` | `control_depth.mp4` + `prompt.json` | 121 frames @ 24 FPS |
| Segmentation | `assets/seg/` | `control_seg.mp4` + `prompt.json` | 121 frames @ 24 FPS |
| World scenario (WSM) | `assets/wsm/` | `control_wsm.mp4` + `prompt.json` | 101 frames @ 10 FPS |

Transfer inference is selected automatically when any hint key is present in the spec.

## Run with Cosmos Framework

### Quickstart

Set up the environment: [Cosmos Framework setup](../../README.md#cosmos-framework).
Activate the framework venv, then run inference (checked-in `specs/*.json` use paths
relative to `specs/`). Transfer on Nano looks like:

```bash
cd cookbooks/cosmos3/generator/transfer

# edge
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  --no-use-torch-compile \
  -i specs/edge.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2025

# blur
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  --no-use-torch-compile \
  -i specs/blur.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2025

# depth
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  --no-use-torch-compile \
  -i specs/depth.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2025

# seg
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  --no-use-torch-compile \
  -i specs/seg.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2025

# wsm
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  --no-guardrails \
  --no-use-torch-compile \
  -i specs/wsm.json \
  -o ./output/ \
  --checkpoint-path Cosmos3-Nano \
  --seed 2025
```

The input spec sets `prompt_path` and a hint block with `control_path` pointing at the
checked-in assets under [`assets/`](./assets) via paths relative to [`specs/`](./specs).

To run one or more controls from this directory:


Outputs are written under `outputs/<control>/` by default (`vision.mp4`, `sample_args.json`,
`console.log`). Batch size must be 1 for transfer.

### Cookbook entrypoints

- [`run_video_transfer_with_cosmos_framework.ipynb`](./run_video_transfer_with_cosmos_framework.ipynb) —
  full tutorial on a **GPU host**: environment setup, `nvidia-smi` check, then five inference blocks
  (edge, blur, depth, seg, wsm) with previews. See [Cosmos3 environment setup](../../README.md).
- [`specs/`](./specs) — checked-in Framework input JSON per control (paths relative to `specs/`).
