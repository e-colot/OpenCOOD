# ONNX Build and Test Tutorial

This tutorial explains how to export an OpenCOOD model to ONNX and evaluate AP with ONNX Runtime.

## Prerequisites

- Working OpenCOOD environment.
- A trained checkpoint directory (for example in logs or opencood/v2x-vit).
- ONNX Runtime installed in the Python environment.

Example install:

```bash
pip install onnx onnxruntime-gpu
```

## 1. Export PyTorch model to ONNX

```bash
python3 opencood/tools/export_onnx.py \
  --model_dir opencood/v2x-vit \
  --fusion_method intermediate \
  --output opencood/v2x-vit/pipeline_a/model.onnx
```

Optional: validate exported ONNX with `onnx.checker`:

```bash
python3 opencood/tools/export_onnx.py \
  --model_dir opencood/v2x-vit \
  --fusion_method intermediate \
  --output opencood/v2x-vit/pipeline_a/model.onnx \
  --validate
```

Notes:

- The exporter uses one batch from the dataset split configured in the model config under `model_dir` (no split override flag in exporter).
- During this search, the exporter prints `Selected trace batch total CAV: N`.
  Here `CAV` means cooperative agents (ego + connected vehicles/infrastructure) in that traced sample.
  The script picks the batch with the largest total CAV count (within `--max_search_batches`) to better capture the richest intermediate-fusion input shape during export.
- Exported outputs are ordered as psm and rm.
- Intermediate fusion models require ONNX opset >= 20 (affine_grid support).

## 2. Run ONNX Runtime inference and AP evaluation

```bash
python3 opencood/tools/inference_onnx.py \
  --model_dir opencood/v2x-vit \
  --onnx_model opencood/v2x-vit/pipeline_a/model.onnx \
  --fusion_method intermediate \
  --output_yaml pipeline_a/eval_onnx.yaml \
  --test
```

Output:

- AP metrics are saved to <MODEL_DIR>/pipeline_a/eval_onnx.yaml.

## 3. Record results in docs/result.md

Add the ONNX AP metrics and the AP delta versus the PyTorch baseline in `docs/result.md`.

## Troubleshooting

- Missing ONNX Runtime provider:
  Use --providers CPUExecutionProvider in inference_onnx.py.
- Input name mismatch:
  Re-export ONNX from the exact same checkpoint and config used for inference.
- Intermediate fusion export errors:
  Start with early or late fusion first, then add intermediate after parity checks.
- Intermediate V2X-ViT export status:
  In this repository state, intermediate V2X-ViT export completes successfully with the provided `export_onnx.py` path.
  If you still see `RuntimeError: Only consecutive 1-d tensor indices are supported in exporting aten::index_put to ONNX`, confirm you are using the current exporter and model code from this branch.
- If you override opset and see:
  UnsupportedOperatorError: Exporting the operator 'aten::affine_grid_generator' ...
  rerun export with --opset 20 (or higher).

## Export Warnings Status

Warnings fixed in this repo:

- Removed tensor-to-NumPy split logic from feature regroup path used by intermediate fusion export.
- Removed explicit symbolic-int cast in BEV backbone debug stride bookkeeping during ONNX export.
- Removed PFN tensor-to-bool branch warning in ONNX export mode.

Warnings that currently remain (not hidden):

- `point_pillar_scatter.py`: `batch_size = int(...sum().item())` emits a tracer warning.
  - Why it remains: this scatter path needs Python-sized allocation/loop boundaries with ragged per-agent voxel placement.
  - Patchability: partially patchable only with a larger refactor of scatter layout assumptions; not a small safe patch.
  - Impact: with fixed export configuration/checkpoint and split, generated ONNX runs correctly for the traced graph.
- `torch_transformation_utils.py`: warnings now mainly report `torch.as_tensor(...)` being captured as constants.
  - Why it remains: image-size terms are shape-derived constants at trace time in the current PyTorch ONNX tracer path.
  - Patchability: low practical value to remove completely; these are informational for this export path.
  - Impact: export succeeds and runtime inference works for the exported graph.
- PyTorch ONNX internal warnings (`constant folding Slice steps=1`, advanced indexing lowering note).
  - Why it remains: emitted by PyTorch ONNX internals/symbolics for opset 20 graph lowering.
  - Impact: informational in this workflow; export still completes and graph is valid.
