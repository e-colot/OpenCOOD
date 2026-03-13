# ONNX Build and Test Tutorial

This tutorial explains how to export an OpenCOOD model to ONNX and evaluate AP with ONNX Runtime.

It also documents the ONNX metadata sidecar used by the TensorRT engine builder.

For the consolidated AP measurement workflow and result locations, see `docs/apEvaluation.md`.

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
  --dynamic_shapes \
  --output opencood/v2x-vit/pipeline_a/model.onnx
```

Optional: validate exported ONNX with `onnx.checker`:

```bash
python3 opencood/tools/export_onnx.py \
  --model_dir opencood/v2x-vit \
  --fusion_method intermediate \
  --dynamic_shapes \
  --output opencood/v2x-vit/pipeline_a/model.onnx \
  --validate
```

Notes:

- The exporter scans dataset batches to choose a trace sample and derive TensorRT profile hints.
- `--test` only controls which split is scanned for these profile hints (switches from test to validation split).
- `--test` does not make TensorRT runtime unbounded. Final runtime limits are still defined by engine min/opt/max profiles.
- During search, the exporter prints selected trace statistics (voxels and CAV count).
- Exported outputs are ordered as psm and rm.
- Intermediate fusion models require ONNX opset >= 20 (affine_grid support).
- `--dynamic_shapes` enables dynamic axes in the exported ONNX graph. This is recommended for intermediate fusion because voxel count and cooperative-agent dimensions vary across samples.

### ONNX export artifacts

The exporter now writes two files:

- `<MODEL_DIR>/pipeline_a/model.onnx`
- `<MODEL_DIR>/pipeline_a/model.onnx.meta.yaml`

`model.onnx.meta.yaml` includes:

- `dynamic_shapes` (whether export used dynamic axes)
- input and output names
- traced input shapes and dtypes
- `suggested_trt_profile` (min/opt/max shape strings for `trtexec`)

This sidecar is consumed by `opencood/tools/build_tensorrt_engine.py` to build dynamic-shape TensorRT engines for intermediate fusion without manually writing shape flags.

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

The complete cross-backend AP procedure is documented in `docs/apEvaluation.md`.

## 3. Record results in docs/result.md

Add the ONNX AP metrics and the AP delta versus the PyTorch baseline in `docs/result.md`.

## 4. Next step for TensorRT

After ONNX export, build TensorRT engine directly from ONNX:

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --engine_path <MODEL_DIR>/pipeline_a/model_fp32.engine
```

If `model.onnx.meta.yaml` exists, the script auto-loads dynamic profile hints.

## Troubleshooting

- Missing ONNX Runtime provider:
  Use --providers CPUExecutionProvider in inference_onnx.py.
- Input name mismatch:
  Re-export ONNX from the exact same checkpoint and config used for inference.
- Missing ONNX sidecar metadata for TensorRT build:
  Re-run `export_onnx.py` so `model.onnx.meta.yaml` is regenerated.
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
