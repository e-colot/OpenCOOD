# Engine Generation Tutorial

Two TensorRT generation pipelines are supported:

- PyTorch -> ONNX -> TensorRT
- PyTorch -> TensorRT

This page gives the minimal commands needed to generate engine artifacts.
Detailed backend-specific behavior is documented in:

- `docs/onnx.md`
- `docs/tensorRT.md`
- `docs/apEvaluation.md`

## Goal

Produce deployable inference artifacts for later AP/runtime validation.

## Pipeline A: PyTorch -> ONNX -> TensorRT

### Step 1. Export ONNX

```bash
python3 opencood/tools/export_onnx.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --dynamic_shapes \
  --output <MODEL_DIR>/pipeline_a/model.onnx \
  --validate
```

Basic explanation:

- Exports the PyTorch checkpoint to ONNX.
- `--dynamic_shapes` guarantees dynamic input/output axes in ONNX export.
- Also writes `<MODEL_DIR>/pipeline_a/model.onnx.meta.yaml`.
- For intermediate fusion, that metadata contains suggested TensorRT dynamic shape profiles.

For more detail, see `docs/onnx.md`.

### Step 2. Build TensorRT engine from ONNX

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx
```

Basic explanation:

- Builds a TensorRT engine with `trtexec`.
- If `--engine_path` is not provided, output defaults to `<onnx_dir>/<onnx_name>.engine`.
- If `model.onnx.meta.yaml` exists, the build script auto-loads dynamic profile hints.
- This is the recommended path for intermediate fusion.

### Deployment Profile Sizing

TensorRT dynamic engines are still bounded by optimization profiles.

For intermediate fusion, the main limiter is usually voxel count on:

- `voxel_features`
- `voxel_coords`
- `voxel_num_points`

Recommended deployment safety setting:

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --max_voxels_margin 2
```

Optional explicit absolute cap:

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --max_voxels 30000
```

Both flags can be combined and the larger value is used.

The builder always prints detected and final max voxel profile counts.

Example with larger deployment headroom:

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --max_voxels_margin 2
```

Optional absolute cap (can be combined with margin):

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --max_voxels_margin 2 \
  --max_voxels 30000
```

When both are provided, the builder uses the larger value for each voxel-related input.

Optional FP16 build:

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --engine_path <MODEL_DIR>/pipeline_a/model_fp16.engine \
  --fp16
```

For more detail, see `docs/tensorRT.md`.

## Pipeline B: PyTorch -> TensorRT

### Step 1. Build direct TensorRT artifact

```bash
python3 opencood/tools/build_tensorrt_engine_direct.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --output <MODEL_DIR>/pipeline_b/model_trt.ts \
  --dynamic_shapes \
  --max_search_batches 200 \
  --max_voxels_margin 2 \
  --precision fp32 \
  --test
```

Basic explanation:

- Compiles the PyTorch model directly to a Torch-TensorRT artifact.
- Supports dynamic input shape ranges when `--dynamic_shapes` is enabled.
- If `<MODEL_DIR>/pipeline_a/model.onnx.meta.yaml` is present, direct build reuses that profile and skips full Stage 3 scan.
- You can pass a custom metadata file with `--profile_yaml <PATH_TO_META_YAML>`.
- Stage 3 profile search scans the dataset; cap it with `--max_search_batches` (for example 200) to avoid long scans.
- Use `--scan_log_interval` to print periodic Stage 3 scan progress (default: every 50 batches).
- `--max_voxels_margin` and `--max_voxels` work the same way as in the ONNX -> TensorRT builder.
- Also writes `<MODEL_DIR>/pipeline_b/model_trt.ts.meta.yaml` with input ordering, precision, and profile metadata.

Optional FP16 build:

```bash
python3 opencood/tools/build_tensorrt_engine_direct.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --output <MODEL_DIR>/pipeline_b/model_trt_fp16.ts \
  --dynamic_shapes \
  --max_search_batches 200 \
  --max_voxels_margin 2 \
  --precision fp16 \
  --test
```

For more detail, see `docs/tensorRT.md`.

## Generated Artifacts

- ONNX path:
  - `<MODEL_DIR>/pipeline_a/model.onnx`
  - `<MODEL_DIR>/pipeline_a/model.onnx.meta.yaml`
  - default: `<MODEL_DIR>/pipeline_a/model.engine`
  - or custom path passed with `--engine_path`
  - optionally `<MODEL_DIR>/pipeline_a/model_fp16.engine`
- Direct TensorRT path:
  - `<MODEL_DIR>/pipeline_b/model_trt.ts`
  - `<MODEL_DIR>/pipeline_b/model_trt.ts.meta.yaml`
  - optionally `<MODEL_DIR>/pipeline_b/model_trt_fp16.ts`

## Validation

AP evaluation commands for ONNX Runtime, ONNX -> TensorRT, and direct TensorRT are documented in `docs/apEvaluation.md`.

## Notes For Intermediate Fusion

- Intermediate fusion uses dynamic fusion dimensions.
- The ONNX -> TensorRT path is the preferred engine-generation route when you want explicit ONNX inspection and TensorRT profile control.
- If engine build or inference fails because of profile mismatch, regenerate ONNX metadata from export or pass wider explicit shape ranges.
