# AP Evaluation Tutorial

This tutorial explains how to measure AP for:

- PyTorch baseline
- ONNX Runtime
- TensorRT engine built from ONNX
- Direct TensorRT artifact built from PyTorch

It also specifies where evaluation results are stored.

For engine generation commands, see:

- `docs/engineGeneration.md`
- `docs/onnx.md`
- `docs/tensorRT.md`

## Goal

Measure AP at IoU 0.3, 0.5, and 0.7 and compare backend parity.

## 1. PyTorch baseline AP

```bash
python3 opencood/tools/inference.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --test
```

Stored results:

- `<MODEL_DIR>/eval.yaml`
- or `<MODEL_DIR>/eval_global_sort.yaml` if global sorting is enabled

## 2. ONNX Runtime AP

Prerequisite:

- `<MODEL_DIR>/pipeline_a/model.onnx` already exists

Command:

```bash
python3 opencood/tools/inference_onnx.py \
  --model_dir <MODEL_DIR> \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --fusion_method <early|late|intermediate> \
  --output_yaml pipeline_a/eval_onnx.yaml \
  --test
```

Stored results:

- `<MODEL_DIR>/pipeline_a/eval_onnx.yaml`

Optional CPU fallback:

```bash
python3 opencood/tools/inference_onnx.py \
  --model_dir <MODEL_DIR> \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --fusion_method <early|late|intermediate> \
  --providers CPUExecutionProvider \
  --output_yaml pipeline_a/eval_onnx.yaml \
  --test
```

## 3. TensorRT AP from ONNX engine

Prerequisite:

- `<MODEL_DIR>/pipeline_a/model_fp32.engine` or another TensorRT engine already exists

Command:

```bash
python3 opencood/tools/inference_tensorrt.py \
  --model_dir <MODEL_DIR> \
  --engine_path <MODEL_DIR>/pipeline_a/model_fp32.engine \
  --fusion_method <early|late|intermediate> \
  --output_yaml pipeline_a/eval_tensorrt.yaml \
  --test
```

Stored results:

- `<MODEL_DIR>/pipeline_a/eval_tensorrt.yaml`

Notes:

- The runtime supports TensorRT legacy bindings and TensorRT 10 I/O tensor execution.
- For intermediate fusion, unresolved output shape errors usually mean the engine profile is too narrow.

## 4. Direct TensorRT AP from Torch-TensorRT artifact

Prerequisite:

- `<MODEL_DIR>/pipeline_b/model_trt.ts` already exists

Command:

```bash
python3 opencood/tools/inference_tensorrt_direct.py \
  --model_dir <MODEL_DIR> \
  --trt_module <MODEL_DIR>/pipeline_b/model_trt.ts \
  --fusion_method <early|late|intermediate> \
  --output_yaml pipeline_b/eval_tensorrt_direct.yaml \
  --test
```

Stored results:

- `<MODEL_DIR>/pipeline_b/eval_tensorrt_direct.yaml`

## 5. Compare AP across backends

Update `docs/results.md` with:

- PyTorch AP values
- ONNX AP values and delta vs PyTorch
- TensorRT from ONNX AP values and delta vs PyTorch
- Direct TensorRT AP values and delta vs PyTorch

Optional helper:

```bash
python3 opencood/tools/compare_backend_ap.py \
  --model_name v2x-vit \
  --split test \
  --pytorch_yaml <MODEL_DIR>/eval.yaml \
  --onnx_yaml <MODEL_DIR>/pipeline_a/eval_onnx.yaml \
  --trt_onnx_yaml <MODEL_DIR>/pipeline_a/eval_tensorrt.yaml \
  --trt_direct_yaml <MODEL_DIR>/pipeline_b/eval_tensorrt_direct.yaml
```

## Result Storage Summary

- PyTorch:
  - `<MODEL_DIR>/eval.yaml`
  - `<MODEL_DIR>/eval_global_sort.yaml`
- ONNX Runtime:
  - `<MODEL_DIR>/pipeline_a/eval_onnx.yaml`
- TensorRT from ONNX:
  - `<MODEL_DIR>/pipeline_a/eval_tensorrt.yaml`
- Direct TensorRT:
  - `<MODEL_DIR>/pipeline_b/eval_tensorrt_direct.yaml`

## Suggested Validation Order

1. Validate PyTorch baseline AP.
2. Validate ONNX Runtime AP.
3. Validate FP32 TensorRT AP from ONNX.
4. Validate FP32 direct TensorRT AP.
5. Only then move to FP16 and later INT8.