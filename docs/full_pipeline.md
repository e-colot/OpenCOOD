# Full Pipeline Tutorial: ONNX -> TensorRT -> AP Testing

This guide validates the full deployment path before quantization work starts.

## Goal

Validate the deployment pipelines:

- PyTorch baseline AP
- ONNX Runtime AP
- TensorRT AP (from ONNX)
- TensorRT AP (direct from PyTorch via Torch-TensorRT)

and confirm AP parity target is met.

## Step 1. Run PyTorch baseline AP

```bash
python3 opencood/tools/inference.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --test
```

Output:

- <MODEL_DIR>/eval.yaml (or eval_global_sort.yaml)

## Step 2. Export ONNX

```bash
python3 opencood/tools/export_onnx.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --output <MODEL_DIR>/pipeline_a/model.onnx
```

## Step 3. Run ONNX Runtime AP

```bash
python3 opencood/tools/inference_onnx.py \
  --model_dir <MODEL_DIR> \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --fusion_method <early|late|intermediate> \
  --output_yaml pipeline_a/eval_onnx.yaml \
  --test
```

## Step 4. Build TensorRT engine

```bash
python3 opencood/tools/build_tensorrt_engine.py \
  --onnx_model <MODEL_DIR>/pipeline_a/model.onnx \
  --engine_path <MODEL_DIR>/pipeline_a/model_fp32.engine
```

## Step 5. Run TensorRT AP

```bash
python3 opencood/tools/inference_tensorrt.py \
  --model_dir <MODEL_DIR> \
  --engine_path <MODEL_DIR>/pipeline_a/model_fp32.engine \
  --fusion_method <early|late|intermediate> \
  --output_yaml pipeline_a/eval_tensorrt.yaml \
  --test
```

## Step 6. Build direct PyTorch -> TensorRT artifact

```bash
python3 opencood/tools/build_tensorrt_engine_direct.py \
  --model_dir <MODEL_DIR> \
  --fusion_method <early|late|intermediate> \
  --output <MODEL_DIR>/pipeline_b/model_trt.ts \
  --precision fp32 \
  --test
```

## Step 7. Run direct TensorRT AP

```bash
python3 opencood/tools/inference_tensorrt_direct.py \
  --model_dir <MODEL_DIR> \
  --trt_module <MODEL_DIR>/pipeline_b/model_trt.ts \
  --fusion_method <early|late|intermediate> \
  --output_yaml pipeline_b/eval_tensorrt_direct.yaml \
  --test
```

## Step 8. Compare AP and write docs/results.md

Fill docs/results.md with:

- PyTorch AP values
- ONNX AP values and absolute AP delta vs PyTorch
- TensorRT (from ONNX) AP values and absolute AP delta vs PyTorch
- TensorRT (direct) AP values and absolute AP delta vs PyTorch
- Pass/fail against parity threshold (<= 0.5% absolute)

Optional helper to generate markdown-ready rows from AP yaml files:

```bash
python3 opencood/tools/compare_backend_ap.py \
  --model_name v2x-vit \
  --split test \
  --pytorch_yaml <MODEL_DIR>/eval.yaml \
  --onnx_yaml <MODEL_DIR>/pipeline_a/eval_onnx.yaml \
  --trt_onnx_yaml <MODEL_DIR>/pipeline_a/eval_tensorrt.yaml \
  --trt_direct_yaml <MODEL_DIR>/pipeline_b/eval_tensorrt_direct.yaml
```

## Suggested comparison checklist

- Same checkpoint for all runs.
- Same dataset split for all runs.
- Same fusion method for all runs.
- Use FP32 TensorRT first, then FP16 only after FP32 parity passes.

## Known caveat for intermediate fusion

Intermediate models can require export-safe constraints for dynamic fusion behavior. If intermediate export/runtime fails while early/late pass, classify intermediate TensorRT as deferred and continue with quantization preparation on validated modes.
