# OpenCOOD

OpenCOOD basic commands are detailed in:

- `docs/basicCommands.md`

AP tracking file:

- `docs/results.md`

## Battle Plan

Target: TensorRT-backed embedded V2X-ViT with balanced AP and runtime.

### Environment lock:
- Python 3.10.12
- torch 2.7.0+cu128
- torchvision 0.22.0+cu128
- torchaudio 2.7.0+cu128
- onnx 1.16.2
- onnxruntime 1.23.2
- onnxruntime-gpu 1.23.2
- tensorrt 10.15.1.29
- torch_tensorrt 2.7.0
- pycuda 2026.1

### Checklist

- Fix current pipeline:
    - [ ] PyTorch -> ONNX -> TensorRT
        - Validate ONNX AP metrics (artifact: `opencood/v2x-vit/pipeline_a/eval_onnx.yaml`)
        - Validate TensorRT AP metrics (artifact: `opencood/v2x-vit/pipeline_a/eval_tensorrt.yaml`)
        - Validate runtime profile yaml files:
            - `opencood/v2x-vit/pipeline_a/profile_onnx.yaml`
            - `opencood/v2x-vit/pipeline_a/profile_tensorrt.yaml`
    - [ ] PyTorch -> TensorRT
        - Validate TensorRT AP metrics (artifact: `opencood/v2x-vit/pipeline_b/eval_tensorrt_direct.yaml`)

- [ ] Validate PyTorch baseline AP metrics (artifact: `opencood/v2x-vit/eval.yaml or eval_global_sort.yaml`)
- [ ] Validate PyTorch runtime profile (artifact: `opencood/v2x-vit/pipeline_a/profile_pytorch.yaml`)

- [ ] Run end-to-end pipelines on platform for runtime metrics
    - Store consolidated run logs under: `opencood/v2x-vit/benchmarks/`

- Add per-module profiling
    - [ ] at least: front-end, BEV backbone, fusion transformer, heads
    - [ ] runtime and memory analysis

- [ ] Implement PTQ on the whole model (reference baseline)

- End-to-end quantized runs
    - [ ] FP16
        - AP metrics
        - on-device runtime and memory analysis
    - [ ] INT8
        - AP metrics
        - on-device runtime and memory analysis

- Reporting discipline
    - [ ] Keep AP and deltas updated in `docs/results.md`
    - [ ] For every milestone, record commands + artifacts + notes under `opencood/v2x-vit/benchmarks/`
        
- ...



