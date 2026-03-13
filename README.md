# Documentation

- OpenCOOD basic commands are detailed in:

    `docs/basicCommands.md`

- TensorRT engine generation:

    `docs/engineGeneration.md`

- AP evaluation:

    `docs/apEvaluation.md`

- AP tracking file:

    `docs/results.md`

## Battle Plan

Target: TensorRT-backed embedded V2X-ViT with balanced AP and runtime.

### Environment
- Python 3.10.12
- torch 2.7.0+cu128
- torchvision 0.22.0+cu128
- torchaudio 2.7.0+cu128
- onnx 1.16.2
- onnxruntime 1.23.2
- onnxruntime-gpu 1.23.2
- tensorrt-cu12 10.15.1.29
- trtexec 10.15.1.29
- torch_tensorrt 2.7.0
- pycuda 2026.1

### About Metrics

Two types of metrics are planned: 
- AP metrics which are related to the model precision (average precision @IoU = 0.3, 0.5, 0.7). Those are device independent and can therefore be ran once.
- Runtime metrics, which depends on the device on which the inference is ran on. Those metrics include memory usage, inference time, power consumption. 

Those metrics can be measured on two different datasets: `test` and `dataset`, both being part of v2xset.

More advanced metrics will allow specific AP/runtime profiling, meaning splitting the model into subblocks and evaluating each one *"independently"*.

### Checklist

- Fix current pipeline:
    - PyTorch -> ONNX -> TensorRT
        - [x] PyTorch -> ONNX generation
        - [x] Validate ONNX AP metrics 
        - [x] ONNX -> TensorRT generation
        - [x] Validate TensorRT AP metrics
    - PyTorch -> TensorRT
        - [ ] Validate TensorRT AP metrics

- [ ] Validate PyTorch baseline AP metrics
- [ ] Validate PyTorch runtime profile

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



