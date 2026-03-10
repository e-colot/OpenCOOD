# AP Results (Deployment Validation)

This file is the canonical AP comparison report for deployment backend validation.

## How to use this file

1. Run baseline PyTorch inference AP.
2. Run ONNX Runtime AP with the same checkpoint and split.
3. Run TensorRT AP with the same checkpoint and split.
4. Record all results in the table below.

## AP Comparison Table

| Date | Model | Fusion | Split | Backend | AP@0.3 | AP@0.5 | AP@0.7 | Delta vs PyTorch (AP@0.5) | Pass (<= 0.5%) | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| TODO | TODO | TODO | TODO | pytorch | TODO | TODO | TODO | 0.0000 | - | baseline |
| TODO | TODO | TODO | TODO | onnxruntime | TODO | TODO | TODO | TODO | TODO | |
| TODO | TODO | TODO | TODO | tensorrt | TODO | TODO | TODO | TODO | TODO | |

## Runtime Table (Optional)

| Date | Model | Fusion | Split | Backend | Avg Latency (ms/frame) | Notes |
| --- | --- | --- | --- | --- | ---: | --- |
| TODO | TODO | TODO | TODO | pytorch | TODO | |
| TODO | TODO | TODO | TODO | onnxruntime | TODO | |
| TODO | TODO | TODO | TODO | tensorrt | TODO | |
