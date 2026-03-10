# AP Results

Store detection AP results here.

Comparison target: absolute AP delta <= 0.5% versus PyTorch baseline per IoU.

## Validate Dataset Results

| Model Name | Backend | AP@IoU=0.3 | Delta@0.3 (%) | AP@IoU=0.5 | Delta@0.5 (%) | AP@IoU=0.7 | Delta@0.7 (%) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| v2x-vit | PyTorch | 0.8924 | - | 0.8410 | - | 0.6164 | - | baseline |
| v2x-vit | ONNX Runtime | 0.8428 | -4.96% | 0.8066 | -3.44% | 0.6087 | -0.77% | regroup+scatter parity fixes |
| v2x-vit | TensorRT (from ONNX) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | TensorRT parser rejects ONNX AffineGrid for intermediate fusion |
| v2x-vit | TensorRT (direct Torch-TensorRT) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | Torch-TensorRT compile segfault (code 139) on this stack |

## Test Dataset Results

| Model Name | Backend | AP@IoU=0.3 | Delta@0.3 (%) | AP@IoU=0.5 | Delta@0.5 (%) | AP@IoU=0.7 | Delta@0.7 (%) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| v2x-vit | PyTorch | 0.8832 | - | 0.8408 | - | 0.6403 | - | baseline |
| v2x-vit | ONNX Runtime | 0.8603 | -2.29% | 0.8177 | -2.31% | 0.6247 | -1.56% | graph_optimization_level=none |
| v2x-vit | TensorRT (from ONNX) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | TensorRT parser rejects ONNX AffineGrid for intermediate fusion |
| v2x-vit | TensorRT (direct Torch-TensorRT) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | Torch-TensorRT compile segfault (code 139) on this stack |

