# AP Results

Store detection AP results here.

## Validate Dataset Results

| Model Name | Backend | AP@IoU=0.3 | Delta@0.3 (%) | AP@IoU=0.5 | Delta@0.5 (%) | AP@IoU=0.7 | Delta@0.7 (%) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| v2x-vit | PyTorch | 0.8924 | - | 0.8410 | - | 0.6164 | - | baseline |
| v2x-vit | ONNX | 0.8141 | -7.83% | 0.7741 | -6.69% | 0.5651 | -5.13% | dynamic shapes enabled |
| v2x-vit | TensorRT (pipeline A) | 0.8043 | -8.81% | 0.7428 | -9.82% | 0.5254 | -9.10% | dynamic shapes enabled |

## Test Dataset Results

| Model Name | Backend | AP@IoU=0.3 | Delta@0.3 (%) | AP@IoU=0.5 | Delta@0.5 (%) | AP@IoU=0.7 | Delta@0.7 (%) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| v2x-vit | PyTorch | 0.8832 | - | 0.8408 | - | 0.6403 | - | baseline |
| v2x-vit | ONNX | 0.8603 | -2.29% | 0.8177 | -2.31% | 0.6247 | -1.56% | dynamic shapes enabled |
| v2x-vit | TensorRT (pipeline A) | 0.8585 | -2.47% | 0.8014 | -3.94% | 0.5904 | -4.99% | dynamic shapes enabled |

