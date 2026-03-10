# Basic OpenCOOD Commands

To run from the repository root.

## Visualize

```bash
python3 opencood/visualization/vis_data_sequence.py --color_mode intensity
```

## Train

```bash
python3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_v2xvit.yaml
```

## Inference

```bash
python3 opencood/tools/inference.py --model_dir opencood/logs/preTrained/v2x-vit --fusion_method intermediate
```
