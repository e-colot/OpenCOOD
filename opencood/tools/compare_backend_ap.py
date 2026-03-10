# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import os

import opencood.hypes_yaml.yaml_utils as yaml_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare AP yaml outputs across backends and print markdown-ready rows'
    )
    parser.add_argument('--model_name', type=str, default='v2x-vit')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--pytorch_yaml', type=str, required=True,
                        help='Baseline PyTorch AP yaml path')
    parser.add_argument('--onnx_yaml', type=str, default='',
                        help='ONNX Runtime AP yaml path')
    parser.add_argument('--trt_onnx_yaml', type=str, default='',
                        help='TensorRT-from-ONNX AP yaml path')
    parser.add_argument('--trt_direct_yaml', type=str, default='',
                        help='Direct Torch-TensorRT AP yaml path')
    return parser.parse_args()


def _extract_ap(metrics):
    return {
        'ap03': float(metrics.get('ap30', 0.0)),
        'ap05': float(metrics.get('ap_50', 0.0)),
        'ap07': float(metrics.get('ap_70', 0.0)),
    }


def _load_optional(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f'AP yaml not found: {path}')
    return _extract_ap(yaml_utils.load_yaml(path))


def _delta(value, baseline):
    return value - baseline


def _format_row(model_name, backend_name, ap, baseline_ap, notes):
    if ap is None:
        return f'| {model_name} | {backend_name} | TODO | TODO | TODO | TODO | TODO | TODO | {notes} |'

    return (
        f'| {model_name} | {backend_name} | '
        f'{ap["ap03"]:.4f} | {_delta(ap["ap03"], baseline_ap["ap03"]):+.4f} | '
        f'{ap["ap05"]:.4f} | {_delta(ap["ap05"], baseline_ap["ap05"]):+.4f} | '
        f'{ap["ap07"]:.4f} | {_delta(ap["ap07"], baseline_ap["ap07"]):+.4f} | '
        f'{notes} |'
    )


def main():
    args = parse_args()

    baseline = _extract_ap(yaml_utils.load_yaml(args.pytorch_yaml))
    onnx = _load_optional(args.onnx_yaml)
    trt_onnx = _load_optional(args.trt_onnx_yaml)
    trt_direct = _load_optional(args.trt_direct_yaml)

    print(f'# Backend AP Comparison ({args.split})')
    print('| Model Name | Backend | AP@IoU=0.3 | Delta@0.3 | AP@IoU=0.5 | Delta@0.5 | AP@IoU=0.7 | Delta@0.7 | Notes |')
    print('| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |')
    print(_format_row(args.model_name,
                      'PyTorch',
                      baseline,
                      baseline,
                      'baseline'))
    print(_format_row(args.model_name,
                      'ONNX Runtime',
                      onnx,
                      baseline,
                      'from --onnx_yaml'))
    print(_format_row(args.model_name,
                      'TensorRT (from ONNX)',
                      trt_onnx,
                      baseline,
                      'from --trt_onnx_yaml'))
    print(_format_row(args.model_name,
                      'TensorRT (direct Torch-TensorRT)',
                      trt_direct,
                      baseline,
                      'from --trt_direct_yaml'))


if __name__ == '__main__':
    main()