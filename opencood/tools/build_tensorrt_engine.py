# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import os
import shlex
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Build TensorRT engine with trtexec')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--engine_path', type=str, required=True,
                        help='Target TensorRT engine path')
    parser.add_argument('--fp16', action='store_true',
                        help='Build FP16 engine')
    parser.add_argument('--workspace', type=int, default=4096,
                        help='Workspace size in MB')
    parser.add_argument('--min_shapes', type=str, default='',
                        help='trtexec minShapes, e.g. voxel_features:1x32x4,voxel_coords:1x4')
    parser.add_argument('--opt_shapes', type=str, default='',
                        help='trtexec optShapes')
    parser.add_argument('--max_shapes', type=str, default='',
                        help='trtexec maxShapes')
    parser.add_argument('--trtexec_path', type=str, default='trtexec',
                        help='Path to trtexec binary')
    return parser.parse_args()


def _add_shape_args(cmd, name, value):
    if value:
        cmd.append(f'--{name}={value}')


def main():
    args = parse_args()

    engine_dir = os.path.dirname(args.engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)

    cmd = [
        args.trtexec_path,
        f'--onnx={args.onnx_model}',
        f'--saveEngine={args.engine_path}',
        f'--memPoolSize=workspace:{args.workspace}M',
        '--skipInference',
    ]

    if args.fp16:
        cmd.append('--fp16')

    _add_shape_args(cmd, 'minShapes', args.min_shapes)
    _add_shape_args(cmd, 'optShapes', args.opt_shapes)
    _add_shape_args(cmd, 'maxShapes', args.max_shapes)

    print('Running TensorRT build command:')
    print(' '.join(shlex.quote(x) for x in cmd))

    subprocess.run(cmd, check=True)

    print(f'Engine written to {args.engine_path}')


if __name__ == '__main__':
    main()
