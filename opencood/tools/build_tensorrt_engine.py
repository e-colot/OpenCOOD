# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import math
import os
import shlex
import subprocess

import opencood.hypes_yaml.yaml_utils as yaml_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Build TensorRT engine with trtexec')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--engine_path', type=str, default='',
                        help='Target TensorRT engine path. Defaults to <onnx_dir>/<onnx_name>.engine')
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
    parser.add_argument('--profile_yaml', type=str, default='',
                        help='Optional ONNX export metadata yaml with suggested_trt_profile')
    parser.add_argument('--max_voxels', type=int, default=0,
                        help='Override max profile voxel count for voxel_* inputs (0 = keep metadata/default)')
    parser.add_argument('--max_voxels_margin', '--max_voxel_margin',
                        dest='max_voxels_margin', type=float, default=1.0,
                        help='Safety factor applied to detected max voxel profile count (default: 1.0)')
    return parser.parse_args()


def _add_shape_args(cmd, name, value):
    if value:
        cmd.append(f'--{name}={value}')


def _onnx_has_dynamic_inputs(onnx_path):
    try:
        import onnx
    except ImportError:
        return False

    model = onnx.load(onnx_path)
    for input_value in model.graph.input:
        tensor_type = input_value.type.tensor_type
        if not tensor_type.HasField('shape'):
            return True
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_param'):
                return True
            if not dim.HasField('dim_value'):
                return True
            if int(dim.dim_value) <= 0:
                return True

    return False


def _onnx_input_names(onnx_path):
    try:
        import onnx
    except ImportError:
        return set()

    model = onnx.load(onnx_path)
    return {value.name for value in model.graph.input}


def _filter_shape_profile_to_inputs(profile_value, valid_input_names):
    if not profile_value:
        return profile_value

    parts = []
    for item in profile_value.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            continue
        name, shape = item.split(':', 1)
        name = name.strip()
        shape = shape.strip()
        if name in valid_input_names and shape:
            parts.append(f'{name}:{shape}')

    return ','.join(parts)


def _read_profile_hints(args):
    if args.min_shapes and args.opt_shapes and args.max_shapes:
        return args.min_shapes, args.opt_shapes, args.max_shapes

    candidates = []
    if args.profile_yaml:
        candidates.append(args.profile_yaml)
    candidates.append(args.onnx_model + '.meta.yaml')

    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        meta = yaml_utils.load_yaml(candidate)
        profile = meta.get('suggested_trt_profile', {})
        min_shapes = profile.get('min_shapes', '')
        opt_shapes = profile.get('opt_shapes', '')
        max_shapes = profile.get('max_shapes', '')
        if min_shapes and opt_shapes and max_shapes:
            print(f'Using TensorRT shape profile from metadata: {candidate}')
            return min_shapes, opt_shapes, max_shapes

    return args.min_shapes, args.opt_shapes, args.max_shapes


def _parse_shape_profile(profile_value):
    if not profile_value:
        return {}

    parsed = {}
    for item in profile_value.split(','):
        item = item.strip()
        if not item or ':' not in item:
            continue
        name, shape = item.split(':', 1)
        name = name.strip()
        shape = shape.strip()
        if not name or not shape:
            continue
        dims = []
        valid = True
        for dim in shape.split('x'):
            dim = dim.strip()
            if not dim:
                valid = False
                break
            try:
                dims.append(max(1, int(dim)))
            except ValueError:
                valid = False
                break
        if valid and dims:
            parsed[name] = dims
    return parsed


def _format_shape_profile(parsed_shapes):
    if not parsed_shapes:
        return ''

    parts = []
    for name, dims in parsed_shapes.items():
        parts.append(f"{name}:{'x'.join(str(int(v)) for v in dims)}")
    return ','.join(parts)


def _apply_voxel_profile_limits(max_shapes, max_voxels, max_voxels_margin):
    parsed = _parse_shape_profile(max_shapes)
    if not parsed:
        print('Detected max voxel profile count: unavailable (no maxShapes profile found).')
        return max_shapes

    target_names = ['voxel_features', 'voxel_coords', 'voxel_num_points']
    detected_values = [parsed[name][0] for name in target_names if name in parsed and parsed[name]]
    if detected_values:
        detected_max = max(detected_values)
        print(f'Detected max voxel profile count: {detected_max}')
    else:
        print('Detected max voxel profile count: unavailable (voxel_* inputs missing in maxShapes).')

    margin = max(1.0, float(max_voxels_margin))
    changed = []
    summary = {}
    for name in target_names:
        dims = parsed.get(name)
        if not dims:
            continue

        base_value = int(dims[0])
        margin_value = int(math.ceil(base_value * margin))
        target_value = max(base_value, int(max_voxels), margin_value)
        if target_value > base_value:
            dims[0] = target_value
            changed.append(name)
        summary[name] = int(dims[0])

    if changed:
        print(
            'Applied voxel max profile override to: '
            f"{', '.join(changed)} (max_voxels={max_voxels}, max_voxels_margin={margin})"
        )

    if summary:
        summary_text = ', '.join(f'{k}={v}' for k, v in summary.items())
        print(f'Final max voxel profile count per input: {summary_text}')

    return _format_shape_profile(parsed)


def main():
    args = parse_args()

    if not args.engine_path:
        onnx_dir = os.path.dirname(args.onnx_model)
        onnx_name = os.path.splitext(os.path.basename(args.onnx_model))[0]
        args.engine_path = os.path.join(onnx_dir, f'{onnx_name}.engine')
        print(f'Engine path not provided, using default: {args.engine_path}')

    valid_input_names = _onnx_input_names(args.onnx_model)
    min_shapes, opt_shapes, max_shapes = _read_profile_hints(args)
    min_shapes = _filter_shape_profile_to_inputs(min_shapes, valid_input_names)
    opt_shapes = _filter_shape_profile_to_inputs(opt_shapes, valid_input_names)
    max_shapes = _filter_shape_profile_to_inputs(max_shapes, valid_input_names)
    max_shapes = _apply_voxel_profile_limits(max_shapes,
                                             args.max_voxels,
                                             args.max_voxels_margin)
    has_dynamic_inputs = _onnx_has_dynamic_inputs(args.onnx_model)
    if has_dynamic_inputs and not (min_shapes and opt_shapes and max_shapes):
        raise RuntimeError(
            'ONNX model has dynamic input shapes but no TensorRT shape profile was provided. '
            'Either pass --min_shapes/--opt_shapes/--max_shapes or export ONNX with '
            'opencood/tools/export_onnx.py and reuse the generated .meta.yaml profile.'
        )

    engine_dir = os.path.dirname(args.engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)

    cmd = [
        args.trtexec_path,
        f'--onnx={args.onnx_model}',
        f'--saveEngine={args.engine_path}',
        f'--memPoolSize=workspace:{args.workspace}',
        '--skipInference',
    ]

    if args.fp16:
        cmd.append('--fp16')

    _add_shape_args(cmd, 'minShapes', min_shapes)
    _add_shape_args(cmd, 'optShapes', opt_shapes)
    _add_shape_args(cmd, 'maxShapes', max_shapes)

    print('Running TensorRT build command:')
    print(' '.join(shlex.quote(x) for x in cmd))

    subprocess.run(cmd, check=True)

    print(f'Engine written to {args.engine_path}')


if __name__ == '__main__':
    main()
