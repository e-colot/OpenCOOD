# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


class OpenCoodExportWrapper(torch.nn.Module):
    """
    Wrap OpenCOOD models so torch.onnx.export receives tensor arguments
    while the underlying model still gets its expected nested dict.
    """

    def __init__(self, model, extra_keys):
        super().__init__()
        self.model = model
        self.extra_keys = extra_keys

    def forward(self,
                voxel_features,
                voxel_coords,
                voxel_num_points,
                *extra_tensors):
        data_dict = {
            'processed_lidar': {
                'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points,
            }
        }

        for key, value in zip(self.extra_keys, extra_tensors):
            data_dict[key] = value

        output = self.model(data_dict)

        # Keep a stable output tuple for inference scripts.
        return output['psm'], output['rm']


def _sanitize_tensor_for_export(tensor):
    # Prevent accidental fp64 inputs from becoming ONNX graph contract.
    if tensor.is_floating_point() and tensor.dtype == torch.float64:
        return tensor.float()
    return tensor


def parse_args():
    parser = argparse.ArgumentParser(description='Export OpenCOOD model to ONNX')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Trained checkpoint directory')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path. Defaults to <model_dir>/pipeline_a/model.onnx')
    parser.add_argument('--opset', type=int, default=20,
                        help='ONNX opset version')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Export device, defaults to cuda if available else cpu')
    parser.add_argument('--validate', action='store_true',
                        help='Run onnx.checker validation on the exported model')
    parser.add_argument('--max_search_batches', type=int, default=128,
                        help='Search this many batches and pick the one with largest total record_len for tracing')
    return parser.parse_args()


def _validate_onnx_model(onnx_path):
    try:
        import onnx
    except ImportError as exc:
        raise ImportError('ONNX validation requested but onnx package is unavailable.') from exc

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)


def _collect_extra_keys(cav_content):
    # Common non-lidar tensor keys used by intermediate fusion variants.
    candidate_keys = [
        'record_len',
        'prior_encoding',
        'spatial_correction_matrix',
        'pairwise_t_matrix',
        'lidar_pose',
    ]
    extra_keys = []
    for key in candidate_keys:
        value = cav_content.get(key)
        if isinstance(value, torch.Tensor):
            extra_keys.append(key)
    return extra_keys


def _dynamic_axes_for_input(name, tensor):
    if name == 'record_len':
        return {0: 'batch'}
    if name == 'pairwise_t_matrix':
        # Usually [B, L, L, 4, 4] with B=1 for eval; keep cav dimensions dynamic.
        if tensor.dim() >= 3:
            return {0: 'batch', 1: 'max_cav', 2: 'max_cav'}
        return {0: 'batch'}
    if name == 'spatial_correction_matrix':
        # Usually [B, L, 4, 4]
        if tensor.dim() >= 2:
            return {0: 'batch', 1: 'max_cav'}
        return {0: 'batch'}
    if name == 'prior_encoding':
        # Usually [B, L, C]
        if tensor.dim() >= 2:
            return {0: 'batch', 1: 'max_cav'}
        return {0: 'batch'}

    # Conservative default if shape semantics are unknown.
    return {0: 'batch'}


def _select_trace_batch(loader, fusion_method, max_search_batches):
    best_batch = None
    best_total_cav = -1

    for idx, batch_data in enumerate(loader):
        if idx >= max_search_batches:
            break

        if fusion_method in ['early', 'intermediate']:
            cav_content = batch_data['ego']
        else:
            first_key = next(iter(batch_data.keys()))
            cav_content = batch_data[first_key]

        record_len = cav_content.get('record_len')
        if isinstance(record_len, torch.Tensor):
            total_cav = int(record_len.to(torch.long).sum().item())
        else:
            total_cav = 0

        if total_cav > best_total_cav:
            best_total_cav = total_cav
            best_batch = batch_data

    if best_batch is None:
        raise RuntimeError('Failed to fetch a batch for ONNX export tracing.')

    print(f'Selected trace batch total CAV: {best_total_cav}')
    return best_batch


def _patch_squeeze_axes_for_tensorrt(onnx_path):
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print('Skip ONNX squeeze patch: onnx package is unavailable.')
        return 0

    model = onnx.load(onnx_path)

    shape_map = {}

    def _record_shape(value_info):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField('shape'):
            return

        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                dims.append(int(dim.dim_value))
            elif dim.HasField('dim_param'):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        shape_map[value_info.name] = dims

    for value_info in list(model.graph.value_info) + \
            list(model.graph.input) + list(model.graph.output):
        _record_shape(value_info)

    patched_count = 0
    new_initializers = []

    for idx, node in enumerate(model.graph.node):
        if node.op_type != 'Squeeze':
            continue
        if len(node.input) >= 2:
            continue

        in_shape = shape_map.get(node.input[0], [])
        out_shape = shape_map.get(node.output[0], [])

        axes = []
        if in_shape and out_shape and len(in_shape) >= len(out_shape):
            i = 0
            j = 0
            while i < len(in_shape) and j < len(out_shape):
                if in_shape[i] == out_shape[j]:
                    i += 1
                    j += 1
                elif in_shape[i] == 1:
                    axes.append(i)
                    i += 1
                else:
                    i += 1

            while i < len(in_shape):
                if in_shape[i] == 1:
                    axes.append(i)
                i += 1

        if not axes:
            # Conservative fallback for common [N,1,C] -> [N,C].
            axes = [1]

        const_name = f'{node.name or "squeeze"}_axes_const_{idx}'
        axes_arr = np.asarray(axes, dtype=np.int64)
        new_initializers.append(numpy_helper.from_array(axes_arr, const_name))
        node.input.append(const_name)
        patched_count += 1

    if new_initializers:
        model.graph.initializer.extend(new_initializers)
        onnx.save(model, onnx_path)

    return patched_count


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    hypes = yaml_utils.load_yaml(None, args)

    print('Building dataset for sample input')
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=0,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)

    print('Creating and loading model')
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    batch_data = _select_trace_batch(loader,
                                     args.fusion_method,
                                     args.max_search_batches)
    batch_data = train_utils.to_device(batch_data, device)

    if args.fusion_method in ['early', 'intermediate']:
        cav_content = batch_data['ego']
    else:
        first_key = next(iter(batch_data.keys()))
        cav_content = batch_data[first_key]

    processed = cav_content['processed_lidar']
    base_inputs = [
        _sanitize_tensor_for_export(processed['voxel_features']),
        _sanitize_tensor_for_export(processed['voxel_coords']),
        _sanitize_tensor_for_export(processed['voxel_num_points']),
    ]
    base_names = ['voxel_features', 'voxel_coords', 'voxel_num_points']

    extra_keys = _collect_extra_keys(cav_content)
    extra_inputs = [_sanitize_tensor_for_export(cav_content[k]) for k in extra_keys]

    wrapper = OpenCoodExportWrapper(model, extra_keys).to(device)
    wrapper.eval()

    output_path = args.output or os.path.join(args.model_dir, 'pipeline_a', 'model.onnx')
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    input_names = base_names + extra_keys
    output_names = ['psm', 'rm']

    dynamic_axes = {
        'voxel_features': {0: 'num_voxels', 1: 'points_per_voxel'},
        'voxel_coords': {0: 'num_voxels'},
        'voxel_num_points': {0: 'num_voxels'},
        'psm': {0: 'batch'},
        'rm': {0: 'batch'},
    }
    for key, tensor in zip(extra_keys, extra_inputs):
        dynamic_axes[key] = _dynamic_axes_for_input(key, tensor)

    print(f'Exporting ONNX to {output_path}')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(base_inputs + extra_inputs),
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    patched_count = _patch_squeeze_axes_for_tensorrt(output_path)
    if patched_count:
        print(f'Patched {patched_count} Squeeze nodes with explicit axes for TensorRT.')

    if args.validate:
        _validate_onnx_model(output_path)
        print('ONNX checker validation passed')

    print('Export completed')
    print(f'Inputs: {input_names}')
    print(f'Outputs: {output_names}')


if __name__ == '__main__':
    main()
