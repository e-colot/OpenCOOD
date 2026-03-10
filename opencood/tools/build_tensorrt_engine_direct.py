# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import os

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.export_onnx import (
    OpenCoodExportWrapper,
    _collect_extra_keys,
    _maybe_replace_validate_split,
    _sanitize_tensor_for_export,
    _select_trace_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build direct PyTorch->TensorRT TorchScript artifact with Torch-TensorRT'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Trained checkpoint directory')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--output', type=str, default=None,
                        help='Output Torch-TensorRT artifact path. Defaults to <model_dir>/trt_direct/model_trt.ts')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Output metadata yaml path. Defaults to <output>.meta.yaml')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Target TensorRT precision mode for compilation')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Build device, defaults to cuda if available else cpu')
    parser.add_argument('--workspace_mb', type=int, default=4096,
                        help='TensorRT workspace size in MB')
    parser.add_argument('--test', action='store_true',
                        help='Use validation split by replacing test with validate')
    parser.add_argument('--max_search_batches', type=int, default=128,
                        help='Search this many batches and pick the one with largest total record_len for tracing')
    return parser.parse_args()


def _default_output_path(model_dir):
    return os.path.join(model_dir, 'pipeline_b', 'model_trt.ts')


def _default_metadata_path(output_path):
    return output_path + '.meta.yaml'


def _enabled_precisions(precision):
    if precision == 'fp16':
        return {torch.float, torch.half}
    return {torch.float}


def main():
    args = parse_args()

    try:
        import torch_tensorrt
    except ImportError as exc:
        raise ImportError(
            'Direct TensorRT build requires torch_tensorrt. '
            'Install Torch-TensorRT first, then rerun.'
        ) from exc

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if device.type != 'cuda':
        raise RuntimeError('Direct Torch-TensorRT compile requires CUDA. Use --device cuda.')

    hypes = yaml_utils.load_yaml(None, args)
    _maybe_replace_validate_split(hypes, args.test)

    print('Building dataset for trace input')
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

    compile_inputs = tuple(base_inputs + extra_inputs)

    print('Tracing wrapper for Torch-TensorRT compile')
    scripted_wrapper = torch.jit.trace(wrapper, compile_inputs, strict=False)

    enabled_precisions = _enabled_precisions(args.precision)
    print(f'Compiling direct TensorRT artifact with precision={args.precision}')
    trt_module = torch_tensorrt.compile(
        scripted_wrapper,
        ir='torchscript',
        inputs=compile_inputs,
        enabled_precisions=enabled_precisions,
        workspace_size=args.workspace_mb * 1024 * 1024,
        truncate_long_and_double=True,
    )

    with torch.no_grad():
        _ = trt_module(*compile_inputs)

    output_path = args.output or _default_output_path(args.model_dir)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f'Saving direct TensorRT artifact to {output_path}')
    torch.jit.save(trt_module, output_path)

    metadata_path = args.metadata or _default_metadata_path(output_path)
    metadata = {
        'artifact': output_path,
        'precision': args.precision,
        'fusion_method': args.fusion_method,
        'input_names': base_names + extra_keys,
    }
    yaml_utils.save_yaml(metadata, metadata_path)

    print('Direct TensorRT build completed')
    print(f'Metadata written to {metadata_path}')
    print(f'Inputs: {metadata["input_names"]}')


if __name__ == '__main__':
    main()