# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import math
import os
import threading
import time

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.export_onnx import (
    OpenCoodExportWrapper,
    _build_trt_profile_hints,
    _collect_extra_keys,
    _maybe_replace_validate_split,
    _sanitize_tensor_for_export,
    _select_trace_batch_and_profile_bounds,
    _shape_profile_for_input,
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
    parser.add_argument('--dynamic_shapes', action='store_true',
                        help='Compile direct TensorRT artifact with dynamic input shape ranges')
    parser.add_argument('--max_search_batches', type=int, default=0,
                        help='Search this many batches for tracing/profile bounds (0 = full dataset)')
    parser.add_argument('--scan_log_interval', type=int, default=50,
                        help='Print Stage 3 scan progress every N batches (0 disables)')
    parser.add_argument('--profile_yaml', type=str, default='',
                        help='Optional metadata yaml with suggested_trt_profile to skip Stage 3 full scan')
    parser.add_argument('--max_voxels', type=int, default=0,
                        help='Override max profile voxel count for voxel_* inputs (0 = keep observed/default)')
    parser.add_argument('--max_voxels_margin', '--max_voxel_margin',
                        dest='max_voxels_margin', type=float, default=1.0,
                        help='Safety factor applied to detected max voxel profile count (default: 1.0)')
    parser.add_argument('--heartbeat_sec', type=int, default=30,
                        help='Print periodic compile heartbeat every N seconds (0 disables)')
    return parser.parse_args()


def _default_output_path(model_dir):
    return os.path.join(model_dir, 'pipeline_b', 'model_trt.ts')


def _default_metadata_path(output_path):
    return output_path + '.meta.yaml'


def _enabled_precisions(precision):
    if precision == 'fp16':
        return {torch.float, torch.half}
    return {torch.float}


def _float_dtype_for_precision(precision):
    if precision == 'fp16':
        return torch.float16
    return torch.float32


def _print_and_apply_voxel_profile_limits(profile_max_shapes, max_voxels, max_voxels_margin):
    if not profile_max_shapes:
        print('Detected max voxel profile count: unavailable (no observed profile shapes).')
        return profile_max_shapes

    updated_shapes = {
        name: [int(v) for v in shape]
        for name, shape in profile_max_shapes.items()
    }
    target_names = ['voxel_features', 'voxel_coords', 'voxel_num_points']
    detected_values = [updated_shapes[name][0]
                       for name in target_names
                       if name in updated_shapes and updated_shapes[name]]

    if detected_values:
        detected_max = max(detected_values)
        print(f'Detected max voxel profile count: {detected_max}')
    else:
        print('Detected max voxel profile count: unavailable (voxel_* inputs missing in observed profile shapes).')

    margin = max(1.0, float(max_voxels_margin))
    summary = {}
    changed = []
    for name in target_names:
        dims = updated_shapes.get(name)
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

    return updated_shapes


def _dynamic_input_spec(torch_tensorrt, name, tensor, observed_max_shapes, max_voxels, max_voxels_margin):
    shape = [int(v) for v in tensor.shape]
    observed_max_shape = observed_max_shapes.get(name)
    min_shape, opt_shape, max_shape = _shape_profile_for_input(name,
                                                               shape,
                                                               observed_max_shape=observed_max_shape)

    if name in ['voxel_features', 'voxel_coords', 'voxel_num_points'] and max_shape:
        base_value = int(max_shape[0])
        margin_value = int(math.ceil(base_value * max(1.0, float(max_voxels_margin))))
        max_shape[0] = max(base_value, int(max_voxels), margin_value)

    return torch_tensorrt.Input(
        min_shape=tuple(min_shape),
        opt_shape=tuple(opt_shape),
        max_shape=tuple(max_shape),
        dtype=tensor.dtype,
        name=name,
    )


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


def _load_profile_max_shapes(args):
    candidates = []
    if args.profile_yaml:
        candidates.append(args.profile_yaml)
    candidates.append(os.path.join(args.model_dir, 'pipeline_a', 'model.onnx.meta.yaml'))

    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue

        try:
            meta = yaml_utils.load_yaml(candidate)
        except Exception as exc:
            print(f'Skip metadata profile {candidate}: {exc}')
            continue

        profile = meta.get('suggested_trt_profile', {})
        max_shapes = profile.get('max_shapes', '')
        parsed = _parse_shape_profile(max_shapes)
        if parsed:
            print(f'Using profile max shapes from metadata: {candidate}')
            return parsed

    return {}


def _run_wrapper_sanity_check(wrapper, compile_inputs):
    print('Sanity check: running wrapper forward pass on trace inputs ...')
    start_t = time.perf_counter()
    with torch.no_grad():
        outputs = wrapper(*compile_inputs)
    elapsed = time.perf_counter() - start_t

    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise RuntimeError('Wrapper sanity check failed: expected tuple/list with at least 2 outputs (psm, rm).')

    psm, rm = outputs[0], outputs[1]
    if not isinstance(psm, torch.Tensor) or not isinstance(rm, torch.Tensor):
        raise RuntimeError('Wrapper sanity check failed: outputs must be torch.Tensor values for psm and rm.')

    print(
        f'Sanity check passed in {elapsed:.2f}s '
        f'(psm: shape={tuple(psm.shape)}, dtype={psm.dtype}; '
        f'rm: shape={tuple(rm.shape)}, dtype={rm.dtype})'
    )


def _compile_with_heartbeat(torch_tensorrt,
                            scripted_wrapper,
                            compile_spec_inputs,
                            enabled_precisions,
                            workspace_bytes,
                            heartbeat_sec):
    stop_event = threading.Event()

    def _heartbeat():
        start_t = time.perf_counter()
        while not stop_event.wait(max(1, heartbeat_sec)):
            elapsed = time.perf_counter() - start_t
            print(f'[compile] still running... elapsed={elapsed:.1f}s')

    hb_thread = None
    if heartbeat_sec > 0:
        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()

    try:
        return torch_tensorrt.compile(
            scripted_wrapper,
            ir='torchscript',
            inputs=compile_spec_inputs,
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_bytes,
            truncate_long_and_double=True,
        )
    finally:
        stop_event.set()
        if hb_thread is not None:
            hb_thread.join(timeout=1.0)


def main():
    args = parse_args()
    total_start_t = time.perf_counter()

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

    print('Stage 1/6: Building dataset for trace input')
    stage_t = time.perf_counter()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=0,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)
    print(f'Stage 1/6 done in {time.perf_counter() - stage_t:.2f}s')

    print('Stage 2/6: Creating and loading model')
    stage_t = time.perf_counter()
    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()
    print(f'Stage 2/6 done in {time.perf_counter() - stage_t:.2f}s')

    print('Stage 3/6: Preparing trace batch/profile bounds')
    stage_t = time.perf_counter()
    profile_max_shapes = _load_profile_max_shapes(args)
    if profile_max_shapes:
        print('Stage 3 fast path: taking first batch for tracing (profile bounds from metadata)')
        batch_data = next(iter(loader))
    else:
        print('Stage 3 metadata profile unavailable; falling back to dataset scan')
        if args.max_search_batches == 0:
            print('Stage 3 scan target: full dataset (set --max_search_batches to limit scan time)')
        else:
            print(f'Stage 3 scan target: first {args.max_search_batches} batches')
        batch_data, profile_max_shapes = _select_trace_batch_and_profile_bounds(
            loader,
            args.fusion_method,
            args.max_search_batches,
            scan_log_interval=args.scan_log_interval,
        )

    profile_max_shapes = _print_and_apply_voxel_profile_limits(profile_max_shapes,
                                                               args.max_voxels,
                                                               args.max_voxels_margin)
    print(f'Stage 3/6 done in {time.perf_counter() - stage_t:.2f}s')

    print('Stage 4/6: Preparing trace tensors')
    stage_t = time.perf_counter()
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
    input_names = base_names + extra_keys
    print(f'Stage 4/6 done in {time.perf_counter() - stage_t:.2f}s')

    print('Stage 5/6: Tracing wrapper for Torch-TensorRT compile')
    stage_t = time.perf_counter()
    _run_wrapper_sanity_check(wrapper, compile_inputs)
    scripted_wrapper = torch.jit.trace(wrapper, compile_inputs, strict=False)
    with torch.no_grad():
        _ = scripted_wrapper(*compile_inputs)
    print(f'Stage 5/6 done in {time.perf_counter() - stage_t:.2f}s')

    enabled_precisions = _enabled_precisions(args.precision)
    compile_spec_inputs = compile_inputs
    if args.dynamic_shapes:
        compile_spec_inputs = [
            _dynamic_input_spec(torch_tensorrt,
                                name,
                                tensor,
                                profile_max_shapes,
                                args.max_voxels,
                                args.max_voxels_margin)
            for name, tensor in zip(input_names, compile_inputs)
        ]
    print(
        'Stage 6/6: Compiling direct TensorRT artifact '
        f'with precision={args.precision} (heartbeat every {args.heartbeat_sec}s)'
    )
    stage_t = time.perf_counter()
    trt_module = _compile_with_heartbeat(torch_tensorrt,
                                         scripted_wrapper,
                                         compile_spec_inputs,
                                         enabled_precisions,
                                         args.workspace_mb * 1024 * 1024,
                                         args.heartbeat_sec)
    print(f'Stage 6/6 done in {time.perf_counter() - stage_t:.2f}s')

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
        'dynamic_shapes': bool(args.dynamic_shapes),
        'input_names': input_names,
        'trace_input_shapes': {
            name: [int(v) for v in tensor.shape]
            for name, tensor in zip(input_names, compile_inputs)
        },
        'trace_input_dtypes': {
            name: str(tensor.dtype).replace('torch.', '')
            for name, tensor in zip(input_names, compile_inputs)
        },
        'suggested_trt_profile': _build_trt_profile_hints(
            input_names,
            compile_inputs,
            observed_max_shapes=profile_max_shapes,
        ),
    }
    yaml_utils.save_yaml(metadata, metadata_path)

    print('Direct TensorRT build completed')
    print(f'Metadata written to {metadata_path}')
    print(f'Inputs: {metadata["input_names"]}')
    print(f'Total elapsed: {time.perf_counter() - total_start_t:.2f}s')


if __name__ == '__main__':
    main()