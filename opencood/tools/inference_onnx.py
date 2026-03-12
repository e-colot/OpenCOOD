# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils

warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in intersection',
    category=RuntimeWarning,
)


def parse_args():
    parser = argparse.ArgumentParser(description='OpenCOOD ONNX inference + AP eval')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Checkpoint directory used for data config and artifacts')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--providers', type=str,
                        default='CUDAExecutionProvider,CPUExecutionProvider',
                        help='Comma-separated ONNX Runtime providers')
    parser.add_argument('--cuda_device_id', type=int, default=0,
                        help='CUDA device id for ONNX Runtime CUDAExecutionProvider')
    parser.add_argument('--allow_cpu_fallback', action='store_true',
                        help='Allow fallback to CPU when CUDAExecutionProvider is requested but unavailable')
    parser.add_argument('--graph_optimization_level', type=str, default='disable',
                        choices=['all', 'extended', 'basic', 'disable'],
                        help='ONNX Runtime graph optimization level')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Dataloader worker count')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='Use global confidence sorting in AP computation')
    parser.add_argument('--save_npy', action='store_true',
                        help='Save prediction/gt arrays in npy_test folder')
    parser.add_argument('--test', action='store_true',
                        help='Use validation split by replacing test with validate')
    parser.add_argument('--output_yaml', type=str, default='pipeline_a/eval_onnx.yaml',
                        help='Output AP yaml filename under model_dir')
    parser.add_argument('--profile_runtime', action='store_true',
                        help='Collect end-to-end per-sample latency and peak GPU memory metrics')
    parser.add_argument('--profile_warmup_steps', type=int, default=10,
                        help='Number of initial samples ignored for runtime/memory statistics')
    parser.add_argument('--profile_max_samples', type=int, default=0,
                        help='Maximum profiled samples after warmup (0 = profile all samples)')
    parser.add_argument('--profile_yaml', type=str, default='pipeline_a/profile_onnx.yaml',
                        help='Profiling output yaml path under model_dir')
    return parser.parse_args()


def _init_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
    }


def _maybe_replace_validate_split(hypes, use_test):
    if not use_test or 'validate_dir' not in hypes:
        return

    validate_dir = hypes['validate_dir'].rstrip('/')
    if validate_dir.endswith('test'):
        hypes['validate_dir'] = validate_dir[:-4] + 'validate'
    else:
        hypes['validate_dir'] = os.path.join(os.path.dirname(validate_dir),
                                             'validate')


def _torch_to_numpy(t):
    t_cpu = t.detach().cpu()
    if t_cpu.dtype == torch.float64:
        t_cpu = t_cpu.float()
    return t_cpu.numpy()


def _onnx_type_to_numpy_dtype(onnx_type):
    mapping = {
        'tensor(float)': np.float32,
        'tensor(double)': np.float64,
        'tensor(int64)': np.int64,
        'tensor(int32)': np.int32,
        'tensor(int16)': np.int16,
        'tensor(int8)': np.int8,
        'tensor(uint8)': np.uint8,
        'tensor(bool)': np.bool_,
    }
    return mapping.get(onnx_type)


def _validate_onnx_inputs(session, model_inputs):
    session_inputs = session.get_inputs()
    missing = [x.name for x in session_inputs if x.name not in model_inputs]
    if missing:
        raise KeyError(f'Missing ONNX inputs: {missing}')

    for input_meta in session_inputs:
        arr = model_inputs[input_meta.name]
        expected_dtype = _onnx_type_to_numpy_dtype(input_meta.type)
        if expected_dtype is not None and arr.dtype != expected_dtype:
            try:
                model_inputs[input_meta.name] = arr.astype(expected_dtype, copy=False)
            except TypeError as exc:
                raise TypeError(
                    f'Input {input_meta.name} dtype mismatch: got {arr.dtype}, '
                    f'expected {expected_dtype}'
                ) from exc

        expected_shape = input_meta.shape
        if isinstance(expected_shape, list):
            actual_shape = model_inputs[input_meta.name].shape
            if len(expected_shape) != len(actual_shape):
                raise ValueError(
                    f'Input {input_meta.name} rank mismatch: '
                    f'expected {expected_shape}, got {actual_shape}'
                )
            for idx, (exp_dim, got_dim) in enumerate(zip(expected_shape, actual_shape)):
                if isinstance(exp_dim, int) and exp_dim != got_dim:
                    raise ValueError(
                        f'Input {input_meta.name} shape mismatch at dim {idx}: '
                        f'expected {exp_dim}, got {got_dim} (full expected={expected_shape}, got={actual_shape})'
                    )


def _build_onnx_inputs(cav_content, input_names):
    processed = cav_content['processed_lidar']
    model_inputs = {}

    for name in input_names:
        if name in processed:
            tensor = processed[name]
        elif name in cav_content and isinstance(cav_content[name], torch.Tensor):
            tensor = cav_content[name]
        else:
            raise KeyError(f'ONNX input {name} not found in cav content')
        model_inputs[name] = _torch_to_numpy(tensor)

    return model_inputs


def _run_onnx_single(session, cav_content, output_device):
    input_names = [x.name for x in session.get_inputs()]
    output_names = [x.name for x in session.get_outputs()]

    model_inputs = _build_onnx_inputs(cav_content, input_names)
    _validate_onnx_inputs(session, model_inputs)
    outputs = session.run(output_names, model_inputs)

    # By convention, export_onnx.py writes psm then rm. Keep fallback by name.
    out_map = dict(zip(output_names, outputs))
    psm_np = out_map.get('psm', outputs[0])
    rm_np = out_map.get('rm', outputs[1])

    # Keep parity with PyTorch inference path by running post-process on the same device.
    psm = torch.from_numpy(psm_np).to(output_device)
    rm = torch.from_numpy(rm_np).to(output_device)
    return {'psm': psm, 'rm': rm}


def _create_session(args):
    requested_providers = [p.strip() for p in args.providers.split(',') if p.strip()]
    available_providers = ort.get_available_providers()

    if not requested_providers:
        raise ValueError('No ONNX Runtime providers were requested.')

    cuda_requested = 'CUDAExecutionProvider' in requested_providers
    cuda_available = 'CUDAExecutionProvider' in available_providers
    if cuda_requested and not cuda_available and not args.allow_cpu_fallback:
        raise RuntimeError(
            'CUDAExecutionProvider requested but not available. '
            'Install onnxruntime-gpu and verify CUDA/cuDNN compatibility, '
            'or rerun with --allow_cpu_fallback.'
        )

    resolved_providers = [p for p in requested_providers if p in available_providers]
    if not resolved_providers:
        raise RuntimeError(
            f'None of requested providers are available. '
            f'requested={requested_providers}, available={available_providers}'
        )

    provider_options = []
    for provider in resolved_providers:
        if provider == 'CUDAExecutionProvider':
            provider_options.append({'device_id': args.cuda_device_id})
        else:
            provider_options.append({})

    optimization_map = {
        'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'disable': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = optimization_map[args.graph_optimization_level]

    session = ort.InferenceSession(
        args.onnx_model,
        sess_options=session_options,
        providers=resolved_providers,
        provider_options=provider_options,
    )

    print(f'ONNX Runtime available providers: {available_providers}')
    print(f'ONNX Runtime requested providers: {requested_providers}')
    print(f'ONNX Runtime active providers: {session.get_providers()}')
    print(f'ONNX Runtime graph optimization: {args.graph_optimization_level}')

    if cuda_requested and 'CUDAExecutionProvider' not in session.get_providers():
        if args.allow_cpu_fallback:
            warnings.warn('CUDA provider not active; running on CPUExecutionProvider.')
        else:
            raise RuntimeError('CUDAExecutionProvider is not active in the created session.')

    return session


def _calculate_and_save(result_stat, model_dir, output_yaml, global_sort_detections):
    dump_dict = {}
    ap_30, _, _ = eval_utils.calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, mrec_50, mpre_50 = eval_utils.calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, mrec_70, mpre_70 = eval_utils.calculate_ap(result_stat, 0.70, global_sort_detections)

    dump_dict.update({
        'ap30': ap_30,
        'ap_50': ap_50,
        'ap_70': ap_70,
        'mpre_50': mpre_50,
        'mrec_50': mrec_50,
        'mpre_70': mpre_70,
        'mrec_70': mrec_70,
    })

    out_path = os.path.join(model_dir, output_yaml)
    yaml_utils.save_yaml(dump_dict, out_path)

    print(f'AP results saved to {out_path}')
    print(
        f'The Average Precision at IOU 0.3 is {ap_30:.4f}\n'
        f'The Average Precision at IOU 0.5 is {ap_50:.4f}\n'
        f'The Average Precision at IOU 0.7 is {ap_70:.4f}\n'
    )


def _maybe_sync_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _build_profile_summary(latencies_ms, peak_mem_mb):
    if not latencies_ms:
        return {'profiled_samples': 0}

    latency_arr = np.array(latencies_ms, dtype=np.float64)
    mem_arr = np.array(peak_mem_mb, dtype=np.float64) if peak_mem_mb else np.array([], dtype=np.float64)

    summary = {
        'profiled_samples': int(latency_arr.shape[0]),
        'latency_ms_mean': float(np.mean(latency_arr)),
        'latency_ms_p50': float(np.percentile(latency_arr, 50)),
        'latency_ms_p95': float(np.percentile(latency_arr, 95)),
        'latency_ms_max': float(np.max(latency_arr)),
    }

    if mem_arr.size > 0:
        summary.update({
            'peak_gpu_mem_mb_mean': float(np.mean(mem_arr)),
            'peak_gpu_mem_mb_p50': float(np.percentile(mem_arr, 50)),
            'peak_gpu_mem_mb_p95': float(np.percentile(mem_arr, 95)),
            'peak_gpu_mem_mb_max': float(np.max(mem_arr)),
        })

    return summary


def main():
    args = parse_args()
    session = _create_session(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hypes = yaml_utils.load_yaml(None, args)
    _maybe_replace_validate_split(hypes, args.test)

    print('Dataset Building')
    dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    result_stat = _init_result_stat()
    latency_ms = []
    peak_mem_mb = []

    pbar = tqdm(enumerate(data_loader),
                total=len(data_loader),
                desc='ONNX Inference')

    for i, batch_data in pbar:
        with torch.no_grad():
            if args.profile_runtime and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                _maybe_sync_cuda(device)
            start_time = time.perf_counter() if args.profile_runtime else None

            batch_data = train_utils.to_device(batch_data, device)
            output_dict = OrderedDict()

            if args.fusion_method == 'late':
                for cav_id, cav_content in batch_data.items():
                    output_dict[cav_id] = _run_onnx_single(session, cav_content, device)
            elif args.fusion_method in ['early', 'intermediate']:
                output_dict['ego'] = _run_onnx_single(session, batch_data['ego'], device)
            else:
                raise NotImplementedError('Only early, late and intermediate fusion are supported.')

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            if args.save_npy:
                npy_save_path = os.path.join(args.model_dir, 'npy_onnx')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego']['origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if args.profile_runtime:
                _maybe_sync_cuda(device)
                sample_latency_ms = (time.perf_counter() - start_time) * 1000.0
                should_record = i >= args.profile_warmup_steps and (
                    args.profile_max_samples <= 0 or len(latency_ms) < args.profile_max_samples
                )
                if should_record:
                    latency_ms.append(sample_latency_ms)
                    if device.type == 'cuda':
                        peak_mem_mb.append(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

    _calculate_and_save(result_stat,
                        args.model_dir,
                        args.output_yaml,
                        args.global_sort_detections)

    if args.profile_runtime:
        profile_summary = _build_profile_summary(latency_ms, peak_mem_mb)
        profile_summary.update({
            'backend': 'onnxruntime',
            'fusion_method': args.fusion_method,
            'warmup_steps': int(args.profile_warmup_steps),
            'profile_max_samples': int(args.profile_max_samples),
        })
        out_path = os.path.join(args.model_dir, args.profile_yaml)
        yaml_utils.save_yaml(profile_summary, out_path)
        print(f'Runtime profile saved to {out_path}')
        print(profile_summary)


if __name__ == '__main__':
    main()
