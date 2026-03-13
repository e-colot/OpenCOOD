# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import importlib
import os
import time
import warnings
from collections import OrderedDict

import numpy as np
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
    parser = argparse.ArgumentParser(description='OpenCOOD TensorRT inference + AP eval')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Checkpoint directory used for data config and artifacts')
    parser.add_argument('--engine_path', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--global_sort_detections', action='store_true')
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='Use validation split by replacing test with validate')
    parser.add_argument('--output_yaml', type=str, default='pipeline_a/eval_tensorrt.yaml',
                        help='Output AP yaml filename under model_dir')
    parser.add_argument('--profile_runtime', action='store_true',
                        help='Collect end-to-end per-sample latency and peak GPU memory metrics')
    parser.add_argument('--profile_warmup_steps', type=int, default=10,
                        help='Number of initial samples ignored for runtime/memory statistics')
    parser.add_argument('--profile_max_samples', type=int, default=0,
                        help='Maximum profiled samples after warmup (0 = profile all samples)')
    parser.add_argument('--profile_yaml', type=str, default='pipeline_a/profile_tensorrt.yaml',
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


class TensorRTRunner:
    def __init__(self, engine_path):
        try:
            self._trt = importlib.import_module('tensorrt')
            importlib.import_module('pycuda.autoinit')
            self._cuda = importlib.import_module('pycuda.driver')
        except ImportError as exc:
            raise ImportError(
                'TensorRT inference requires tensorrt and pycuda. '
                'Install TensorRT Python bindings and pycuda first.'
            ) from exc

        trt = self._trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine from {engine_path}')
        self.context = self.engine.create_execution_context()

        self._use_io_tensor_api = hasattr(self.engine, 'num_io_tensors')
        if self._use_io_tensor_api:
            self._tensor_names = [self.engine.get_tensor_name(i)
                                  for i in range(self.engine.num_io_tensors)]
            input_mode = self._trt.TensorIOMode.INPUT
            self._input_names = [name for name in self._tensor_names
                                 if self.engine.get_tensor_mode(name) == input_mode]
            self._output_names = [name for name in self._tensor_names
                                  if self.engine.get_tensor_mode(name) != input_mode]
        else:
            self._binding_names = [self.engine.get_binding_name(i)
                                   for i in range(self.engine.num_bindings)]
            self._binding_index = {name: i for i, name in enumerate(self._binding_names)}
            self._input_names = [name for i, name in enumerate(self._binding_names)
                                 if self.engine.binding_is_input(i)]
            self._output_names = [name for i, name in enumerate(self._binding_names)
                                  if not self.engine.binding_is_input(i)]

    def _expected_numpy_dtype(self, name):
        trt = self._trt
        if self._use_io_tensor_api:
            return trt.nptype(self.engine.get_tensor_dtype(name))
        binding_idx = self._binding_index[name]
        return trt.nptype(self.engine.get_binding_dtype(binding_idx))

    def _prepare_input_array(self, name, array):
        expected_dtype = self._expected_numpy_dtype(name)
        if array.dtype != expected_dtype:
            array = array.astype(expected_dtype, copy=False)
        return np.ascontiguousarray(array)

    def _infer_legacy_bindings(self, inputs):
        trt = self._trt
        cuda = self._cuda
        bindings = [None] * self.engine.num_bindings
        host_outputs = {}
        device_buffers = []

        for idx, name in enumerate(self._binding_names):
            if self.engine.binding_is_input(idx):
                if name not in inputs:
                    raise KeyError(f'Missing TensorRT input binding: {name}')
                arr = self._prepare_input_array(name, inputs[name])
                self.context.set_binding_shape(idx, arr.shape)
                dptr = cuda.mem_alloc(arr.nbytes)
                cuda.memcpy_htod(dptr, arr)
                bindings[idx] = int(dptr)
                device_buffers.append(dptr)

        for idx, name in enumerate(self._binding_names):
            if self.engine.binding_is_input(idx):
                continue

            shape = tuple(self.context.get_binding_shape(idx))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f'Unresolved TensorRT output shape for {name}: {shape}. '
                    'This usually means input shape is outside the engine optimization profile.'
                )

            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            host_arr = np.empty(shape, dtype=dtype)
            dptr = cuda.mem_alloc(host_arr.nbytes)
            bindings[idx] = int(dptr)
            host_outputs[name] = (host_arr, dptr)
            device_buffers.append(dptr)

        self.context.execute_v2(bindings)

        outputs = {}
        for name, (host_arr, dptr) in host_outputs.items():
            cuda.memcpy_dtoh(host_arr, dptr)
            outputs[name] = host_arr

        for dptr in device_buffers:
            dptr.free()

        return outputs

    def _infer_io_tensors(self, inputs):
        trt = self._trt
        cuda = self._cuda
        stream = cuda.Stream()
        host_outputs = {}
        device_buffers = []

        for name in self._input_names:
            if name not in inputs:
                raise KeyError(f'Missing TensorRT input binding: {name}')
            arr = self._prepare_input_array(name, inputs[name])
            self.context.set_input_shape(name, arr.shape)
            dptr = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(dptr, arr, stream)
            self.context.set_tensor_address(name, int(dptr))
            device_buffers.append(dptr)

        for name in self._output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f'Unresolved TensorRT output shape for {name}: {shape}. '
                    'This usually means input shape is outside the engine optimization profile.'
                )
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_arr = np.empty(shape, dtype=dtype)
            dptr = cuda.mem_alloc(host_arr.nbytes)
            self.context.set_tensor_address(name, int(dptr))
            host_outputs[name] = (host_arr, dptr)
            device_buffers.append(dptr)

        if not self.context.execute_async_v3(stream.handle):
            raise RuntimeError('TensorRT execution failed (execute_async_v3 returned False).')

        outputs = {}
        for name, (host_arr, dptr) in host_outputs.items():
            cuda.memcpy_dtoh_async(host_arr, dptr, stream)
            outputs[name] = host_arr

        stream.synchronize()

        for dptr in device_buffers:
            dptr.free()

        return outputs

    def infer(self, inputs):
        if self._use_io_tensor_api:
            return self._infer_io_tensors(inputs)
        return self._infer_legacy_bindings(inputs)


def _build_backend_inputs(cav_content, input_names):
    processed = cav_content['processed_lidar']
    model_inputs = {}

    for name in input_names:
        if name in processed:
            tensor = processed[name]
        elif name in cav_content and isinstance(cav_content[name], torch.Tensor):
            tensor = cav_content[name]
        else:
            raise KeyError(f'TensorRT input {name} not found in cav content')

        model_inputs[name] = tensor.detach().cpu().numpy()

    return model_inputs


def _run_trt_single(runner, cav_content):
    model_inputs = _build_backend_inputs(cav_content, runner._input_names)
    outputs = runner.infer(model_inputs)

    # Keep psm/rm convention with fallback to first two outputs.
    if 'psm' in outputs:
        psm_np = outputs['psm']
    else:
        psm_np = outputs[runner._output_names[0]]

    if 'rm' in outputs:
        rm_np = outputs['rm']
    else:
        rm_np = outputs[runner._output_names[1]]

    device = cav_content['processed_lidar']['voxel_features'].device
    psm = torch.from_numpy(psm_np).to(device)
    rm = torch.from_numpy(rm_np).to(device)

    return {'psm': psm, 'rm': rm}


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

    # Initialize TensorRT runtime only after worker processes are created.
    # This avoids CUDA context inheritance issues when num_workers > 0.
    runner = TensorRTRunner(args.engine_path)

    # Keep tensors on CPU for TensorRT path; mixing torch CUDA transfers with
    # pycuda runtime management can trigger unstable execution.
    device = torch.device('cpu')
    result_stat = _init_result_stat()
    latency_ms = []
    peak_mem_mb = []

    pbar = tqdm(enumerate(data_loader),
                total=len(data_loader),
                desc='TensorRT Inference')

    for i, batch_data in pbar:
        with torch.no_grad():
            if args.profile_runtime and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                _maybe_sync_cuda(device)
            start_time = time.perf_counter() if args.profile_runtime else None

            output_dict = OrderedDict()

            if args.fusion_method == 'late':
                for cav_id, cav_content in batch_data.items():
                    output_dict[cav_id] = _run_trt_single(runner, cav_content)
            elif args.fusion_method in ['early', 'intermediate']:
                output_dict['ego'] = _run_trt_single(runner, batch_data['ego'])
            else:
                raise NotImplementedError('Only early, late and intermediate fusion are supported.')

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            if args.save_npy:
                npy_save_path = os.path.join(args.model_dir, 'npy_tensorrt')
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
            'backend': 'tensorrt',
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
