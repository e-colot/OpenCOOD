# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import importlib
import os
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

        self._binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        self._input_names = [name for i, name in enumerate(self._binding_names)
                             if self.engine.binding_is_input(i)]
        self._output_names = [name for i, name in enumerate(self._binding_names)
                              if not self.engine.binding_is_input(i)]

    def infer(self, inputs):
        trt = self._trt
        cuda = self._cuda
        bindings = [None] * self.engine.num_bindings
        host_outputs = {}
        device_buffers = []

        for idx, name in enumerate(self._binding_names):
            if self.engine.binding_is_input(idx):
                if name not in inputs:
                    raise KeyError(f'Missing TensorRT input binding: {name}')
                arr = np.ascontiguousarray(inputs[name])
                self.context.set_binding_shape(idx, arr.shape)
                dptr = cuda.mem_alloc(arr.nbytes)
                cuda.memcpy_htod(dptr, arr)
                bindings[idx] = int(dptr)
                device_buffers.append(dptr)

        for idx, name in enumerate(self._binding_names):
            if self.engine.binding_is_input(idx):
                continue

            shape = tuple(self.context.get_binding_shape(idx))
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


def main():
    args = parse_args()

    runner = TensorRTRunner(args.engine_path)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_stat = _init_result_stat()

    pbar = tqdm(enumerate(data_loader),
                total=len(data_loader),
                desc='TensorRT Inference')

    for i, batch_data in pbar:
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
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

    _calculate_and_save(result_stat,
                        args.model_dir,
                        args.output_yaml,
                        args.global_sort_detections)


if __name__ == '__main__':
    main()
