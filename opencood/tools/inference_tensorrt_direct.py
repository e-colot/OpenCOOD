# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
import os
import warnings
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils, inference_utils
from opencood.tools.export_onnx import _maybe_replace_validate_split
from opencood.utils import eval_utils

warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in intersection',
    category=RuntimeWarning,
)


def parse_args():
    parser = argparse.ArgumentParser(description='OpenCOOD direct Torch-TensorRT inference + AP eval')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Checkpoint directory used for data config and artifacts')
    parser.add_argument('--trt_module', type=str, required=True,
                        help='Path to direct Torch-TensorRT TorchScript artifact')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Metadata yaml written during direct build. Defaults to <trt_module>.meta.yaml')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--global_sort_detections', action='store_true')
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--test', action='store_true',
                        help='Use validation split by replacing test with validate')
    parser.add_argument('--output_yaml', type=str, default='pipeline_b/eval_tensorrt_direct.yaml',
                        help='Output AP yaml filename under model_dir')
    return parser.parse_args()


def _init_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
    }


def _default_metadata_path(trt_module_path):
    return trt_module_path + '.meta.yaml'


def _resolve_precision_dtype(metadata):
    precision = metadata.get('precision', 'fp32')
    if precision == 'fp16':
        return torch.float16
    return torch.float32


def _build_inputs(cav_content, input_names, float_dtype):
    processed = cav_content['processed_lidar']
    model_inputs = []

    for name in input_names:
        if name in processed:
            tensor = processed[name]
        elif name in cav_content and isinstance(cav_content[name], torch.Tensor):
            tensor = cav_content[name]
        else:
            raise KeyError(f'Direct TensorRT input {name} not found in cav content')

        if tensor.is_floating_point():
            tensor = tensor.to(float_dtype)
        model_inputs.append(tensor)

    return tuple(model_inputs)


def _run_single(trt_module, cav_content, input_names, float_dtype):
    model_inputs = _build_inputs(cav_content, input_names, float_dtype)
    outputs = trt_module(*model_inputs)

    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        psm, rm = outputs[0], outputs[1]
    else:
        raise RuntimeError('Direct TensorRT module output format is invalid. Expected tuple/list with psm and rm.')

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metadata_path = args.metadata or _default_metadata_path(args.trt_module)
    metadata = yaml_utils.load_yaml(metadata_path)
    input_names = metadata.get('input_names', [])
    if not input_names:
        raise ValueError(f'No input_names found in metadata: {metadata_path}')
    float_dtype = _resolve_precision_dtype(metadata)

    print(f'Loading direct TensorRT module: {args.trt_module}')
    trt_module = torch.jit.load(args.trt_module, map_location=device)
    trt_module.eval()

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

    pbar = tqdm(enumerate(data_loader),
                total=len(data_loader),
                desc='Direct TensorRT Inference')

    for i, batch_data in pbar:
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = OrderedDict()

            if args.fusion_method == 'late':
                for cav_id, cav_content in batch_data.items():
                    output_dict[cav_id] = _run_single(trt_module,
                                                      cav_content,
                                                      input_names,
                                                      float_dtype)
            elif args.fusion_method in ['early', 'intermediate']:
                output_dict['ego'] = _run_single(trt_module,
                                                 batch_data['ego'],
                                                 input_names,
                                                 float_dtype)
            else:
                raise NotImplementedError('Only early, late and intermediate fusion are supported.')

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)

            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

            if args.save_npy:
                npy_save_path = os.path.join(args.model_dir, 'npy_tensorrt_direct')
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