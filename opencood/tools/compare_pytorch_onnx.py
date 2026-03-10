# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors

import argparse
from collections import OrderedDict

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and ONNX raw outputs (psm/rm) before post-process'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Trained checkpoint directory')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--fusion_method', required=True, type=str,
                        choices=['late', 'early', 'intermediate'])
    parser.add_argument('--providers', type=str,
                        default='CUDAExecutionProvider,CPUExecutionProvider',
                        help='Comma-separated ONNX Runtime providers')
    parser.add_argument('--graph_optimization_level', type=str, default='disable',
                        choices=['all', 'extended', 'basic', 'disable'],
                        help='ONNX Runtime graph optimization level')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to compare')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Dataloader worker count')
    parser.add_argument('--test', action='store_true',
                        help='Use validation split by replacing test with validate')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='PyTorch device, defaults to cuda if available else cpu')
    return parser.parse_args()


def _maybe_replace_validate_split(hypes, use_test):
    if not use_test or 'validate_dir' not in hypes:
        return

    validate_dir = hypes['validate_dir'].rstrip('/')
    if validate_dir.endswith('test'):
        hypes['validate_dir'] = validate_dir[:-4] + 'validate'


def _torch_to_numpy(t):
    t_cpu = t.detach().cpu()
    if t_cpu.dtype == torch.float64:
        t_cpu = t_cpu.float()
    return t_cpu.numpy()


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


def _adapt_inputs_to_session_contract(session, model_inputs):
    for input_meta in session.get_inputs():
        arr = model_inputs[input_meta.name]
        expected_dtype = _onnx_type_to_numpy_dtype(input_meta.type)
        if expected_dtype is not None and arr.dtype != expected_dtype:
            model_inputs[input_meta.name] = arr.astype(expected_dtype, copy=False)


def _create_session(onnx_model, providers, graph_optimization_level):
    requested = [p.strip() for p in providers.split(',') if p.strip()]
    available = ort.get_available_providers()
    resolved = [p for p in requested if p in available]
    if not resolved:
        raise RuntimeError(f'No requested providers available. requested={requested}, available={available}')

    optimization_map = {
        'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'disable': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = optimization_map[graph_optimization_level]

    session = ort.InferenceSession(onnx_model,
                                   sess_options=session_options,
                                   providers=resolved)
    print(f'ONNX providers available: {available}')
    print(f'ONNX providers active: {session.get_providers()}')
    print(f'ONNX graph optimization: {graph_optimization_level}')
    return session


def _to_ego_content(batch_data, fusion_method):
    if fusion_method in ['early', 'intermediate']:
        return batch_data['ego']

    first_key = next(iter(batch_data.keys()))
    return batch_data[first_key]


def main():
    args = parse_args()

    device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)

    hypes = yaml_utils.load_yaml(None, args)
    _maybe_replace_validate_split(hypes, args.test)

    dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    model = train_utils.create_model(hypes).to(device)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.eval()

    session = _create_session(args.onnx_model,
                              args.providers,
                              args.graph_optimization_level)
    input_names = [x.name for x in session.get_inputs()]
    output_names = [x.name for x in session.get_outputs()]

    max_psm_abs = 0.0
    max_rm_abs = 0.0
    mean_psm_abs = []
    mean_rm_abs = []

    total = min(args.num_samples, len(data_loader))
    pbar = tqdm(enumerate(data_loader), total=total, desc='Compare PT vs ONNX')

    for idx, batch_data in pbar:
        if idx >= args.num_samples:
            break

        with torch.no_grad():
            batch_data_torch = train_utils.to_device(batch_data, device)
            output_dict_pt = OrderedDict()

            if args.fusion_method == 'late':
                for cav_id, cav_content in batch_data_torch.items():
                    output_dict_pt[cav_id] = model(cav_content)
                cav_content_onnx = _to_ego_content(batch_data_torch, 'late')
            else:
                output_dict_pt['ego'] = model(batch_data_torch['ego'])
                cav_content_onnx = batch_data_torch['ego']

            model_inputs = _build_onnx_inputs(cav_content_onnx, input_names)
            _adapt_inputs_to_session_contract(session, model_inputs)
            outputs = session.run(output_names, model_inputs)
            out_map = dict(zip(output_names, outputs))
            psm_onnx = torch.from_numpy(out_map.get('psm', outputs[0])).to(device)
            rm_onnx = torch.from_numpy(out_map.get('rm', outputs[1])).to(device)

            if args.fusion_method == 'late':
                pt_ref = output_dict_pt[next(iter(output_dict_pt.keys()))]
            else:
                pt_ref = output_dict_pt['ego']

            psm_pt = pt_ref['psm']
            rm_pt = pt_ref['rm']

            if psm_pt.shape != psm_onnx.shape or rm_pt.shape != rm_onnx.shape:
                raise ValueError(
                    f'Shape mismatch at sample {idx}: '
                    f'psm {psm_pt.shape} vs {psm_onnx.shape}, '
                    f'rm {rm_pt.shape} vs {rm_onnx.shape}'
                )

            psm_abs = torch.abs(psm_pt - psm_onnx)
            rm_abs = torch.abs(rm_pt - rm_onnx)

            max_psm_abs = max(max_psm_abs, psm_abs.max().item())
            max_rm_abs = max(max_rm_abs, rm_abs.max().item())
            mean_psm_abs.append(psm_abs.mean().item())
            mean_rm_abs.append(rm_abs.mean().item())

    print('=== PyTorch vs ONNX Raw Output Delta ===')
    print(f'Samples compared: {total}')
    print(f'PSM max abs diff: {max_psm_abs:.8f}')
    print(f'PSM mean abs diff: {float(np.mean(mean_psm_abs)):.8f}')
    print(f'RM max abs diff: {max_rm_abs:.8f}')
    print(f'RM mean abs diff: {float(np.mean(mean_rm_abs)):.8f}')


if __name__ == '__main__':
    main()
