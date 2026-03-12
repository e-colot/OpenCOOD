# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
import warnings
from tqdm import tqdm

import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt

# removes a single warning at start of inference
warnings.filterwarnings(
    'ignore',
    message='invalid value encountered in intersection',
    category=RuntimeWarning,
)


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--test', action='store_true',
                        help='use validation split by replacing ../test/ with ../validate/ in validate_dir')
    parser.add_argument('--profile_runtime', action='store_true',
                        help='Collect end-to-end per-sample latency and peak GPU memory metrics')
    parser.add_argument('--profile_warmup_steps', type=int, default=10,
                        help='Number of initial samples ignored for runtime/memory statistics')
    parser.add_argument('--profile_max_samples', type=int, default=0,
                        help='Maximum profiled samples after warmup (0 = profile all samples)')
    parser.add_argument('--profile_yaml', type=str, default='pipeline_a/profile_pytorch.yaml',
                        help='Profiling output yaml path under model_dir')
    opt = parser.parse_args()
    return opt


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
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    if opt.test and 'validate_dir' in hypes:
        validate_dir = hypes['validate_dir'].rstrip('/')
        if validate_dir.endswith('test'):
            hypes['validate_dir'] = validate_dir[:-4] + 'validate'
        else:
            hypes['validate_dir'] = os.path.join(os.path.dirname(validate_dir),
                                                 'validate')

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    total_batches = len(data_loader)
    latency_ms = []
    peak_mem_mb = []
    pbar = tqdm(enumerate(data_loader),
                total=total_batches,
                desc='Inference')
    for i, batch_data in pbar:
        # print(i)
        with torch.no_grad():
            if opt.profile_runtime and device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                _maybe_sync_cuda(device)
            start_time = time.perf_counter() if opt.profile_runtime else None

            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

            if opt.profile_runtime:
                _maybe_sync_cuda(device)
                sample_latency_ms = (time.perf_counter() - start_time) * 1000.0
                should_record = i >= opt.profile_warmup_steps and (
                    opt.profile_max_samples <= 0 or len(latency_ms) < opt.profile_max_samples
                )
                if should_record:
                    latency_ms.append(sample_latency_ms)
                    if device.type == 'cuda':
                        peak_mem_mb.append(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)

    if opt.profile_runtime:
        profile_summary = _build_profile_summary(latency_ms, peak_mem_mb)
        profile_summary.update({
            'backend': 'pytorch',
            'fusion_method': opt.fusion_method,
            'warmup_steps': int(opt.profile_warmup_steps),
            'profile_max_samples': int(opt.profile_max_samples),
        })
        out_path = os.path.join(opt.model_dir, opt.profile_yaml)
        yaml_utils.save_yaml(profile_summary, out_path)
        print(f'Runtime profile saved to {out_path}')
        print(profile_summary)

    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
