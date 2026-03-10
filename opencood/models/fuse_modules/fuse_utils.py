# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch


def regroup(dense_feature, record_len, max_len):
    """
    Regroup the data based on the record_len.

    Parameters
    ----------
    dense_feature : torch.Tensor
        N, C, H, W
    record_len : list
        [sample1_len, sample2_len, ...]
    max_len : int
        Maximum cav number

    Returns
    -------
    regroup_feature : torch.Tensor
        B, L, C, H, W
    """
    if torch.onnx.is_in_onnx_export():
        # Export path uses batch_size=1. Avoid tensor_split, which lowers to
        # ONNX Sequence ops unsupported by TensorRT parser.
        feature_shape = dense_feature.shape
        cur_len = feature_shape[0]
        padding_len = max_len - cur_len

        padding_tensor = torch.zeros(
            padding_len,
            feature_shape[1],
            feature_shape[2],
            feature_shape[3],
            device=dense_feature.device,
            dtype=dense_feature.dtype,
        )
        regroup_features = torch.cat([dense_feature, padding_tensor], dim=0)
        regroup_features = regroup_features.unsqueeze(0)

        mask = torch.cat([
            torch.ones(cur_len, device=dense_feature.device, dtype=dense_feature.dtype),
            torch.zeros(padding_len, device=dense_feature.device, dtype=dense_feature.dtype),
        ], dim=0).unsqueeze(0)

        return regroup_features, mask

    split_features = torch.split(dense_feature,
                                 record_len.to(torch.long).tolist(),
                                 dim=0)
    regroup_features = []
    mask = []

    for split_feature in split_features:
        # M, C, H, W
        feature_shape = split_feature.shape

        # the maximum M is 5 as most 5 cavs
        padding_len = max_len - feature_shape[0]
        valid_mask = torch.cat([
            torch.ones(feature_shape[0], device=split_feature.device,
                       dtype=dense_feature.dtype),
            torch.zeros(padding_len, device=split_feature.device,
                        dtype=dense_feature.dtype)
        ], dim=0)
        mask.append(valid_mask)

        padding_tensor = torch.zeros(
            padding_len,
            feature_shape[1],
            feature_shape[2],
            feature_shape[3],
            device=split_feature.device,
            dtype=split_feature.dtype,
        )

        split_feature = torch.cat([split_feature, padding_tensor],
                                  dim=0)

        # 1, 5C, H, W
        split_feature = split_feature.view(-1,
                                           feature_shape[2],
                                           feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    # B, 5C, H, W
    regroup_features = torch.cat(regroup_features, dim=0)
    # B, L, C, H, W
    regroup_features = regroup_features.view(
        regroup_features.shape[0],
        max_len,
        dense_feature.shape[1],
        regroup_features.shape[2],
        regroup_features.shape[3],
    )
    mask = torch.stack(mask, dim=0).to(regroup_features.device)

    return regroup_features, mask
