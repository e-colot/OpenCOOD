import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']
        voxel_num_points = batch_dict.get('voxel_num_points', None)

        # Ignore padded empty voxels so static-shape padding does not pollute BEV.
        if voxel_num_points is not None:
            valid_mask = voxel_num_points > 0
            pillar_features = pillar_features[valid_mask, :]
            coords = coords[valid_mask, :]

        batch_size = batch_dict['record_len'].to(torch.long).sum()
        batch_spatial_features = torch.zeros(
            batch_size,
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        batch_indices = coords[:, 0].long()
        flat_indices = coords[:, 1] + \
                       coords[:, 2] * self.nx + \
                       coords[:, 3]
        flat_indices = flat_indices.long()

        batch_spatial_features[batch_indices, :, flat_indices] = pillar_features
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict

