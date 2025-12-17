import numpy as np
import torch
import torch.nn as nn

# Reuse components from the main PointPillars implementation
from .pointpillars import Backbone2, Neck, Head, Anchors


PILLAR_FEATURES = 64  # default channels for pillar feature map
NARROW_RANGE = 'wide'


class PointPillarsPostVoxel(nn.Module):
    """
    Post-voxel variant of PointPillars that accepts pre-computed pillar feature maps
    of shape (B, C, H, W) and runs Backbone + Neck + Head.

    This model excludes the voxelization and pillar encoder stages.
    """

    if NARROW_RANGE == 'small':
        point_cloud_range = [0, -10.24, -3, 69.12, 10.24, 1]
    elif NARROW_RANGE == 'mid':
        point_cloud_range = [0, -20.48, -3, 40.96, 20.48, 1]
    elif NARROW_RANGE == 'wide':
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        # point_cloud_range = [0, -40.32, -3, 70.2, 40.32, 1]

    def __init__(self, nclasses: int = 3, in_channels: int = PILLAR_FEATURES):
        super().__init__()
        self.nclasses = nclasses

        # Backbone (MobileNetV1-based) and Neck
        self.backbone = Backbone2(in_channel=in_channels)
        if in_channels == 64:
            self.neck = Neck(in_channels=[64, 128, 256],
                             upsample_strides=[1, 2, 4],
                             out_channels=[128, 128, 128])
            neck_out_channels = 384
        elif in_channels == 32:
            self.neck = Neck(in_channels=[32, 64, 128],
                             upsample_strides=[1, 2, 4],
                             out_channels=[64, 64, 96])
            neck_out_channels = 224
        else:
            # Fallback: assume three scales doubling channels
            c1, c2, c3 = in_channels, in_channels * 2, in_channels * 4
            self.neck = Neck(in_channels=[c1, c2, c3],
                             upsample_strides=[1, 2, 4],
                             out_channels=[c2, c2, c2])
            neck_out_channels = c2 * 3

        # Detection head
        self.head = Head(in_channel=neck_out_channels, n_anchors=2 * nclasses, n_classes=nclasses)

        # Anchors (match ranges from main model for KITTI)
        if NARROW_RANGE == 'small':
            ranges = [[0, -10.24, -0.6, 69.12, 10.24, -0.6],
                      [0, -10.24, -0.6, 69.12, 10.24, -0.6],
                      [0, -10.24, -1.78, 69.12, 10.24, -1.78]]
        elif NARROW_RANGE == 'mid':
            ranges = [[0, -20.48, -0.6, 40.96, 20.48, -0.6],
                      [0, -20.48, -0.6, 40.96, 20.48, -0.6],
                      [0, -20.48, -1.78, 40.96, 20.48, -1.78]]
        else:  # wide
            ranges = [[0, -40.32, -0.6, 70.2, 40.32, -0.6],
                      [0, -40.32, -0.6, 70.2, 40.32, -0.6],
                      [0, -39.68, -3, 69.12, 39.68, 1]]

        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]

        self.anchors_generator = Anchors(ranges=ranges, sizes=sizes, rotations=rotations)

    def forward(self, pillar_features: torch.Tensor):
        """
        Args:
            pillar_features: Tensor of shape (B, C, H, W) with precomputed pillar features
        Returns:
            Tuple of (bbox_cls_pred, bbox_pred, bbox_dir_cls_pred)
        """
        xs = self.backbone(pillar_features)
        neck_out = self.neck(xs)
        return self.head(neck_out)
