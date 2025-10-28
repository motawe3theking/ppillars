"""
PointPillars Camera-LiDAR Fusion Model

This version fuses RGB camera images with LiDAR depth information.
Instead of pillar-based voxelization, it creates a 4-channel image (RGB + Depth)
by projecting point cloud onto the camera plane using KITTI calibration.

Architecture:
- Input: RGB image (H, W, 3) + Point Cloud (N, 4)
- Depth projection: Create depth channel from point cloud
- Backbone: MobileNetV1 (same as original, adapted for 4 input channels)
- Neck: Same FPN-style neck
- Head: Same detection head
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointpillars.model.anchors import Anchors, anchor_target, anchors2bboxes
from pointpillars.model.backbones.mobilenetv1 import MobileNetV1
from pointpillars.ops import nms_cuda
from pointpillars.utils import limit_period


class DepthProjection(nn.Module):
    """
    Projects LiDAR point cloud onto image plane to create depth channel.
    Uses KITTI calibration matrices for accurate projection.
    """
    def __init__(self, image_size=(375, 1242), point_cloud_range=[0, -40, -3, 70.4, 40, 1]):
        """
        Args:
            image_size: (H, W) of output depth map
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] in LiDAR coords
        """
        super().__init__()
        self.image_h, self.image_w = image_size
        self.pc_range = point_cloud_range
        
    def forward(self, batched_pts, batched_calib):
        """
        Args:
            batched_pts: list of tensors, each (N_i, 4) [x, y, z, intensity]
            batched_calib: list of dicts with keys:
                - 'P2': (3, 4) camera projection matrix
                - 'R0_rect': (4, 4) rectification matrix
                - 'Tr_velo_to_cam': (4, 4) transformation from velodyne to camera
        
        Returns:
            depth_maps: (B, 1, H, W) depth channel
        """
        batch_size = len(batched_pts)
        device = batched_pts[0].device
        
        depth_maps = []
        
        for i in range(batch_size):
            pts = batched_pts[i]  # (N, 4)
            calib = batched_calib[i]
            
            # Extract calibration matrices
            P2 = torch.tensor(calib['P2'], dtype=torch.float32, device=device)  # (3, 4)
            R0 = torch.tensor(calib['R0_rect'], dtype=torch.float32, device=device)  # (4, 4)
            Tr = torch.tensor(calib['Tr_velo_to_cam'], dtype=torch.float32, device=device)  # (4, 4)
            
            # Transform points from velodyne to camera coordinates
            # Add homogeneous coordinate
            pts_homo = torch.cat([pts[:, :3], torch.ones(pts.shape[0], 1, device=device)], dim=1)  # (N, 4)
            
            # Velodyne -> Camera
            pts_cam = (Tr @ pts_homo.T).T  # (N, 4)
            
            # Apply rectification
            pts_rect = (R0 @ pts_cam.T).T  # (N, 4)
            
            # Keep only points in front of camera
            valid_mask = pts_rect[:, 2] > 0
            pts_rect = pts_rect[valid_mask]
            
            if pts_rect.shape[0] == 0:
                # No valid points, return empty depth map
                depth_map = torch.zeros(1, self.image_h, self.image_w, device=device)
                depth_maps.append(depth_map)
                continue
            
            # Project to image plane
            pts_img = (P2 @ pts_rect.T).T  # (N, 3)
            
            # Normalize by depth
            depths = pts_img[:, 2]
            pts_img = pts_img[:, :2] / (depths.unsqueeze(1) + 1e-6)
            
            # Filter points within image bounds
            u = pts_img[:, 0]
            v = pts_img[:, 1]
            valid_u = (u >= 0) & (u < self.image_w)
            valid_v = (v >= 0) & (v < self.image_h)
            valid_mask = valid_u & valid_v
            
            u = u[valid_mask].long()
            v = v[valid_mask].long()
            depths = depths[valid_mask]
            
            # Create depth map
            depth_map = torch.zeros(self.image_h, self.image_w, device=device)
            
            # Scatter depths to image (take max depth if multiple points map to same pixel)
            if u.shape[0] > 0:
                # Use scatter to handle multiple points per pixel
                # Convert to linear indices
                indices = v * self.image_w + u
                
                # For each pixel, keep the closest (minimum) depth
                for idx, d in zip(indices, depths):
                    pixel_v = idx // self.image_w
                    pixel_u = idx % self.image_w
                    if depth_map[pixel_v, pixel_u] == 0 or d < depth_map[pixel_v, pixel_u]:
                        depth_map[pixel_v, pixel_u] = d
            
            depth_maps.append(depth_map.unsqueeze(0))  # (1, H, W)
        
        return torch.stack(depth_maps, dim=0)  # (B, 1, H, W)


class Neck(nn.Module):
    """
    FPN-style neck for multi-scale feature fusion.
    Same as original PointPillars neck.
    """
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides) == len(out_channels)
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels[i], 
                                 out_channels[i], 
                                 upsample_strides[i], 
                                 stride=upsample_strides[i], 
                                 bias=False),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(decoder_block)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        x: list of features [(bs, C1, H1, W1), (bs, C2, H2, W2), (bs, C3, H3, W3)]
        return: (bs, sum(out_channels), H1, W1)
        """
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i])
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    """
    Detection head for bounding box prediction.
    Same as original PointPillars head.
    """
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # Initialize weights (consistent with mmdet3d)
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        """
        x: (bs, in_channel, H, W)
        return: 
            bbox_cls_pred: (bs, n_anchors*n_classes, H, W)
            bbox_pred: (bs, n_anchors*7, H, W)
            bbox_dir_cls_pred: (bs, n_anchors*2, H, W)
        """
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class FusionBackbone(nn.Module):
    """
    MobileNetV1 backbone adapted for 4-channel input (RGB + Depth).
    """
    def __init__(self, in_channels=4):
        super().__init__()
        # Use MobileNetV1 with modified input channels
        self.backbone = MobileNetV1(in_channels=in_channels)
        
    def forward(self, x):
        """
        x: (B, 4, H, W) - RGBD image
        return: list of features at different scales
            [(B, 64, H/2, W/2), (B, 128, H/4, W/4), (B, 256, H/8, W/8)]
        """
        return self.backbone(x)


class PointPillarsFusion(nn.Module):
    """
    Camera-LiDAR Fusion version of PointPillars.
    
    Instead of pillar encoding, this model:
    1. Projects LiDAR points onto image plane to create depth channel
    2. Concatenates with RGB to form 4-channel input
    3. Uses same MobileNetV1 backbone (adapted for 4 channels)
    4. Uses same Neck and Head as original PointPillars
    """
    def __init__(self,
                 nclasses=3,
                 image_size=(375, 1242),  # KITTI image size
                 point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                 max_num_points=32,
                 max_voxels=(12000, 40000)):
        super().__init__()
        
        self.nclasses = nclasses
        self.image_size = image_size
        self.point_cloud_range = point_cloud_range
        
        # Store backbone type for summary
        self.backbone_type = 'mobilenet_fusion'
        
        # Depth projection module
        self.depth_projection = DepthProjection(
            image_size=image_size,
            point_cloud_range=point_cloud_range
        )
        
        # Backbone: MobileNetV1 for 4-channel input
        self.backbone = FusionBackbone(in_channels=4)
        
        # Neck: FPN-style feature pyramid
        # MobileNetV1 outputs: [64, 128, 256] channels
        self.neck = Neck(
            in_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            out_channels=[128, 128, 128]
        )
        
        # Head: Detection head
        self.head = Head(
            in_channel=384,  # 128*3 from neck
            n_anchors=2*nclasses,
            n_classes=nclasses
        )
        
        # Anchors (same as original PointPillars)
        ranges = [
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],
            [0, -39.68, -0.6, 69.12, 39.68, -0.6],
            [0, -39.68, -1.78, 69.12, 39.68, -1.78]
        ]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations = [0, 1.57]
        self.anchors_generator = Anchors(
            ranges=ranges,
            sizes=sizes,
            rotations=rotations
        )
        
        # Training: anchor assignment
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]
        
        # Inference: NMS parameters
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.2
        self.max_num = 50
    
    def summary(self):
        """Print model architecture summary."""
        print("\n" + "="*80)
        print("PointPillars Fusion Model Summary")
        print("="*80)
        
        print("\n[Input Configuration]")
        print(f"  Image Size: {self.image_size} (H, W)")
        print(f"  Input Channels: 4 (RGB + Depth)")
        print(f"  Point Cloud Range: {self.point_cloud_range}")
        
        print("\n[Depth Projection]")
        print(f"  Type: Camera-LiDAR Fusion")
        print(f"  Method: Project point cloud to image plane using KITTI calibration")
        print(f"  Output: Depth map {self.image_size}")
        
        print("\n[Backbone]")
        print(f"  Type: {self.backbone_type}")
        print(f"  Architecture: MobileNetV1 (4-channel input)")
        print(f"  Input: (B, 4, {self.image_size[0]}, {self.image_size[1]})")
        print(f"  Output Scales: [64, 128, 256] channels")
        
        print("\n[Neck]")
        print(f"  Type: FPN-style feature pyramid")
        print(f"  Input Channels: [64, 128, 256]")
        print(f"  Output Channels: [128, 128, 128]")
        print(f"  Upsample Strides: [1, 2, 4]")
        print(f"  Total Output Channels: 384")
        
        print("\n[Head]")
        print(f"  Input Channels: 384")
        print(f"  Number of Classes: {self.nclasses}")
        print(f"  Anchors per Location: {2*self.nclasses}")
        print(f"  Outputs:")
        print(f"    - Classification: {2*self.nclasses*self.nclasses} channels")
        print(f"    - Regression: {2*self.nclasses*7} channels")
        print(f"    - Direction: {2*self.nclasses*2} channels")
        
        print("\n[Inference Parameters]")
        print(f"  NMS Pre-filter: {self.nms_pre}")
        print(f"  NMS IoU Threshold: {self.nms_thr}")
        print(f"  Score Threshold: {self.score_thr}")
        print(f"  Max Detections: {self.max_num}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n[Model Parameters]")
        print(f"  Depth Projection: {sum(p.numel() for p in self.depth_projection.parameters()):,}")
        print(f"  Backbone: {sum(p.numel() for p in self.backbone.parameters()):,}")
        print(f"  Neck: {sum(p.numel() for p in self.neck.parameters()):,}")
        print(f"  Head: {sum(p.numel() for p in self.head.parameters()):,}")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        print("\n" + "="*80 + "\n")
    
    def forward(self, batched_pts, batched_imgs, batched_calib, mode='test', 
                batched_gt_bboxes=None, batched_gt_labels=None):
        """
        Args:
            batched_pts: list of tensors, each (N_i, 4) [x, y, z, intensity]
            batched_imgs: (B, 3, H, W) RGB images
            batched_calib: list of dicts with calibration matrices
            mode: 'train', 'val', or 'test'
            batched_gt_bboxes: list of (M_i, 7) ground truth boxes (train mode only)
            batched_gt_labels: list of (M_i,) ground truth labels (train mode only)
        
        Returns:
            If mode == 'train': loss components
            If mode == 'val' or 'test': detection results
        """
        batch_size = batched_imgs.shape[0]
        device = batched_imgs.device
        
        # 1. Create depth channel from point cloud
        depth_maps = self.depth_projection(batched_pts, batched_calib)  # (B, 1, H, W)
        
        # 2. Normalize depth (optional, helps training)
        # Clip and normalize to [0, 1] range
        max_depth = 80.0  # meters
        depth_maps = torch.clamp(depth_maps, 0, max_depth) / max_depth
        
        # 3. Concatenate RGB + Depth
        rgbd = torch.cat([batched_imgs, depth_maps], dim=1)  # (B, 4, H, W)
        
        # 4. Backbone forward pass
        features = self.backbone(rgbd)  # List of [(B, 64, H/2, W/2), ...]
        
        # 5. Neck forward pass
        neck_out = self.neck(features)  # (B, 384, H_out, W_out)
        
        # 6. Head forward pass
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(neck_out)
        
        # 7. Generate anchors
        # For fusion model, anchors are generated in image/BEV space
        # We need to match the spatial dimensions of neck output
        H_out, W_out = neck_out.shape[2], neck_out.shape[3]
        device = neck_out.device
        batched_anchors = self.anchors_generator.get_multi_anchors(batch_size)
        batched_anchors = [anchors.to(device) for anchors in batched_anchors]
        
        # 8. Mode-specific outputs
        if mode == 'train':
            # Compute anchor targets
            anchor_target_dict = anchor_target(
                batched_anchors=batched_anchors,
                batched_gt_bboxes=batched_gt_bboxes,
                batched_gt_labels=batched_gt_labels,
                assigners=self.assigners,
                nclasses=self.nclasses
            )
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        
        elif mode == 'val':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_anchors=batched_anchors
            )
            return results
        
        elif mode == 'test':
            results = self.get_predicted_bboxes(
                bbox_cls_pred=bbox_cls_pred,
                bbox_pred=bbox_pred,
                bbox_dir_cls_pred=bbox_dir_cls_pred,
                batched_anchors=batched_anchors
            )
            return results
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        """
        Decode predictions for a single sample.
        Same as original PointPillars.
        """
        # Reshape predictions
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        # Apply sigmoid to classification scores
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]
        
        # 1. Get top-k predictions before NMS
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]
        
        # 2. Decode bounding boxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)
        
        # 3. Apply NMS per class
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([
            bbox_pred2d_xy - bbox_pred2d_lw / 2,
            bbox_pred2d_xy + bbox_pred2d_lw / 2,
            bbox_pred[:, 6:]
        ], dim=-1)
        
        ret_bboxes, ret_labels, ret_scores = [], [], []
        
        for i in range(self.nclasses):
            # Filter by score threshold
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            
            if score_inds.sum() == 0:
                continue
            
            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # NMS
            keep_inds = nms_cuda(
                boxes=cur_bbox_pred2d,
                scores=cur_bbox_cls_pred,
                thresh=self.nms_thr,
                pre_maxsize=None,
                post_max_size=None
            )
            
            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            
            # Adjust rotation based on direction classification
            cur_bbox_pred[:, -1] = limit_period(
                cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi
            ).to(cur_bbox_pred)
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi
            
            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)
        
        # 4. Limit total number of detections
        if len(ret_bboxes) == 0:
            return {
                'lidar_bboxes': np.zeros((0, 7), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int64),
                'scores': np.zeros((0,), dtype=np.float32)
            }
        
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result
    
    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        """
        Decode predictions for entire batch.
        Same as original PointPillars.
        """
        batch_size = bbox_cls_pred.shape[0]
        results = []
        
        for i in range(batch_size):
            result = self.get_predicted_bboxes_single(
                bbox_cls_pred=bbox_cls_pred[i],
                bbox_pred=bbox_pred[i],
                bbox_dir_cls_pred=bbox_dir_cls_pred[i],
                anchors=batched_anchors[i]
            )
            results.append(result)
        
        return results
