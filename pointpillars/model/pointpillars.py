import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointpillars.model.anchors import Anchors, anchor_target, anchors2bboxes
from pointpillars.model.backbones.mobilenetv1 import MobileNetV1
from pointpillars.ops import Voxelization, nms_cuda
from pointpillars.utils import limit_period
import time
from collections import defaultdict

BACKBONE_SELECT = 'mobilenet'   # Options: 'default', 'mobilenet'
ENCODER = 'HCF'                 # Options: 'default', 'HCF'
NARROW_RANGE = False            # Whether to use narrow point cloud range
ENABLE_TIMING = False           # Timing measurement switch
USE_HARD_VOXELIZATION = False   # Voxelization method switch: False=default, True=hard_pillar
PILLAR_FEATURES = 32            # Number of pillar features after encoding
UPSAMPLE = True                 # Whether to use ConvTranspose2d for upsampling in the decoder (Neck)


class FLOPsCounter:
    """Utility class to calculate FLOPs for common layer types."""
    
    @staticmethod
    def count_conv2d(module, input_shape, output_shape):
        """
        Calculate FLOPs for Conv2d layer.
        FLOPs = 2 * Cin * Kh * Kw * Cout * Hout * Wout
        (multiply-add counted as 2 ops)
        """
        batch, cin, hin, win = input_shape
        bout, cout, hout, wout = output_shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * cin
        output_size = hout * wout
        flops = kernel_ops * cout * output_size * batch
        if module.bias is not None:
            flops += cout * output_size * batch
        return flops
    
    @staticmethod
    def count_conv1d(module, input_shape, output_shape):
        """
        Calculate FLOPs for Conv1d layer.
        FLOPs = 2 * Cin * K * Cout * Lout
        """
        batch, cin, lin = input_shape
        bout, cout, lout = output_shape
        kernel_ops = module.kernel_size[0] * cin
        flops = kernel_ops * cout * lout * batch
        if module.bias is not None:
            flops += cout * lout * batch
        return flops
    
    @staticmethod
    def count_batchnorm(module, input_shape):
        """
        Calculate FLOPs for BatchNorm.
        FLOPs = 2 * num_elements (normalize + scale/shift)
        """
        return 2 * np.prod(input_shape)
    
    @staticmethod
    def count_linear(module, input_shape, output_shape):
        """Calculate FLOPs for Linear layer."""
        batch = input_shape[0]
        in_features = module.in_features
        out_features = module.out_features
        flops = batch * in_features * out_features
        if module.bias is not None:
            flops += batch * out_features
        return flops
    
    @staticmethod
    def count_upsample(scale_factor, input_shape):
        """
        Calculate FLOPs for nearest neighbor upsampling.
        Minimal computation - mainly memory ops.
        """
        return 0  # Nearest neighbor is essentially memory copy
    
    @staticmethod
    def count_activation(input_shape):
        """Calculate FLOPs for activation functions (ReLU, etc)."""
        return np.prod(input_shape)


class HardPillarVoxelization(nn.Module):
    """
    Hard pillar voxelization using deterministic sampling and padding.
    Implements the FPGA-optimized grouping method from hardware papers.
    """
    def __init__(self, pillar_size, grid_range, feature_dim, max_num_points, max_voxels):
        super().__init__()
        self.max_points = max_num_points
        self.max_voxels_train = max_voxels[0]
        self.max_voxels_test = max_voxels[1]

        # Store as Python lists/tuples
        self.pillar_size_list = pillar_size
        self.grid_range_list = grid_range

        # Calculate grid size as integers
        self.grid_size_x = int(round((grid_range[3] - grid_range[0]) / pillar_size[0]))
        self.grid_size_y = int(round((grid_range[4] - grid_range[1]) / pillar_size[1]))
        
        self.feature_dim = feature_dim

    def forward(self, point_clouds):
        dev = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create tensors on the correct device
        pillar_size = torch.tensor(self.pillar_size_list, dtype=torch.float32, device=dev)
        grid_range = torch.tensor(self.grid_range_list, dtype=torch.float32, device=dev)
        # grid_size = torch.tensor([self.grid_size_x, self.grid_size_y], dtype=torch.long, device=dev) # Not used here

        max_voxels = self.max_voxels_train if self.training else self.max_voxels_test
        
        # --- This Python 'for' loop is a MAJOR GPU BOTTLENECK ---
        batch_pillars = []
        batch_coors = []
        batch_npoints = []
        
        for batch_idx, pc in enumerate(point_clouds):
            pc = pc.to(dev)
            
            # Use self.feature_dim in case pc is empty
            current_feature_dim = pc.shape[1] if pc.shape[0] > 0 else self.feature_dim

            # Filter points within grid range
            mask = (pc[:, 0] >= grid_range[0]) & (pc[:, 0] < grid_range[3]) & \
                   (pc[:, 1] >= grid_range[1]) & (pc[:, 1] < grid_range[4]) & \
                   (pc[:, 2] >= grid_range[2]) & (pc[:, 2] <= grid_range[5])
            pc = pc[mask]

            if pc.shape[0] == 0:
                continue

            # Compute pillar coordinates
            # --- FIX: Slice pillar_size to match the other tensors ---
            coords = ((pc[:, :2] - grid_range[:2]) / pillar_size[:2]).long()
            
            # --- This is the "Hard Voxelization" logic (Grouping, Sampling, Padding) ---
            
            # 1. Get unique pillar coordinates and inverse indices
            unique_coords, inverse_indices = torch.unique(coords, return_inverse=True, dim=0)
            
            # 2. Apply pillar limit (max_voxels)
            num_pillars = unique_coords.shape[0]
            if num_pillars > max_voxels:
                # Keep the first 'max_voxels' pillars found
                # This is not random, but it's fast
                
                # We need a mask for points belonging to the kept pillars
                
                # Create a lookup table
                # Ensure lookup is large enough for max coordinates
                max_x_coord = int(round((self.grid_range_list[3] - self.grid_range_list[0]) / self.pillar_size_list[0]))
                max_y_coord = int(round((self.grid_range_list[4] - self.grid_range_list[1]) / self.pillar_size_list[1]))
                
                lookup = torch.full((max_x_coord + 1, max_y_coord + 1), -1, dtype=torch.long, device=dev)
                
                # Clamp unique_coords before indexing lookup
                clamped_unique_coords_x = torch.clamp(unique_coords[:,0], 0, max_x_coord)
                clamped_unique_coords_y = torch.clamp(unique_coords[:,1], 0, max_y_coord)
                
                lookup[clamped_unique_coords_x, clamped_unique_coords_y] = torch.arange(num_pillars, device=dev)
                
                # Clamp coords before indexing lookup
                clamped_coords_x = torch.clamp(coords[:,0], 0, max_x_coord)
                clamped_coords_y = torch.clamp(coords[:,1], 0, max_y_coord)
                
                point_pillar_indices = lookup[clamped_coords_x, clamped_coords_y]
                
                point_mask = (point_pillar_indices < max_voxels) & (point_pillar_indices != -1)
                
                # Filter points and indices
                pc = pc[point_mask]
                inverse_indices = point_pillar_indices[point_mask]
                unique_coords = unique_coords[:max_voxels]
                num_pillars = max_voxels
            
            # 3. Create tensors to hold the dense pillar data
            
            # --- START: Vectorized Inner Loop ---
            # This replaces the slow Python 'for' loop
            
            num_points = pc.shape[0]
            
            # Create a one-hot matrix: [num_points, num_pillars]
            # This is memory-intensive but runs on GPU
            one_hot = F.one_hot(inverse_indices, num_classes=num_pillars)
            
            # Compute cumulative sum to get "in-pillar" indices
            # point_counts_matrix[i, j] = "how many points for pillar j have we seen up to point i"
            time_start = time.time()
            point_counts_matrix = torch.cumsum(one_hot, dim=0) # [num_points, num_pillars]
            time_end = time.time()
            print(f"Cumsum time: {time_end - time_start:.6f} seconds")  
            # Get the "in-pillar" index for each point
            # We use a padded (shifted) version to get the count *before* the current point
            # gather() selects the count corresponding to the correct pillar for each point
            point_in_pillar_idx = F.pad(point_counts_matrix, (0, 0, 1, -1)).gather(1, inverse_indices.unsqueeze(1)).squeeze(1)

            # Create a mask for points that are within the 'max_points' limit
            point_mask = point_in_pillar_idx < self.max_points

            # Filter all tensors based on this mask
            time_start = time.time()
            filtered_pc = pc[point_mask]
            filtered_pillar_idx = inverse_indices[point_mask]
            filtered_point_in_pillar_idx = point_in_pillar_idx[point_mask]
            time_end = time.time()
            print(f"Filtering time: {time_end - time_start:.6f} seconds")
            # Create the output pillars tensor (all zeros)
            pillars = torch.zeros((num_pillars, self.max_points, current_feature_dim), dtype=torch.float32, device=dev)

            # Scatter the points into the correct (pillar_idx, point_idx) slots
            time_start = time.time()
            pillars[filtered_pillar_idx, filtered_point_in_pillar_idx] = filtered_pc
            time_end = time.time()
            print(f"Scatter time: {time_end - time_start:.6f} seconds")
            # Get the final point counts for each pillar
            # The last row of the original cumsum matrix has the total counts
            total_counts_per_pillar = point_counts_matrix[-1]
            npoints = torch.clamp(total_counts_per_pillar, max=self.max_points)
            
            # --- END: Vectorized Inner Loop ---
            
            
            # --- End of "Hard Voxelization" logic ---
            
            # Create coordinates with batch index
            batch_idx_tensor = torch.full((num_pillars, 1), batch_idx, dtype=torch.long, device=dev)
            # Re-order to (batch_idx, x, y, z=0)
            coors_out = torch.cat([
                batch_idx_tensor,
                unique_coords,
                torch.zeros((num_pillars, 1), dtype=torch.long, device=dev)
            ], dim=1)

            batch_pillars.append(pillars)
            batch_coors.append(coors_out)
            batch_npoints.append(npoints)

        # Combine all data from the batch
        if not batch_pillars:
            # Handle empty batch case
            # Return empty tensors with correct shapes
            pillars = torch.zeros((0, self.max_points, self.feature_dim), device=dev)
            coors_batch = torch.zeros((0, 4), dtype=torch.long, device=dev)
            npoints_per_pillar = torch.zeros((0,), dtype=torch.long, device=dev)
            return pillars, coors_batch, npoints_per_pillar
            
        pillars = torch.cat(batch_pillars, dim=0)
        coors_batch = torch.cat(batch_coors, dim=0)
        npoints_per_pillar = torch.cat(batch_npoints, dim=0)

        # Return the pillars, coordinates, and point counts directly.
        return pillars, coors_batch, npoints_per_pillar

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        if USE_HARD_VOXELIZATION:
            self.voxel_layer = HardPillarVoxelization(pillar_size=voxel_size,
                                                    grid_range=point_cloud_range,
                                                    feature_dim=4,
                                                    max_num_points=max_num_points,
                                                    max_voxels=max_voxels)
            print("Using Hard Pillar Voxelization (FPGA-optimized)")
        else:
            self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range,
                                          max_num_points=max_num_points,
                                          max_voxels=max_voxels,
                                          deterministic=False)
            print("Using Default Voxelization")

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        if USE_HARD_VOXELIZATION:
            # Hard voxelization returns (pillars, coordinates, num_points_per_pillar)
            pillars, coors_batch, npoints_per_pillar = self.voxel_layer.forward(batched_pts)
            # Convert coordinates format: [batch_idx, z_coord, y_coord, x_coord] -> [batch_idx, x_coord, y_coord, z_coord]
            if coors_batch.shape[0] > 0:
                coors_batch = coors_batch[:, [0, 3, 2, 1]]  # Reorder to match original format
            print(f"Hard voxelization: {pillars.shape[0]} pillars")
            return pillars, coors_batch, npoints_per_pillar
        else:
            # Original voxelization logic
            pillars, coors, npoints_per_pillar = [], [], []
            # print(f"Processing batch of size {len(batched_pts)} in PillarLayer")
            for i, pts in enumerate(batched_pts):
                voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
                # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
                # num_points_per_voxel_out: (max_voxel, )
                # print(f"number of pillars per batch: {voxels_out.shape}")
                pillars.append(voxels_out)
                coors.append(coors_out.long())
                npoints_per_pillar.append(num_points_per_voxel_out)
            
            pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
            npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
            coors_batch = []
            for i, cur_coors in enumerate(coors):
                coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
            coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

            return pillars, coors_batch, npoints_per_pillar

if ENCODER == 'default':
    class PillarEncoder(nn.Module):
        def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
            super().__init__()
            self.out_channel = out_channel
            self.vx, self.vy = voxel_size[0], voxel_size[1]
            self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
            self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
            self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
            self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

            self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
            self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

        def forward(self, pillars, coors_batch, npoints_per_pillar):
            '''
            pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
            coors_batch: (p1 + p2 + ... + pb, 1 + 3)
            npoints_per_pillar: (p1 + p2 + ... + pb, )
            return:  (bs, out_channel, y_l, x_l)
            '''
            device = pillars.device
            # 1. calculate offset to the points center (in each pillar)
            offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

            # 2. calculate offset to the pillar center
            x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
            y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

            # 3. encoder
            features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
            features[:, :, 0:1] = x_offset_pi_center # tmp
            features[:, :, 1:2] = y_offset_pi_center # tmp
            # In consitent with mmdet3d. 
            # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

            # 4. find mask for (0, 0, 0) and update the encoded features
            # a very beautiful implementation
            voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
            mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
            mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
            features *= mask[:, :, None]

            # 5. embedding
            features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
            features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
            pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

            # 6. pillar scatter
            batched_canvas = []
            bs = coors_batch[-1, 0] + 1
            for i in range(bs):
                cur_coors_idx = coors_batch[:, 0] == i
                cur_coors = coors_batch[cur_coors_idx, :]
                cur_features = pooling_features[cur_coors_idx]

                canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
                canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
                canvas = canvas.permute(2, 1, 0).contiguous()
                batched_canvas.append(canvas)
            batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
            return batched_canvas
elif ENCODER == 'HCF':
    class PillarEncoder(nn.Module):
        """
        GPU-optimized Pillar Feature Encoder implementing paper equations (4)
        Transforms tensor (P, N, 4) to (P, N, 9) with HCF, then to (P, 64) via PW conv + pooling
        """
        def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
            super().__init__()
            self.out_channel = out_channel
            self.vx, self.vy = voxel_size[0], voxel_size[1]
            self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
            self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
            self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
            self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

            # Point-wise (PW) convolution as specified in paper
            self.conv = nn.Conv1d(9, out_channel, 1, bias=False)  # 9 features after HCF
            self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

        def forward(self, pillars, coors_batch, npoints_per_pillar):
            '''
            GPU-optimized forward implementing paper equation (4): ni=[x,y,z,r,xm,ym,zm,xc,yc]
            
            Input:
            - pillars: (P, N, 4) - tensor from equation (3): ni=[x,y,z,r]  
            - coors_batch: (P, 4) - pillar coordinates
            - npoints_per_pillar: (P,) - actual points per pillar
            
            Output:
            - (bs, 64, y_l, x_l) - 2D feature map with 64 channels
            '''
            device = pillars.device
            P, N = pillars.shape[:2]
            
            # Step 1: GPU-vectorized HCF computation (equation 4)
            # time_start = time.time()
            features_with_hcf = self._compute_hcf_vectorized(pillars, coors_batch, npoints_per_pillar)
            # time_end = time.time()
            # print(f"HCF computation time: {time_end - time_start:.6f} seconds")
            # Step 2: Create valid point mask (GPU operation)
            # time_start = time.time()
            valid_mask = self._create_valid_mask_vectorized(P, N, npoints_per_pillar, device)
            # time_end = time.time()
            # print(f"Valid mask creation time: {time_end - time_start:.6f} seconds")
            # Step 3: Apply mask to features (equation 4 compliance)
            # time_start = time.time()
            features_with_hcf = features_with_hcf * valid_mask.unsqueeze(-1)
            # time_end = time.time()
            # print(f"Mask application time: {time_end - time_start:.6f} seconds")
            # Step 4: Point-wise convolution + BatchNorm + ReLU (P, N, 9) -> (P, N, 64)
            # time_start = time.time()
            features_conv = self._pointwise_convolution(features_with_hcf)
            # time_end = time.time()
            # print(f"Point-wise convolution time: {time_end - time_start:.6f} seconds")
            # Step 5: Max pooling along N dimension (P, N, 64) -> (P, 64)
            # time_start = time.time()
            pooling_features = torch.max(features_conv, dim=1)[0]
            # time_end = time.time()
            # print(f"Max pooling time: {time_end - time_start:.6f} seconds")

            # Step 6: GPU-vectorized scatter to 2D plane
            # time_start = time.time()
            feature_map = self._scatter_to_2d_vectorized(pooling_features, coors_batch)
            # time_end = time.time()
            # print(f"Scatter to 2D plane time: {time_end - time_start:.6f} seconds")
            # print(f"Feature map shape: {feature_map.shape}")

            return feature_map
        
        def _compute_hcf_vectorized(self, pillars, coors_batch, npoints_per_pillar):
            """
            GPU-vectorized HCF computation implementing equation (4)
            ni=[x,y,z,r,xm,ym,zm,xc,yc]
            """
            P, N = pillars.shape[:2]
            device = pillars.device
            
            # Extract original features: [x,y,z,r]
            xyz = pillars[:, :, :3]  # (P, N, 3)
            reflectivity = pillars[:, :, 3:4]  # (P, N, 1)
            
            # Compute arithmetic mean coordinates (xm, ym, zm) per pillar
            # Use broadcasting to avoid division by zero
            valid_point_sums = torch.sum(xyz, dim=1, keepdim=True)  # (P, 1, 3)
            point_means = valid_point_sums / npoints_per_pillar[:, None, None].clamp(min=1)  # (P, 1, 3)
            
            # Compute differences (xm, ym, zm) = mean - point coordinates
            mean_diffs = xyz - point_means  # (P, N, 3) - broadcasting
            
            # Compute pillar center coordinates (xc, yc)
            pillar_centers_x = coors_batch[:, 1:2] * self.vx + self.x_offset  # (P, 1)
            pillar_centers_y = coors_batch[:, 2:3] * self.vy + self.y_offset  # (P, 1)
            
            # Compute differences (xc, yc) = pillar_center - point coordinates
            center_diffs_x = xyz[:, :, 0:1] - pillar_centers_x.unsqueeze(1)  # (P, N, 1)
            center_diffs_y = xyz[:, :, 1:2] - pillar_centers_y.unsqueeze(1)  # (P, N, 1)
            
            # Concatenate all features: [x,y,z,r,xm,ym,zm,xc,yc] - equation (4)
            features_with_hcf = torch.cat([
                xyz[:, :, 0:1],  # x coordinate (1)
                xyz[:, :, 1:2],  # y coordinate (1)
                xyz[:, :, 2:3],  # z coordinate (1)
                reflectivity,    # r (reflectivity) (1)
                mean_diffs,      # xm, ym, zm differences (3)
                center_diffs_x,  # xc difference (1)
                center_diffs_y   # yc difference (1)
            ], dim=-1)  # (P, N, 9)
            
            return features_with_hcf
        
        def _create_valid_mask_vectorized(self, P, N, npoints_per_pillar, device):
            """
            GPU-vectorized valid point mask creation
            """
            # Create point indices tensor
            point_indices = torch.arange(N, device=device).unsqueeze(0).expand(P, -1)  # (P, N)
            
            # Create mask: point_index < actual_points_in_pillar
            valid_mask = point_indices < npoints_per_pillar.unsqueeze(1)  # (P, N)
            
            return valid_mask.float()
        
        def _pointwise_convolution(self, features_with_hcf):
            """
            Point-wise convolution as specified in paper: (P, N, 9) -> (P, N, 64)
            """
            P, N, _ = features_with_hcf.shape
            
            # Reshape for Conv1d: (P, 9, N)
            features_transposed = features_with_hcf.transpose(1, 2)  # (P, 9, N)
            
            # Point-wise convolution + BatchNorm + ReLU
            features_conv = self.conv(features_transposed)  # (P, 64, N)
            features_conv = self.bn(features_conv)
            features_conv = F.relu(features_conv)
            
            # Transpose back: (P, N, 64)
            return features_conv.transpose(1, 2)
        
        def _scatter_to_2d_vectorized(self, pooling_features, coors_batch):
            """
            GPU-vectorized scatter operation to create 2D feature map
            Replaces the Python loop with efficient GPU operations
            """
            device = pooling_features.device
            
            # Get batch size from coordinates
            batch_size = int(coors_batch[:, 0].max().item()) + 1
            
            # Pre-allocate feature maps for all batches
            feature_maps = torch.zeros((batch_size, self.out_channel, self.y_l, self.x_l), 
                                    device=device, dtype=pooling_features.dtype)
            
            # GPU-vectorized scatter using advanced indexing
            batch_indices = coors_batch[:, 0].long()
            y_indices = coors_batch[:, 2].long()
            x_indices = coors_batch[:, 1].long()
            
            # Use scatter_add for efficient GPU operation
            for batch_id in range(batch_size):
                batch_mask = batch_indices == batch_id
                if batch_mask.any():
                    batch_features = pooling_features[batch_mask]  # (num_pillars_in_batch, 64)
                    batch_y = y_indices[batch_mask]
                    batch_x = x_indices[batch_mask]
                    
                    # Advanced indexing for efficient scatter
                    feature_maps[batch_id, :, batch_y, batch_x] = batch_features.transpose(0, 1)
            
            return feature_maps

   

class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]):
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            # Use Upsample + Conv2d instead of ConvTranspose2d
            if UPSAMPLE:
                decoder_block.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=upsample_strides[i], mode='nearest'),
                        nn.Conv2d(in_channels[i], 
                                out_channels[i], 
                                kernel_size=3,
                                padding=1,
                                bias=False)
                    )
                )
            else:
                decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
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
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class Backbone2(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.backbone = MobileNetV1(in_channels=in_channel)
        
    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        return self.backbone(x)


class PointPillars(nn.Module):
    if NARROW_RANGE:
        point_cloud_range = [0, -10.24, -3, 69.12, 10.24, 1]
        pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)
    else:
        # point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        point_cloud_range = [0, -20.48, -3, 40.96, 20.48, 1]
    #    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
    def __init__(self,
                 nclasses=3, 
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=point_cloud_range,
                # point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000),
                 backbone_type=BACKBONE_SELECT):
        super().__init__()
        self.nclasses = nclasses
        self.backbone_type = backbone_type
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=9, 
                                            out_channel=PILLAR_FEATURES)
        # Choose backbone based on type
        if backbone_type == 'default':
            self.backbone = Backbone(in_channel=PILLAR_FEATURES, 
                                    out_channels=[64, 128, 256], 
                                    layer_nums=[3, 5, 5])
            # Neck for default backbone: [64, 128, 256] -> 384 channels
            self.neck = Neck(in_channels=[64, 128, 256], 
                            upsample_strides=[1, 2, 4], 
                            out_channels=[128, 128, 128])
            neck_out_channels = 384  # 128 + 128 + 128
        elif backbone_type == 'mobilenet':
            self.backbone = Backbone2(in_channel=PILLAR_FEATURES)
            # Neck for MobileNet: [32, 64, 128] -> 224 channels
            self.neck = Neck(in_channels=[32, 64, 128], 
                            upsample_strides=[1, 2, 4], 
                            out_channels=[64, 64, 96])
            neck_out_channels = 224  # 64 + 64 + 96
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        self.head = Head(in_channel=neck_out_channels, n_anchors=2*nclasses, n_classes=nclasses)
        
        # anchors
        if NARROW_RANGE:
            ranges = [[0, -10.24, -0.6, 69.12, 10.24, -0.6],
                      [0, -10.24, -0.6, 69.12, 10.24, -0.6],
                      [0, -10.24, -1.78, 69.12, 10.24, -1.78]]
        else:
            # ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
            #         [0, -39.68, -0.6, 69.12, 39.68, -0.6],
            #         [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
            ranges = [
                [0, -20.48, -0.6, 40.96, 20.48, -0.6],
                [0, -20.48, -0.6, 40.96, 20.48, -0.6],
                [0, -20.48, -1.78, 40.96, 20.48, -1.78]
            ]
        sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
        rotations=[0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, 
                                         sizes=sizes, 
                                         rotations=rotations)
        
        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.2
        self.max_num = 50

    def _get_voxelizer_info(self):
        """Collects voxelization configuration from the wrapped voxel layer."""
        vl = self.pillar_layer.voxel_layer
        info = {}
        # Common
        info['type'] = type(vl).__name__
        # Voxel size and range
        if hasattr(vl, 'voxel_size'):
            info['voxel_size'] = vl.voxel_size
        elif hasattr(vl, 'pillar_size_list'):
            info['voxel_size'] = vl.pillar_size_list
        if hasattr(vl, 'point_cloud_range'):
            info['point_cloud_range'] = vl.point_cloud_range
        elif hasattr(vl, 'grid_range_list'):
            info['point_cloud_range'] = vl.grid_range_list
        # Limits
        if hasattr(vl, 'max_num_points'):
            info['max_num_points'] = vl.max_num_points
        if hasattr(vl, 'max_voxels'):
            info['max_voxels'] = vl.max_voxels
        elif hasattr(vl, 'max_voxels_train') and hasattr(vl, 'max_voxels_test'):
            info['max_voxels'] = (vl.max_voxels_train, vl.max_voxels_test)
        # Determinism (only for default Voxelization)
        if hasattr(vl, 'deterministic'):
            info['deterministic'] = vl.deterministic
        else:
            info['deterministic'] = None
        # Grid size (if available)
        if hasattr(self.pillar_encoder, 'x_l') and hasattr(self.pillar_encoder, 'y_l'):
            info['grid_size'] = (self.pillar_encoder.x_l, self.pillar_encoder.y_l)
        return info

    @staticmethod
    def _count_params(module: nn.Module):
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        return total, trainable
    
    def _calculate_flops_detailed(self, batched_pts):
        """
        Calculate FLOPs for each layer by doing a forward pass with hooks.
        Returns a dictionary with layer-wise FLOPs breakdown.
        """
        self.eval()
        flops_dict = defaultdict(lambda: {'flops': 0, 'params': 0, 'name': ''})
        
        def conv2d_hook(module, input, output, name):
            input_shape = input[0].shape
            output_shape = output.shape
            flops = FLOPsCounter.count_conv2d(module, input_shape, output_shape)
            params = sum(p.numel() for p in module.parameters())
            flops_dict[name] = {'flops': flops, 'params': params, 'type': 'Conv2d', 
                               'shape': f"{tuple(input_shape)} -> {tuple(output_shape)}"}
        
        def conv1d_hook(module, input, output, name):
            input_shape = input[0].shape
            output_shape = output.shape
            flops = FLOPsCounter.count_conv1d(module, input_shape, output_shape)
            params = sum(p.numel() for p in module.parameters())
            flops_dict[name] = {'flops': flops, 'params': params, 'type': 'Conv1d',
                               'shape': f"{tuple(input_shape)} -> {tuple(output_shape)}"}
        
        def bn_hook(module, input, output, name):
            input_shape = input[0].shape
            flops = FLOPsCounter.count_batchnorm(module, input_shape)
            params = sum(p.numel() for p in module.parameters())
            flops_dict[name] = {'flops': flops, 'params': params, 'type': 'BatchNorm',
                               'shape': f"{tuple(input_shape)}"}
        
        def relu_hook(module, input, output, name):
            input_shape = input[0].shape
            flops = FLOPsCounter.count_activation(input_shape)
            flops_dict[name] = {'flops': flops, 'params': 0, 'type': 'ReLU',
                               'shape': f"{tuple(input_shape)}"}
        
        # Register hooks
        hooks = []
        module_count = defaultdict(int)
        
        for name, module in self.named_modules():
            if len(list(module.children())) > 0:
                continue  # Skip container modules
            
            module_type = type(module).__name__
            module_count[module_type] += 1
            layer_name = f"{name}_{module_type}_{module_count[module_type]}"
            
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=layer_name: conv2d_hook(m, i, o, n)))
            elif isinstance(module, nn.Conv1d):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=layer_name: conv1d_hook(m, i, o, n)))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=layer_name: bn_hook(m, i, o, n)))
            elif isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=layer_name: relu_hook(m, i, o, n)))
        
        # Forward pass
        with torch.no_grad():
            try:
                _ = self.forward(batched_pts, mode='test')
            except Exception as e:
                print(f"FLOPs calculation failed during forward pass: {e}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return dict(flops_dict)

    def summary(self, batched_pts=None, verbose: bool = True, calculate_flops: bool = False):
        """
        Print a concise model summary including configuration, parameter counts,
        FLOPs breakdown, and (optionally) tensor shapes.

        Args:
            batched_pts: Optional list[Tensor(N,4)] to run a dry pass and print shapes.
            verbose: Whether to print to stdout.
            calculate_flops: Whether to calculate detailed FLOPs breakdown (requires batched_pts).
        """
        lines = []
        lines.append("=== PointPillars Summary ===")
        # Config
        vox_info = self._get_voxelizer_info()
        lines.append(f"Backbone: {self.backbone_type}")
        lines.append(f"Encoder: {ENCODER}")
        lines.append(f"Voxelizer: {vox_info.get('type')}")
        if vox_info.get('deterministic') is not None:
            lines.append(f"  deterministic: {vox_info['deterministic']}")
        if 'voxel_size' in vox_info:
            lines.append(f"  voxel_size: {vox_info['voxel_size']}")
        if 'point_cloud_range' in vox_info:
            lines.append(f"  point_cloud_range: {vox_info['point_cloud_range']}")
        if 'grid_size' in vox_info:
            gx, gy = vox_info['grid_size']
            lines.append(f"  grid_size (x,y): ({gx}, {gy})")
        if 'max_num_points' in vox_info:
            lines.append(f"  max_points_per_voxel: {vox_info['max_num_points']}")
        if 'max_voxels' in vox_info:
            lines.append(f"  max_voxels (train,test): {vox_info['max_voxels']}")

        # Params
        sections = [
            ("pillar_layer", self.pillar_layer),
            ("pillar_encoder", self.pillar_encoder),
            ("backbone", self.backbone),
            ("neck", self.neck),
            ("head", self.head),
            ("full_model", self),
        ]
        lines.append("\nParameters:")
        for name, module in sections:
            tot, tri = self._count_params(module)
            lines.append(f"- {name:14s} total={tot:,} trainable={tri:,}")

        # FLOPs calculation
        if calculate_flops and batched_pts is not None:
            lines.append("\n=== Detailed FLOPs Breakdown ===")
            flops_info = self._calculate_flops_detailed(batched_pts)
            
            # Group by module
            encoder_flops = sum(v['flops'] for k, v in flops_info.items() if 'pillar_encoder' in k)
            backbone_flops = sum(v['flops'] for k, v in flops_info.items() if 'backbone' in k)
            neck_flops = sum(v['flops'] for k, v in flops_info.items() if 'neck' in k)
            head_flops = sum(v['flops'] for k, v in flops_info.items() if 'head' in k)
            total_flops = sum(v['flops'] for v in flops_info.values())
            
            lines.append(f"\nModule-wise FLOPs:")
            lines.append(f"- Pillar Encoder: {encoder_flops/1e9:.3f} GFLOPs ({encoder_flops/total_flops*100:.1f}%)")
            lines.append(f"- Backbone:       {backbone_flops/1e9:.3f} GFLOPs ({backbone_flops/total_flops*100:.1f}%)")
            lines.append(f"- Neck:           {neck_flops/1e9:.3f} GFLOPs ({neck_flops/total_flops*100:.1f}%)")
            lines.append(f"- Head:           {head_flops/1e9:.3f} GFLOPs ({head_flops/total_flops*100:.1f}%)")
            lines.append(f"- TOTAL:          {total_flops/1e9:.3f} GFLOPs")
            
            # Detailed layer breakdown (top 20 layers by FLOPs)
            lines.append(f"\nTop 20 Layers by FLOPs:")
            sorted_layers = sorted(flops_info.items(), key=lambda x: x[1]['flops'], reverse=True)[:20]
            lines.append(f"{'Layer Name':<50} {'Type':<12} {'FLOPs (M)':<12} {'Params':<12} {'Shape'}")
            lines.append("-" * 140)
            for layer_name, info in sorted_layers:
                name_short = layer_name[-47:] if len(layer_name) > 47 else layer_name
                lines.append(f"{name_short:<50} {info['type']:<12} {info['flops']/1e6:<12.2f} "
                           f"{info['params']:<12,} {info.get('shape', 'N/A')}")

        # Optional dry-run to report shapes
        if batched_pts is not None and not calculate_flops:
            self.eval()
            with torch.no_grad():
                try:
                    pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
                    lines.append("\nShapes:")
                    lines.append(f"pillars: {tuple(pillars.shape)}")
                    lines.append(f"coors_batch: {tuple(coors_batch.shape)}")
                    lines.append(f"npoints_per_pillar: {tuple(npoints_per_pillar.shape)}")

                    feat2d = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                    lines.append(f"pillar_features(2D): {tuple(feat2d.shape)}")

                    xs = self.backbone(feat2d)
                    for i, xi in enumerate(xs):
                        lines.append(f"backbone[{i}]: {tuple(xi.shape)}")

                    neck_out = self.neck(xs)
                    lines.append(f"neck_out: {tuple(neck_out.shape)}")

                    bcls, breg, bdir = self.head(neck_out)
                    lines.append(f"head.cls: {tuple(bcls.shape)}")
                    lines.append(f"head.reg: {tuple(breg.shape)}")
                    lines.append(f"head.dir: {tuple(bdir.shape)}")
                except Exception as e:
                    lines.append(f"[summary] Dry-run failed: {repr(e)}")

        text = "\n".join(lines)
        if verbose:
            print(text)
        return text

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            # Return empty dictionary with correct keys for consistency
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
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        
        if ENABLE_TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_voxel = time.time()
                
        # Voxelization stage
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        
        if ENABLE_TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_voxel = time.time()
            start_encoder = time.time()

        # Pillar encoding stage
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
        
        if ENABLE_TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_encoder = time.time()
            start_backbone = time.time()

        # Backbone stage
        xs = self.backbone(pillar_features)
        
        if ENABLE_TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_backbone = time.time()
            start_neck = time.time()

        # Neck stage
        x = self.neck(xs)
        
        if ENABLE_TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_neck = time.time()
            start_head = time.time()

        # Head stage
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)
        
        if ENABLE_TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_head = time.time()
            
            voxel_time = end_voxel - start_voxel
            encoder_time = end_encoder - start_encoder
            backbone_time = end_backbone - start_backbone
            neck_time = end_neck - start_neck
            head_time = end_head - start_head
            total_time = end_head - start_voxel
            
            print("\nPointPillars timing breakdown:")
            print(f"- Voxelization:    {voxel_time*1000:.3f}ms")
            print(f"- Pillar Encoder:  {encoder_time*1000:.3f}ms")
            print(f"- Backbone:        {backbone_time*1000:.3f}ms")
            print(f"- Neck:            {neck_time*1000:.3f}ms")
            print(f"- Head:            {head_time*1000:.3f}ms")
            print(f"- Total time:      {total_time*1000:.3f}ms ({1/total_time:.1f} FPS)")

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   
