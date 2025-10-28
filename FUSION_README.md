# PointPillars Camera-LiDAR Fusion Model

This is a camera-LiDAR fusion version of PointPillars that leverages both RGB images and LiDAR point clouds for 3D object detection.

## Architecture Overview

### Key Differences from Original PointPillars

| Component | Original PointPillars | Fusion PointPillars |
|-----------|----------------------|---------------------|
| **Input** | Point Cloud only | RGB Image + Point Cloud |
| **Feature Extraction** | Pillar-based voxelization + Pillar Encoder | Depth projection onto image plane |
| **Backbone Input** | Pseudo-image from pillars (64 channels) | 4-channel RGBD image |
| **Backbone** | MobileNetV1 (64 input channels) | MobileNetV1 (4 input channels) |
| **Neck** | Same FPN | Same FPN |
| **Head** | Same detection head | Same detection head |

### Architecture Pipeline

```
Input:
  - RGB Image: (B, 3, H, W)
  - Point Cloud: List[(N_i, 4)]  # [x, y, z, intensity]
  - Calibration: List[Dict]      # P2, R0_rect, Tr_velo_to_cam

    ↓
    
1. Depth Projection Module
   - Projects LiDAR points onto image plane using KITTI calibration
   - Creates depth channel: (B, 1, H, W)
   
    ↓
    
2. RGBD Concatenation
   - Combines RGB + Depth: (B, 4, H, W)
   
    ↓
    
3. Backbone (MobileNetV1)
   - Extracts multi-scale features
   - Output: [(B, 64, H/2, W/2), (B, 128, H/4, W/4), (B, 256, H/8, W/8)]
   
    ↓
    
4. Neck (FPN)
   - Fuses multi-scale features
   - Output: (B, 384, H_out, W_out)
   
    ↓
    
5. Head
   - Classification: (B, n_anchors*n_classes, H_out, W_out)
   - Regression: (B, n_anchors*7, H_out, W_out)
   - Direction: (B, n_anchors*2, H_out, W_out)
```

## Key Components

### 1. DepthProjection Module

Transforms LiDAR point cloud into a depth map aligned with the camera image.

**Algorithm:**
1. Convert points from Velodyne to Camera coordinates using `Tr_velo_to_cam`
2. Apply rectification using `R0_rect`
3. Project to image plane using camera matrix `P2`
4. Filter points within image bounds
5. Scatter depths to create depth map (taking minimum depth per pixel)

**Input:**
- `batched_pts`: List of point clouds, each (N, 4) [x, y, z, intensity]
- `batched_calib`: List of calibration dicts with P2, R0_rect, Tr_velo_to_cam

**Output:**
- `depth_maps`: (B, 1, H, W) normalized depth channel

### 2. FusionBackbone

MobileNetV1 adapted for 4-channel input (RGB + Depth).

**Input:** (B, 4, H, W)
**Output:** Multi-scale features
- Scale 1: (B, 64, H/2, W/2)
- Scale 2: (B, 128, H/4, W/4)  
- Scale 3: (B, 256, H/8, W/8)

### 3. Neck & Head

Same as original PointPillars - no modifications needed.

## Usage

### Basic Inference

```python
from pointpillars.model import PointPillarsFusion
from pointpillars.utils import read_points, read_calib
import cv2
import torch

# Initialize model
model = PointPillarsFusion(nclasses=3).cuda()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Load data
pc = torch.from_numpy(read_points('000000.bin')).cuda()
img = cv2.imread('000000.png')
img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
img = img.unsqueeze(0).cuda()  # (1, 3, H, W)

calib_info = read_calib('000000.txt')
calib_dict = {
    'P2': calib_info['P2'],
    'R0_rect': calib_info['R0_rect'],
    'Tr_velo_to_cam': calib_info['Tr_velo_to_cam']
}

# Run inference
with torch.no_grad():
    results = model(
        batched_pts=[pc],
        batched_imgs=img,
        batched_calib=[calib_dict],
        mode='test'
    )

# Get detections
bboxes = results[0]['lidar_bboxes']  # (N, 7)
labels = results[0]['labels']         # (N,)
scores = results[0]['scores']         # (N,)
```

### Using the Test Script

```bash
# Run inference with model summary
python test_fusion.py \
    --pc_path pointpillars/dataset/kitti/kitti/testing/velodyne/000002.bin \
    --img_path pointpillars/dataset/kitti/kitti/testing/image_2/000002.png \
    --calib_path pointpillars/dataset/kitti/kitti/testing/calib/000002.txt \
    --ckpt checkpoint.pth \
    --summary

# Dry run (summary only)
python test_fusion.py \
    --pc_path dummy.bin \
    --img_path dummy.png \
    --calib_path dummy.txt \
    --summary \
    --dry_run
```

## Calibration Format (KITTI)

The model expects KITTI-format calibration files with the following matrices:

```
P2: 3x4 camera projection matrix
    [[fx,  0, cx, tx],
     [ 0, fy, cy, ty],
     [ 0,  0,  1,  0]]

R0_rect: 4x4 rectification matrix
    [[r11, r12, r13, 0],
     [r21, r22, r23, 0],
     [r31, r32, r33, 0],
     [  0,   0,   0, 1]]

Tr_velo_to_cam: 4x4 transformation from velodyne to camera
    [[r11, r12, r13, tx],
     [r21, r22, r23, ty],
     [r31, r32, r33, tz],
     [  0,   0,   0,  1]]
```

## Model Configuration

### Default Parameters

```python
PointPillarsFusion(
    nclasses=3,                           # Number of classes (Car, Pedestrian, Cyclist)
    image_size=(375, 1242),              # KITTI image size (H, W)
    point_cloud_range=[0, -40, -3, 70.4, 40, 1],  # LiDAR range [x_min, y_min, z_min, x_max, y_max, z_max]
    max_num_points=32,                   # Not used in fusion model (kept for compatibility)
    max_voxels=(12000, 40000)           # Not used in fusion model (kept for compatibility)
)
```

### Inference Parameters

- `nms_pre`: 100 (keep top-100 before NMS)
- `nms_thr`: 0.01 (IoU threshold for NMS)
- `score_thr`: 0.2 (confidence threshold)
- `max_num`: 50 (maximum final detections)

## Shape Compatibility

### Input Shapes
- **RGB Image**: (B, 3, H, W) where H=375, W=1242 for KITTI
- **Point Cloud**: List of (N_i, 4) tensors [x, y, z, intensity]
- **Calibration**: List of B dictionaries

### Internal Shapes
1. **Depth Map**: (B, 1, 375, 1242)
2. **RGBD**: (B, 4, 375, 1242)
3. **Backbone Output**:
   - Level 1: (B, 64, 188, 621)
   - Level 2: (B, 128, 94, 311)
   - Level 3: (B, 256, 47, 156)
4. **Neck Output**: (B, 384, 188, 621)
5. **Head Output**:
   - Classification: (B, 18, 188, 621) for 3 classes
   - Regression: (B, 42, 188, 621)
   - Direction: (B, 12, 188, 621)

### Output Shapes
- **Bounding Boxes**: (N, 7) [x, y, z, w, l, h, theta]
- **Labels**: (N,) class indices
- **Scores**: (N,) confidence scores

## Advantages of Fusion Approach

1. **Rich Semantic Information**: RGB provides texture, color, and context
2. **Precise Depth**: LiDAR provides accurate 3D measurements
3. **Complementary Modalities**: 
   - Camera excels at texture/appearance discrimination
   - LiDAR excels at geometric understanding
4. **No Pillar Encoding**: Simpler architecture, potentially faster inference
5. **Unified Representation**: RGBD image is intuitive and well-suited for CNNs

## Limitations

1. **Calibration Dependency**: Requires accurate camera-LiDAR calibration
2. **Sparse Depth**: LiDAR creates sparse depth maps (many zero pixels)
3. **Image Size**: Fixed to KITTI dimensions (can be adapted)
4. **Memory**: 4-channel input may use more memory than pillar representation

## Training Considerations

1. **Data Augmentation**: Need to maintain RGB-LiDAR alignment
2. **Depth Normalization**: Important for training stability
3. **Sparse Depth Handling**: Consider using sparse convolutions or depth completion
4. **Multi-task Learning**: Can add auxiliary tasks (depth completion, segmentation)

## Comparison with Original PointPillars

| Metric | Original | Fusion |
|--------|----------|--------|
| Input Modalities | LiDAR only | RGB + LiDAR |
| Input Size | Variable (pillars) | Fixed (image) |
| Backbone Input Channels | 64 | 4 |
| Preprocessing | Voxelization + Encoding | Depth Projection |
| Parameter Count | ~1.3M (backbone) | ~1.3M (backbone) |
| Inference Speed | Fast | Similar (depends on depth projection) |

## Future Improvements

1. **Depth Completion**: Fill sparse depth map using RGB information
2. **Attention Mechanisms**: Learn to weight RGB vs Depth features
3. **Multi-Scale Fusion**: Fuse at multiple levels, not just input
4. **Temporal Fusion**: Use video sequences for better depth estimation
5. **Adaptive Calibration**: Learn calibration refinement end-to-end

## File Structure

```
pointpillars/
├── model/
│   ├── pointpillars.py          # Original pillar-based model
│   ├── pointpillars_fusion.py   # NEW: Camera-LiDAR fusion model
│   └── __init__.py              # Updated to export PointPillarsFusion
test_fusion.py                    # NEW: Example inference script
FUSION_README.md                  # This file
```

## Citation

If you use this fusion model, please cite the original PointPillars paper:

```bibtex
@inproceedings{lang2019pointpillars,
  title={PointPillars: Fast encoders for object detection from point clouds},
  author={Lang, Alex H and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12697--12705},
  year={2019}
}
```

## License

Same as the original PointPillars implementation.
