# FLOPs Calculation for PointPillars

This document explains how to calculate FLOPs (Floating Point Operations) for the PointPillars model.

## Quick Start

### Option 1: Using the standalone script

```bash
# Basic usage with default settings
python calculate_flops.py

# With a trained checkpoint
python calculate_flops.py --checkpoint pretrained/epoch_160.pth

# With CUDA acceleration
python calculate_flops.py --cuda --batch_size 2 --num_points 15000
```

### Option 2: Using the model's summary() method directly

```python
import torch
from pointpillars.model.pointpillars import PointPillars

# Create model
model = PointPillars(nclasses=3)
model.eval()

# Create dummy input (list of point cloud tensors)
batched_pts = [torch.rand(10000, 4) for _ in range(1)]  # 1 sample with 10k points

# Calculate FLOPs with detailed breakdown
model.summary(batched_pts=batched_pts, calculate_flops=True)
```

## Output Example

The FLOPs summary includes:

1. **Module-wise FLOPs**: Total FLOPs for each major component (Encoder, Backbone, Neck, Head)
2. **Layer-wise breakdown**: Top 20 layers sorted by computational cost
3. **Percentage distribution**: How FLOPs are distributed across modules

Example output:
```
=== Detailed FLOPs Breakdown ===

Module-wise FLOPs:
- Pillar Encoder: 0.234 GFLOPs (5.2%)
- Backbone:       3.456 GFLOPs (76.8%)
- Neck:           0.789 GFLOPs (17.5%)
- Head:           0.023 GFLOPs (0.5%)
- TOTAL:          4.502 GFLOPs

Top 20 Layers by FLOPs:
Layer Name                                          Type         FLOPs (M)    Params       Shape
--------------------------------------------------------------------------------
backbone.conv_layer_1                              Conv2d       1234.56      65,536       (1, 64, 128, 128) -> (1, 128, 64, 64)
...
```

## FLOPs Calculation Details

The FLOPs counter calculates operations for:
- **Conv2d**: 2 × C_in × K_h × K_w × C_out × H_out × W_out
- **Conv1d**: 2 × C_in × K × C_out × L_out
- **BatchNorm**: 2 × num_elements (normalize + scale/shift)
- **ReLU**: num_elements
- **Linear**: batch × in_features × out_features

## Notes

- FLOPs are calculated for inference (eval mode)
- The calculation requires a forward pass, so input data is needed
- Results may vary slightly based on input point cloud density
- Voxelization is not included in FLOPs (it's a data preprocessing step)

## Performance Analysis

Use FLOPs to:
- Compare different model configurations (default vs MobileNet backbone)
- Identify computational bottlenecks
- Estimate deployment requirements for edge devices
- Optimize model architecture

## Comparison with Other Models

Typical 3D object detection models:
- **PointPillars (default)**: ~4-6 GFLOPs
- **PointPillars (MobileNet)**: ~2-3 GFLOPs
- **VoxelNet**: ~15-20 GFLOPs
- **PointNet++**: ~8-12 GFLOPs

Lower FLOPs generally means faster inference and lower power consumption.
