# Training Guide for PointPillarsFusion

## Quick Start

### Basic Training

```bash
python train_fusion.py \
    --data_root /path/to/kitti/dataset \
    --batch_size 2 \
    --max_epoch 160 \
    --log_dir ./logs_fusion \
    --ckpt_dir ./logs_fusion/checkpoints
```

### Training with Evaluation

```bash
python train_fusion.py \
    --data_root /path/to/kitti/dataset \
    --batch_size 2 \
    --max_epoch 160 \
    --eval_map \
    --eval_freq_epoch 5 \
    --log_dir ./logs_fusion \
    --ckpt_dir ./logs_fusion/checkpoints
```

### Resume from Checkpoint

```bash
python train_fusion.py \
    --data_root /path/to/kitti/dataset \
    --resume_from ./logs_fusion/checkpoints/checkpoint_epoch_20.pth \
    --batch_size 2 \
    --max_epoch 160
```

### Training with Model Summary

```bash
python train_fusion.py \
    --data_root /path/to/kitti/dataset \
    --summary \
    --batch_size 2 \
    --max_epoch 160
```

## Command-Line Arguments

### Data Arguments
- `--data_root`: Path to KITTI dataset root (REQUIRED)
- `--batch_size`: Training batch size (default: 2)
  - Recommended: 2-4 for fusion model (higher memory usage than pillar-only)
- `--num_workers`: Number of data loading workers (default: 4)

### Model Arguments
- `--nclasses`: Number of object classes (default: 3 for Car, Pedestrian, Cyclist)
- `--no_cuda`: Use CPU instead of GPU (not recommended for training)
- `--summary`: Print model architecture summary before training

### Training Arguments
- `--max_epoch`: Maximum number of training epochs (default: 160)
- `--init_lr`: Initial learning rate (default: 0.0001)
  - OneCycleLR will increase it to 0.001 max

### Checkpoint Arguments
- `--log_dir`: TensorBoard log directory (default: ./pillar_logs_fusion)
- `--ckpt_dir`: Checkpoint save directory (default: ./pillar_logs_fusion/checkpoints)
- `--ckpt_freq_epoch`: Save checkpoint every N epochs (default: 20)
- `--log_freq`: Log metrics to TensorBoard every N steps (default: 10)
- `--resume_from`: Path to checkpoint to resume training from

### Evaluation Arguments
- `--eval_map`: Enable mAP evaluation during training
- `--eval_freq_epoch`: Evaluate every N epochs (default: 5)

## Dataset Structure

The script expects KITTI dataset in the following structure:

```
data_root/
├── kitti_infos_train.pkl      # Training metadata
├── kitti_infos_val.pkl         # Validation metadata
├── kitti_dbinfos_train.pkl     # Database info for augmentation
├── training/
│   ├── velodyne/               # Point cloud .bin files
│   ├── image_2/                # RGB images .png files
│   ├── calib/                  # Calibration .txt files
│   └── label_2/                # Ground truth labels
└── testing/
    ├── velodyne/
    ├── image_2/
    └── calib/
```

## Training Process

### 1. Data Loading
- Point clouds are loaded from `velodyne/` directory
- RGB images are loaded from `image_2/` directory  
- Calibration matrices (P2, R0_rect, Tr_velo_to_cam) from `calib/`
- Ground truth boxes and labels from preprocessed pickle files

### 2. Data Preprocessing
Each batch is processed as follows:
- **Point clouds**: (N, 4) tensors [x, y, z, intensity]
- **Images**: Loaded, normalized to [0, 1], transposed to (3, H, W)
- **Calibration**: Converted to numpy arrays in dict format
- **Ground truth**: 3D bounding boxes (M, 7) and labels (M,)

### 3. Forward Pass
```python
# Model input
batched_pts: List[(N_i, 4)]
batched_imgs: (B, 3, H, W)
batched_calib: List[Dict]

# Depth projection creates 4th channel
depth_maps: (B, 1, H, W)

# RGBD concatenation
rgbd: (B, 4, H, W)

# Through backbone → neck → head
predictions: (cls, reg, dir)
```

### 4. Loss Computation
Three loss components:
- **Classification Loss**: Focal loss for object classification
- **Regression Loss**: Smooth L1 loss for bbox parameters
- **Direction Loss**: Cross-entropy for orientation bins

Total loss = cls_loss + reg_loss + dir_loss

### 5. Optimization
- **Optimizer**: AdamW (weight_decay=0.01, betas=(0.95, 0.99))
- **Scheduler**: OneCycleLR
  - Warmup: 40% of training
  - Max LR: 10x initial LR
  - Cosine annealing
- **Gradient clipping**: max_norm=35

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs_fusion
```

View metrics:
- `train/total_loss`: Total training loss
- `train/loss_cls`: Classification loss
- `train/loss_reg`: Regression loss  
- `train/loss_dir`: Direction classification loss
- `lr`: Current learning rate
- `val/AP_Car`, `val/AP_Pedestrian`, `val/AP_Cyclist`: Per-class AP
- `val/mAP`: Mean average precision

### Console Output

During training, you'll see:
```
Epoch 1/160
================================================================================
Training: 100%|████████| 1234/1234 [12:34<00:00, loss=1.2345, cls=0.5, reg=0.6, dir=0.1, lr=0.000123]

Checkpoint saved: ./logs_fusion/checkpoints/checkpoint_epoch_20.pth

================================================================================
Running validation evaluation...
================================================================================
Evaluating: 100%|████████| 234/234 [02:34<00:00]

Validation Results (Epoch 20):
  AP_Car: 65.32%
  AP_Pedestrian: 42.18%
  AP_Cyclist: 38.45%
  mAP: 48.65%
```

## Memory Considerations

### GPU Memory Usage
Fusion model requires more memory than pillar-only:
- 4-channel RGBD images (vs 64-channel pseudo-images)
- Full resolution images vs sparse pillar representation

**Recommended batch sizes by GPU:**
- 8GB VRAM: batch_size=1
- 12GB VRAM: batch_size=2
- 16GB VRAM: batch_size=4
- 24GB VRAM: batch_size=6-8

### Reducing Memory Usage
If you encounter OOM errors:
1. Reduce batch size: `--batch_size 1`
2. Reduce image size (requires code modification)
3. Use gradient accumulation (requires code modification)
4. Use mixed precision training (requires code modification)

## Expected Training Time

On a single GPU:
- **RTX 3090**: ~8-10 hours for 160 epochs (batch_size=2)
- **RTX 4090**: ~6-8 hours for 160 epochs (batch_size=4)
- **V100**: ~10-12 hours for 160 epochs (batch_size=2)

## Checkpoints

Checkpoints are saved with the following structure:
```python
{
    'epoch': 20,
    'global_step': 24680,
    'train_step': 24680,
    'model_state_dict': OrderedDict(...),
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...}
}
```

To load for inference:
```python
checkpoint = torch.load('checkpoint_epoch_20.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### Error: "CUDA out of memory"
**Solution**: Reduce batch size or use smaller images

### Error: "Image file not found"
**Solution**: Check that `image_2/` directory exists and paths in pickle files are correct

### Error: "Calibration matrix missing"
**Solution**: Ensure calibration files exist in `calib/` directory with P2, R0_rect, Tr_velo_to_cam

### Warning: "No valid points after projection"
**Solution**: This is normal for some frames - the model handles empty depth maps

### Loss is NaN
**Solutions**:
1. Check learning rate (try reducing to 0.00005)
2. Check for corrupted data
3. Ensure gradients are clipped
4. Verify calibration matrices are correct

## Best Practices

1. **Start with pretrained backbone**: If available, initialize MobileNetV1 with ImageNet weights
2. **Monitor validation mAP**: Early stopping if mAP plateaus
3. **Save multiple checkpoints**: Keep every 20 epochs for comparison
4. **Use data augmentation**: Already included via Kitti dataset
5. **Experiment with depth normalization**: Current max is 80m, adjust if needed

## Differences from Original PointPillars Training

| Aspect | Original | Fusion |
|--------|----------|--------|
| Input preparation | Voxelization | Image loading + depth projection |
| Batch preparation | Point clouds only | Point clouds + images + calibration |
| Memory usage | Lower | Higher (full images) |
| Training speed | Faster | Slightly slower (image loading) |
| Data augmentation | 3D only | Must maintain RGB-LiDAR alignment |

## Next Steps After Training

1. **Evaluate on test set**: Use `evaluate_fusion.py` (to be created)
2. **Visualize results**: Use `test_fusion.py` with visualization
3. **Compare with baseline**: Train original PointPillars for comparison
4. **Fine-tune hyperparameters**: Adjust LR, batch size, augmentation
5. **Export model**: Convert to ONNX or TorchScript for deployment
