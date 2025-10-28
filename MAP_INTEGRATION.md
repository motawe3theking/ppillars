# mAP Metric Integration Guide

## Overview
Added mAP (mean Average Precision) metric computation to both training and testing pipelines for the PointPillars 3D object detection model.

## Changes Made

### 1. Training Script (`train.py`)

#### New Functions Added:
- `get_score_thresholds()`: Calculates score thresholds for PR curve evaluation
- `compute_mAP_3d()`: Computes mAP for 3D bounding boxes using KITTI evaluation protocol
- `evaluate_model()`: Runs full evaluation on validation set and returns mAP metrics

#### New Command-Line Arguments:
- `--eval_map`: Enable mAP evaluation during training (flag)
- `--eval_freq_epoch`: Frequency of epochs to run mAP evaluation (default: 5)

#### Integration:
- mAP evaluation runs at specified epoch intervals
- Results logged to TensorBoard under `val/AP_{class}` and `val/mAP`
- Per-class AP and overall mAP printed to console

### 2. Evaluation Script (`evaluate.py`)

#### Enhancements:
- Added summary mAP output showing average mAP across difficulties
- Modified `do_eval()` to return `overall_results` dict for programmatic access
- Enhanced output formatting for better readability

### 3. Evaluation Metrics

The implementation computes:
- **3D AP** @ IoU thresholds:
  - Car: 0.7
  - Pedestrian: 0.5
  - Cyclist: 0.5
- **Per-class AP** for each difficulty level (Easy, Moderate, Hard)
- **Overall mAP** averaged across all classes
- Uses **11-point interpolation** method (KITTI standard)

## Usage

### During Training

#### Enable mAP Evaluation:
```powershell
python train.py --data_root <path> --eval_map --eval_freq_epoch 1
```

#### Example with Full Options:
```powershell
python train.py \
    --data_root C:\Users\AIT\Desktop\GIU\Bachelor\manual_repo_migrate\PointPillars\pointpillars\dataset\kitti\kitti \
    --resume C:\Users\AIT\Desktop\GIU\Bachelor\manual_repo_migrate\PointPillars\pillar_logs\checkpoints\checkpoint_epoch_2.pth \
    --eval_freq_epoch 1 \
    --eval_map \
    --batch_size 2 \
    --max_epoch 50
```

#### Parameters:
- `--eval_map`: Enables mAP computation (without this flag, only loss is tracked)
- `--eval_freq_epoch N`: Runs evaluation every N epochs (default: 5)

### Standalone Evaluation

Run the enhanced evaluate.py script:
```powershell
python evaluate.py \
    --data_root <path_to_kitti> \
    --ckpt <path_to_checkpoint> \
    --saved_path results
```

Output includes:
- Per-class AP for 2D bbox, BEV, and 3D bbox
- Overall mAP for each metric type
- Summary mAP averaged across difficulties
- Results saved to `results/eval_results.txt`

## TensorBoard Visualization

View mAP metrics in TensorBoard:
```powershell
tensorboard --logdir pillar_logs/summary
```

Available metrics:
- `val/AP_Car`: Car class Average Precision
- `val/AP_Pedestrian`: Pedestrian class Average Precision  
- `val/AP_Cyclist`: Cyclist class Average Precision
- `val/mAP`: Overall mean Average Precision

## Output Format

### Console Output (Training):
```
==================== Evaluating Epoch 5 ====================

mAP Results (Epoch 5):
  Car: 85.32%
  Pedestrian: 72.45%
  Cyclist: 68.91%
  mAP: 75.56%
============================================================
```

### Evaluation File Output:
```
==========BBOX_3D==========
Car AP@0.7: 85.3245 78.2341 72.1234
Pedestrian AP@0.5: 72.4512 68.3421 65.2134
Cyclist AP@0.5: 68.9123 65.7823 62.4512

==========Overall==========
bbox_3d AP: 75.5627 70.7862 66.5960

==========Summary mAP==========
bbox_3d mAP (avg over difficulties): 70.9816
```

## Implementation Details

### mAP Calculation Method:
1. **IoU Computation**: 3D bounding box IoU in camera coordinates
2. **Matching**: Greedy matching of detections to ground truth
3. **Filtering**: Height-based difficulty thresholds (KITTI standard)
4. **PR Curve**: 41-point sampling for precision-recall curve
5. **Interpolation**: 11-point interpolation for final AP
6. **Averaging**: Mean across all classes for overall mAP

### Performance Considerations:
- Evaluation runs on GPU for faster IoU computation
- Progress bars show evaluation progress
- Results cached to avoid redundant computation
- Validation loss calculation remains separate from mAP evaluation

## Troubleshooting

### Issue: Out of Memory During Evaluation
**Solution**: Reduce batch size or evaluate less frequently
```powershell
python train.py --eval_freq_epoch 10 --batch_size 1 --eval_map
```

### Issue: Evaluation Takes Too Long
**Solution**: 
- Increase `--eval_freq_epoch` to evaluate less often
- Use smaller validation set
- Ensure CUDA is enabled (`--no_cuda` flag not set)

### Issue: mAP Not Showing in TensorBoard
**Solution**: Ensure `--eval_map` flag is set when running training

## Notes

- mAP evaluation uses moderate difficulty by default (height >= 25 pixels)
- Evaluation follows KITTI 3D object detection benchmark protocol
- Results are comparable to official KITTI evaluation server metrics
- Training can continue without mAP evaluation if `--eval_map` is not set
