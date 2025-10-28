# PointPillarsFusion Training Error Fix

## Issue Fixed

**Error:** `Sizes of tensors must match except in dimension 1. Expected size 621 but got size 622 for tensor number 1 in the list.`

### Root Cause
KITTI dataset images have varying dimensions (not all exactly 375×1242). When stacking images into a batch tensor, PyTorch requires all tensors to have the same shape. The error occurred because images in the same batch had slightly different widths (621 vs 622 pixels).

### Solution
Modified `load_image()` function to resize all images to a fixed target size before processing:

```python
def load_image(img_path, target_size=(375, 1242)):
    """Load and preprocess image for fusion model.
    
    Args:
        img_path: Path to image file
        target_size: (H, W) to resize image to ensure batch consistency
    
    Returns:
        img: Resized and normalized image (H, W, 3)
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size to ensure all images in batch have same dimensions
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    img = img.astype(np.float32) / 255.0
    return img
```

**Why this works:**
- All images are now resized to exactly (375, 1242) before batching
- `torch.stack()` can now combine them into (B, 3, 375, 1242) tensor
- Uses LINEAR interpolation to preserve image quality
- Only resizes if dimensions don't match (efficiency)

---

## New Feature: Enhanced Checkpoint Management

### Changes Made

1. **Checkpoint Frequency Control:**
   - `--ckpt_freq_epoch` (default: 1): Save checkpoint every N epochs
   - `--ckpt_freq_iter` (default: 0): Save checkpoint every N iterations (disabled by default)

2. **Automatic Checkpoint Cleanup:**
   - `--keep_latest_n` (default: 5): Keep only the latest N checkpoints
   - Automatically deletes older checkpoints to save disk space
   - Prevents disk from filling up during long training runs

3. **Per-Iteration Checkpointing:**
   - Enable with `--ckpt_freq_iter 100` to save every 100 iterations
   - Useful for recovery from crashes during long epochs
   - **Warning:** Can create many files, use sparingly

### Usage Examples

**Default (save every epoch, keep 5 latest):**
```bash
python train_fusion.py --data_root <path>
```

**Save every 2 epochs, keep 10 checkpoints:**
```bash
python train_fusion.py --data_root <path> --ckpt_freq_epoch 2 --keep_latest_n 10
```

**Save every 500 iterations (very frequent):**
```bash
python train_fusion.py --data_root <path> --ckpt_freq_iter 500
```

**Disable automatic cleanup (keep all checkpoints):**
```bash
python train_fusion.py --data_root <path> --keep_latest_n 0
```

### Implementation Details

**Checkpoint Management Function:**
```python
def manage_checkpoints(checkpoint_dir, keep_latest_n=5):
    """
    Keep only the latest N checkpoints to save disk space.
    Deletes older checkpoints based on epoch number.
    """
    - Finds all checkpoint_epoch_*.pth files
    - Sorts by epoch number
    - Deletes oldest checkpoints beyond keep_latest_n
    - Safely handles errors during deletion
```

**Checkpoint Save Locations:**
- Epoch checkpoints: `checkpoint_epoch_<N>.pth`
- Iteration checkpoints: `checkpoint_iter_<N>.pth`

**Information Saved in Checkpoints:**
```python
{
    'epoch': current_epoch,
    'global_step': total_steps_across_all_epochs,
    'train_step': total_training_steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}
```

---

## Testing the Fix

**Run training with fixed configuration:**
```bash
python train_fusion.py \
    --data_root C:\Users\AIT\Desktop\GIU\Bachelor\manual_repo_migrate\PointPillars\pointpillars\dataset\kitti\kitti \
    --batch_size 2 \
    --max_epoch 160 \
    --ckpt_freq_epoch 1 \
    --keep_latest_n 5
```

**Expected behavior:**
- ✅ No more tensor size mismatch errors
- ✅ All images resized to 375×1242 before batching
- ✅ Checkpoint saved after each epoch
- ✅ Only latest 5 checkpoints kept on disk
- ✅ Training progress logged every 10 steps

---

## Performance Considerations

### Image Resizing Impact
- **Minimal overhead:** cv2.resize is highly optimized
- **Quality:** LINEAR interpolation preserves features well
- **Alternative:** Could use original sizes with dynamic batching (more complex)

### Checkpoint Storage
With default settings (every epoch, keep 5):
- ~5 checkpoint files on disk at any time
- Each checkpoint: ~50-100MB (depending on model size)
- Total disk usage: ~250-500MB for checkpoints

With per-iteration saving (e.g., every 100 iters):
- Can create hundreds of checkpoints per epoch
- **Recommendation:** Only use for debugging or critical runs
- Combine with low `keep_latest_n` to control disk usage

---

## Troubleshooting

**If training still fails with size mismatch:**
1. Check KITTI dataset integrity (corrupted images?)
2. Verify all images can be loaded with cv2.imread()
3. Add debug print in load_image() to check shapes

**If running out of disk space:**
1. Reduce `keep_latest_n` (e.g., 3 or 2)
2. Increase `ckpt_freq_epoch` (e.g., 5 or 10)
3. Disable `ckpt_freq_iter` (set to 0)

**If need to recover from crash:**
1. Use `--resume_from <checkpoint_path>` to continue training
2. Training will resume from saved epoch/step

---

## Summary

✅ **Fixed:** Tensor dimension mismatch by resizing all images to uniform size
✅ **Added:** Flexible checkpoint saving (per-epoch and per-iteration)
✅ **Added:** Automatic checkpoint cleanup to manage disk space
✅ **Improved:** Training robustness and recovery capabilities

The fusion model training should now run smoothly without dimension errors, and checkpoints are saved more frequently with better management.
