# Low mAP Analysis for PointPillars

## Your Results Summary
```
==========BBOX_3D==========
Pedestrian AP@0.5: 14.55 | 13.40 | 12.96
Cyclist AP@0.5:    18.27 | 14.08 | 13.85
Car AP@0.7:        10.11 | 10.50 | 10.01

Overall bbox_3d AP: 14.31 | 12.66 | 12.27
```

## Expected Results (from README)
```
==========BBOX_3D==========
Pedestrian AP@0.5: 51.46 | 47.94 | 43.80
Cyclist AP@0.5:    81.87 | 63.66 | 60.91
Car AP@0.7:        86.65 | 76.74 | 74.17

Overall bbox_3d AP: 73.33 | 62.78 | 59.63
```

**Your performance is ~5-8x lower than expected!**

---

## Root Cause Analysis

### 🔴 **CRITICAL ISSUE #1: Point Cloud Range Mismatch**

The model architecture is **fundamentally incompatible** with the evaluation setup:

#### Model Configuration (pointpillars.py line 668):
```python
point_cloud_range=[0, -10.24, -3, 69.12, 10.24, 1]
# Y-range: -10.24 to 10.24 (ONLY 20.48 METERS WIDTH!)
```

#### Evaluation/Ground Truth Range (evaluate.py line 350):
```python
pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0])
# Y-range: -40 to 40 (80 METERS WIDTH!)
```

#### **Impact:**
- **The model only "sees" the middle ~26% of the scene width!**
- Objects outside Y=[-10.24, 10.24] are **invisible** to the model during training
- Grid size: Only 128 pillars in Y-direction vs needed 500 pillars
- **Most ground truth boxes are outside the model's receptive field**

**This is like training a model to detect cars only in the center lane, then testing it on all lanes.**

---

### 🔴 **CRITICAL ISSUE #2: Commented-Out Correct Configuration**

Look at line 669 in `pointpillars.py`:
```python
point_cloud_range=[0, -10.24, -3, 69.12, 10.24, 1],     # CURRENT (WRONG!)
# point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],   # THIS IS COMMENTED OUT (CORRECT!)
```

**Someone changed the range from 79.36m to 20.48m width and forgot to change it back!**

---

## Secondary Issues

### 📊 **Issue #3: Voxelization Bottleneck**
Your code shows:
```python
USE_HARD_VOXELIZATION = False  # Using default (soft) voxelization
ENABLE_TIMING = False
```

The `HardPillarVoxelization` implementation has severe performance issues:
- **Line 164-175**: Python for-loop iterating over batch samples (GPU → CPU → GPU transfer bottleneck)
- **Line 140-145**: Cumulative sum operation with timing debug prints still in code
- These debug prints indicate someone was profiling performance issues

**Impact:** Training is probably very slow, limiting epochs/experimentation.

---

### 📐 **Issue #4: NMS Configuration**
```python
self.nms_pre = 100       # Only keep top 100 before NMS
self.nms_thr = 0.01      # VERY LOW threshold (keeps almost everything)
self.score_thr = 0.2     # Relatively low confidence threshold
self.max_num = 50        # Final detection limit
```

**Analysis:**
- `nms_thr=0.01` is extremely permissive (typical is 0.45-0.7)
- This means overlapping detections are rarely suppressed
- Could lead to many duplicate/poor quality predictions
- But this is a **symptom**, not the root cause (range mismatch is)

---

### 🎯 **Issue #5: Anchor Configuration vs Point Cloud Range**
```python
# Anchor ranges (line 701):
ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],  # Pedestrian
          [0, -39.68, -0.6, 69.12, 39.68, -0.6],  # Cyclist
          [0, -39.68, -1.78, 69.12, 39.68, -1.78]] # Car

# But point cloud range is:
point_cloud_range=[0, -10.24, -3, 69.12, 10.24, 1]
```

**MASSIVE INCONSISTENCY:**
- Anchors are generated over Y=[-39.68, 39.68] (79.36m)
- But features only exist for Y=[-10.24, 10.24] (20.48m)
- **74% of anchor locations have NO FEATURE DATA underneath them!**
- The model is making predictions in areas it cannot see

This is like asking someone to find objects in a dark room while standing in a spotlight.

---

## Verification

### Expected Grid Dimensions:
With correct range `[0, -39.68, -3, 69.12, 39.68, 1]` and `voxel_size=[0.16, 0.16, 4]`:
```python
grid_x = (69.12 - 0) / 0.16 = 432
grid_y = (39.68 - (-39.68)) / 0.16 = 496
grid_z = (1 - (-3)) / 4 = 1
# Grid: 432 × 496 × 1 = 214,272 pillars
```

### Current (Wrong) Grid Dimensions:
With current range `[0, -10.24, -3, 69.12, 10.24, 1]`:
```python
grid_x = (69.12 - 0) / 0.16 = 432
grid_y = (10.24 - (-10.24)) / 0.16 = 128
grid_z = (1 - (-3)) / 4 = 1
# Grid: 432 × 128 × 1 = 55,296 pillars (4X SMALLER!)
```

---

## Why Some Detections Still Occur

You're getting ~10-15% AP instead of 0% because:
1. **Center-biased dataset**: KITTI objects tend to cluster near the vehicle's path
2. **Some objects are in range**: Objects in Y=[-10.24, 10.24] can still be detected
3. **2D metrics are higher**: Your 2D bbox AP (43%) is better than 3D (10%) because 2D projections still work for visible objects

---

## How This Happened

Looking at the code structure:
1. Original implementation had correct range (line 669 commented code)
2. Someone experimented with a narrower range (perhaps for speed/memory)
3. The change was committed but never reverted
4. The pretrained checkpoint `pretrained/epoch_160.pth` was likely trained with the **correct** range
5. **You're loading a checkpoint trained on 79m width into a model expecting 20m width!**

This explains:
- Why checkpoint loading works (architectures are identical)
- Why performance is catastrophic (feature maps are wrong size/scale)
- Why the README shows good results (they used the correct range)

---

## Solutions

### ✅ **IMMEDIATE FIX** (Critical)

**Option A: Use the correct range (RECOMMENDED)**
```python
# In pointpillars/model/pointpillars.py line 668
point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],  # CORRECT!
```

**Option B: Train from scratch with narrow range**
- Keep `[0, -10.24, -3, 69.12, 10.24, 1]`
- Delete pretrained checkpoint
- Train for 160 epochs from random initialization
- Update anchor ranges to match
- **Not recommended**: You lose 74% of training data

---

### ✅ **Secondary Fixes**

1. **Fix NMS threshold:**
```python
self.nms_thr = 0.5  # Standard IoU threshold for NMS
```

2. **Clean up debug code:**
```python
ENABLE_TIMING = False  # Remove timing prints from HardPillarVoxelization
# Delete lines 140-145 (cumsum timing)
# Delete lines 170-175 (filtering timing)
```

3. **Verify checkpoint compatibility:**
```bash
# After fixing range, retrain from scratch OR
# Find the original checkpoint trained with correct range
```

---

## Testing the Fix

After changing line 668 to use the correct range:

```bash
# Re-evaluate with corrected model
python evaluate.py --ckpt pretrained/epoch_160.pth --data_root <your_path>

# Expected results should jump from ~14% to ~73% mAP (3D)
```

If results don't improve:
- The checkpoint may have been trained with the wrong range too
- Need to retrain from scratch with correct configuration

---

## Conclusion

**Primary Issue:** Point cloud range mismatch (20m vs 80m width)
- **Severity:** CRITICAL - Model cannot see 74% of the scene
- **Fix Effort:** 1 line change + potential retraining
- **Expected Improvement:** 5-8x mAP increase

**Secondary Issues:**
- NMS threshold too low
- Performance bottlenecks in voxelization
- Debug code still present

The good news: This is a **configuration error**, not an algorithmic problem. The architecture is sound, just configured incorrectly.

---

## Recommended Action Plan

1. ✅ **Change line 668** to use correct range
2. ✅ **Test with existing checkpoint** (may or may not work)
3. ✅ **If still low mAP**: Retrain from scratch for 160 epochs
4. ✅ **Fix NMS threshold** to 0.5
5. ✅ **Remove debug prints** from voxelization
6. ✅ **Document the correct configuration** to prevent future issues

This should recover full performance and match the README benchmarks.
