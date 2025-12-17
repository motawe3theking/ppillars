"""Precompute pillar feature maps (post-voxel inputs) for calibration.

This script loads KITTI samples, runs voxelization + pillar encoder, and
saves the resulting 2D pillar feature maps to disk as .npy files which can
be used later as calibration inputs for the post-voxel quantizer.

Usage example:
  python tools/precompute_pillar_features.py \
      --data-root /path/to/kitti \
      --out-dir calib_features/train \
      --num-samples 500
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add repo root to path so we can import local modules
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from pointpillars.dataset import Kitti
from pointpillars.model.pointpillars import PillarLayer, PillarEncoder, PointPillars


def save_feature(arr: np.ndarray, out_path: str):
    np.save(out_path, arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="C:\\Users\\AIT\\Desktop\\GIU\\Bachelor\\manual_repo_migrate\\PointPillars\\pointpillars\\dataset\\kitti\\kitti", help='KITTI data root (where kitti_infos_*.pkl lives)')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'trainval', 'test'])
    parser.add_argument('--out-dir', required=True, help='Directory to save .npy feature files')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--pts-prefix', default='velodyne_reduced')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Instantiate a full PointPillars instance only to reuse pillar_layer and pillar_encoder config
    model = PointPillars()
    pillar_layer = model.pillar_layer
    pillar_encoder = model.pillar_encoder

    pillar_layer.eval()
    pillar_encoder.eval()

    device = torch.device(args.device)
    # PillarLayer and PillarEncoder are pure-PyTorch; move parameters if any
    pillar_layer.to(device)
    pillar_encoder.to(device)

    dataset = Kitti(data_root=args.data_root, split=args.split, pts_prefix=args.pts_prefix)

    end_idx = min(len(dataset), args.start + args.num_samples)
    print(f"Precomputing features for samples {args.start}..{end_idx-1}")

    for i in range(args.start, end_idx):
        data = dataset[i]
        pts = data['pts']  # numpy (N, >=3 or 4)
        pts_t = torch.from_numpy(pts).float().to(device)

        with torch.no_grad():
            # PillarLayer expects a list of tensors (batch)
            pillars, coors_batch, npoints = pillar_layer([pts_t])
            # pillar_encoder returns (B, C, H, W)
            feat2d = pillar_encoder(pillars, coors_batch, npoints)

        feat_np = feat2d.cpu().numpy()
        # Save per-sample. Use zero-padded index for ordering
        out_path = os.path.join(args.out_dir, f"feat_{i:06d}.npy")
        print(f"Saving {out_path} -> shape {feat_np.shape}")
        save_feature(feat_np, out_path)


if __name__ == '__main__':
    main()
