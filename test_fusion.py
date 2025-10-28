"""
Example usage of PointPillarsFusion model.

This script demonstrates how to use the camera-LiDAR fusion model with KITTI data.
"""

import argparse
import cv2
import numpy as np
import os
import torch

from pointpillars.utils import setup_seed, read_points, read_calib, read_label
from pointpillars.model import PointPillarsFusion


def load_kitti_image(img_path):
    """Load and preprocess KITTI image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Transpose to (C, H, W)
    img = img.transpose(2, 0, 1)
    
    return img


def prepare_calib_dict(calib_info):
    """Convert calib_info to the format expected by the fusion model."""
    calib_dict = {
        'P2': calib_info['P2'],  # (3, 4)
        'R0_rect': calib_info['R0_rect'],  # (4, 4)
        'Tr_velo_to_cam': calib_info['Tr_velo_to_cam']  # (4, 4)
    }
    return calib_dict


def main(args):
    setup_seed()
    
    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2
    }
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    
    # Initialize fusion model
    if not args.no_cuda:
        model = PointPillarsFusion(nclasses=len(CLASSES)).cuda()
        if args.ckpt:
            checkpoint = torch.load(args.ckpt)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    else:
        model = PointPillarsFusion(nclasses=len(CLASSES))
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    
    # Print model summary if requested
    if args.summary:
        model.summary()
        if args.dry_run:
            return
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Point Cloud: {args.pc_path}")
    print(f"  Image: {args.img_path}")
    print(f"  Calibration: {args.calib_path}")
    
    # Load point cloud
    pc = read_points(args.pc_path)
    pc_torch = torch.from_numpy(pc)
    
    # Load image
    img = load_kitti_image(args.img_path)
    img_torch = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
    
    # Load calibration
    calib_info = read_calib(args.calib_path)
    calib_dict = prepare_calib_dict(calib_info)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        if not args.no_cuda:
            pc_torch = pc_torch.cuda()
            img_torch = img_torch.cuda()
        
        print(f"\nRunning inference...")
        print(f"  Image shape: {img_torch.shape}")
        print(f"  Point cloud shape: {pc_torch.shape}")
        
        result = model(
            batched_pts=[pc_torch],
            batched_imgs=img_torch,
            batched_calib=[calib_dict],
            mode='test'
        )[0]
    
    # Display results
    lidar_bboxes = result['lidar_bboxes']
    labels = result['labels']
    scores = result['scores']
    
    print(f"\nDetection Results:")
    print(f"  Number of detections: {len(lidar_bboxes)}")
    
    for i, (bbox, label, score) in enumerate(zip(lidar_bboxes, labels, scores)):
        class_name = LABEL2CLASSES[label]
        print(f"  [{i}] {class_name}: score={score:.3f}, bbox={bbox}")
    
    print(f"\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointPillarsFusion Inference')
    
    # Model
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Use CPU instead of GPU')
    
    # Data paths
    parser.add_argument('--pc_path', type=str, required=True,
                        help='Path to point cloud file (.bin)')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to image file (.png)')
    parser.add_argument('--calib_path', type=str, required=True,
                        help='Path to calibration file (.txt)')
    
    # Optional
    parser.add_argument('--summary', action='store_true',
                        help='Print model summary')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only print summary, do not run inference')
    
    args = parser.parse_args()
    main(args)
