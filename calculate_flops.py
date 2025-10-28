"""
Script to calculate and display FLOPs for PointPillars model.
Usage: python calculate_flops.py [--checkpoint path/to/checkpoint.pth]
"""

import torch
import argparse
import numpy as np
from pointpillars.model.pointpillars import PointPillars

def create_dummy_batch(batch_size=1, num_points=1000):
    """Create a dummy batch of point clouds for FLOPs calculation."""
    batched_pts = []
    for _ in range(batch_size):
        # Create random point cloud: (N, 4) with [x, y, z, intensity]
        # x: [0, 69.12], y: [-10.24, 10.24], z: [-3, 1], intensity: [0, 1]
        pts = torch.rand(num_points, 4)
        pts[:, 0] = pts[:, 0] * 69.12  # x
        pts[:, 1] = pts[:, 1] * 20.48 - 10.24  # y
        pts[:, 2] = pts[:, 2] * 4 - 3  # z
        pts[:, 3] = pts[:, 3]  # intensity
        batched_pts.append(pts)
    return batched_pts


def main():
    parser = argparse.ArgumentParser(description='Calculate FLOPs for PointPillars')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for FLOPs calculation')
    parser.add_argument('--num_points', type=int, default=10000,
                       help='Number of points per sample')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\nCreating PointPillars model...")
    model = PointPillars(nclasses=3)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    
    # Create dummy input
    print(f"\nCreating dummy input (batch_size={args.batch_size}, num_points={args.num_points})...")
    batched_pts = create_dummy_batch(args.batch_size, args.num_points)
    batched_pts = [pts.to(device) for pts in batched_pts]
    
    # Calculate FLOPs
    print("\nCalculating FLOPs (this may take a moment)...\n")
    print("=" * 140)
    summary = model.summary(batched_pts=batched_pts, verbose=True, calculate_flops=True)
    print("=" * 140)
    
    # Save to file
    output_file = 'flops_summary.txt'
    with open(output_file, 'w') as f:
        f.write(summary)
    print(f"\n✓ Summary saved to {output_file}")


if __name__ == '__main__':
    main()
