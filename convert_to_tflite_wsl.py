"""
Convert PyTorch PointPillars model to TFLite using Google's ai-edge-torch (WSL2 version)
This is the modern, official way to convert PyTorch models to TFLite.
Works on Linux (including WSL2).
"""

import torch
import ai_edge_torch
import sys
import os

def load_model_from_checkpoint(checkpoint_path):
    """Load the PointPillars post-voxel model from checkpoint"""
    # Import the model architecture
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    from pointpillars.model.pointpillars_post_voxel import PointPillarsPostVoxel
    
    # Initialize model (post-voxel variant - no pillar_layer)
    model = PointPillarsPostVoxel(nclasses=3)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state_dict from common checkpoint wrappers
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Adapt mismatched parameter shapes when possible (crop or pad channels)
    model_state = model.state_dict()
    adapted_state = {}

    for k, v in state_dict.items():
        # Skip pillar_layer and pillar_encoder weights (not in post-voxel model)
        if k.startswith('pillar_layer.') or k.startswith('pillar_encoder.'):
            print(f"Skipping voxelization weight (not needed): {k}")
            continue
            
        if k in model_state:
            m = model_state[k]
            if v.shape == m.shape:
                adapted_state[k] = v
            else:
                # attempt to adapt by cropping or padding along mismatched dimensions
                try:
                    adapted = v
                    if v.dim() == m.dim():
                        for dim in range(v.dim()):
                            if v.shape[dim] != m.shape[dim]:
                                if v.shape[dim] > m.shape[dim]:
                                    # crop the extra channels
                                    slices = [slice(None)] * v.dim()
                                    slices[dim] = slice(0, m.shape[dim])
                                    adapted = adapted[tuple(slices)]
                                else:
                                    # pad with zeros on this dim
                                    pad_shape = list(adapted.shape)
                                    pad_shape[dim] = m.shape[dim] - adapted.shape[dim]
                                    pad = torch.zeros(*pad_shape, dtype=adapted.dtype)
                                    adapted = torch.cat([adapted, pad], dim=dim)
                        if adapted.shape == m.shape:
                            adapted_state[k] = adapted
                            print(f"Adapted checkpoint param '{k}': {v.shape} -> {m.shape}")
                        else:
                            print(f"Could not adapt '{k}': checkpoint {v.shape} vs model {m.shape}; using model param")
                            adapted_state[k] = m
                    else:
                        print(f"Skipping incompatible param '{k}': differing ranks {v.dim()} vs {m.dim()}")
                        adapted_state[k] = m
                except Exception as e:
                    print(f"Error adapting param '{k}': {e}; using model param")
                    adapted_state[k] = m
        else:
            # Unexpected key in checkpoint; skip it
            print(f"Skipping unexpected key in checkpoint: {k}")

    # Fill missing model params with model defaults
    for k, m in model_state.items():
        if k not in adapted_state:
            adapted_state[k] = m

    # Load adapted state dict
    model.load_state_dict(adapted_state, strict=False)
    model.eval()
    
    return model

def convert_to_tflite(checkpoint_path, output_path, input_shape=(1, 64, 496, 432)):
    """
    Convert PyTorch model (post-voxel) to TFLite format
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Path to save .tflite file
        input_shape: Input tensor shape (batch, channels, height, width) for pillar features
                     Default: (1, 64, 496, 432) matches standard PointPillars pillar feature map
    """
    print(f"Loading post-voxel model from {checkpoint_path}...")
    model = load_model_from_checkpoint(checkpoint_path)
    
    print(f"Model loaded. Creating sample pillar_features input with shape {input_shape}...")
    # Create sample input: pre-computed pillar features (batch, C, H, W)
    sample_input = torch.randn(*input_shape, dtype=torch.float32)
    
    print("Converting to TFLite using ai-edge-torch...")
    print(f"  Model type: {type(model)}")
    print(f"  Sample input shape: {sample_input.shape}")
    print(f"  Sample input dtype: {sample_input.dtype}")
    
    # Test forward pass first
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample_input)
            print(f"  Forward pass successful!")
            print(f"  Output types: {[type(o) for o in output]}")
            print(f"  Output shapes: {[o.shape for o in output]}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")
            raise
    
    print("\nStarting ai-edge-torch conversion...")
    try:
        # Convert to TFLite (post-voxel model accepts pillar_features tensor directly)
        edge_model = ai_edge_torch.convert(
            model,
            (sample_input,)
        )
        # Save the TFLite model
        edge_model.export(output_path)
        print(f"✓ Successfully converted to TFLite: {output_path}")
        
        # Print model info
        import os
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Model size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PointPillars PyTorch model to TFLite')
    parser = argparse.ArgumentParser(description='Convert PointPillars PyTorch model (post-voxel) to TFLite')
    parser.add_argument('--checkpoint', type=str, 
                        default='/mnt/c/Users/AIT/Desktop/GIU/Bachelor/manual_repo_migrate/PointPillars/pretrained/epoch_160.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, 
                        default='pointpillars_post_voxel.tflite',
                        help='Output TFLite file path')
    parser.add_argument('--channels', type=int, default=64,
                        help='Pillar feature channels (default: 64)')
    parser.add_argument('--height', type=int, default=496,
                        help='Pillar feature map height (default: 496)')
    parser.add_argument('--width', type=int, default=432,
                        help='Pillar feature map width (default: 432)')
    
    args = parser.parse_args()
    
    # Input shape: (batch_size, channels, height, width) for pillar features
    input_shape = (1, args.channels, args.height, args.width)
    
    print("=" * 70)
    print("PyTorch to TFLite Converter - Post-Voxel Model (ai-edge-torch)")
    print("=" * 70)
    print("NOTE: This exports only the post-voxelization part of PointPillars.")
    print("      Voxelization must be done separately in your deployment pipeline.")
    print("=" * 70)
    
    convert_to_tflite(args.checkpoint, args.output, input_shape)
