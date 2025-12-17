"""
Convert PyTorch PointPillars model to TFLite using Google's ai-edge-torch
This is the modern, official way to convert PyTorch models to TFLite.
"""

import torch
import ai_edge_torch
import sys
import os

# import numpy
# import torchvision

def load_model_from_checkpoint(checkpoint_path):
    """Load the PointPillars model from checkpoint"""
    # Import the model architecture
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pointpillars.model.pointpillars import PointPillars
    
    # Initialize model
    model = PointPillars(nclasses=3, alpha=1.0, nhead_conv=64)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def convert_to_tflite(checkpoint_path, output_path, input_shape=(1, 10000, 64)):
    """
    Convert PyTorch model to TFLite format
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Path to save .tflite file
        input_shape: Input tensor shape (batch, max_pillars, features)
    """
    print(f"Loading model from {checkpoint_path}...")
    model = load_model_from_checkpoint(checkpoint_path)
    
    print(f"Model loaded. Creating sample input with shape {input_shape}...")
    # Create sample input
    sample_input = torch.randn(*input_shape)
    
    print("Converting to TFLite using ai-edge-torch...")
    try:
        # Convert to TFLite
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
        print("\nTrying with quantization...")
        
        try:
            # Try with dynamic range quantization
            edge_model = ai_edge_torch.convert(
                model,
                (sample_input,),
                quant_config=ai_edge_torch.config.QuantConfig(
                    quantize_weight=True
                )
            )
            edge_model.export(output_path)
            print(f"✓ Successfully converted with quantization: {output_path}")
            
        except Exception as e2:
            print(f"✗ Quantized conversion also failed: {e2}")
            raise

if __name__ == "__main__":
    # Paths
    checkpoint_path = "C:\\Users\\AIT\\Desktop\\GIU\\Bachelor\\manual_repo_migrate\\PointPillars\\mobilenet_Ratio_0.16m_64f_Upsample\\checkpoints\\checkpoint_epoch_103_fp32_bn_fused.pth"
    output_path = "C:\\Users\\AIT\\Desktop\\GIU\\Bachelor\\manual_repo_migrate\\PointPillars\\pointpillars_aiedge.tflite"
    
    # Input shape: (batch_size, max_pillars, features_per_pillar)
    # Based on your model: max 10000 pillars, 64 features each
    input_shape = (1, 10000, 64)
    
    print("=" * 60)
    print("PyTorch to TFLite Converter (ai-edge-torch)")
    print("=" * 60)
    
    convert_to_tflite(checkpoint_path, output_path, input_shape)
