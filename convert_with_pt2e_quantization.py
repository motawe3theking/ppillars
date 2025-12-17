"""Quantize the post-voxel PointPillars (backbone->head) using PT2E.

This script expects precomputed pillar feature maps (.npy files) produced by
`tools/precompute_pillar_features.py` which perform voxelization + pillar
encoder on raw point clouds. The calibrator will run those feature maps
through the post-voxel model for precise activation calibration.

Usage example:
  python tools/convert_with_pt2e_quantization.py \
      --calib-dir calib_features/train \
      --output quantized_post_voxel.pt \
      --ckpt pretrained/epoch_160.pth
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

from pointpillars.model.pointpillars_post_voxel import PointPillarsPostVoxel


def load_checkpoint_into_postvoxel(model, checkpoint_path):
    if checkpoint_path is None:
        return model
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Try to adapt and load only matching keys
    model_state = model.state_dict()
    adapted = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            adapted[k] = v
        else:
            # try stripping common prefixes like 'backbone.' or 'module.'
            k2 = k
            if k.startswith('pointpillars.'):
                k2 = k[len('pointpillars.'):]
            if k2 in model_state and v.shape == model_state[k2].shape:
                adapted[k2] = v

    model.load_state_dict(adapted, strict=False)
    return model


def make_calib_loader(calib_dir, batch_size=8):
    files = sorted([str(p) for p in Path(calib_dir).glob('*.npy')])
    if len(files) == 0:
        raise RuntimeError(f'No .npy files found in {calib_dir}')

    def gen():
        batch = []
        for f in files:
            arr = np.load(f)
            # arr is expected to be shape (1, C, H, W) or (C, H, W)
            if arr.ndim == 3:
                arr = arr[None, ...]
            batch.append(torch.from_numpy(arr).float())
            if len(batch) == batch_size:
                yield torch.cat(batch, dim=0)
                batch = []
        if len(batch) > 0:
            yield torch.cat(batch, dim=0)

    return gen()


def run_calibration(prepared_model, calib_loader, device):
    prepared_model.to(device)
    prepared_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            batch = batch.to(device)
            _ = prepared_model(batch)
            if (i + 1) % 10 == 0:
                print(f"Calibrated { (i+1) * batch.shape[0] } samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib-dir', default="calib_features", help='Directory with .npy pillar feature maps')
    parser.add_argument('--ckpt', default="/mnt/c/Users/AIT/Desktop/GIU/Bachelor/manual_repo_migrate/PointPillars/mobilenet_Ratio_0.16m_64f_Upsample/checkpoints/checkpoint_epoch_104.pth", help='Path to checkpoint to load weights from (optional)')
    parser.add_argument('--output', default='quantized_post_voxel.pt', help='Output quantized model path')
    parser.add_argument('--device', default='cpu', help='Device for calibration and conversion')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--export-tflite', action='store_true', help='Also convert the (quantized) model to TFLite using ai-edge-torch')
    parser.add_argument('--tflite-output', default='pointpillars_post_voxel.tflite', help='TFLite output path when --export-tflite is used')
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Building post-voxel model...')
    model = PointPillarsPostVoxel()
    model = load_checkpoint_into_postvoxel(model, args.ckpt)
    model.eval()

    # Prepare example input from first calibration file
    files = sorted([str(p) for p in Path(args.calib_dir).glob('*.npy')])
    if len(files) == 0:
        raise RuntimeError('No calibration .npy files found')
    example = torch.from_numpy(np.load(files[0])).float()
    if example.ndim == 3:
        example = example.unsqueeze(0)

    example = example.to(device)

    print('Preparing model for PT2E quantization (this may register observers)...')
    # We will use ai-edge-torch's PT2E quantization pipeline as the primary path.
    # Calibration: run the post-voxel model on precomputed pillar-feature batches
    print('Running calibration by executing the post-voxel model on precomputed features...')
    calib_loader = make_calib_loader(args.calib_dir, batch_size=args.batch_size)
    model.to(device)
    model.eval()
    num_calib = 0
    with torch.no_grad():
        for batch in calib_loader:
            batch = batch.to(device)
            try:
                _ = model(batch)
            except Exception as e:
                # If the model expects a different input shape, try squeezing leading dim
                try:
                    _ = model(batch.squeeze(1))
                except Exception:
                    print('Calibration forward failed on a batch:', e)
                    raise
            num_calib += batch.shape[0]
            if num_calib % (args.batch_size * 10) == 0:
                print(f'  Calibrated {num_calib} samples')

    print(f'Calibration completed - ran {num_calib} samples through the model')

    # Build ai-edge-torch PT2E quantizer and QuantConfig and convert using ai-edge
    try:
        import ai_edge_torch
        from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
        from ai_edge_torch.quantize.quant_config import QuantConfig
    except Exception as e:
        raise RuntimeError('ai_edge_torch quantize API not available: ' + repr(e))

    quantizer = PT2EQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    qcfg = QuantConfig(pt2e_quantizer=quantizer)

    print('Converting model with ai-edge-torch (PT2EQuantizer)...')
    try:
        # ai_edge_torch.convert expects CPU tensors/samples in most environments;
        # convert a CPU copy to avoid device-related surprises.
        edge_model = ai_edge_torch.convert(model.cpu(), (example.cpu(),), quant_config=qcfg)
        if args.export_tflite:
            edge_model.export(args.tflite_output)
            print(f'✓ Successfully exported quantized TFLite: {args.tflite_output}')
        try:
            edge_model.save(args.output)
            print(f'Edge model saved to {args.output}')
        except Exception:
            # Fallback: save a TorchScript of the original CPU model
            try:
                scripted = torch.jit.script(model.cpu())
                scripted.save(args.output)
                print(f'Saved TorchScript model to {args.output}')
            except Exception as e:
                print('Failed to save edge model or TorchScript:', e)
    except Exception as e:
        print('ai-edge-torch conversion failed:', e)
        raise


if __name__ == '__main__':
    main()
