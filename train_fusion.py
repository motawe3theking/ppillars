"""
Training script for PointPillarsFusion model.

This script trains the camera-LiDAR fusion version of PointPillars.
It requires RGB images, point clouds, and calibration files from KITTI dataset.
"""

import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import cv2

from pointpillars.utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, iou3d_camera
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillarsFusion
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter


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


def prepare_batch_fusion(data_dict, device, data_root):
    """
    Prepare batch data for fusion model.
    
    Args:
        data_dict: Dictionary from dataloader containing:
            - 'batched_pts': list of point cloud tensors
            - 'batched_gt_bboxes': list of ground truth boxes
            - 'batched_labels': list of ground truth labels
            - 'batched_img_info': list of dicts with 'image_path'
            - 'batched_calib_info': list of dicts with calibration matrices
        device: torch device
        data_root: root directory of KITTI dataset
    
    Returns:
        batched_pts, batched_imgs, batched_calib, batched_gt_bboxes, batched_gt_labels
    """
    batch_size = len(data_dict['batched_pts'])
    
    # Prepare point clouds
    batched_pts = [pts.to(device) for pts in data_dict['batched_pts']]
    
    # Load and prepare images
    batched_imgs = []
    for i in range(batch_size):
        # Get image path from data_dict (relative path)
        img_info = data_dict['batched_img_info'][i]
        img_rel_path = img_info['image_path']
        
        # Construct full path
        img_full_path = os.path.join(data_root, img_rel_path)
        
        # Load image
        img = load_image(img_full_path)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        batched_imgs.append(img_tensor)
    
    batched_imgs = torch.stack(batched_imgs, dim=0).to(device)  # (B, 3, H, W)
    
    # Prepare calibration
    batched_calib = []
    for i in range(batch_size):
        calib_info = data_dict['batched_calib_info'][i]
        calib_dict = {
            'P2': calib_info['P2'].numpy() if isinstance(calib_info['P2'], torch.Tensor) else calib_info['P2'],
            'R0_rect': calib_info['R0_rect'].numpy() if isinstance(calib_info['R0_rect'], torch.Tensor) else calib_info['R0_rect'],
            'Tr_velo_to_cam': calib_info['Tr_velo_to_cam'].numpy() if isinstance(calib_info['Tr_velo_to_cam'], torch.Tensor) else calib_info['Tr_velo_to_cam']
        }
        batched_calib.append(calib_dict)
    
    # Prepare ground truth
    batched_gt_bboxes = [gt.to(device) for gt in data_dict['batched_gt_bboxes']]
    batched_gt_labels = [gt.to(device) for gt in data_dict['batched_labels']]
    
    return batched_pts, batched_imgs, batched_calib, batched_gt_bboxes, batched_gt_labels


def compute_mAP_3d(det_results, gt_results, CLASSES, min_ious={'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}):
    """Compute mAP for 3D bounding boxes."""
    ids = list(sorted(gt_results.keys()))
    
    # Calculate 3D IoUs
    ious_3d = []
    for id in ids:
        gt_result = gt_results[id]['annos']
        det_result = det_results[id]
        
        gt_location = gt_result['location'].astype(np.float32)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32)
        det_dimensions = det_result['dimensions'].astype(np.float32)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)
        
        # Handle empty arrays
        if len(gt_rotation_y) == 0:
            gt_bboxes3d = np.zeros((0, 7), dtype=np.float32)
        else:
            gt_bboxes3d = np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
        
        if len(det_rotation_y) == 0:
            det_bboxes3d = np.zeros((0, 7), dtype=np.float32)
        else:
            det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
        
        iou3d_v = iou3d_camera(torch.from_numpy(gt_bboxes3d).cuda(), torch.from_numpy(det_bboxes3d).cuda())
        ious_3d.append(iou3d_v.cpu().numpy())
    
    MIN_HEIGHT = [40, 25, 25]
    ap_results = {}
    
    for cls in CLASSES:
        CLS_MIN_IOU = min_ious[cls]
        difficulty = 1  # Moderate difficulty
        
        total_gt_ignores, total_det_ignores, total_scores = [], [], []
        
        for id in ids:
            gt_result = gt_results[id]['annos']
            det_result = det_results[id]
            
            # GT filtering
            gt_names = gt_result['name']
            gt_heights = gt_result['bbox'][:, 3] - gt_result['bbox'][:, 1]
            gt_ignores = []
            for j in range(len(gt_names)):
                if gt_names[j] == cls:
                    if gt_heights[j] >= MIN_HEIGHT[difficulty]:
                        gt_ignores.append(0)
                    else:
                        gt_ignores.append(1)
                else:
                    gt_ignores.append(-1)
            total_gt_ignores.append(gt_ignores)
            
            # Det filtering
            det_names = det_result['name']
            det_ignores = []
            for j in range(len(det_names)):
                if det_names[j] == cls:
                    det_ignores.append(0)
                else:
                    det_ignores.append(-1)
            total_det_ignores.append(det_ignores)
            total_scores.append(det_result['score'])
        
        # Calculate AP (simplified version - you may want to use KITTI's official eval)
        tp_scores = []
        for i, id in enumerate(ids):
            cur_eval_ious = ious_3d[i]
            gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
            scores = total_scores[i]
            
            nn, mm = cur_eval_ious.shape
            assigned = np.zeros((mm,), dtype=np.bool_)
            for j in range(nn):
                if gt_ignores[j] == -1:
                    continue
                match_id = -1
                match_iou = CLS_MIN_IOU
                for k in range(mm):
                    if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > match_iou:
                        match_id = k
                        match_iou = cur_eval_ious[j, k]
                if match_id >= 0:
                    assigned[match_id] = True
                    if gt_ignores[j] == 0:
                        tp_scores.append(scores[match_id])
        
        # Calculate total valid GT
        total_num_valid_gt = sum([sum([1 for ignore in ignores if ignore == 0]) 
                                  for ignores in total_gt_ignores])
        
        # Simple AP calculation (11-point interpolation)
        if len(tp_scores) == 0 or total_num_valid_gt == 0:
            ap_results[cls] = 0.0
        else:
            tp_scores = sorted(tp_scores)[::-1]
            precision = []
            recall = []
            for i in range(len(tp_scores)):
                tp = i + 1
                precision.append(tp / (i + 1))
                recall.append(tp / total_num_valid_gt)
            
            # 11-point interpolation
            ap = 0
            for t in np.linspace(0, 1, 11):
                p_interp = 0
                for i in range(len(recall)):
                    if recall[i] >= t:
                        p_interp = max(p_interp, precision[i])
                ap += p_interp / 11
            
            ap_results[cls] = ap * 100  # Convert to percentage
    
    # Calculate overall mAP
    ap_results['mAP'] = np.mean([ap_results[cls] for cls in CLASSES])
    
    return ap_results


def evaluate_model(model, val_dataloader, CLASSES, LABEL2CLASSES, device, data_root):
    """Run evaluation on validation set."""
    model.eval()
    
    format_results = {}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
    
    # Get sorted IDs from dataset
    sorted_ids = val_dataloader.dataset.sorted_ids
    batch_size = val_dataloader.batch_size
    global_idx = 0
    
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(val_dataloader, desc='Evaluating')):
            # Prepare batch
            batched_pts, batched_imgs, batched_calib, _, _ = prepare_batch_fusion(data_dict, device, data_root)
            
            # Forward pass
            results = model(
                batched_pts=batched_pts,
                batched_imgs=batched_imgs,
                batched_calib=batched_calib,
                mode='val'
            )
            
            # Process results
            for batch_idx, result in enumerate(results):
                # Calculate the correct dataset index
                idx = sorted_ids[global_idx + batch_idx]
                
                # Filter results (similar to original)
                lidar_bboxes = result['lidar_bboxes']
                labels = result['labels']
                scores = result['scores']
                
                # Keep only valid detections
                if len(lidar_bboxes) > 0:
                    # Filter by range
                    mask = (lidar_bboxes[:, 0] >= pcd_limit_range[0]) & \
                           (lidar_bboxes[:, 0] <= pcd_limit_range[3]) & \
                           (lidar_bboxes[:, 1] >= pcd_limit_range[1]) & \
                           (lidar_bboxes[:, 1] <= pcd_limit_range[4])
                    
                    lidar_bboxes = lidar_bboxes[mask]
                    labels = labels[mask]
                    scores = scores[mask]
                
                # Format results
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                
                for bbox, label, score in zip(lidar_bboxes, labels, scores):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    format_result['alpha'].append(0.0)  # Simplified
                    format_result['bbox'].append(np.zeros(4))  # Placeholder
                    format_result['dimensions'].append(bbox[3:6])
                    format_result['location'].append(bbox[:3])
                    format_result['rotation_y'].append(bbox[6])
                    format_result['score'].append(score)
                
                format_results[idx] = {k: np.array(v) for k, v in format_result.items()}
            
            # Increment global index counter
            global_idx += len(results)
    
    # Get GT data
    gt_results = val_dataloader.dataset.data_infos
    
    # Compute mAP
    mAP_results = compute_mAP_3d(format_results, gt_results, CLASSES)
    
    return mAP_results


def save_summary(writer, loss_dict, global_step, tag, lr=None):
    """Save training metrics to TensorBoard."""
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)


def save_checkpoint(state, filename):
    """Save training checkpoint."""
    print(f"\nSaving checkpoint to {filename}...")
    torch.save(state, filename)
    print("Checkpoint saved successfully!")


def manage_checkpoints(checkpoint_dir, keep_latest_n=5):
    """
    Keep only the latest N checkpoints to save disk space.
    Deletes older checkpoints based on epoch number.
    """
    import glob
    import re
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= keep_latest_n:
        return
    
    # Extract epoch numbers and sort
    checkpoint_epochs = []
    for ckpt_path in checkpoints:
        match = re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt_path)
        if match:
            epoch = int(match.group(1))
            checkpoint_epochs.append((epoch, ckpt_path))
    
    # Sort by epoch number
    checkpoint_epochs.sort(key=lambda x: x[0])
    
    # Delete oldest checkpoints
    to_delete = checkpoint_epochs[:-keep_latest_n]
    for epoch, ckpt_path in to_delete:
        try:
            os.remove(ckpt_path)
            print(f"Deleted old checkpoint: {os.path.basename(ckpt_path)}")
        except Exception as e:
            print(f"Warning: Could not delete {ckpt_path}: {e}")


def save_checkpoint(state, filename):
    """Save training checkpoint."""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, scheduler):
    """Load checkpoint with backward compatibility."""
    print(f"Loading checkpoint from {filename}...")
    checkpoint = torch.load(filename)
    
    # Load model state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✓ Model state loaded successfully")
    except RuntimeError as e:
        print(f"⚠ Warning: Loading with strict=False...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Model state loaded with compatibility mode")
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['global_step'], checkpoint['train_step']


def main(args):
    setup_seed()
    
    # Setup datasets
    train_dataset = Kitti(data_root=args.data_root, split='train')
    val_dataset = Kitti(data_root=args.data_root, split='val')
    
    train_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # Setup model
    device = torch.device('cuda' if not args.no_cuda else 'cpu')
    model = PointPillarsFusion(nclasses=args.nclasses).to(device)
    
    # Print model summary
    if args.summary:
        model.summary()
    
    # Setup loss and optimizer
    loss_func = Loss()
    
    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=init_lr,
        betas=(0.95, 0.99),
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=init_lr * 10,
        total_steps=max_iters,
        pct_start=0.4,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.95 * 0.895,
        max_momentum=0.95
    )
    
    # Setup directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    train_step = 0
    
    if args.resume_from:
        start_epoch, global_step, train_step = load_checkpoint(
            args.resume_from, model, optimizer, scheduler
        )
    
    # Training loop
    CLASSES = train_dataset.CLASSES
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    
    print(f"\nStarting training...")
    print(f"  Max epochs: {args.max_epoch}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial LR: {init_lr}")
    print(f"  Device: {device}")
    print(f"  Classes: {list(CLASSES.keys())}")
    
    for epoch in range(start_epoch, args.max_epoch):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.max_epoch}")
        print(f"{'='*80}")
        
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Training')
        
        for i, data_dict in enumerate(pbar):
            try:
                # Prepare batch
                batched_pts, batched_imgs, batched_calib, batched_gt_bboxes, batched_gt_labels = \
                    prepare_batch_fusion(data_dict, device, args.data_root)
                
                # Forward pass
                optimizer.zero_grad()
                
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(
                    batched_pts=batched_pts,
                    batched_imgs=batched_imgs,
                    batched_calib=batched_calib,
                    batched_gt_bboxes=batched_gt_bboxes,
                    batched_gt_labels=batched_gt_labels,
                    mode='train'
                )
                
                # Compute loss
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
                
                anchor_target_dict['bbox_cls_target'] = anchor_target_dict['bbox_cls_target'].reshape(-1)
                anchor_target_dict['bbox_reg_target'] = anchor_target_dict['bbox_reg_target'].reshape(-1, 7)
                anchor_target_dict['reg_weights'] = anchor_target_dict['reg_weights'].reshape(-1)
                anchor_target_dict['bbox_dir_cls_target'] = anchor_target_dict['bbox_dir_cls_target'].reshape(-1)
                anchor_target_dict['dir_weights'] = anchor_target_dict['dir_weights'].reshape(-1)
                
                loss_dict = loss_func(
                    bbox_cls_pred=bbox_cls_pred,
                    bbox_pred=bbox_pred,
                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                    anchor_target_dict=anchor_target_dict
                )
                
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
                
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                global_step += 1
                train_step += 1
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'cls': f"{loss_dict['loss_cls'].item():.4f}",
                    'reg': f"{loss_dict['loss_reg'].item():.4f}",
                    'dir': f"{loss_dict['loss_dir'].item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to tensorboard
                if train_step % args.log_freq == 0:
                    save_summary(
                        writer,
                        loss_dict,
                        global_step,
                        tag='train',
                        lr=optimizer.param_groups[0]['lr']
                    )
                
                # Save checkpoint per iteration (if enabled)
                if args.ckpt_freq_iter > 0 and train_step % args.ckpt_freq_iter == 0:
                    ckpt_path = os.path.join(args.ckpt_dir, f'checkpoint_iter_{train_step}.pth')
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'train_step': train_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }, ckpt_path)
                    print(f"\n✓ Saved iteration checkpoint: checkpoint_iter_{train_step}.pth")
            
            except Exception as e:
                print(f"\n⚠ Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save checkpoint
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'train_step': train_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, ckpt_path)
            print(f"\n✓ Saved epoch checkpoint: checkpoint_epoch_{epoch + 1}.pth")
            
            # Manage checkpoints (keep only latest N)
            if args.keep_latest_n > 0:
                manage_checkpoints(args.ckpt_dir, args.keep_latest_n)
        
        # Evaluation
        if args.eval_map and (epoch + 1) % args.eval_freq_epoch == 0:
            print(f"\n{'='*80}")
            print(f"Running validation evaluation...")
            print(f"{'='*80}")
            
            mAP_results = evaluate_model(model, val_dataloader, CLASSES, LABEL2CLASSES, device, args.data_root)
            
            print(f"\nValidation Results (Epoch {epoch + 1}):")
            for cls in CLASSES:
                print(f"  AP_{cls}: {mAP_results[cls]:.2f}%")
            print(f"  mAP: {mAP_results['mAP']:.2f}%")
            
            # Log to tensorboard
            for cls in CLASSES:
                writer.add_scalar(f'val/AP_{cls}', mAP_results[cls], epoch + 1)
            writer.add_scalar('val/mAP', mAP_results['mAP'], epoch + 1)
            
            model.train()
    
    writer.close()
    print("\n✓ Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PointPillarsFusion')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of KITTI dataset')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model
    parser.add_argument('--nclasses', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--summary', action='store_true',
                        help='Print model summary before training')
    
    # Training
    parser.add_argument('--max_epoch', type=int, default=160,
                        help='Maximum number of epochs')
    parser.add_argument('--init_lr', type=float, default=0.0001,
                        help='Initial learning rate')
    
    # Checkpointing
    parser.add_argument('--log_dir', type=str, default='./pillar_logs_fusion',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--ckpt_dir', type=str, default='./pillar_logs_fusion/checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--ckpt_freq_epoch', type=int, default=1,
                        help='Save checkpoint every N epochs (default: 1 = every epoch)')
    parser.add_argument('--ckpt_freq_iter', type=int, default=0,
                        help='Save checkpoint every N iterations (0 = disabled, use with caution)')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log metrics every N steps')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--keep_latest_n', type=int, default=5,
                        help='Keep only the latest N checkpoints to save disk space')
    
    # Evaluation
    parser.add_argument('--eval_map', action='store_true',
                        help='Evaluate mAP during training')
    parser.add_argument('--eval_freq_epoch', type=int, default=5,
                        help='Evaluate every N epochs')
    
    args = parser.parse_args()
    main(args)
