import argparse
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np

from pointpillars.utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, iou2d, iou3d_camera, iou_bev
from pointpillars.dataset import Kitti, get_dataloader
from pointpillars.model import PointPillars
from pointpillars.loss import Loss
from torch.utils.tensorboard import SummaryWriter
NARROW_RANGE = True # Whether to use narrow point cloud range for evaluation

def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    """Calculate score thresholds for PR curve evaluation."""
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds


def compute_mAP_3d(det_results, gt_results, CLASSES, min_ious={'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}):
    """
    Compute mAP for 3D bounding boxes.
    Returns: dict with mAP per class and overall mAP
    """
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
        
        # Handle empty arrays - ensure they remain 2D
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
        
        # Calculate score thresholds
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
                match_id, match_score = -1, -1
                for k in range(mm):
                    if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                        match_id = k
                        match_score = scores[k]
                if match_id != -1:
                    assigned[match_id] = True
                    if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                        tp_scores.append(match_score)
        
        total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
        if total_num_valid_gt == 0:
            ap_results[cls] = 0.0
            continue
            
        score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)
        
        # Calculate precision-recall
        tps, fps, fns = [], [], []
        for score_threshold in score_thresholds:
            tp, fn, fp = 0, 0, 0
            for i, id in enumerate(ids):
                cur_eval_ious = ious_3d[i]
                gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                scores = total_scores[i]
                
                nn, mm = cur_eval_ious.shape
                assigned = np.zeros((mm,), dtype=np.bool_)
                for j in range(nn):
                    if gt_ignores[j] == -1:
                        continue
                    match_id, match_iou = -1, -1
                    for k in range(mm):
                        if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
                            if det_ignores[k] == 0 and cur_eval_ious[j, k] > match_iou:
                                match_iou = cur_eval_ious[j, k]
                                match_id = k
                            elif det_ignores[k] == 1 and match_iou == -1:
                                match_id = k
                    
                    if match_id != -1:
                        assigned[match_id] = True
                        if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                            tp += 1
                    else:
                        if gt_ignores[j] == 0:
                            fn += 1
                
                for k in range(mm):
                    if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                        fp += 1
            
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
        
        tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)
        recalls = tps / (tps + fns + 1e-6)
        precisions = tps / (tps + fps + 1e-6)
        
        # Smooth precisions
        for i in range(len(score_thresholds)):
            precisions[i] = np.max(precisions[i:])
        
        # Calculate mAP (11-point interpolation)
        sums_AP = 0
        for i in range(0, len(score_thresholds), 4):
            sums_AP += precisions[i]
        mAP = sums_AP / 11 * 100
        ap_results[cls] = mAP
    
    # Overall mAP
    ap_results['mAP'] = np.mean([ap_results[cls] for cls in CLASSES])
    return ap_results


def evaluate_model(model, val_dataloader, nclasses, no_cuda=False):
    """
    Evaluate model and return mAP metrics.
    """
    CLASSES = Kitti.CLASSES
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    if NARROW_RANGE:
        pcd_limit_range = np.array([0, -10.24, -3, 69.12, 10.24, 1], dtype=np.float32)
    else:
        pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    model.eval()
    format_results = {}
    
    with torch.no_grad():
        for i, data_dict in enumerate(tqdm(val_dataloader, desc='Evaluating')):
            if not no_cuda:
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batch_results = model(batched_pts=batched_pts, 
                                  mode='val',
                                  batched_gt_bboxes=batched_gt_bboxes, 
                                  batched_gt_labels=batched_labels)
            
            for j, result in enumerate(batch_results):
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
                
                calib_info = data_dict['batched_calib_info'][j]
                tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                r0_rect = calib_info['R0_rect'].astype(np.float32)
                P2 = calib_info['P2'].astype(np.float32)
                image_shape = data_dict['batched_img_info'][j]['image_shape']
                idx = data_dict['batched_img_info'][j]['image_idx']
                
                result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
                result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
                
                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                    zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    format_result['name'].append(LABEL2CLASSES[label])
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                    format_result['alpha'].append(alpha)
                    format_result['bbox'].append(bbox2d)
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['rotation_y'].append(camera_bbox[6])
                    format_result['score'].append(score)
                
                format_results[idx] = {k: np.array(v) for k, v in format_result.items()}
    
    # Get GT data
    gt_results = val_dataloader.dataset.data_infos
    
    # Compute mAP
    mAP_results = compute_mAP_3d(format_results, gt_results, CLASSES)
    
    return mAP_results


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


def save_checkpoint(state, filename):
    print(f"\nSaving checkpoint to {filename}...", end='\r', flush=True)
    torch.save(state, filename)
    print("Checkpoint saved successfully!", end='\r', flush=True)


def load_checkpoint(filename, model, optimizer, scheduler, load_scheduler=True):
    """Load checkpoint with backward compatibility for model state dict."""
    print(f"Loading checkpoint from {filename}...")
    checkpoint = torch.load(filename)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # New format: checkpoint is a dict with 'model_state_dict', 'optimizer_state_dict', etc.
        model_state = checkpoint['model_state_dict']
        is_full_checkpoint = True
    else:
        # Old format: checkpoint is directly the model state dict
        model_state = checkpoint
        is_full_checkpoint = False
        print("⚠ Legacy checkpoint format detected (model state dict only)")
    
    # Load model state dict with compatibility
    try:
        model.load_state_dict(model_state, strict=True)
        print("✓ Model state loaded successfully (strict mode)")
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            print(f"⚠ Warning: Model checkpoint incompatibility detected")
            print(f"  Loading with strict=False...")
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)} (randomly initialized)")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)} (ignored)")
            print("✓ Model state loaded with compatibility mode")
        else:
            raise e
    
    # Only load optimizer/scheduler if full checkpoint format
    if is_full_checkpoint:
        if load_scheduler:
            # Load both optimizer and scheduler (normal resume)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Optimizer and scheduler states loaded (continuing with old LR schedule)")
        else:
            # Skip optimizer too when resetting LR schedule (fresh optimizer starts with init_lr)
            print("⚠ Skipping optimizer and scheduler states - using NEW LR schedule from code")
        
        return checkpoint['epoch'], checkpoint['global_step'], checkpoint['train_step']
    else:
        print("⚠ Optimizer and scheduler states not found - starting fresh")
        return 0, 0, 0  # Start from epoch 0 with fresh optimizer/scheduler



def main(args):
    setup_seed()
    train_dataset = Kitti(data_root=args.data_root,
                          split='train')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses)
    loss_func = Loss()

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    
    # Reduced peak LR to prevent overshooting
    # Balanced: 4x multiplier with longer annealing phase
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*4,  # Reduced from 10x
                                                    total_steps=max_iters, 
                                                    pct_start=0.3,  # 30% warmup, 70% decay
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    # Initialize or load checkpoint
    start_epoch = 0
    global_step = 0
    train_step = 0
    if args.resume:
        checkpoint_path = args.resume + '.pth' if not args.resume.endswith('.pth') else args.resume
        if os.path.isfile(checkpoint_path):
            print(f"Found checkpoint at {checkpoint_path}")
            start_epoch, global_step, train_step = load_checkpoint(
                checkpoint_path, pointpillars, optimizer, scheduler, 
                load_scheduler=not args.reset_lr_schedule)
            print(f"Successfully resumed from epoch {start_epoch}, global_step {global_step}, train_step {train_step}")
            args.resume = True  # Set flag to true for the entire training session
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            print(f"Available checkpoints in {os.path.dirname(checkpoint_path)}:")
            if os.path.exists(os.path.dirname(checkpoint_path)):
                checkpoints = [f for f in os.listdir(os.path.dirname(checkpoint_path)) if f.endswith('.pth')]
                for ckpt in checkpoints:
                    print(f"  - {ckpt}")
            args.resume = False
    else:
        args.resume = False

    for epoch in range(start_epoch, args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        if not args.resume or epoch > start_epoch:
            train_step = 0
        val_step = 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                pointpillars(batched_pts=batched_pts, 
                             mode='train',
                             batched_gt_bboxes=batched_gt_bboxes, 
                             batched_gt_labels=batched_labels)
            
            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
            
            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            # Gradient clipping to prevent regression/direction overshooting
            # Reduced from 35 to 10 for more stable training
            torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=10.0)
            
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1
            print(f"\nEpoch {epoch:3d}, Loss: total={loss_dict['total_loss']:.3f}, cls={loss_dict['cls_loss']:.3f}, reg={loss_dict['reg_loss']:.3f}, dir={loss_dict['dir_cls_loss']:.3f}      \n", end='\r', flush=True)

            if global_step % args.log_freq == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'], 
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1

            # Save checkpoint on keyboard interrupt
            try:
                if (epoch + 1) % args.ckpt_freq_epoch == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'train_step': train_step,
                        'model_state_dict': pointpillars.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    save_checkpoint(checkpoint, os.path.join(saved_ckpt_path, f'checkpoint_epoch_{epoch+1}.pth'))
            except KeyboardInterrupt:
                print("\nInterrupt received! Saving checkpoint...")
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'train_step': train_step,
                    'model_state_dict': pointpillars.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                save_checkpoint(checkpoint, os.path.join(saved_ckpt_path, f'checkpoint_interrupted_epoch_{epoch}.pth'))
                print(f"Training was interrupted at epoch {epoch}. To resume training, use:")
                print(f"python train.py --resume {os.path.join(saved_ckpt_path, f'checkpoint_interrupted_epoch_{epoch}.pth')} [other args...]")
                raise KeyboardInterrupt

        if epoch % 2 == 0:
            continue
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_freq_epoch == 0 and args.eval_map:
            print(f'\n{"="*20} Evaluating Epoch {epoch+1} {"="*20}')
            mAP_results = evaluate_model(pointpillars, val_dataloader, args.nclasses, args.no_cuda)
            
            print(f'\nmAP Results (Epoch {epoch+1}):')
            for cls, ap in mAP_results.items():
                print(f'  {cls}: {ap:.2f}%')
                if cls != 'mAP':
                    writer.add_scalar(f'val/AP_{cls}', ap, epoch + 1)
            writer.add_scalar('val/mAP', mAP_results['mAP'], epoch + 1)
            print(f'{"="*60}\n')
        
        # Original validation loss calculation (if not skipped by epoch % 2)
        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_bboxes=batched_gt_bboxes, 
                                batched_gt_labels=batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)
                
                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                    bbox_pred=bbox_pred,
                                    bbox_dir_cls_pred=bbox_dir_cls_pred,
                                    batched_labels=batched_bbox_labels, 
                                    num_cls_pos=num_cls_pos, 
                                    batched_bbox_reg=batched_bbox_reg, 
                                    batched_dir_labels=batched_dir_labels)
                
                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--reset_lr_schedule', action='store_true',
                        help='when resuming, use new LR schedule instead of checkpoint LR')
    parser.add_argument('--eval_freq_epoch', type=int, default=5,
                        help='frequency of epochs to run mAP evaluation (default: 5)')
    parser.add_argument('--eval_map', action='store_true',
                        help='whether to evaluate mAP during training')
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupt received! Saving checkpoint...")
        # Save checkpoint code will be added in the training loop
        print("You can resume training using --resume [checkpoint_path]")
