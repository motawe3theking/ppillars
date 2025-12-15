import argparse
import cv2
import numpy as np
import os
import torch
import pdb

from pointpillars.utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from pointpillars.model import PointPillars
NARROW_RANGE = 'small' # Whether to use narrow point cloud range for evaluation


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 


def main(args):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    
    if NARROW_RANGE == 'small':
        point_cloud_range = [0, -10.24, -3, 69.12, 10.24, 1]
        pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)
    elif NARROW_RANGE == 'mid':
        point_cloud_range = [0, -20.48, -3, 40.96, 20.48, 1]
        pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)
    elif NARROW_RANGE == 'wide':
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        pcd_limit_range = np.array(point_cloud_range, dtype=np.float32)

    if not args.no_cuda:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        checkpoint = torch.load(args.ckpt)
    else:
        model = PointPillars(nclasses=len(CLASSES))
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    
    # Handle both old-style (just model state_dict) and new-style (full checkpoint) formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint format")
    
    # Fuse BatchNorm into Conv for 10-20% speedup (no retraining needed!)
    if not args.no_fuse:
        model.fuse_bn()
    
    if not os.path.exists(args.pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {args.pc_path}")
    
    pc_all = read_points(args.pc_path)
    print(f"Initial point cloud size: {pc_all.shape}")
    if pc_all.shape[0] == 0:
        raise ValueError(f"Empty point cloud loaded from {args.pc_path}. Please check if the file is corrupted or empty.")
    
    pc = point_range_filter(pc_all)
    print(f"Point cloud size after filtering: {pc.shape}")
    if pc.shape[0] == 0:
        raise ValueError(f"No points remained after range filtering. Check if point_range filter is too restrictive or if points are outside the expected range.")
    
    pc_torch = torch.from_numpy(pc)
    if os.path.exists(args.calib_path):
        calib_info = read_calib(args.calib_path)
    else:
        calib_info = None
    
    if os.path.exists(args.gt_path):
        gt_label = read_label(args.gt_path)
    else:
        gt_label = None

    if os.path.exists(args.img_path):
        img = cv2.imread(args.img_path, 1)
    else:
        img = None

    # Optional model summary
    if getattr(args, 'summary', False):
        # Provide a tiny dry-run if possible
        dry_pts = torch.from_numpy(pc[:min(len(pc), 4096)])
        if not args.no_cuda:
            dry_pts = dry_pts.cuda()
        try:
            print(model.summary(batched_pts=[dry_pts], verbose=True))
        except Exception as e:
            print(f"Model summary (without dry-run) due to error: {e}")
            print(model.summary(batched_pts=None, verbose=True))

    model.eval()
    with torch.no_grad():
        try:
            if not args.no_cuda:
                pc_torch = pc_torch.cuda()
                print(f"Point cloud shape: {pc_torch.shape}, dtype: {pc_torch.dtype}, device: {pc_torch.device}")
            
            # Ensure the point cloud has correct dimensions
            if len(pc_torch.shape) != 2 or pc_torch.shape[1] != 4:
                raise ValueError(f"Expected point cloud shape (N, 4), got {pc_torch.shape}")
            
            # Perform warm-up inferences
            print("Performing warm-up inferences...")
            num_warmup = 5
            for i in range(num_warmup):
                _ = model(batched_pts=[pc_torch], mode='test')[0]
                if not args.no_cuda:
                    torch.cuda.synchronize()
                print(f"Warm-up inference {i+1}/{num_warmup} completed")
            
            print("\nStarting timed inference...")
            
            # Enable CUDA error debugging and timing
            if not args.no_cuda:
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
            else:
                start_time = torch.time.time()
            
            result_filter = model(batched_pts=[pc_torch], mode='test')[0]
            
            if not args.no_cuda:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                inference_time = torch.time.time() - start_time
            
            print(f"\nInference time: {inference_time:.3f} seconds ({1/inference_time:.1f} FPS)")
            if not args.no_cuda:
                torch.cuda.synchronize()
        except Exception as e:
            print(f"\nError details:")
            print(f"Point cloud statistics:")
            print(f"- Shape: {pc_torch.shape}")
            print(f"- Device: {pc_torch.device}")
            print(f"- Has NaN: {torch.isnan(pc_torch).any()}")
            print(f"- Has Inf: {torch.isinf(pc_torch).any()}")
            print(f"- Value range: [{pc_torch.min().item()}, {pc_torch.max().item()}]")
            raise e
    if calib_info is not None and img is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        P2 = calib_info['P2'].astype(np.float32)

        image_shape = img.shape[:2]
        result_filter = keep_bbox_from_image_range(result_filter, tr_velo_to_cam, r0_rect, P2, image_shape)

    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)

    if calib_info is not None and img is not None:
        bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes'] 
        point_heights = result_filter.get('point_heights', None)  # Get point-level heights if available
        print(f"[TEST_MID DEBUG] point_heights: {point_heights[:3] if point_heights is not None and len(point_heights) > 3 else point_heights}")
        print(f"[TEST_MID DEBUG] camera_bboxes shape: {camera_bboxes.shape}")
        bboxes_corners = bbox3d2corners_camera(camera_bboxes)
        image_points = points_camera2image(bboxes_corners, P2)
        img = vis_img_3d(img, image_points, labels, camera_bboxes=camera_bboxes, point_heights=point_heights, rt=True)

    if calib_info is not None and gt_label is not None:
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)

        dimensions = gt_label['dimensions']
        location = gt_label['location']
        rotation_y = gt_label['rotation_y']
        gt_labels = np.array([CLASSES.get(item, -1) for item in gt_label['name']])
        sel = gt_labels != -1
        gt_labels = gt_labels[sel]
        bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1)
        gt_lidar_bboxes = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)
        bboxes_camera = bboxes_camera[sel]
        gt_lidar_bboxes = gt_lidar_bboxes[sel]

        gt_labels = [-1] * len(gt_label['name']) # to distinguish between the ground truth and the predictions
        
        pred_gt_lidar_bboxes = np.concatenate([lidar_bboxes, gt_lidar_bboxes], axis=0)
        pred_gt_labels = np.concatenate([labels, gt_labels])
        vis_pc(pc, pred_gt_lidar_bboxes, labels=pred_gt_labels)

        if img is not None:
            bboxes_corners = bbox3d2corners_camera(bboxes_camera)
            image_points = points_camera2image(bboxes_corners, P2)
            gt_labels = [-1] * len(gt_label['name'])
            img = vis_img_3d(img, image_points, gt_labels, camera_bboxes=bboxes_camera, rt=True)
    
    if calib_info is not None and img is not None:
        cv2.imshow(f'{os.path.basename(args.img_path)}-3d bbox', img)
        cv2.waitKey(0)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--ckpt', default='pretrained/epoch_160.pth', help='your checkpoint for kitti')
    parser.add_argument('--pc_path', help='your point cloud path')
    parser.add_argument('--calib_path', default='', help='your calib file path')
    parser.add_argument('--gt_path', default='', help='your ground truth path')
    parser.add_argument('--img_path', default='', help='your image path')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--no_fuse', action='store_true',
                        help='disable BatchNorm fusion (slower but keeps original layers)')
    parser.add_argument('--summary', action='store_true',
                        help='print model summary and exit')
    args = parser.parse_args()

    main(args)
