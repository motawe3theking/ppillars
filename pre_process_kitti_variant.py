import argparse
import numpy as np
import os
import pickle
import pdb
from tqdm import tqdm

from pointpillars.utils import read_points, write_points, read_calib, read_label, \
    remove_outside_points, get_points_num_in_bbox, points_in_bboxes_v2, bbox_camera2lidar

# Macro to control image processing
IMAGE_PROCESS = False  # Set to False to disable image processing

def get_calib_info(calib):
    tr_velo_to_cam = calib['Tr_velo_to_cam'].astype(np.float32)
    r0_rect = calib['R0_rect'].astype(np.float32)

    calib_info = {
        'Tr_velo_to_cam': tr_velo_to_cam,
        'R0_rect': r0_rect
    }
    return calib_info

def get_image_info(img_path, img_shape=None):
    if IMAGE_PROCESS and img_path is not None:
        if img_shape is None:
            raise ValueError("Image shape must be provided when IMAGE_PROCESS is True")
        image_info = {
            'image_idx': img_path[-10:-4] if img_path else '000000',
            'image_path': img_path,
            'image_shape': img_shape,
        }
        return image_info
    else:
        # Return minimal image info when image processing is disabled
        # Use the idx from the path if available, otherwise use default
        idx = os.path.basename(img_path)[:-4] if img_path else '000000'
        return {
            'image_idx': idx,
            'image_path': None,
            'image_shape': None,
        }

def get_label_info(label_path, classes, calib_info):
    annotations = read_label(label_path)
    annos_name = annotations['name']
    keepind = [(name in classes) for name in annos_name]
    annos_name = annos_name[keepind]
    
    # Process annotations directly in LiDAR coordinates
    dims = annotations['dimensions'][keepind]  # lhw format
    locs = annotations['location'][keepind]  # Use location directly as LiDAR coordinates
    rots = annotations['rotation_y'][keepind]
    
    gt_boxes_lidar = np.concatenate([locs, dims, rots[:, None]], axis=1)
    difficulty = annotations['occluded'][keepind]  # Using occluded as difficulty

    label_info = {
        'name': annos_name,
        'gt_boxes_camera': gt_boxes_lidar,  # Using same boxes since we're not doing camera transforms
        'gt_boxes_lidar': gt_boxes_lidar,
        'dimensions': gt_boxes_lidar[:, 3:6],
        'location': gt_boxes_lidar[:, 0:3],
        'rotation_y': gt_boxes_lidar[:, 6],
        'difficulty': difficulty
    }
    return label_info

def process_single_scene(args, classes, split, idx):
    """Process a single scene from the dataset.
    
    Args:
        args: Command line arguments
        classes: Dictionary of class names and their IDs
        split: Dataset split (train/val/test)
        idx: Scene index as string (e.g., '000000')
    Returns:
        info: Dictionary containing scene information
    """
    # Construct paths for sample_training structure
    velodyne_path = os.path.join('velodyne', f'{idx}.bin')  # Store relative path
    velodyne_reduced_path = os.path.join(args.data_root, 'velodyne_reduced', f'{idx}.bin')  # Full path for saving
    label_path = os.path.join(args.data_root, 'label_2', f'{idx}.txt')
    
    # Handle calibration path
    calib_dir = os.path.join(args.data_root, 'calib')
    calib_path = os.path.join(calib_dir, f'{idx}.txt') if os.path.exists(calib_dir) else None
    
    # Handle image path only if image processing is enabled
    if IMAGE_PROCESS:
        img_dir = os.path.join(args.data_root, 'image_2')
        img_path = os.path.join(img_dir, f'{idx}.png') if os.path.exists(img_dir) else None
        img_shape = (1024, 1920) if img_path and os.path.exists(img_path) else None
    else:
        img_path = None
        img_shape = None
    img_shape = (1024, 1920) if IMAGE_PROCESS else None
    
    try:
        points = read_points(velodyne_path)
    except FileNotFoundError:
        points = np.zeros((0, 4), dtype=np.float32)  # Empty point cloud if no file exists
        print(f"Note: No point cloud file for {idx}, using empty point cloud")
    
    # Handle the case when we don't have calibration files
    if calib_path is not None and os.path.exists(calib_path):
        calib = read_calib(calib_path)
        calib_info = get_calib_info(calib)
        # Process point cloud with calibration
        reduced_lidar_points = remove_outside_points(
            points=points,
            r0_rect=calib['R0_rect'],
            tr_velo_to_cam=calib['Tr_velo_to_cam'],
            P2=calib['P2'],
            image_shape=img_shape if IMAGE_PROCESS else None)
    else:
        # When no calibration is available, just use the points as is
        print(f"No calibration file found at {calib_path}, using raw points")
        reduced_lidar_points = points
        # Create default calibration info
        calib_info = {
            'Tr_velo_to_cam': np.eye(4)[:3],
            'R0_rect': np.eye(4)[:3, :3]
        }

    image_info = get_image_info(img_path, img_shape)
    os.makedirs(os.path.dirname(velodyne_reduced_path), exist_ok=True)
    write_points(reduced_lidar_points, velodyne_reduced_path)
    
    if split != 'test':
        label_info = get_label_info(label_path, classes, calib_info)
        info = {
            'velodyne_path': os.path.join(split, 'velodyne', f'{idx}.bin'),
            'image': image_info,
            'calib': calib_info,
            'annos': label_info
        }
    else:
        info = {
            'velodyne_path': os.path.join(split, 'velodyne', f'{idx}.bin'),
            'image': image_info,
            'calib': calib_info
        }
    
    return info

def create_db_info(args, info_with_db_path, split='train'):
    if not os.path.exists(info_with_db_path):
        print(f"Warning: {info_with_db_path} not found, skipping database creation")
        return
        
    with open(info_with_db_path, 'rb') as f:
        infos = pickle.load(f)
        
    # Initialize database info structure
    db_infos = {'Car': [], 'Pedestrian': [], 'Cyclist': []}
    # Map each scene's objects to their respective classes
    for scene_id, info in infos.items():
        if 'annos' not in info:
            continue
        for i, name in enumerate(info['annos']['name']):
            if name in db_infos:
                db_infos[name].append({
                    'velodyne_path': info['velodyne_path'],
                    'gt_box_lidar': info['annos']['gt_boxes_lidar'][i],
                    'num_points_in_gt': 0,  # We'll update this if we find points
                    'difficulty': info['annos']['difficulty'][i],
                    'unique_id': f"{scene_id}_{i}"
                })
    
    print('Creating database...')
    database_save_path = os.path.join(args.data_root, 'kitti_gt_database')
    os.makedirs(database_save_path, exist_ok=True)
    db_info_save_path = os.path.join(args.data_root, 'kitti_dbinfos_train.pkl')

    all_db_infos = {'Car': [], 'Pedestrian': [], 'Cyclist': []}  # Initialize empty lists for all classes
    for name in list(db_infos.keys()):
        print(f'Process {name} data...')
        for info in tqdm(db_infos[name]):
            try:
                points = read_points(os.path.join(args.data_root, info['velodyne_path']))
            except FileNotFoundError:
                print(f"Warning: No point cloud found at {info['velodyne_path']}, skipping...")
                continue

            gt_box = info['gt_box_lidar']
            points_in_box = points  # Without calibration, we'll use all points as is

            save_point_path = os.path.join(database_save_path, f"{name}_{info['unique_id']}.bin")
            points_in_box.tofile(save_point_path)

            if name not in all_db_infos:
                all_db_infos[name] = []
            info['velodyne_path'] = save_point_path
            all_db_infos[name].append(info)

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def get_available_files(data_root, split='train'):
    """Get list of available file indices from label_2 directory."""
    label_dir = os.path.join(data_root, 'label_2')
    files = sorted(os.listdir(label_dir))
    return [os.path.splitext(f)[0] for f in files]

def create_infos(args, classes, split='train'):
    print(f'Generate info. split = {split}')
    ids = get_available_files(args.data_root, split)
    infos = {}
    for idx in tqdm(ids):
        info = process_single_scene(args, classes, split, idx)
        infos[idx] = info  # Store info with index as key
        if split != 'test':
            gt_names = info['annos']['name']
            num_gt = gt_names.shape[0]
            index = info['image']['image_idx']
            gt_boxes = info['annos']['gt_boxes_lidar']
            difficulty = info['annos']['difficulty']
            for j in range(num_gt):
                if gt_names[j] in classes:
                    if gt_names[j] not in infos:
                        infos[gt_names[j]] = []
                    info_obj = {
                        'velodyne_path': info['velodyne_path'],
                        'image_idx': index,
                        'gt_box_lidar': gt_boxes[j],
                        'name': gt_names[j],
                        'difficulty': difficulty[j],
                        'unique_id': f"{index}_{j}"
                    }
                    infos[gt_names[j]].append(info_obj)
    return infos

def main(args):
    # Convert relative path to absolute path if needed
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(args.data_root)
    
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
    }
    
    # Create training set infos
    train_infos = create_infos(args, CLASSES.keys(), 'train')
    if train_infos:
        train_info_path = os.path.join(args.data_root, 'kitti_infos_train.pkl')
        with open(train_info_path, 'wb') as f:
            pickle.dump(train_infos, f)
        # Create database for data augmentation
        create_db_info(args, train_info_path)
    else:
        print("No training data found. Skipping training info creation.")

    # Create validation set infos
    val_infos = create_infos(args, CLASSES.keys(), 'val')
    if val_infos:
        val_info_path = os.path.join(args.data_root, 'kitti_infos_val.pkl')
        with open(val_info_path, 'wb') as f:
            pickle.dump(val_infos, f)
    else:
        print("No validation data found. Skipping validation info creation.")

    # Create test set infos if available
    test_infos = create_infos(args, CLASSES.keys(), 'test')
    if test_infos:
        test_info_path = os.path.join(args.data_root, 'kitti_infos_test.pkl')
        with open(test_info_path, 'wb') as f:
            pickle.dump(test_infos, f)
    else:
        print("No test data found. Skipping test info creation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process KITTI dataset')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Path to KITTI dataset')
    args = parser.parse_args()
    main(args)