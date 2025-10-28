def get_label_info(label_path, classes, calib_info):
    if calib_info is None:
        return {'name': np.array([]), 'difficulty': np.array([]), 
                'gt_boxes_camera': np.zeros((0, 7)), 'gt_boxes_lidar': np.zeros((0, 7))}
                
    annotations = read_label(label_path)
    annos_name = annotations['name']
    keepind = [(name in classes) for name in annos_name]
    annos_name = annos_name[keepind]
    
    # Convert annotations to camera coordinates
    dims = annotations['dimensions'][keepind]  # lhw format
    locs = annotations['location'][keepind]
    rots = annotations['rotation_y'][keepind]
    gt_boxes_camera = np.concatenate([locs, dims, rots[:, None]], axis=1)
    difficulty = annotations['occluded'][keepind]  # Using occluded as difficulty
    
    if calib_info is None:
        gt_boxes_lidar = np.copy(gt_boxes_camera)
    else:
        _, _, _, gt_boxes_lidar, _ = points_in_bboxes_v2(
            np.zeros((0, 3), dtype=np.float32),
            calib_info['R0_rect'],
            calib_info['Tr_velo_to_cam'],
            dims,
            locs,
            rots,
            annos_name)
            
    return {
        'name': annos_name,
        'difficulty': difficulty,
        'gt_boxes_camera': gt_boxes_camera,
        'gt_boxes_lidar': gt_boxes_lidar
    }