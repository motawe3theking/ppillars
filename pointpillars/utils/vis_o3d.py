import cv2
import numpy as np
import open3d as o3d
import os
from pointpillars.utils import bbox3d2corners


COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
COLORS_IMG = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]

LINES = [
        [0, 1],
        [1, 2], 
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [2, 6],
        [7, 3],
        [1, 5],
        [4, 0]
    ]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def ply2npy(ply):
    return np.array(ply.points)


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    PAR = os.path.dirname(os.path.abspath(__file__))
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'))
    for ply in plys:
        vis.add_geometry(ply)
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters(os.path.join(PAR, 'viewpoint.json'), param)
    vis.destroy_window()


def create_ruler(bottom_point, top_point, color=[1, 1, 1]):
    '''Create a vertical ruler with tick marks at 0.5m intervals'''
    ruler_objects = []
    
    # Main vertical line
    ruler_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([bottom_point, top_point]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ruler_line.colors = o3d.utility.Vector3dVector([color])
    ruler_objects.append(ruler_line)
    
    # Add tick marks every 0.5m
    height = top_point[2] - bottom_point[2]
    num_ticks = int(height / 0.5) + 1
    
    for i in range(num_ticks):
        tick_z = bottom_point[2] + i * 0.5
        if tick_z > top_point[2]:
            break
        
        # Horizontal tick mark (0.2m wide)
        tick_points = [
            [bottom_point[0] - 0.1, bottom_point[1], tick_z],
            [bottom_point[0] + 0.1, bottom_point[1], tick_z]
        ]
        tick_line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(tick_points),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        tick_line.colors = o3d.utility.Vector3dVector([color])
        ruler_objects.append(tick_line)
    
    return ruler_objects


def vis_pc(pc, bboxes=None, labels=None):
    '''
    pc: ply or np.ndarray (N, 4)
    bboxes: np.ndarray, (n, 7) or (n, 8, 3)
    labels: (n, )
    '''
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10, origin=[0, 0, 0])

    if bboxes is None:
        vis_core([pc, mesh_frame])
        return
    
    if len(bboxes.shape) == 2:
        bboxes_corners = bbox3d2corners(bboxes)
    else:
        bboxes_corners = bboxes
    
    vis_objs = [pc, mesh_frame]
    for i in range(len(bboxes_corners)):
        bbox = bboxes_corners[i]
        if labels is None:
            color = [1, 1, 0]
        else:
            if labels[i] >= 0 and labels[i] < 3:
                color = COLORS[labels[i]]
            else:
                color = COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))
        
        # Add ruler next to the bbox
        # Find bottom center and top center of bbox
        bottom_center = bbox[[0, 1, 2, 3]].mean(axis=0)  # Bottom 4 corners
        top_center = bbox[[4, 5, 6, 7]].mean(axis=0)  # Top 4 corners
        
        # Offset ruler to the side of the bbox
        ruler_offset = 0.5  # 0.5m to the side
        ruler_bottom = bottom_center.copy()
        ruler_bottom[0] += ruler_offset
        ruler_top = top_center.copy()
        ruler_top[0] += ruler_offset
        
        ruler_objs = create_ruler(ruler_bottom, ruler_top, color=color)
        vis_objs.extend(ruler_objs)
    
    vis_core(vis_objs)


def vis_img_3d(img, image_points, labels, camera_bboxes=None, point_heights=None, rt=True):
    '''
    img: (h, w, 3)
    image_points: (n, 8, 2)
    labels: (n, )
    camera_bboxes: (n, 7) [x, y, z, l, h, w, ry] - optional, for displaying bbox height
    point_heights: (n,) - bbox heights from model prediction
    '''
    for i in range(len(image_points)):
        label = labels[i]
        bbox_points = image_points[i] # (8, 2)
        if label >= 0 and label < 3:
            color = COLORS_IMG[label]
        else:
            color = COLORS_IMG[-1]
        for line_id in LINES:
            x1, y1 = bbox_points[line_id[0]]
            x2, y2 = bbox_points[line_id[1]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        
        # Display height from bbox prediction
        height_to_display = None
        
        if point_heights is not None:
            height_to_display = point_heights[i]
        elif camera_bboxes is not None:
            # camera_bboxes format: [x, y, z, l, h, w, ry]
            height_to_display = camera_bboxes[i][4]  # h is at index 4
        
        if height_to_display is not None:
            # Find top center of the box for text placement
            top_points = bbox_points[[0, 1, 2, 3]]  # Top 4 corners
            text_x = int(top_points[:, 0].mean())
            text_y = int(top_points[:, 1].min()) - 5  # Slightly above the box
            
            # Draw text with background for better visibility
            text = f"{height_to_display:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(img, 
                         (text_x - 2, text_y - text_height - 2),
                         (text_x + text_width + 2, text_y + 2),
                         (0, 0, 0), -1)
            # Draw text
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    
    if rt:
        return img
    cv2.imshow('bbox', img)
    cv2.waitKey(0)
