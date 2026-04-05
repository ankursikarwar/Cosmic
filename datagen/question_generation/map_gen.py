import copy
import os
import argparse
import random
import re
import json
import numpy as np
import itertools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_category, allowed_categories
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import hsv_to_rgb
import colorsys
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import matplotlib.cm as cm

from shapely.geometry import Polygon
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from consistent_color_mapping import get_color_for_category, CATEGORY_COLORS
from matplotlib.colors import to_rgba

pastel_colors = [
    "#C2185B", "#7B1FA2", "#512DA8", "#303F9F",
    "#1976D2", "#0097A7", "#00796B", "#388E3C",
    "#FBC02D", "#F57C00", "#5D4037", "#455A64", "#616161",
    "#AFB42B", "#E64A19", "#9E9E9E", "#212121"
]
pastel_cmap = ListedColormap(pastel_colors, name="pastel20")

def convert_color_to_rgb_255(color):
    """
    Convert a color to RGB tuple in 0-255 range.
    color can be RGB tuple (0-1 range), tuple/list, or numpy array.
    Returns [R, G, B] as integers in 0-255 range.
    """
    if isinstance(color, (tuple, list, np.ndarray)):
        rgb = tuple(color[:3])  # Take first 3 values, ignore alpha if present
        # If values are > 1, assume they're already in 0-255 range
        if max(rgb) > 1.0:
            return [int(round(c)) for c in rgb]
        else:
            # Convert from 0-1 range to 0-255 range
            return [int(round(c * 255)) for c in rgb]
    else:
        # Try to convert using matplotlib
        from matplotlib.colors import to_rgb
        rgb = to_rgb(color)
        return [int(round(c * 255)) for c in rgb]

# Extend allowed_categories to include Table Dining and Chair
allowed_categories = list(allowed_categories) + ["Table Dining", "Chair"]
random.seed(42)
MAP_NORM_DIM = 256
# Minimum distance (in pixels) that counting distractor objects must be from other objects
# Set to None to disable minimum distance checking, or a positive number (e.g., 5.0) to enforce spacing
COUNTING_DISTRACTOR_MIN_DISTANCE = 0.5  # Adjust this value as needed
# Minimum distance (in pixels) that swapped objects (type 2 and type 3) must be from other objects
# Set to None to disable minimum distance checking, or a positive number (e.g., 0.5) to enforce spacing
SWAP_DISTRACTOR_MIN_DISTANCE = 0.5  # Adjust this value as needed


def get_2d(coords):
    coords = np.array(coords)[:,[0,1]][::2]
    x_min, x_max, y_min, y_max = np.min(coords[:,0]), np.max(coords[:,0]), np.min(coords[:,1]), np.max(coords[:,1])
    return x_min, x_max, y_min, y_max

def rotate_point(cx, cy, width, height, rotation_angle = 0):
    if rotation_angle == 90:
        new_cx = cy
        new_cy = 1 - cx
        new_width = height
        new_height = width
    elif rotation_angle == 180:
        new_cx = 1 - cx
        new_cy = 1 - cy
        new_width = width
        new_height = height
    elif rotation_angle == 270:
        new_cx = 1 - cy
        new_cy = cx
        new_width = height
        new_height = width
    else:
        new_cx = cx
        new_cy = cy
        new_width = width
        new_height = height
    return new_cx, new_cy, new_width, new_height

def get_camera_2d_angle(T_matrix):
    T = np.array(T_matrix)
    R_w2c = T[0:3, 0:3]
    view_dir_world = -R_w2c[2, :]
    x = view_dir_world[0]
    y = view_dir_world[1]
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def convert_angle(input_angle):
    input_angle = 90 * round(input_angle / 90)
    return (input_angle + 270) % 360

    print(f"Identified {len(objects_on_top)} objects that are on top of others: {objects_on_top}")
    return {top: bottom for top, bottom in zip(objects_on_top, [list(all_objs.keys())[0]]*len(objects_on_top))} # Placeholder, need real base object

def filter_objects_on_top(ground_json, on_height_thresh=0.025, overlap_thresh=0.25):
    """
    Identify objects that are positioned on top of other objects.
    
    Returns:
        dict: A dictionary mapping top_object -> bottom_object
    """
    all_objs = {}
    for cam in ground_json:
        for obj, data in ground_json[cam].items():
            if "bbox_3d_corners" in data:
                all_objs[obj] = np.array(data["bbox_3d_corners"])

    objects_on_top = {}

    for obj, obj_coords in all_objs.items():
        obj_min_z = obj_coords[:, 2].min()
        obj_bottom_face = obj_coords[np.isclose(obj_coords[:, 2], obj_min_z, atol=1e-3)][:, :2]
        if len(obj_bottom_face) < 3:
            continue
        # Exclude wall objects from being "on top" of things
        if get_category(obj) in ["Window", "Door", "Wall Art"]:
            continue

        obj_poly = Polygon(obj_bottom_face[ConvexHull(obj_bottom_face).vertices])

        for neighbor, neigh_coords in all_objs.items():
            if neighbor == obj:
                continue

            neigh_max_z = neigh_coords[:, 2].max()
            neigh_top_face = neigh_coords[np.isclose(neigh_coords[:, 2], neigh_max_z, atol=1e-3)][:, :2]
            if len(neigh_top_face) < 3:
                continue
            neigh_poly = Polygon(neigh_top_face[ConvexHull(neigh_top_face).vertices])

            # Compute planar overlap
            overlap_ratio = 0
            if obj_poly.is_valid and neigh_poly.is_valid:
                intersection = obj_poly.intersection(neigh_poly)
                smaller = min(obj_poly.area, neigh_poly.area)
                overlap_ratio = intersection.area / smaller if smaller > 0 else 0

            # Check if obj is slightly above neighbor
            vertical_gap = obj_min_z - neigh_max_z
            is_on_top = -0.06 <= vertical_gap <= on_height_thresh and overlap_ratio >= overlap_thresh

            if is_on_top:
                objects_on_top[obj] = neighbor
                break

    print(f"Identified {len(objects_on_top)} objects that are on top of others: {list(objects_on_top.keys())}")
    return objects_on_top

def get_all_coordinates(ground_json, angle, objects_on_top=None):
    """
    Calculate normalized coordinates for all objects in the scene.
    
    Args:
        ground_json: Dictionary containing object data for all cameras
        angle: Rotation angle (0, 90, 180, 270)
        objects_on_top: Optional set of object names that are on top of other objects.
                       These objects will have their boxes shrunk by 80%.
    
    Returns:
        Tuple of (all_coordinates, map_width_pixels, map_height_pixels)
    """
    if objects_on_top is None:
        objects_on_top = {}
    
    all_xmin, all_xmax, all_ymin, all_ymax = [], [], [], []
    for cam in ground_json:
        for obj in ground_json[cam]:
            # EXCLUDE wall objects from normalization bounds to tighten the room around furniture
            if get_category(obj) in ["Window", "Door", "Wall Art"]:
                continue

            if "bbox_3d_corners" in ground_json[cam][obj]:
                x_min, x_max, y_min, y_max = get_2d(ground_json[cam][obj]["bbox_3d_corners"])
                all_xmin.append(x_min)
                all_xmax.append(x_max)
                all_ymin.append(y_min)
                all_ymax.append(y_max)

    if not all_xmin:
        print("Error: No ground truth objects found for normalization!")
        return {}, MAP_NORM_DIM, MAP_NORM_DIM

    global_xmin, global_xmax, global_ymin, global_ymax = min(all_xmin), max(all_xmax), min(all_ymin), max(all_ymax)
    
    # Calculate room dimensions
    room_width = global_xmax - global_xmin
    room_height = global_ymax - global_ymin
    
    print(f"Normalization bounds (from ALL ground truth objects):")
    print(f"  X: [{global_xmin:.3f}, {global_xmax:.3f}], Y: [{global_ymin:.3f}, {global_ymax:.3f}]")
    print(f"  Room dimensions: {room_width:.3f} x {room_height:.3f}")
    
    # Use MAP_NORM_DIM as the size for the LARGER dimension to preserve aspect ratio
    max_room_dim = max(room_width, room_height)
    scale_factor = MAP_NORM_DIM / max_room_dim
    
    # Calculate pixel dimensions preserving aspect ratio
    map_width_pixels = room_width * scale_factor
    map_height_pixels = room_height * scale_factor
    
    print(f"  Map pixel dimensions: {map_width_pixels:.1f} x {map_height_pixels:.1f} (aspect ratio preserved)")

    all_coordinates = {}
    processed_objects = set()
    for cam in ground_json:
        for obj in ground_json[cam]:
            if obj in processed_objects:
                continue
            # Only include objects in allowed_categories
            if get_category(obj) not in allowed_categories:
                continue

            x_min, x_max, y_min, y_max = get_2d(ground_json[cam][obj]["bbox_3d_corners"])
            # Extract z-coordinate for z-ordering (use max_z so objects on top are drawn last)
            bbox_3d = np.array(ground_json[cam][obj]["bbox_3d_corners"])
            max_z = bbox_3d[:, 2].max()  # Maximum z-coordinate (top of object)
            
            # Normalize coordinates using the scale factor (preserves aspect ratio)
            x_min_norm = (x_min - global_xmin) * scale_factor
            x_max_norm = (x_max - global_xmin) * scale_factor
            y_min_orig = (y_min - global_ymin) * scale_factor
            y_max_orig = (y_max - global_ymin) * scale_factor

            # Convert to normalized coordinates [0, 1] for rotation
            cx = (x_min_norm + x_max_norm) / 2 / map_width_pixels
            cy = (y_min_orig + y_max_orig) / 2 / map_height_pixels
            width = (x_max_norm - x_min_norm) / map_width_pixels
            height = (y_max_orig - y_min_orig) / map_height_pixels

            new_cx, new_cy, new_width, new_height = rotate_point(cx, cy, width, height, angle)
            new_cy = 1.0 - new_cy
            
            # Determine map dimensions after rotation
            if angle in [90, 270]:
                final_map_width = map_height_pixels
                final_map_height = map_width_pixels
            else:
                final_map_width = map_width_pixels
                final_map_height = map_height_pixels

            # Convert back to pixel coordinates
            x_min_norm = np.round(final_map_width * (new_cx - new_width / 2))
            x_max_norm = np.round(final_map_width * (new_cx + new_width / 2))
            y_min_norm = np.round(final_map_height * (new_cy - new_height / 2))
            y_max_norm = np.round(final_map_height * (new_cy + new_height / 2))

            width = x_max_norm - x_min_norm
            height = y_max_norm - y_min_norm
            
            # Shrink box by 80% if object is on top of another object AND is a Plant Container
            # ONLY if it occupies more than a threshold of area on the object over which it is kept
            if obj in objects_on_top and "Plant Container" in obj:
                base_obj = objects_on_top[obj]
                base_coords_3d = None
                for c in ground_json:
                    if base_obj in ground_json[c]:
                        base_coords_3d = ground_json[c][base_obj]["bbox_3d_corners"]
                        break
                
                if base_coords_3d:
                    p_xmin, p_xmax, p_ymin, p_ymax = get_2d(ground_json[cam][obj]["bbox_3d_corners"])
                    b_xmin, b_xmax, b_ymin, b_ymax = get_2d(base_coords_3d)
                    plant_area = (p_xmax - p_xmin) * (p_ymax - p_ymin)
                    base_area = (b_xmax - b_xmin) * (b_ymax - b_ymin)
                    
                    area_ratio = plant_area / base_area if base_area > 0 else 0
                    AREA_THRESHOLD = 0.5
                    
                    if area_ratio > AREA_THRESHOLD:
                        # Calculate center point
                        center_x = x_min_norm + width / 2
                        center_y = y_min_norm + height / 2
                        
                        # Shrink width and height (using the existing factors in the code)
                        new_width = width * 0.6
                        new_height = height * 0.6
                        
                        # Recalculate position to keep center the same
                        x_min_norm = center_x - new_width / 2
                        y_min_norm = center_y - new_height / 2
                        width = new_width
                        height = new_height
                    else:
                        # Calculate center point
                        center_x = x_min_norm + width / 2
                        center_y = y_min_norm + height / 2
                        
                        # Shrink width and height (using the existing factors in the code)
                        new_width = width * 0.8
                        new_height = height * 0.8
                        
                        # Recalculate position to keep center the same
                        x_min_norm = center_x - new_width / 2
                        y_min_norm = center_y - new_height / 2
                        width = new_width
                        height = new_height

            # NEW: Standardize thickness and snap to boundary for Windows, Doors, Wall Arts
            obj_category = get_category(obj)
            if obj_category in ["Window", "Door", "Wall Art"]:
                # Determine orientation (Vertical or Horizontal)
                # Vertical: attached to Left or Right wall (aligns with Y axis) - Height > Width usually
                # Horizontal: attached to Top or Bottom wall (aligns with X axis) - Width > Height usually
                is_vertical = height > width
                
                # Constant thickness in pixels
                WALL_OBJ_THICKNESS = 8.0
                
                if is_vertical: # Left or Right wall
                    # Fix width (thickness)
                    width = WALL_OBJ_THICKNESS
                    
                    # Snap to nearest vertical boundary (Left=0 or Right=final_map_width)
                    dist_to_left = x_min_norm
                    dist_to_right = abs(final_map_width - (x_min_norm + width))
                    
                    if dist_to_left < dist_to_right: # Snap to Left (Outside)
                        x_min_norm = -width
                    else: # Snap to Right (Outside)
                        x_min_norm = final_map_width
                        
                else: # Top or Bottom wall
                    # Fix height (thickness)
                    height = WALL_OBJ_THICKNESS
                    
                    # Snap to nearest horizontal boundary (Top=0 or Bottom=final_map_height)
                    dist_to_top = y_min_norm
                    dist_to_bottom = abs(final_map_height - (y_min_norm + height))
                    
                    if dist_to_top < dist_to_bottom: # Snap to Top (Outside)
                        y_min_norm = -height
                    else: # Snap to Bottom (Outside)
                        y_min_norm = final_map_height
                
                # Update max coordinates based on new min and dimensions
                x_max_norm = x_min_norm + width
                y_max_norm = y_min_norm + height
                

            
            description_text = get_category(obj)
            all_coordinates[obj] = (x_min_norm, y_min_norm, width, height, description_text, max_z)
            processed_objects.add(obj)

    print(f"Calculated coordinates for {len(all_coordinates)} ground truth objects")
    
    # Return map dimensions after rotation
    # Swap dimensions if rotated 90 or 270 degrees
    if abs(angle) in [90, 270]:
        return all_coordinates, map_height_pixels, map_width_pixels
    else:
        return all_coordinates, map_width_pixels, map_height_pixels

# def plot(all_coordinates, ax, color_map, boundary, title=""):
#     all_objects = list(all_coordinates.keys())
#     all_descriptions = {all_coordinates[o][-1] for o in all_objects}
#     cmap = pastel_cmap
#     description_list = sorted(list(all_descriptions))
#     for idx, desc in enumerate(description_list):
#         color_map[desc] = cmap(idx / (len(description_list) - 1))


#     # Draw objects
#     for obj in all_objects:
#         x_min_norm, y_min_norm, width, height, description = all_coordinates[obj]
#         if description == "Door":
#             print(obj)
#             print(x_min_norm, y_min_norm, width, height)
#             print(boundary['left'], boundary['top'], boundary['right'], boundary['bottom'])

#         color = color_map[description]
#         rect = Rectangle((x_min_norm, y_min_norm), width, height,
#                          facecolor=color, linewidth=2, alpha=0.9)
#         ax.add_patch(rect)

#     rect = Rectangle(
#         (boundary['left'], boundary['top']),
#         boundary['right'] - boundary['left'],
#         boundary['bottom'] - boundary['top'],
#         fill=False, color='black', linewidth=1.5
#     )
#     ax.add_patch(rect)

#     ax.set_xlim(-20, 276)
#     ax.set_ylim(-20, 276)
#     ax.set_aspect('equal')
#     ax.set_xticks(np.arange(0, 257, 20))
#     ax.set_yticks(np.arange(0, 257, 20))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.axis('off')
#     ax.invert_yaxis()

#     ax.set_title(title, fontsize=20, fontweight='bold', pad=10)

def plot(all_coordinates, ax, color_map, boundary, title=""):
    all_objects = list(all_coordinates.keys())
    all_descriptions = {all_coordinates[o][4] for o in all_coordinates}  # Description is at index 4
    
    # Use consistent colors from the predefined mapping
    for desc in all_descriptions:
        hex_color = get_color_for_category(desc)
        color_map[desc] = to_rgba(hex_color)

    displaced_coords = {}

    # --- STEP 1: Move windows and wall art outside their corresponding wall ---
    for obj in all_objects:
        x, y, w, h, desc, max_z = all_coordinates[obj]
        x2, y2 = x + w, y + h

        # Remove debug prints
        
        if desc == "Window" or desc == "Wall Art" or desc == "Door":
            vertical = h > w
            horizontal = not vertical

            dist_top = abs(y - boundary['top'])
            dist_bottom = abs(boundary['bottom'] - y2)
            dist_left = abs(x - boundary['left'])
            dist_right = abs(boundary['right'] - x2)

            # Determine which wall the object belongs to
            if vertical:
                wall = 'left' if dist_left <= dist_right else 'right'
            else:
                wall = 'top' if dist_top <= dist_bottom else 'bottom'


            # Use offset to push windows and wall art outside the boundary
            # Wall art has zero thickness so needs offset, windows should touch boundary
            # UPDATE: Removed offset for Wall Art to standardize behavior
            offset = 0
            



            if wall == 'left':
                x_new = boundary['left'] - w - offset
                y_new = y
            elif wall == 'right':
                x_new = boundary['right'] + offset
                y_new = y
            elif wall == 'top':
                x_new = x
                y_new = boundary['top'] - h - offset
            elif wall == 'bottom':
                x_new = x
                y_new = boundary['bottom'] + offset
            
            displaced_coords[obj] = (x_new, y_new, w, h, desc, max_z)
        else:
            displaced_coords[obj] = (x, y, w, h, desc, max_z)

    # --- STEP 2: Global shift if any negative coordinates ---
    # Consider all objects for the shift to ensure visibility on canvas
    print("displaced_coords", displaced_coords)
    inside_coords = [(x, y) for (x, y, w, h, desc, max_z) in displaced_coords.values()]
    # if you want to exclude some specific meta-objects, do it here, but Windows/Doors/Wall Art should be included if they shouldn't be cropped
    
    if inside_coords:
        min_x = min(x for (x, y) in inside_coords)
        min_y = min(y for (x, y) in inside_coords)
    else:
        # Fallback if no inside objects
        min_x = min(x for (x, _, _, _, _, _) in displaced_coords.values())
        min_y = min(y for (_, y, _, _, _, _) in displaced_coords.values())

    print("min_x, min_y", min_x, min_y)
    shift_x = abs(min(0, min_x)) if min_x < 0 else 0
    shift_y = abs(min(0, min_y)) if min_y < 0 else 0

    print("shift_x, shift_y", shift_x, shift_y)

    if shift_x or shift_y:
        print(f"Global shift applied: ({shift_x}, {shift_y})")
        boundary = {
            'left': boundary['left'] + shift_x,
            'right': boundary['right'] + shift_x,
            'top': boundary['top'] + shift_y,
            'bottom': boundary['bottom'] + shift_y
        }

        for obj, (x, y, w, h, desc, max_z) in displaced_coords.items():
            displaced_coords[obj] = (x + shift_x, y + shift_y, w, h, desc, max_z)

    # --- NEW STEP: Expand boundary LENGTHWISE only for protruding windows/doors ---
    # This ensures that windows/doors extending past the room corners are included in the map view,
    # but their "thickness" protrusion (out of the wall) does NOT expand the boundary,
    # keeping them attached to the "outside" of the room.
    
    # Calculate new boundary limits based on lengthwise extent of wall objects
    new_top = boundary['top']
    new_bottom = boundary['bottom']
    new_left = boundary['left']
    new_right = boundary['right']

    for obj, (x, y, w, h, desc, z) in displaced_coords.items():
        if desc in ["Window", "Door", "Wall Art"]:
            vertical = h > w
            
            if vertical: # Attached to Left or Right wall
                # Check Y limits (Lengthwise) - Extend Top/Bottom if needed
                new_top = min(new_top, y)
                new_bottom = max(new_bottom, y + h)
                # Do NOT check X limits (Thickness)
            else: # Attached to Top or Bottom wall
                # Check X limits (Lengthwise) - Extend Left/Right if needed
                new_left = min(new_left, x)
                new_right = max(new_right, x + w)
                # Do NOT check Y limits (Thickness)

    # Apply the lengthwise expansions
    boundary['top'] = new_top
    boundary['bottom'] = new_bottom
    boundary['left'] = new_left
    boundary['right'] = new_right

    # --- STEP 3: Compute new bounds for plotting ---
    all_x2 = [x + w for (x, _, w, _, _, _) in displaced_coords.values()]
    all_y2 = [y + h for (_, y, _, h, _, _) in displaced_coords.values()]

    plot_xmin = min(boundary['left'], min(x for (x, _, _, _, _, _) in displaced_coords.values()))
    plot_ymin = min(boundary['top'], min(y for (_, y, _, _, _, _) in displaced_coords.values()))
    plot_xmax = max(boundary['right'], max(all_x2))
    plot_ymax = max(boundary['bottom'], max(all_y2))

    # --- STEP 4: Sort objects by z-coordinate (lower z first, higher z last) ---
    # Objects with higher z-coordinates should be drawn last so they appear on top
    sorted_objects = sorted(displaced_coords.items(), key=lambda item: item[1][5])  # Sort by max_z (6th element)

    # --- STEP 5: Draw all objects in z-order ---
    for obj, (x, y, w, h, desc, max_z) in sorted_objects:
        if desc == "Wall Art" or desc == "Window":
            # Make wall objects thicker for better visibility
            # Logic: Inflate thickness (w for vertical, h for horizontal)
            # And shift position if on Left/Top wall to grow OUTWARD instead of inward
            min_thickness = 5
            
            if w < h:
                # Vertical object (Left/Right Wall)
                old_w = w
                w = max(w * 1, min_thickness)
                
                # Check if on Left Wall (near boundary['left'])
                if abs(x - boundary['left']) < 20: 
                    x -= (w - old_w) # Shift left to grow outward
            else:
                # Horizontal object (Top/Bottom Wall)
                old_h = h
                h = max(h * 1, min_thickness)
                
                # Check if on Top Wall (near boundary['top'])
                if abs(y - boundary['top']) < 20:
                    y -= (h - old_h) # Shift up to grow outward

            # Update plot limits to include this expanded object
            plot_xmin = min(plot_xmin, x)
            plot_ymin = min(plot_ymin, y)
            plot_xmax = max(plot_xmax, x + w)
            plot_ymax = max(plot_ymax, y + h)
        
        color = color_map[desc]
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)

    # --- STEP 6: Draw the boundary box ---
    rect = Rectangle(
        (boundary['left'], boundary['top']),
        boundary['right'] - boundary['left'],
        boundary['bottom'] - boundary['top'],
        fill=False, color='black', linewidth=1.5
    )
    ax.add_patch(rect)

    # --- STEP 7: Dynamically expand plot limits ---
    margin = 10  # add some breathing space
    ax.set_xlim(plot_xmin - margin, plot_xmax + margin)
    ax.set_ylim(plot_ymin - margin, plot_ymax + margin)

    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 257, 20))
    ax.set_yticks(np.arange(0, 257, 20))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=20, fontweight='bold', pad=10)

from matplotlib.patches import Polygon as MPLPolygon, Circle
import matplotlib.patches as mpatches

def calculate_camera_fov(camera_data, floor_z=0.0, max_distance=50.0):
    K = np.array(camera_data['K'])
    T = np.array(camera_data['T'])
    HW = camera_data['HW']
    height, width = HW[0], HW[1]
    T_inv = np.linalg.inv(T)
    camera_pos_3d = T_inv[:3, 3]
    camera_pos = (camera_pos_3d[0], camera_pos_3d[1])
    R = T_inv[:3, :3]

    corners_pixel = np.array([
        [0, 0, 1],           # top-left
        [width, 0, 1],       # top-right
        [width, height, 1],  # bottom-right
        [0, height, 1]       # bottom-left
    ]).T

    K_inv = np.linalg.inv(K)
    corners_cam = K_inv @ corners_pixel
    fov_points = []

    for i in range(corners_cam.shape[1]):
        dir_cam = corners_cam[:, i]
        dir_cam = dir_cam / np.linalg.norm(dir_cam)
        dir_cam = -dir_cam
        dir_world = R @ dir_cam

        if abs(dir_world[2]) > 1e-8:  # Not parallel to floor
            t = (floor_z - camera_pos_3d[2]) / dir_world[2]
            if t > 0:  # In front of camera
                intersection = camera_pos_3d + t * dir_world
                fov_points.append((intersection[0], intersection[1]))
            else:
                # Ray goes upward, extend to max_distance
                intersection = camera_pos_3d + max_distance * dir_world
                fov_points.append((intersection[0], intersection[1]))
        else:
            # Parallel to floor, extend horizontally
            intersection = camera_pos_3d + max_distance * dir_world
            fov_points.append((intersection[0], intersection[1]))

    # Also add center ray for better coverage
    center_pixel = np.array([width/2, height/2, 1])
    center_cam = K_inv @ center_pixel
    center_cam = center_cam / np.linalg.norm(center_cam)
    center_cam = -center_cam
    fov_polygon = []

    if len(fov_points) >= 3:
        # Calculate angles from camera position
        angles = []
        for point in fov_points:
            dx = point[0] - camera_pos[0]
            dy = point[1] - camera_pos[1]
            angle = np.arctan2(dy, dx)
            angles.append((angle, point))

        # Sort by angle
        angles.sort(key=lambda x: x[0])

        # Create polygon: camera position -> sorted points
        fov_polygon = [camera_pos] + [point for _, point in angles]

        # Extend the polygon to ensure full room coverage
        # Calculate distances and extend if needed
        extended_polygon = [camera_pos]
        for point in [p for _, p in angles]:
            dx = point[0] - camera_pos[0]
            dy = point[1] - camera_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Extend to at least max_distance
            if dist < max_distance:
                scale = max_distance / dist
                extended_point = (
                    camera_pos[0] + dx * scale,
                    camera_pos[1] + dy * scale
                )
                extended_polygon.append(extended_point)
            else:
                extended_polygon.append(point)

        fov_polygon = extended_polygon

    return fov_polygon, camera_pos

def is_valid_position(obj_to_move, new_box, all_coordinates_state, plotted_objects, boundary=None, verbose=False, min_distance=None):
    new_x, new_y, new_w, new_h = new_box
    epsilon = 1e-9

    # Use boundary if provided, otherwise use default map dimensions
    if boundary is not None:
        left_bound = boundary['left']
        right_bound = boundary['right']
        top_bound = boundary['top']
        bottom_bound = boundary['bottom']
    else:
        left_bound = 0
        right_bound = MAP_NORM_DIM
        top_bound = 0
        bottom_bound = MAP_NORM_DIM

    # Check if box is within boundary limits
    if new_x < left_bound - epsilon:
        if verbose:
            print(f"      [VALIDATION FAIL] {obj_to_move}: X position {new_x:.2f} is outside left boundary {left_bound:.2f}")
        return False
    if new_y < top_bound - epsilon:
        if verbose:
            print(f"      [VALIDATION FAIL] {obj_to_move}: Y position {new_y:.2f} is outside top boundary {top_bound:.2f}")
        return False
    if (new_x + new_w) > right_bound + epsilon:
        if verbose:
            print(f"      [VALIDATION FAIL] {obj_to_move}: Right edge {new_x + new_w:.2f} exceeds right boundary {right_bound:.2f}")
        return False
    if (new_y + new_h) > bottom_bound + epsilon:
        if verbose:
            print(f"      [VALIDATION FAIL] {obj_to_move}: Bottom edge {new_y + new_h:.2f} exceeds bottom boundary {bottom_bound:.2f}")
        return False

    # Check for overlaps with other objects
    for obj_name in plotted_objects:
        if obj_name == obj_to_move:
            continue
        (x, y, w, h, _, _) = all_coordinates_state[obj_name]
        other_box = (x, y, w, h)
        if boxes_overlap(new_box, other_box):
            if verbose:
                print(f"      [VALIDATION FAIL] {obj_to_move}: Overlaps with existing object {obj_name}")
                print(f"        New box: x={new_x:.2f}, y={new_y:.2f}, w={new_w:.2f}, h={new_h:.2f}")
                print(f"        Existing box: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")
            return False

        # Check minimum distance if specified
        if min_distance is not None and min_distance > 0:
            distance = boxes_min_distance(new_box, other_box)
            if distance < min_distance:
                if verbose:
                    print(f"      [VALIDATION FAIL] {obj_to_move}: Too close to existing object {obj_name}")
                    print(f"        Distance: {distance:.2f}, Required: {min_distance:.2f}")
                    print(f"        New box: x={new_x:.2f}, y={new_y:.2f}, w={new_w:.2f}, h={new_h:.2f}")
                    print(f"        Existing box: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")
                return False

    if verbose:
        print(f"      [VALIDATION PASS] {obj_to_move}: Position is valid")
    return True

def group_by_category(object_set, all_coordinates_map):
    """Group objects by their category."""
    category_groups = {}
    for obj in object_set:
        if obj in all_coordinates_map:
            category = get_category(obj)
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(obj)
    return category_groups

def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box1_x_max, box1_y_max = x1 + w1, y1 + h1
    box2_x_max, box2_y_max = x2 + w2, y2 + h2
    if (box1_x_max <= x2 or x1 >= box2_x_max or
        box1_y_max <= y2 or y1 >= box2_y_max):
        return False
    return True

def boxes_min_distance(box1, box2):
    """
    Calculate the minimum distance between two boxes.
    Returns 0 if boxes overlap, otherwise returns the minimum Euclidean distance
    between any two points on the boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box1_x_max, box1_y_max = x1 + w1, y1 + h1
    box2_x_max, box2_y_max = x2 + w2, y2 + h2

    # If boxes overlap, distance is 0
    if boxes_overlap(box1, box2):
        return 0.0

    # Calculate horizontal and vertical distances
    if box1_x_max <= x2:
        # box1 is to the left of box2
        dx = x2 - box1_x_max
    elif box2_x_max <= x1:
        # box2 is to the left of box1
        dx = x1 - box2_x_max
    else:
        # boxes overlap horizontally
        dx = 0

    if box1_y_max <= y2:
        # box1 is above box2
        dy = y2 - box1_y_max
    elif box2_y_max <= y1:
        # box2 is above box1
        dy = y1 - box2_y_max
    else:
        # boxes overlap vertically
        dy = 0

    # Return Euclidean distance
    return np.sqrt(dx * dx + dy * dy)

def states_are_equal(state1, state2):
    """Check if two states have the same object positions (ignoring description)."""
    if set(state1.keys()) != set(state2.keys()):
        return False
    for obj in state1:
        x1, y1, w1, h1, _, _ = state1[obj]
        x2, y2, w2, h2, _, _ = state2[obj]
        if not (np.isclose(x1, x2) and np.isclose(y1, y2) and
                np.isclose(w1, w2) and np.isclose(h1, h2)):
            return False
    return True

def is_state_unique(new_state, existing_states):
    """Check if a state is unique compared to existing states."""
    for existing_state in existing_states:
        if states_are_equal(new_state, existing_state):
            return False
    return True

def is_box_blocking_door(box, all_coordinates_state, door_buffer=10.0):
    """
    Check if a box (x, y, w, h) is blocking or too close to any door.

    Args:
        box: Tuple (x, y, w, h) in pixel coordinates
        all_coordinates_state: Dictionary of all object coordinates
        door_buffer: Minimum distance (in pixels) that objects must maintain from doors

    Returns:
        True if box is blocking or too close to a door, False otherwise
    """
    x, y, w, h = box
    box_x_max, box_y_max = x + w, y + h

    for obj_name, coords in all_coordinates_state.items():
        obj_x, obj_y, obj_w, obj_h, desc, _ = coords
        if desc.lower() != 'door':
            continue

        obj_x_max, obj_y_max = obj_x + obj_w, obj_y + obj_h

        # Check if boxes overlap (with buffer)
        # Expand door box by buffer
        door_expanded = (
            obj_x - door_buffer,
            obj_y - door_buffer,
            obj_w + 2 * door_buffer,
            obj_h + 2 * door_buffer
        )

        if boxes_overlap(box, door_expanded):
            return True

        # Also check if box center is very close to door center (within buffer distance)
        box_center = (x + w/2, y + h/2)
        door_center = (obj_x + obj_w/2, obj_y + obj_h/2)
        distance = np.sqrt((box_center[0] - door_center[0])**2 + (box_center[1] - door_center[1])**2)

        if distance < door_buffer:
            return True

    return False

def is_box_in_fov(box, fov_polygon_normalized):
    """
    Check if a box (x, y, w, h) intersects with or is inside the FOV polygon.

    Args:
        box: Tuple (x, y, w, h) in normalized pixel coordinates (0-256)
        fov_polygon_normalized: List of (x, y) points forming FOV polygon in normalized pixel coordinates

    Returns:
        True if box intersects or is inside FOV, False otherwise
    """
    if len(fov_polygon_normalized) < 3:
        return False

    try:
        # Create shapely polygon for FOV
        fov_poly = Polygon(fov_polygon_normalized)
        if not fov_poly.is_valid:
            # Try to fix invalid polygon
            fov_poly = fov_poly.buffer(0)
            if not fov_poly.is_valid:
                return False
    except:
        return False

    x, y, w, h = box
    # Get box corners
    box_corners = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]

    try:
        # Create box polygon
        box_poly = Polygon(box_corners)
        if not box_poly.is_valid:
            return False

        # Check if box polygon intersects or is contained in FOV
        if fov_poly.intersects(box_poly) or fov_poly.contains(box_poly):
            return True

        # Check if box center is in FOV (using point-in-polygon)
        from shapely.geometry import Point
        center = Point(x + w/2, y + h/2)
        if fov_poly.contains(center):
            return True
    except:
        return False

    return False

def normalize_fov_to_pixel_space(fov_polygon_world, cam_pos_world, global_xmin, global_xmax, global_ymin, global_ymax, angle):
    """
    Normalize FOV polygon from world coordinates to pixel space (0-256) after rotation.

    Args:
        fov_polygon_world: FOV polygon in world coordinates
        cam_pos_world: Camera position in world coordinates
        global_xmin, global_xmax, global_ymin, global_ymax: Normalization bounds
        angle: Rotation angle (0, 90, 180, 270)

    Returns:
        FOV polygon in normalized pixel space (0-256)
    """
    normalized = []
    for point in fov_polygon_world:
        if point == fov_polygon_world[0]:  # Camera position
            # Normalize camera position
            x_norm = (cam_pos_world[0] - global_xmin) / (global_xmax - global_xmin)
            y_norm = (cam_pos_world[1] - global_ymin) / (global_ymax - global_ymin)
            # Apply rotation
            cx, cy, _, _ = rotate_point(x_norm, y_norm, 0, 0, angle)
            cy = 1.0 - cy
            # Convert to pixel space
            x_pixel = MAP_NORM_DIM * cx
            y_pixel = MAP_NORM_DIM * cy
            normalized.append((x_pixel, y_pixel))
        else:
            # Normalize point
            x_norm = (point[0] - global_xmin) / (global_xmax - global_xmin)
            y_norm = (point[1] - global_ymin) / (global_ymax - global_ymin)
            # Apply rotation
            cx, cy, _, _ = rotate_point(x_norm, y_norm, 0, 0, angle)
            cy = 1.0 - cy
            # Convert to pixel space
            x_pixel = MAP_NORM_DIM * cx
            y_pixel = MAP_NORM_DIM * cy
            normalized.append((x_pixel, y_pixel))
    return normalized

def generate_fov_distractor(obj_to_move, all_coordinates, boundary, fov_polygon_normalized,
                           global_xmin, global_xmax, global_ymin, global_ymax, angle,
                           other_agent_fov_polygon=None,
                           max_tries=100, min_distance=0.5, min_displacement=10.0):
    """
    Generate a distractor by moving an object not in view to a valid position near walls,
    ensuring it's not in the FOV of the current camera but IS in the FOV of the other agent.
    The object is rotated so its longer side is parallel to the nearest wall.

    Args:
        obj_to_move: Object name to move
        all_coordinates: Dictionary of all object coordinates (in pixel space after rotation)
        boundary: Room boundary dictionary
        fov_polygon_normalized: FOV polygon of CURRENT agent in normalized pixel space
        global_xmin, global_xmax, global_ymin, global_ymax: Normalization bounds for FOV calculation
        angle: Rotation angle
        other_agent_fov_polygon: FOV polygon of OTHER agent (new position must be in this FOV)
        max_tries: Maximum number of attempts
        min_distance: Minimum distance from other objects
        min_displacement: Minimum distance the object must be moved from its original position

    Returns:
        New state with moved object, or None if no valid position found
    """
    if obj_to_move not in all_coordinates:
        return None

    x, y, w, h, desc, max_z = all_coordinates[obj_to_move]
    original_box = (x, y, w, h)

    # Never move Wall Art, Window, or Door
    if desc in ["Wall Art", "Window", "Door"]:
        return None
    new_state = copy.deepcopy(all_coordinates)

    # Calculate room dimensions
    room_width = boundary['right'] - boundary['left']
    room_height = boundary['bottom'] - boundary['top']

    # Define wall proximity zones (objects should be within 5 pixels of walls)
    wall_proximity = 5

    sides = ["left", "right", "top", "bottom"]
    temp_state = {k: v for k, v in new_state.items() if k != obj_to_move}
    temp_keys = list(temp_state.keys())

    for attempt in range(max_tries):
        side = random.choice(sides)

        # Determine if we should rotate (longer side parallel to wall)
        # For left/right walls (vertical): we want height (parallel to wall) to be the longer dimension
        #   - If width > height (object is wider than tall), rotate so height becomes longer
        # For top/bottom walls (horizontal): we want width (parallel to wall) to be the longer dimension
        #   - If height > width (object is taller than wide), rotate so width becomes longer
        should_rotate = False
        if side in ["left", "right"]:
            # For vertical walls: rotate if width > height (so height becomes longer after rotation)
            should_rotate = (w > h)
        else:  # top or bottom
            # For horizontal walls: rotate if height > width (so width becomes longer after rotation)
            should_rotate = (h > w)

        # Use rotated dimensions if needed
        use_w, use_h = (h, w) if should_rotate else (w, h)

        # Generate candidate position near the selected wall
        if side == "left":
            cand_x = boundary['left'] + random.uniform(3, wall_proximity)
            cand_y = random.uniform(boundary['top'] + 5, boundary['bottom'] - use_h - 5)
        elif side == "right":
            cand_x = boundary['right'] - use_w - random.uniform(3, wall_proximity)
            cand_y = random.uniform(boundary['top'] + 5, boundary['bottom'] - use_h - 5)
        elif side == "top":
            cand_x = random.uniform(boundary['left'] + 5, boundary['right'] - use_w - 5)
            cand_y = boundary['top'] + random.uniform(3, wall_proximity)
        else:  # bottom
            cand_x = random.uniform(boundary['left'] + 5, boundary['right'] - use_w - 5)
            cand_y = boundary['bottom'] - use_h - random.uniform(3, wall_proximity)

        new_box = (cand_x, cand_y, use_w, use_h)

        # Check if position is valid (no collisions, within boundary)
        if not is_valid_position(obj_to_move, new_box, temp_state, temp_keys, boundary,
                                verbose=False, min_distance=min_distance):
            continue

        # Check if position is NOT in current agent's FOV
        if is_box_in_fov(new_box, fov_polygon_normalized):
            continue

        # NEW: Check if position IS in other agent's FOV (if provided)
        if other_agent_fov_polygon is not None:
            if not is_box_in_fov(new_box, other_agent_fov_polygon):
                continue  # Skip if NOT visible to other agent

        # Check if position is blocking a door
        if is_box_blocking_door(new_box, temp_state, door_buffer=10.0):
            continue

        # Check if the displacement is sufficiently large
        displacement_distance = boxes_min_distance(original_box, new_box)
        if displacement_distance < min_displacement:
            continue

        # Found valid position!
        new_state[obj_to_move] = (cand_x, cand_y, use_w, use_h, desc, max_z)
        return new_state

    return None


def group_objects_by_wall(objects, all_coordinates, boundary, tolerance=10):
    """
    Group objects by which wall they're closest to.
    
    Args:
        objects: List of object names to group
        all_coordinates: Dict of object coordinates
        boundary: Room boundary dict
        tolerance: Maximum distance from wall to be considered "on wall"
    
    Returns:
        Dict with keys 'left', 'right', 'top', 'bottom' containing object lists
    """
    walls = {'left': [], 'right': [], 'top': [], 'bottom': []}
    
    for obj in objects:
        if obj not in all_coordinates:
            continue
            
        x, y, w, h = all_coordinates[obj][:4]
        
        # Calculate distances to each wall
        dist_left = x - boundary['left']
        dist_right = boundary['right'] - (x + w)
        dist_top = y - boundary['top']
        dist_bottom = boundary['bottom'] - (y + h)
        
        # Assign to closest wall if within tolerance
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        if min_dist <= tolerance:
            if min_dist == dist_left:
                walls['left'].append(obj)
            elif min_dist == dist_right:
                walls['right'].append(obj)
            elif min_dist == dist_top:
                walls['top'].append(obj)
            else:
                walls['bottom'].append(obj)
    
    return walls


def reorder_objects_on_wall(wall, wall_objects, all_coordinates, boundary, objects_on_top=None):
    """
    Shuffle objects along a wall and adjust bbox positions to fit the new order.
    
    Args:
        wall: 'left', 'right', 'top', or 'bottom'
        wall_objects: List of object names on this wall
        all_coordinates: Dict of all coordinates
        boundary: Room boundary dict
        objects_on_top: Dict mapping top_object -> bottom_object
    
    Returns:
        New state dict with reordered objects and adjusted bboxes, or None if can't reorder
    """
    if objects_on_top is None:
        objects_on_top = {}

    if len(wall_objects) < 2:
        return None, None, None  # Need at least 2 objects to reorder
    
    
    # Check if wall has objects from different categories
    # Reordering same categories doesn't create meaningful distractors
    categories = [get_category(obj) for obj in wall_objects]
    unique_categories = set(categories)
    
    if len(unique_categories) == 1:
        return None, None, None  # All objects are same category - skip this wall
    
    # Get original positions and sort by position along wall
    obj_data = []
    for obj in wall_objects:
        x, y, w, h, desc, z = all_coordinates[obj]
        obj_data.append((obj, x, y, w, h, desc, z))
    
    # Sort by position along wall
    if wall in ['left', 'right']:
        obj_data.sort(key=lambda o: o[2])  # Sort by Y (top to bottom)
        is_vertical = True
    else:  # top or bottom
        obj_data.sort(key=lambda o: o[1])  # Sort by X (left to right)
        is_vertical = False
    
    # Extract object info
    objects = [o[0] for o in obj_data]
    
    # Shuffle the object order
    shuffled_objects = random.sample(objects, len(objects))
    
    # Ensure order actually changed
    if shuffled_objects == objects:
        return None, None, None
    
    # Check if category sequence changed (not just object instances)
    # E.g., [Cabinet1, Bed, Cabinet2] → [Cabinet2, Bed, Cabinet1] is invalid
    # because category sequence [Cabinet, Bed, Cabinet] didn't change
    original_categories = [get_category(obj) for obj in objects]
    shuffled_categories = [get_category(obj) for obj in shuffled_objects]
    
    if original_categories == shuffled_categories:
        return None, None, None  # Category sequence didn't change - not meaningful
    

    
    # Get the start position from first object in original order
    first_obj_data = obj_data[0]
    last_obj_data = obj_data[-1]
    
    start_x = first_obj_data[1]
    start_y = first_obj_data[2]
    
    # Calculate the original span (from first object start to last object end)
    if is_vertical:
        end_y = last_obj_data[2] + last_obj_data[4]  # last Y + last height
        original_span = end_y - start_y
    else:
        end_x = last_obj_data[1] + last_obj_data[3]  # last X + last width
        original_span = end_x - start_x
    
    # Calculate total size of shuffled objects
    if is_vertical:
        total_size = sum(all_coordinates[obj][3] for obj in shuffled_objects)  # Sum of heights
    else:
        total_size = sum(all_coordinates[obj][2] for obj in shuffled_objects)  # Sum of widths
    
    # Calculate gap to distribute (original span - total object size)
    total_gap = original_span - total_size
    gap_per_object = total_gap / len(shuffled_objects) if len(shuffled_objects) > 1 else 0
    
    # Create new state with truly independent copies
    # Explicitly unpack and repack tuples to avoid shared references
    new_state = {}
    for obj, coords in all_coordinates.items():
        x, y, w, h, desc, z = coords
        new_state[obj] = (x, y, w, h, desc, z)
    
    # Find all dependent objects (those that sit ON TOP of other objects)
    # We need to map base_object -> list of dependent objects
    dependents_map = {}
    for dependent, base in objects_on_top.items():
        if base not in dependents_map:
            dependents_map[base] = []
        dependents_map[base].append(dependent)

    # For vertical walls (left/right): X stays fixed, only Y changes
    # For horizontal walls (top/bottom): Y stays fixed, only X changes
    if is_vertical:
        # Left/Right walls: each object keeps its own X, only Y changes
        current_y = start_y
        
        for obj in shuffled_objects:
            # Get this object's original dimensions and X position
            orig_x, orig_y, w, h, desc, z = all_coordinates[obj]
            
            # Place object: keep original X, update Y based on new order
            new_state[obj] = (orig_x, current_y, w, h, desc, z)
            
            # Update dependent objects (if any)
            if obj in dependents_map:
                delta_y = current_y - orig_y  # How much the base object moved
                for dep_obj in dependents_map[obj]:
                    if dep_obj in all_coordinates:
                        d_x, d_y, d_w, d_h, d_desc, d_z = all_coordinates[dep_obj]
                        # Move dependent object by same delta
                        new_state[dep_obj] = (d_x, d_y + delta_y, d_w, d_h, d_desc, d_z)

            # Move down for next object (including proportional gap)
            current_y += h + gap_per_object
        
        # Validate: check if last object exceeds bottom boundary
        last_obj = shuffled_objects[-1]
        last_y, last_h = new_state[last_obj][1], new_state[last_obj][3]
        if (last_y + last_h) > boundary['bottom']:
            return None, None, None  # Would exceed boundary
            
    else:
        # Top/Bottom walls: each object keeps its own Y, only X changes
        current_x = start_x
        
        for obj in shuffled_objects:
            # Get this object's original dimensions and Y position
            orig_x, orig_y, w, h, desc, z = all_coordinates[obj]
            
            # Place object: keep original Y, update X based on new order
            new_state[obj] = (current_x, orig_y, w, h, desc, z)
            
            # Update dependent objects (if any)
            if obj in dependents_map:
                delta_x = current_x - orig_x  # How much the base object moved
                for dep_obj in dependents_map[obj]:
                    if dep_obj in all_coordinates:
                        d_x, d_y, d_w, d_h, d_desc, d_z = all_coordinates[dep_obj]
                        # Move dependent object by same delta
                        new_state[dep_obj] = (d_x + delta_x, d_y, d_w, d_h, d_desc, d_z)

            # Move right for next object (including proportional gap)
            current_x += w + gap_per_object
        
        # Validate: check if last object exceeds right boundary
        last_obj = shuffled_objects[-1]
        last_x, last_w = new_state[last_obj][0], new_state[last_obj][2]
        if (last_x + last_w) > boundary['right']:
            return None, None, None  # Would exceed boundary
            
    # --- COLLISION CHECK ---
    # Check if any moved object overlaps with other objects in the room
    # or with each other (though spacing handles most internal overlaps)
    
    def check_overlap(rect1, rect2, padding=0.0):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        # Use a small negative padding to allow touching but not overlapping
        return not (x1 + w1 - padding <= x2 or x2 + w2 - padding <= x1 or 
                    y1 + h1 - padding <= y2 or y2 + h2 - padding <= y1)

    moved_objects = set(new_state.keys())
    
    # Categories to ignore during collision checks (floor items that are okay to overlap)
    ignore_categories = ["Rug", "Carpet", "Mat", "Floor", "Ceiling Lamp", "Pendant Lamp"]

    for obj1 in moved_objects:
        x1, y1, w1, h1, desc1, z1 = new_state[obj1]
        rect1 = (x1, y1, w1, h1)
        
        # 1. Check against static objects (not moved)
        for obj2, coords2 in all_coordinates.items():
            if obj2 in moved_objects:
                continue
                
            x2, y2, w2, h2, desc2, z2 = coords2
            
            # Skip if either object is an ignored category
            if desc1 in ignore_categories or desc2 in ignore_categories:
                continue
                
            # Skip if one object is explicitly sitting on top of the other
            if objects_on_top.get(obj1) == obj2 or objects_on_top.get(obj2) == obj1:
                continue
                
            if check_overlap(rect1, (x2, y2, w2, h2)):
                 return None, None, None  # Collision detected
        
        # 2. Check against other moved objects (internal collision)
        for obj2 in moved_objects:
            if obj1 == obj2:
                continue
                
            x2, y2, w2, h2, desc2, z2 = new_state[obj2]
            
            if desc1 in ignore_categories or desc2 in ignore_categories:
                continue
            
            # Skip if one object is explicitly sitting on top of the other
            if objects_on_top.get(obj1) == obj2 or objects_on_top.get(obj2) == obj1:
                continue
                
            if check_overlap(rect1, (x2, y2, w2, h2)):
                 return None, None, None # Internal collision

    
    # Return new state with ordering info for logging
    return new_state, objects, shuffled_objects


def main():
    parser = argparse.ArgumentParser(description="Generate map questions with distractors")
    parser.add_argument("--input_json", type=str, default="./object_info.json",
                        help="Path to visible objects JSON file")
    parser.add_argument("--cam_data_json", type=str, default="./camera_info.json",
                        help="Path to camera data JSON file")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for map images and JSON files")
    parser.add_argument("--output_json", type=str, default="map_questions.json",
                        help="Output filename for questions JSON file")
    args = parser.parse_args()

    # from types import SimpleNamespace

    # # Simulate argparse args
    # args = SimpleNamespace(
    #     input_json="./scenes/temp/visible_objects.json",
    #     cam_data_json="./scenes/temp/cameras.json",
    #     output_dir="./scenes/temp/map_questions_binary2",
    #     output_json="map_questions.json"
    # )


    with open(args.input_json, "r") as f:
        visible_objects = json.load(f)
    with open(args.cam_data_json, "r") as f:
        camera_json = json.load(f)

    # Kitchen/Sink Exclusion Check
    is_kitchen = "kitchen" in os.path.abspath(args.input_json).lower()
    has_sink = False
    for cam in visible_objects:
        for obj_name in visible_objects[cam]:
            if get_category(obj_name).lower() == "sink":
                has_sink = True
                break
        if has_sink: break
    
    if is_kitchen and has_sink:
        print(f"[{os.path.basename(args.input_json)}] Skipping map generation: Kitchen with Sink detected.")
        return

    int_count = len(set(visible_objects["camera_0_0"].keys()).intersection(set(visible_objects["camera_1_0"].keys())))
    union_count = len(set(visible_objects["camera_0_0"].keys()).union(set(visible_objects["camera_1_0"].keys())))
    camera_angle_1 = get_camera_2d_angle(camera_json["camera_0_0"]["T"])
    camera_angle_2 = get_camera_2d_angle(camera_json["camera_1_0"]["T"])
    input_angle_1 = convert_angle(camera_angle_1)
    input_angle_2 = convert_angle(camera_angle_2)

    # Identify objects positioned on top of other objects
    objects_on_top = filter_objects_on_top(visible_objects)

    all_coordinates_1, map_width_1, map_height_1 = get_all_coordinates(visible_objects, input_angle_1, objects_on_top)
    all_coordinates_2, map_width_2, map_height_2 = get_all_coordinates(visible_objects, input_angle_2, objects_on_top)

    cam_0_visible = set(visible_objects["camera_0_0"].keys())
    cam_1_visible = set(visible_objects["camera_1_0"].keys())
    unique_agent1 = cam_0_visible - cam_1_visible
    unique_agent2 = cam_1_visible - cam_0_visible
    common_objs = cam_0_visible.intersection(cam_1_visible)
    all_objs = cam_0_visible.union(cam_1_visible)

    avoid_swap = ["Door", "Wall Art", "Window", "Chair", "Table Dining"]

    # Helper function to compute boundary
    def compute_boundary(all_coords, map_width, map_height, room_min=0):
        """
        Compute a boundary box for the room based on door positions:
        - Detects which wall each door lies on by comparing edge alignment and aspect ratio.
        - Shrinks the boundary inward along that wall by the door's inner edge.
        - The opposite sides remain unchanged.
        """

        boundary = {
            'left': room_min,
            'right': map_width,
            'top': room_min,
            'bottom': map_height
        }

        door_objs = [v for k, v in all_coords.items() if v[4].lower() == 'door']  # Description is at index 4
        if not door_objs:
            return boundary

        for x, y, w, h, desc, _ in door_objs:
            x2, y2 = x + w, y + h

            # Determine whether door is vertical or horizontal
            vertical = h > w  # vertical door spans up-down → left/right wall
            horizontal = not vertical  # horizontal door spans left-right → top/bottom wall

            # Compute distances from all walls
            dist_top = abs(y - room_min)
            dist_bottom = abs(map_height - y2)
            dist_left = abs(x - room_min)
            dist_right = abs(map_width - x2)

            # Determine which walls the door could plausibly lie on
            if vertical:
                # Door likely on left or right wall
                if dist_left <= dist_right:
                    wall = 'left'
                else:
                    wall = 'right'
            else:
                # Door likely on top or bottom wall
                if dist_top <= dist_bottom:
                    wall = 'top'
                else:
                    wall = 'bottom'

            # --- Shrink boundary along that wall's inner edge ---
            if wall == 'left':
                # Door starts at left edge, inner edge = x + w
                boundary['left'] = max(boundary['left'], x + w)
            elif wall == 'right':
                # Door ends at right edge, inner edge = x
                boundary['right'] = min(boundary['right'], x)
            elif wall == 'top':
                # Door starts at top edge, inner edge = y + h
                boundary['top'] = max(boundary['top'], y + h)
            elif wall == 'bottom':
                # Door ends at bottom edge, inner edge = y
                boundary['bottom'] = min(boundary['bottom'], y)

        return boundary

    boundary_1 = compute_boundary(all_coordinates_1, map_width_1, map_height_1)
    boundary_2 = compute_boundary(all_coordinates_2, map_width_2, map_height_2)

    # Calculate global bounds for FOV normalization
    all_xmin, all_xmax, all_ymin, all_ymax = [], [], [], []
    for cam in visible_objects:
        for obj in visible_objects[cam]:
            if "bbox_3d_corners" in visible_objects[cam][obj]:
                x_min, x_max, y_min, y_max = get_2d(visible_objects[cam][obj]["bbox_3d_corners"])
                all_xmin.append(x_min)
                all_xmax.append(x_max)
                all_ymin.append(y_min)
                all_ymax.append(y_max)

    global_xmin = min(all_xmin) if all_xmin else 0
    global_xmax = max(all_xmax) if all_xmax else 1
    global_ymin = min(all_ymin) if all_ymin else 0
    global_ymax = max(all_ymax) if all_ymax else 1

    # Calculate FOV for both cameras
    camera_data_1 = camera_json["camera_0_0"]
    camera_data_2 = camera_json["camera_1_0"]
    fov_polygon_1_world, cam_pos_1_world = calculate_camera_fov(camera_data_1, floor_z=0.0, max_distance=100.0)
    fov_polygon_2_world, cam_pos_2_world = calculate_camera_fov(camera_data_2, floor_z=0.0, max_distance=100.0)

    # Normalize FOV to pixel space for both angles
    fov_polygon_1_norm = normalize_fov_to_pixel_space(fov_polygon_1_world, cam_pos_1_world,
                                                      global_xmin, global_xmax, global_ymin, global_ymax, input_angle_1)
    fov_polygon_2_norm = normalize_fov_to_pixel_space(fov_polygon_2_world, cam_pos_2_world,
                                                      global_xmin, global_xmax, global_ymin, global_ymax, input_angle_2)

    # IMPORTANT: For distractor generation, we need Agent 2's FOV in Agent 1's coordinate system (and vice versa)
    # This is because distractors for Agent 1 use all_coordinates_1 which is in Agent 1's rotated space
    fov_polygon_2_in_agent1_coords = normalize_fov_to_pixel_space(fov_polygon_2_world, cam_pos_2_world,
                                                                   global_xmin, global_xmax, global_ymin, global_ymax, input_angle_1)
    fov_polygon_1_in_agent2_coords = normalize_fov_to_pixel_space(fov_polygon_1_world, cam_pos_1_world,
                                                                   global_xmin, global_xmax, global_ymin, global_ymax, input_angle_2)

    # Helper function to generate distractors for an agent
    def generate_distractors_for_agent(asking_to, all_coordinates, boundary, fov_polygon_normalized, angle, other_agent_fov_polygon, objects_on_top):
        # Determine which objects are visible only to other agent
        if asking_to == "agent_1":
            not_in_my_view_objects = list(unique_agent2)
        else:
            not_in_my_view_objects = list(unique_agent1)
        
        # Filter valid objects
        not_in_view_pool = [
            item for item in not_in_my_view_objects
            if item in all_coordinates
            and get_category(item) not in avoid_swap
        ]
        
        if len(not_in_view_pool) == 0:
            print(f"[{asking_to}] No objects available for distractor generation")
            return [], [], []

        # Remove objects that are on top from the pool (they move with their base)
        # We only want to reorder base objects
        # EXCLUDE WALL OBJECTS from reordering to preserve their strict boundary snapping
        base_objects = [
            obj for obj in not_in_view_pool 
            if obj not in objects_on_top 
            and get_category(obj) not in ["Window", "Door", "Wall Art"]
        ]
        
        # Group objects by wall (using base objects only)
        walls = group_objects_by_wall(base_objects, all_coordinates, boundary, tolerance=15)
        
        # Filter walls with at least 2 objects
        valid_walls = {wall_name: objs for wall_name, objs in walls.items() if len(objs) >= 2}
        
        if not valid_walls:
            print(f"[{asking_to}] No walls with 2+ objects for reordering")
            return [], [], []
        
        print(f"[{asking_to}] Found walls with objects: {[(w, len(objs)) for w, objs in valid_walls.items()]}")
        
        # Track used category sequences to avoid duplicate permutations
        used_category_sequences = set()
        
        # Generate 3 distractors by reordering objects on random walls
        distractor_maps = []
        max_attempts = 50
        
        for distractor_idx in range(3):
            success = False
            for attempt in range(max_attempts):
                # Choose a random wall
                wall_name = random.choice(list(valid_walls.keys()))
                wall_objs = valid_walls[wall_name]
                
                # Reorder objects on this wall
                new_state, original_order, shuffled_order = reorder_objects_on_wall(
                    wall_name, wall_objs, all_coordinates, boundary, objects_on_top
                )
                
                if new_state is not None:
                    # Check if this category sequence has been used
                    shuf_cats = [get_category(obj) for obj in shuffled_order]
                    cat_seq_tuple = tuple(shuf_cats)
                    
                    if cat_seq_tuple in used_category_sequences:
                        continue  # Skip this duplicate category sequence
                    
                    if is_state_unique(new_state, distractor_maps):
                        distractor_maps.append(new_state)
                        used_category_sequences.add(cat_seq_tuple)
                        
                    # Get categories for both orderings
                    orig_cats = [get_category(obj) for obj in original_order]
                    shuf_cats = [get_category(obj) for obj in shuffled_order]
                    
                    print(f"[{asking_to}] Generated distractor {distractor_idx + 1}/3: reordered {len(wall_objs)} objects on {wall_name} wall")
                    print(f"  Original objects: {original_order}")
                    print(f"  Shuffled objects: {shuffled_order}")
                    print(f"  Original: {' → '.join(orig_cats)}")
                    print(f"  Shuffled: {' → '.join(shuf_cats)}")
                    success = True
                    break
            
            if not success:
                print(f"[{asking_to}] Could not generate distractor {distractor_idx + 1}/3 after {max_attempts} attempts")
        
        # Split into three lists
        distractor_2_maps = distractor_maps[:1] if len(distractor_maps) >= 1 else []
        distractor_3_maps = distractor_maps[1:2] if len(distractor_maps) >= 2 else []
        counting_maps = distractor_maps[2:3] if len(distractor_maps) >= 3 else []
        
        return distractor_2_maps, distractor_3_maps, counting_maps

    # Generate distractors for both agents
    print("\n=== Generating Distractors for Agent 1 ===")
    distractor_2_maps_1, distractor_3_maps_1, counting_maps_1 = generate_distractors_for_agent(
        "agent_1", all_coordinates_1, boundary_1, fov_polygon_1_norm, input_angle_1, 
        fov_polygon_2_in_agent1_coords, objects_on_top  # Agent 2's FOV in Agent 1's coordinate system
    )
    print(f"Agent 1 - Distractors generated: {len(distractor_2_maps_1) + len(distractor_3_maps_1) + len(counting_maps_1)}")

    print("\n=== Generating Distractors for Agent 2 ===")
    distractor_2_maps_2, distractor_3_maps_2, counting_maps_2 = generate_distractors_for_agent(
        "agent_2", all_coordinates_2, boundary_2, fov_polygon_2_norm, input_angle_2,
        fov_polygon_1_in_agent2_coords, objects_on_top  # Agent 1's FOV in Agent 2's coordinate system
    )
    print(f"Agent 2 - Distractors generated: {len(distractor_2_maps_2) + len(distractor_3_maps_2) + len(counting_maps_2)}")

    # Helper function to extract base object name from clone names
    def extract_base_object_name(obj_name):
        """Extract base object name, removing _clone suffixes."""
        if "_clone" in obj_name:
            base_name = obj_name.split("_clone")[0]
            return base_name
        return obj_name

    # Helper function to format coordinates as JSON map
    def format_coordinates_as_json_map(all_coords):
        """Format coordinates as JSON map with category + number."""
        formatted_map = {}
        # Group objects by category, using base object names
        category_groups = {}
        for obj, (x_min, y_min, width, height, description, _) in all_coords.items():
            try:
                # Extract base object name (remove _clone suffix)
                base_obj = extract_base_object_name(obj)
                # Get category from base object name
                category = get_category(base_obj)
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append((obj, x_min, y_min, width, height, base_obj))
            except Exception as e:
                print(f"  WARNING: Error processing object {obj} in format_coordinates_as_json_map: {e}")
                continue

        # Sort objects within each category for consistent ordering
        for category in category_groups:
            category_groups[category].sort(key=lambda x: (x[5], x[0]))  # Sort by base_obj name, then obj name

        # Number objects within each category starting from 1
        for category, objects in category_groups.items():
            for idx, (obj, x_min, y_min, width, height, base_obj) in enumerate(objects, start=1):
                x_max = x_min + width
                y_max = y_min + height
                key = f"{category} {idx}"
                formatted_map[key] = [int(x_min), int(y_min), int(x_max), int(y_max)]
        return formatted_map

    # Helper function to create Yes/No questions for an agent
    def create_yesno_questions(asking_to, all_coordinates, distractor_2_maps, distractor_3_maps, counting_maps, boundary, output_dir):
        questions = []

        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"  ERROR: Failed to create output directory {output_dir}: {e}")
            return questions

        # Collect all available distractors
        all_distractors = []
        all_distractors.extend([('type2', d) for d in distractor_2_maps])
        all_distractors.extend([('type3', d) for d in distractor_3_maps])
        # all_distractors.extend([('counting', d) for d in counting_maps])

        print(f"\n[{asking_to} Question Generation]")
        print(f"  Available distractors: {len(all_distractors)}")

        # Determine number of questions to generate
        # Generate equal number of correct and wrong questions
        max_questions = 10
        num_questions = min(max_questions, len(all_distractors) + 1)  # +1 for at least one correct question

        if num_questions == 0:
            print(f"  WARNING: No questions can be generated for {asking_to} (no distractors available)")
            return questions

        # Generate questions: 50% correct, 50% wrong
        num_correct = num_questions // 2
        num_wrong = num_questions - num_correct

        # Shuffle distractors to randomize selection
        random.shuffle(all_distractors)

        q_idx = 0

        # Generate correct questions
        for _ in range(num_correct):
            try:
                # Use correct map
                map_coords = all_coordinates
                is_correct = True

                # Format coordinates as JSON map
                json_map = format_coordinates_as_json_map(map_coords)

                # Generate and save plot
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                object_color_map = {}
                print("map_coords:", map_coords)
                plot(map_coords, ax, object_color_map, boundary, "")

                # Create legend
                from matplotlib.patches import Patch
                if object_color_map:
                    legend_elements = [Patch(facecolor=object_color_map[desc], label=desc) for desc in sorted(object_color_map.keys())]
                    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                              ncol=4, fontsize=18, frameon=False)

                # Save image
                image_filename = f"map_question_{asking_to}_{q_idx}.png"
                print("Saving image:", image_filename)
                image_path = os.path.join(output_dir, image_filename)
                plt.savefig(image_path, bbox_inches='tight')
                plt.close(fig)

                # Create question object
                question = {
                    "image_path": image_path,
                    "json_map": json_map,
                    "is_correct": is_correct,
                    "asking_to": asking_to,
                }

                questions.append(question)
                q_idx += 1
            except Exception as e:
                print(f"  ERROR generating correct question {q_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Generate wrong questions
        for i in range(num_wrong):
            if i >= len(all_distractors):
                break

            try:
                # Use a distractor map
                dist_type, distractor_state = all_distractors[i]
                map_coords = distractor_state
                is_correct = False

                # Format coordinates as JSON map
                json_map = format_coordinates_as_json_map(map_coords)

                # Generate and save plot
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                object_color_map = {}
                plot(map_coords, ax, object_color_map, boundary, "")

                # Create legend
                from matplotlib.patches import Patch
                if object_color_map:
                    legend_elements = [Patch(facecolor=object_color_map[desc], label=desc) for desc in sorted(object_color_map.keys())]
                    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                              ncol=4, fontsize=18, frameon=False)

                # Save image
                image_filename = f"map_question_{asking_to}_{q_idx}.png"
                image_path = os.path.join(output_dir, image_filename)
                plt.savefig(image_path, bbox_inches='tight')
                plt.close(fig)

                # Create question object
                question = {
                    "image_path": image_path,
                    "json_map": json_map,
                    "is_correct": is_correct,
                    "asking_to": asking_to,
                }

                questions.append(question)
                q_idx += 1
            except Exception as e:
                print(f"  ERROR generating wrong question {q_idx} (distractor {i}): {e}")
                import traceback
                traceback.print_exc()
                continue

        # Shuffle questions to randomize order
        random.shuffle(questions)

        print(f"  Total questions generated: {len(questions)} (Correct: {num_correct}, Wrong: {num_wrong})")
        return questions

    # Create questions for both agents
    try:
        questions_agent_1 = create_yesno_questions("agent_1", all_coordinates_1, distractor_2_maps_1, distractor_3_maps_1, counting_maps_1, boundary_1, args.output_dir)
    except Exception as e:
        print(f"ERROR creating questions for agent_1: {e}")
        import traceback
        traceback.print_exc()
        questions_agent_1 = []

    try:
        questions_agent_2 = create_yesno_questions("agent_2", all_coordinates_2, distractor_2_maps_2, distractor_3_maps_2, counting_maps_2, boundary_2, args.output_dir)
    except Exception as e:
        print(f"ERROR creating questions for agent_2: {e}")
        import traceback
        traceback.print_exc()
        questions_agent_2 = []

    # Save questions to JSON
    all_questions = {
        "agent_1": questions_agent_1,
        "agent_2": questions_agent_2
    }

    # Handle output paths - if filename contains path separators, use as-is, otherwise join with output_dir
    if os.sep in args.output_json or '/' in args.output_json:
        output_json_path = args.output_json
    else:
        output_json_path = os.path.join(args.output_dir, args.output_json)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else args.output_dir, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\n=== Final Summary ===")
    print(f"Agent 1 Distractors:")
    print(f"  Total: {len(distractor_2_maps_1) + len(distractor_3_maps_1) + len(counting_maps_1)}")
    print(f"\nAgent 2 Distractors:")
    print(f"  Total: {len(distractor_2_maps_2) + len(distractor_3_maps_2) + len(counting_maps_2)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()