import bpy
import json
import numpy as np
import os
import argparse
import sys

# Filter out Blender's arguments and only parse our own
script_args = []
i = 0
while i < len(sys.argv):
    if sys.argv[i] == "--output_json" and i + 1 < len(sys.argv):
        script_args.extend([sys.argv[i], sys.argv[i + 1]])
        i += 2
    elif sys.argv[i].startswith("--output_json="):
        script_args.append(sys.argv[i])
        i += 1
    else:
        i += 1

parser = argparse.ArgumentParser()
parser.add_argument("--output_json", type=str, default="./cameras.json")
args = parser.parse_args(script_args)
OUTPUT_JSON = args.output_json

# Pick your cameras
camera_names = ["camera_0_0", "camera_1_0"]

def get_camera_info(cam_obj, scene):
    cam_data = cam_obj.data

    # --- Intrinsics ---
    f_in_mm = cam_data.lens
    sensor_width_in_mm = cam_data.sensor_width
    sensor_height_in_mm = cam_data.sensor_height

    # Image size
    render = scene.render
    width = render.resolution_x
    height = render.resolution_y
    scale = render.resolution_percentage / 100.0
    width = int(width * scale)
    height = int(height * scale)

    # Focal length in pixels
    f_x = f_in_mm / sensor_width_in_mm * width
    f_y = f_in_mm / sensor_height_in_mm * height

    # Principal point
    c_x = width / 2.0
    c_y = height / 2.0

    K = np.array([
        [f_x,   0,   c_x],
        [0,   f_y,   c_y],
        [0,     0,     1]
    ])

    # --- Extrinsics ---
    # Blender uses right-handed Z-up, camera looks -Z in its local space.
    # world->camera = inverse(camera->world)
    cam_matrix_world = np.array(cam_obj.matrix_world)
    T = np.linalg.inv(cam_matrix_world)

    return {
        "K": K.tolist(),
        "T": T.tolist(),
        "HW": [height, width]
    }

scene = bpy.context.scene
all_cams = {}

for name in camera_names:
    cam_obj = bpy.data.objects.get(name)
    if cam_obj and cam_obj.type == 'CAMERA':
        all_cams[name] = get_camera_info(cam_obj, scene)

# Save JSON
output_path = OUTPUT_JSON
with open(output_path, "w") as f:
    json.dump(all_cams, f, indent=2)

print(f"Saved camera info to {output_path}")
