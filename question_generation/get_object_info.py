# save_visible_objects.py
import bpy
import re
import json
import random
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector, Matrix
import os
import sys
import argparse
import numpy as np

EXCLUDE_KEYWORDS = ["camera", "camrig", "staircase", "rug", "light", "bookstack", "spawn_placeholder", "NatureShelfTrinkets", "CeilingClassicLamp", "warehouses"]
# MIN_VISIBLE_POINTS = 5 # No longer used for corner-based sampling
VOLUME_THRESHOLD = 0.02
# NUM_SAMPLES = 250 # No longer used for corner-based sampling

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def is_excluded(obj_name: str) -> bool:
    lname = obj_name.lower()
    return any(keyword in lname for keyword in EXCLUDE_KEYWORDS)

def clean_name(name: str) -> str:
    name = re.sub(r'\.\d+$', '', name)
    name = re.split(r'Factory', name, maxsplit=1)[0]
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    name = re.sub(r'spawn_asset', '', name)
    return name.strip()

def compute_volume(obj) -> float:
    if not hasattr(obj, "dimensions"):
        return 0.0
    dims = obj.dimensions
    return dims.x * dims.y * dims.z

def get_2d_bbox(scene, cam, obj):
    """Return normalized 2D bounding box of object in [0,1] image space"""
    coords = []
    for corner in obj.bound_box:
        world_coord = obj.matrix_world @ Vector(corner)
        co_ndc = world_to_camera_view(scene, cam, world_coord)
        if co_ndc.z >= 0:  # only keep points in front of camera
            coords.append((co_ndc.x, co_ndc.y))

    if not coords:
        return None
    xs, ys = zip(*coords)
    return [min(xs), min(ys), max(xs), max(ys)]  # [x_min, y_min, x_max, y_max]

def get_3d_bbox(obj):
    """Return 3D bounding box corners in world coordinates."""
    return [list(obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]

def check_visibility(camera_key, target_obj, scene_dir, threshold = 1000) -> bool:
    print(f"Checking visibility for {target_obj} in {scene_dir}")
    object_json_map = {
        "camera_0_0": scene_dir+"/frames/Objects/camera_0/Objects_0_0_0048_0.json",
        "camera_1_0": scene_dir+"/frames/Objects/camera_0/Objects_1_0_0048_0.json"}
    object_seg_map = {
        "camera_0_0": scene_dir+"/frames/ObjectSegmentation/camera_0/ObjectSegmentation_0_0_0048_0.npz",
        "camera_1_0": scene_dir+"/frames/ObjectSegmentation/camera_0/ObjectSegmentation_1_0_0048_0.npz"}

    with open(object_json_map[camera_key], "r") as f:
        cam_json = json.load(f)

    target_index = None
    for obj in cam_json:
        if obj["name"] == target_obj:
            target_index = obj['object_index']

    if target_index is None:
        return False

    data = np.load(object_seg_map[camera_key])
    mapping = data["vals"]
    pointwise_labels_original = data["indices"]
    pointwise_labels_mapped = np.array([mapping[val] for val in pointwise_labels_original])
    sum_val = np.sum(pointwise_labels_mapped==target_index)
    if sum_val<threshold:
        print(f"Too Small or Not Visible: {target_obj}")
    return sum_val>=threshold

def get_local_axes(obj):
    """Return local axes (x, y, z) in world coordinates."""
    x_axis = list(obj.matrix_world.col[0].xyz)
    y_axis = list(obj.matrix_world.col[1].xyz)
    z_axis = list(obj.matrix_world.col[2].xyz)
    return {'x': x_axis, 'y': y_axis, 'z': z_axis}

def get_material_names(obj):
    """Return a list of material names applied to the object."""
    return [slot.material.name for slot in obj.material_slots if slot.material]

def main():
    import sys
    script_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--output_json" and i + 1 < len(sys.argv):
            script_args.extend([sys.argv[i], sys.argv[i + 1]])
            i += 2
        elif sys.argv[i].startswith("--output_json="):
            script_args.append(sys.argv[i])
            i += 1
        elif sys.argv[i] == "--scene_dir" and i + 1 < len(sys.argv):
            script_args.extend([sys.argv[i], sys.argv[i + 1]])
            i += 2
        elif sys.argv[i].startswith("--scene_dir="):
            script_args.append(sys.argv[i])
            i += 1
        else:
            i += 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", type=str, default="./visible_objects.json")
    parser.add_argument("--scene_dir", type=str, default="")
    args = parser.parse_args(script_args)
    OUTPUT_JSON = args.output_json

    try:
        cam1 = bpy.data.objects["camera_0_0"]
        cam2 = bpy.data.objects["camera_1_0"]
    except KeyError as e:
        print(f"❌ Error: Camera not found: {e}")
        return

    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene

    result = {"camera_0_0": {}, "camera_1_0": {}}

    # Store objects to be processed, grouped by their base name
    objects_to_process = {}

    for obj in bpy.data.objects:
        # Exclude placeholders and other keywords
        if obj.type != 'MESH' or is_excluded(obj.name):
            continue
        if "spawn_asset" not in obj.name:
            continue

        # VOLUME FILTERING
        volume = compute_volume(obj)
        if volume <= VOLUME_THRESHOLD and not any(word in obj.name.lower() for word in ["art", "lamp", "door", "window", "plant"]):
            print(f"{obj.name} skipped in Volume Filtering")
            continue

        # BASE NAME FILTERING
        base_name = re.split(r'\.\d+$', obj.name)[0]
        if base_name not in objects_to_process:
            objects_to_process[base_name] = []
        objects_to_process[base_name].append(obj)

    # Now, iterate through the grouped objects and assign a number
    # based on the group, not the individual components
    unique_name_counts = {}

    for base_name, obj_list in objects_to_process.items():
        cleaned_name = clean_name(base_name)

        # Determine if this cleaned name has been seen before
        count = unique_name_counts.get(cleaned_name, 0) + 1
        unique_name_counts[cleaned_name] = count
        unique_name_numbering = f"{cleaned_name} {count}"

        # Now process each individual object in the group
        for obj in obj_list:
            # Collect all metadata for this object
            loc = list(obj.location)
            rot = list(obj.rotation_euler)
            scale = list(obj.scale)

            # Use the local z-axis as a representative normal vector
            normals = list(obj.matrix_world.col[2].xyz)

            for cam, key in [(cam1, "camera_0_0"), (cam2, "camera_1_0")]:
                obj_name = obj.name
                if check_visibility(camera_key=key, target_obj=obj_name, scene_dir=args.scene_dir):
                    bbox_2d = get_2d_bbox(scene, cam, obj)

                    if bbox_2d:
                        print(f"Saving data for {key} | {unique_name_numbering}")
                        result[key][unique_name_numbering] = {
                            "name": unique_name_numbering,
                            "id": base_name,
                            "bbox_2d": bbox_2d,
                            "bbox_3d_corners": get_3d_bbox(obj),
                            "location": loc,
                            "rotation": rot,
                            "scale": scale,
                            "local_axes": get_local_axes(obj),
                            "normals": normals,
                            "volume_fraction": compute_volume(obj),
                            "material_names": get_material_names(obj)
                        }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved visible object data to {os.path.abspath(OUTPUT_JSON)}")

if __name__ == "__main__":
    main()