import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import re
import numpy as np
from scipy.spatial.distance import cdist
import random
from collections import defaultdict
from shapely.geometry import LineString, Polygon
import argparse
from utils import get_category, allowed_categories
from scipy.spatial import ConvexHull

def get_sets(data_json, ground_json):
    cam_0_visible = set(data_json["camera_0_0"].keys())
    cam_0_visible_ground = set(ground_json["camera_0_0"].keys())
    cam_1_visible = set(data_json["camera_1_0"].keys())
    cam_1_visible_ground = set(ground_json["camera_1_0"].keys())

    common_objs_or = cam_0_visible.intersection(cam_1_visible)
    unique_agent1_or = cam_0_visible - cam_1_visible_ground
    unique_agent2_or = cam_1_visible - cam_0_visible_ground

    not_visible_agent1 = cam_0_visible_ground - cam_0_visible
    not_visible_agent2 = cam_1_visible_ground - cam_1_visible

    print("\n====== Not visible ======")
    print(f"Not visible to agent 1: {not_visible_agent1}")
    print(f"Not visible to agent 2: {not_visible_agent2}")
    avoid_set = set([get_category(item) for item in not_visible_agent1]).union(set([get_category(item) for item in not_visible_agent2]))

    common_objs = set([item for item in common_objs_or if (get_category(item) not in avoid_set and item not in avoid_set)])
    unique_agent1 = set([item for item in unique_agent1_or if (get_category(item) not in avoid_set and item not in avoid_set)])
    unique_agent2 = set([item for item in unique_agent2_or if (get_category(item) not in avoid_set and item not in avoid_set)])

    print("\n====== Eliminating ======")
    print(f"Eliminating: {len(common_objs_or)-len(common_objs)} from Common Objects: {common_objs_or-common_objs}")
    print(f"Eliminating: {len(unique_agent1_or)-len(unique_agent1)} from Agent 1 Objects: {unique_agent1_or-unique_agent1}")
    print(f"Eliminating: {len(unique_agent2_or)-len(unique_agent2)} from Agent 2 Objects: {unique_agent2_or-unique_agent2}")

    print("\n====== Lengths ======")
    print(f"{len(common_objs)} in Common Objects")
    print(f"{len(unique_agent1)} in Agent 1 Objects")
    print(f"{len(unique_agent2)} in Agent 2 Objects\n")

    all_objs = common_objs.union(unique_agent1).union(unique_agent2)

    common_objs = set([obj for obj in list(common_objs) if get_category(obj) in allowed_categories])
    unique_agent1 = set([obj for obj in list(unique_agent1) if get_category(obj) in allowed_categories])
    unique_agent2 = set([obj for obj in list(unique_agent2) if get_category(obj) in allowed_categories])
    all_objs = set([obj for obj in list(all_objs) if get_category(obj) in allowed_categories])
    return common_objs, unique_agent1, unique_agent2, all_objs

def get_neighboring_objects(data, room_min_dist, input_dictionary, input_pool, object_pool, neighbor_dict):
    on_height_thresh=0.02
    overlap_thresh = 0.7

    for obj in input_pool:
        obj_coords = np.array(input_dictionary[obj]["bbox_3d_corners"])
        obj_centre = obj_coords.mean(axis=0)
        neighbor_dict[obj] = {"near":[], "next":[], "on":[], "on_which":[]}

        obj_min_z = obj_coords[:, 2].min()
        obj_bottom_face = obj_coords[obj_coords[:, 2] == obj_min_z][:, :2]
        if len(obj_bottom_face) < 4:
            obj_bottom_face = obj_coords[np.argsort(obj_coords[:, 2])[:4], :2]
        hull = ConvexHull(obj_bottom_face)
        ordered_points = obj_bottom_face[hull.vertices]
        poly_obj = Polygon(ordered_points)

        for potential_neighbor in object_pool:
            if get_category(obj) == get_category(potential_neighbor):
                continue
            neighbor_coords = np.array(input_dictionary[potential_neighbor]["bbox_3d_corners"])

            # Find distance between bounding box all 8 corner pairs
            dist_matrix = cdist(obj_coords, neighbor_coords, metric="euclidean")
            min_dist = dist_matrix.min()
            distance_frac = min_dist / room_min_dist
            if distance_frac > 0.15:
                continue

            neigh_max_z = neighbor_coords[:, 2].max()
            neigh_top_face = neighbor_coords[neighbor_coords[:, 2] == neigh_max_z][:, :2]
            if len(neigh_top_face) < 4:
                neigh_top_face = neighbor_coords[np.argsort(-neighbor_coords[:, 2])[:4], :2]
            hull = ConvexHull(neigh_top_face)
            ordered_points = neigh_top_face[hull.vertices]
            poly_neigh = Polygon(ordered_points)

            overlap_ratio = 0
            if poly_obj.is_valid and poly_neigh.is_valid:
                intersection_area = poly_obj.intersection(poly_neigh).area
                smaller_area = min(poly_obj.area, poly_neigh.area)
                overlap_ratio = intersection_area / smaller_area if smaller_area > 0 else 0

            is_on_top = abs(obj_min_z - neigh_max_z) <= on_height_thresh and overlap_ratio>=overlap_thresh

            # Find if objects lie in between them
            neighbor_centre = neighbor_coords.mean(axis=0)
            line_AB = LineString([obj_centre, neighbor_centre])
            between_boxes = []
            for cam in data:
                for other in data[cam]:
                    if other in [obj, potential_neighbor]:
                        continue
                    other_coords = np.array(data[cam][other]["bbox_3d_corners"])
                    poly_other = Polygon(other_coords)
                    if line_AB.intersects(poly_other):
                        between_boxes.append(other)

            if is_on_top:
                neighbor_dict[obj]["on"].append(potential_neighbor)
                neighbor_dict[obj]["near"].append(potential_neighbor)
            else:
                if distance_frac <= 0.10 and len(between_boxes)==0:
                    neighbor_dict[obj]["next"].append(potential_neighbor)
                neighbor_dict[obj]["near"].append(potential_neighbor)

def get_objects_in_category(category, all_keys, get_category_func):
    return [key for key in all_keys if get_category_func(key) == category]

def get_object_height(bbox_corners):
    bbox_corners = np.array(bbox_corners)
    return bbox_corners[:, 2].max() - bbox_corners[:, 2].min()

def detect_size_buckets(heights):
    heights = np.array(sorted(heights))
    n = len(heights)

    for i in range(1, n):
        small = heights[:i]
        big = heights[i:]
        if big.min() < 2 * small.max():
            continue
        small_var = small.max() / small.min() if small.min() > 0 else np.inf
        big_var = big.max() / big.min() if big.min() > 0 else np.inf

        if small_var <= 1.3 and big_var <= 1.3:
            return small, big
    return None, None

def assign_object_sizes(full_dict, category_groups):
    output_sizes = {}
    for cat, objs in category_groups.items():
        if cat.lower() == "lamp":
            for obj in objs:
                if "desk" in obj.lower():
                    output_sizes[obj] = "desk"
                elif "floor" in obj.lower():
                    output_sizes[obj] = "floor"
                else:
                    output_sizes[obj] = ""
            continue

        if len(objs) == 1:
            output_sizes[objs[0]] = ""
            continue

        heights = [get_object_height(full_dict[obj]["bbox_3d_corners"]) for obj in objs]
        small, big = detect_size_buckets(heights)

        if small is None:
            for obj in objs:
                output_sizes[obj] = ""
        else:
            for obj, h in zip(objs, heights):
                if h in small:
                    output_sizes[obj] = "short"
                elif h in big:
                    output_sizes[obj] = "tall"
                else:
                    output_sizes[obj] = ""
    return output_sizes

def is_unique_by_property(target_key, category_keys, property_names, description_data):
    target_values = tuple(description_data[target_key].get(p, '') for p in property_names)
    if any(not v for v in target_values):
        return False
    count = 0
    for key in category_keys:
        if tuple(description_data[key].get(p, '') for p in property_names) == target_values:
            count += 1
    return count == 1

def is_unique_by_near(target_key, category_keys, near_item_category, description_data):
    count = 0
    for key in category_keys:
        near_items = description_data[key].get('near', [])
        if near_item_category in near_items:
            count += 1
    return count == 1

def is_unique_by_property_and_near(target_key, category_keys, property_names, near_item_category, description_data):
    """Check if combination of intrinsic properties + near relationship is unique"""
    # First check if target has all required properties and the near relationship
    target_prop_values = tuple(description_data[target_key].get(p, '') for p in property_names)
    if any(not v for v in target_prop_values):
        return False
    target_near_items = description_data[target_key].get('near', [])
    if near_item_category not in target_near_items:
        return False

    # Count how many objects in category have the same property combination AND the same near relationship
    count = 0
    for key in category_keys:
        key_prop_values = tuple(description_data[key].get(p, '') for p in property_names)
        key_near_items = description_data[key].get('near', [])
        if key_prop_values == target_prop_values and near_item_category in key_near_items:
            count += 1
    return count == 1

def generate_unique_descriptions(description_data, get_category_func):
    outputs = {}
    all_keys = list(description_data.keys())
    category_map = {key: get_category_func(key) for key in all_keys}
    category_groups = defaultdict(list)

    for key, cat in category_map.items():
        category_groups[cat].append(key)

    for obj_key in all_keys:
        category = category_map[obj_key]
        category_keys = category_groups[category]

        if len(category_keys) == 1:
            # --- This object is the only one in its category ---
            # Build description in format: {size} {color} {object category} {neighboring object}
            obj_category = get_category_func(obj_key)
            size = description_data[obj_key].get('size', '')
            color = description_data[obj_key].get('color', '')
            near_categories = description_data[obj_key].get('near', [])

            # Build parts list in order: size, color, category, neighbor
            descriptor_parts = []
            marker_array = []
            chosen_neighbor = None

            if size:
                descriptor_parts.append(size)
                marker_array.append(1)

            if color:
                descriptor_parts.append(color)
                marker_array.append(0)

            # Add neighboring relationship if available
            neighbor_phrase = None
            if near_categories:
                neighbor_chosen = random.choice(near_categories)
                if neighbor_chosen in description_data[obj_key].get("on_which", []):
                    neighbor_phrase = f"on which a {neighbor_chosen}"
                elif neighbor_chosen in description_data[obj_key].get("on", []):
                    neighbor_phrase = f"on a {neighbor_chosen}"
                elif neighbor_chosen in description_data[obj_key].get("next", []):
                    neighbor_phrase = f"next to a {neighbor_chosen}"
                else:
                    neighbor_phrase = f"near a {neighbor_chosen}"

                if neighbor_phrase:
                    descriptor_parts.append(neighbor_phrase)
                    marker_array.append(2)
                    chosen_neighbor = neighbor_chosen

            # If no descriptors (size, color, neighbor), return empty string
            if len(descriptor_parts) == 0:
                outputs[obj_key] = ("", [], None)
            else:
                # Build final description: {size} {color} {category} {neighbor}
                final_parts = []
                # Add size and color first
                if size:
                    final_parts.append(size)
                if color:
                    final_parts.append(color)
                # Add category name (required when we have at least one descriptor)
                final_parts.append(obj_category)
                # Add neighbor phrase if present
                if neighbor_phrase:
                    final_parts.append(neighbor_phrase)

                final_desc = " ".join(final_parts)
                outputs[obj_key] = (final_desc, marker_array, chosen_neighbor)
            continue

        # --- This is the "multiple items in category" logic ---
        intrinsic_props = ['color', 'size']
        descriptor_pool = [p for p in intrinsic_props if description_data[obj_key].get(p)]
        near_categories = description_data[obj_key].get('near', [])
        sampling_pool = descriptor_pool + ['near_item'] * len(near_categories)

        if not sampling_pool:
            outputs[obj_key] = ("", [], None)
            continue

        random.shuffle(sampling_pool)
        final_descriptor_values = []
        final_descriptor_keys = []
        marker_array = []
        unique_found = False
        chosen_neighbor = None

        for item_to_sample in sampling_pool:
            if unique_found:
                break  # Stop sampling if unique

            if item_to_sample in intrinsic_props:
                # Intrinsic Property Check (Rules 3c, 3d)
                prop = item_to_sample
                prop_value = description_data[obj_key].get(prop)

                if prop not in final_descriptor_keys and prop_value:
                    temp_keys = final_descriptor_keys + [prop]

                    if is_unique_by_property(obj_key, category_keys, temp_keys, description_data):
                        # Unique combination found!
                        final_descriptor_keys = temp_keys
                        final_descriptor_values = [description_data[obj_key][k] for k in final_descriptor_keys]
                        marker_array = [0 if k == 'color' else 1 for k in final_descriptor_keys]
                        unique_found = True
                        break  # Found unique, stop and finalize
                    else:
                        # Non-unique, but add it and continue trying (Rule 3d)
                        final_descriptor_keys.append(prop)
                        final_descriptor_values.append(prop_value)
                        if prop == 'color':
                            marker_array.append(0)
                        elif prop == 'size':
                            marker_array.append(1)

            elif item_to_sample == 'near_item':
                # 'Near' Property Check (Rule 3e)
                near_categories_used = [
                    p.replace("near a ", "").strip()
                    for p in final_descriptor_values
                    if p.startswith("near a ")
                ]
                remaining_near_cats = [c for c in near_categories if c not in near_categories_used]

                chosen_cat = None
                random.shuffle(remaining_near_cats)
                for cat in remaining_near_cats:
                    # First check if near relationship alone is unique
                    if is_unique_by_near(obj_key, category_keys, cat, description_data):
                        chosen_cat = cat
                        unique_found = True
                        break
                    # If not unique alone, check if (intrinsic properties + near) is unique
                    elif final_descriptor_keys and is_unique_by_property_and_near(
                        obj_key, category_keys, final_descriptor_keys, cat, description_data
                    ):
                        chosen_cat = cat
                        unique_found = True
                        break

                if chosen_cat:
                    # Found unique 'near' descriptor (alone or with intrinsic properties), so we stop and finalize
                    final_descriptor_values.append(f"near a {chosen_cat}")
                    marker_array.append(2)
                    chosen_neighbor = chosen_cat
                    break  # Found unique, stop and finalize

        # --- Final Description Construction (Rule 3f & Blank Check) ---
        if not unique_found:
            outputs[obj_key] = ("", [], None)  # Leave blank as instructed
            continue

        # Build description in format: {size} {color} {object category} {neighboring object}
        obj_category = get_category_func(obj_key)
        size = ""
        color = ""
        neighbor_phrase = None

        # Extract size and color from final_descriptor_keys and final_descriptor_values
        for i, key in enumerate(final_descriptor_keys):
            if key == 'size':
                size = final_descriptor_values[i]
            elif key == 'color':
                color = final_descriptor_values[i]

        # Extract neighbor phrase from final_descriptor_values
        for val in final_descriptor_values:
            if val.startswith("near a ") or val.startswith("on a ") or val.startswith("next to a ") or val.startswith("on which a "):
                neighbor_phrase = val
                break

        # If we have a neighbor phrase, check if it should be "on which", "on", or "next to" instead of "near"
        if neighbor_phrase:
            obj_info = description_data[obj_key]
            on_which_items = obj_info.get('on_which', [])
            on_items = obj_info.get('on', [])
            next_items = obj_info.get('next', [])

            # Extract the neighbor item from the phrase
            near_item = neighbor_phrase.replace("near a ", "").replace("on a ", "").replace("next to a ", "").replace("on which a ", "").strip()

            # Set chosen_neighbor if it wasn't already set
            if chosen_neighbor is None:
                chosen_neighbor = near_item

            if near_item in on_which_items:
                neighbor_phrase = f"on which a {near_item}"
            elif near_item in on_items:
                neighbor_phrase = f"on a {near_item}"
            elif near_item in next_items:
                neighbor_phrase = f"next to a {near_item}"
            else:
                neighbor_phrase = f"near a {near_item}"

        # Build parts list in order: size, color, category, neighbor
        parts = []
        final_marker_array = []

        if size:
            parts.append(size)
            final_marker_array.append(1)

        if color:
            parts.append(color)
            final_marker_array.append(0)

        parts.append(obj_category)

        if neighbor_phrase:
            parts.append(neighbor_phrase)
            final_marker_array.append(2)

        final_description = " ".join(parts)
        outputs[obj_key] = (final_description, final_marker_array, chosen_neighbor)

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="./llm_detected_objects_color.json")
    parser.add_argument("--ground_truth_json", type=str, default="./visible_objects.json")
    parser.add_argument("--output_json", type=str, default="./visible_objects_with_descriptions.json")
    parser.add_argument("--full_description_json", type=str, default="./full_description.json")

    args = parser.parse_args()
    output_json_path = args.output_json

    with open(args.input_json, "r") as f:
        data = json.load(f)
    with open(args.ground_truth_json, "r") as f:
        ground_data = json.load(f)

    arr = []
    common_objs, unique_agent1, unique_agent2, all_objs = get_sets(data, ground_data)
    cam_0_visible = unique_agent1.union(common_objs)
    cam_1_visible = unique_agent2.union(common_objs)

    for cam in data:
        for item in data[cam]:
            arr.append(data[cam][item]["bbox_3d_corners"])
    all_coords = np.vstack(arr)
    room_min_dist = min(np.max(all_coords[:,0]) - np.min(all_coords[:,0]), np.max(all_coords[:,1]) - np.min(all_coords[:,1]))
    full_dict = data["camera_0_0"]
    for key in unique_agent2:
        full_dict[key] = data["camera_1_0"][key]

    # Get object colors
    colors_dict = {}
    for object in cam_0_visible:
        if "color" in data["camera_0_0"][object].keys():
            colors_dict[object] = data["camera_0_0"][object]["color"].lower().strip()
        else:
            print(f"Object {object} has no color")
            colors_dict[object] = ""
    for object in cam_1_visible:
        if "color" in data["camera_1_0"][object].keys():
            colors_dict[object] = data["camera_1_0"][object]["color"].lower().strip()
        else:
            print(f"Object {object} has no color")
            colors_dict[object] = ""

    # Get object neighbors
    neighbor_dict = {}
    get_neighboring_objects(data, room_min_dist, data["camera_0_0"], unique_agent1, unique_agent1, neighbor_dict)
    get_neighboring_objects(data, room_min_dist, data["camera_1_0"], unique_agent2, unique_agent2, neighbor_dict)
    get_neighboring_objects(data, room_min_dist, full_dict, common_objs, all_objs, neighbor_dict)

    for key in neighbor_dict:
        on_array = neighbor_dict[key]["on"]
        for on_obj in on_array:
            # Ensure on_obj is initialized in neighbor_dict
            if on_obj not in neighbor_dict:
                neighbor_dict[on_obj] = {"near":[], "next":[], "on":[], "on_which":[]}
            if key not in neighbor_dict[on_obj]["near"]:
                neighbor_dict[on_obj]["near"].append(key)
            if key not in neighbor_dict[on_obj]["on_which"]:
                neighbor_dict[on_obj]["on_which"].append(key)
            if key in neighbor_dict[on_obj]["next"]:
                neighbor_dict[on_obj]["next"].remove(key)

    for key in neighbor_dict:
        obj_near = []
        obj_on = []
        obj_next = []
        obj_on_which = []

        for obj in neighbor_dict[key]["near"]:
            if obj not in colors_dict:
                continue
            obj_near.append(f"{colors_dict[obj]} {get_category(obj)}")

        for obj in neighbor_dict[key]["on"]:
            if obj not in colors_dict:
                continue
            obj_on.append(f"{colors_dict[obj]} {get_category(obj)}")

        for obj in neighbor_dict[key]["next"]:
            if obj not in colors_dict:
                continue
            obj_next.append(f"{colors_dict[obj]} {get_category(obj)}")

        for obj in neighbor_dict[key]["on_which"]:
            if obj not in colors_dict:
                continue
            obj_on_which.append(f"{colors_dict[obj]} {get_category(obj)}")

        neighbor_dict[key]["near"] = list(set(obj_near))
        neighbor_dict[key]["on"] = list(set(obj_on))
        neighbor_dict[key]["next"] = list(set(obj_next))
        neighbor_dict[key]["on_which"] = list(set(obj_on_which))

    # Get object sizes
    all_keys = list(full_dict.keys())
    category_map = {key: get_category(key) for key in all_keys}
    category_groups = defaultdict(list)
    for key, cat in category_map.items():
        category_groups[cat].append(key)
    sizes_dict = assign_object_sizes(full_dict, category_groups)


    # Generating combined descriptions
    description_dict = {}
    for object in all_objs:
        # Defensive checks for missing keys
        if object not in colors_dict:
            print(f"Warning: Object {object} not found in colors_dict, using empty string")
            colors_dict[object] = ""
        if object not in sizes_dict:
            print(f"Warning: Object {object} not found in sizes_dict, using empty string")
            sizes_dict[object] = ""
        if object not in neighbor_dict:
            print(f"Warning: Object {object} not found in neighbor_dict, initializing empty neighbor lists")
            neighbor_dict[object] = {"near":[], "next":[], "on":[], "on_which":[]}
        info_dict = {"color" : colors_dict[object], "size": sizes_dict[object], "near":neighbor_dict[object]["near"], "next":neighbor_dict[object]["next"], "on":neighbor_dict[object]["on"], "on_which":neighbor_dict[object]["on_which"]}
        description_dict[object] = info_dict
    import pprint
    pprint.pprint(description_dict)

    final_descriptions = {"camera_0_0":{}, "camera_1_0":{}}
    for cam in ground_data:
        for obj in ground_data[cam]:
            if obj in description_dict:
                final_descriptions[cam][obj] = description_dict[obj]
    with open(args.full_description_json, "w") as f:
        json.dump(final_descriptions, f, indent=2)

    # description_dict generation
    description_outputs = generate_unique_descriptions(description_dict, get_category)
    print("Generated Descriptions (outputs dict):")
    keys_to_delete = [key for key, value in description_outputs.items() if not value or (isinstance(value, tuple) and not value[0])]

    print("Below items: insufficient descriptions: ")
    if len(keys_to_delete)==0:
        print("NONE\n")
    for key in keys_to_delete:
        print(key)

    for key in keys_to_delete:
        description_outputs[key] = ("", [], None)
    print(description_outputs)

    for key, value in description_outputs.items():
        print(f"'{key}': '{value}'")

    with open(args.input_json, "r") as f:
        data_fresh = json.load(f)
    for key in data_fresh:
        data_fresh[key] = {
            obj: data_fresh[key][obj]
            for obj in data_fresh[key]
            if obj in description_outputs
        }

    for key in data_fresh:
        for obj in data_fresh[key]:
            data_fresh[key][obj]["description"] = description_outputs[obj][0]
            data_fresh[key][obj]["description_difficulty"] = description_outputs[obj][1]
            val = None if len(description_outputs[obj])!=3 or description_outputs[obj][2] is None else description_outputs[obj][2]
            data_fresh[key][obj]["chosen_neighbor"] = val

    with open(output_json_path, "w") as f:
        json.dump(data_fresh, f, indent=2)
    print(f"Saved updated JSON to {output_json_path}")

if __name__ == "__main__":
    main()