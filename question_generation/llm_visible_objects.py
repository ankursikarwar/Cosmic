import re
import os
import json
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

from question_generation_v2.utils import get_category, allowed_categories
from question_generation_v2.llm_utils import create_vllm_client, create_openai_client, encode_image

# --- Constants ---
# It's good practice to define constants at the top
WIDTH, HEIGHT = 1280, 720
GENERATIONS = 10
TEMPERATURE = 0.8
MAX_TOKENS = 8192

# --- Helper Functions (Unchanged) ---

def flip_blender_bbox_y(bbox: List[float]) -> List[float]:
    """Flips the y-coordinates of a normalized bounding box from Blender's coordinate system."""
    x_min, y_min, x_max, y_max = bbox
    return [x_min, 1 - y_max, x_max, 1 - y_min]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def filter_visible_objects(camera_key, llm_output, original_objects):
    filtered = []
    for obj in original_objects[camera_key]:
        # print("Original Object: ", obj)
        object_dict = original_objects[camera_key][obj]
        gt_box = flip_blender_bbox_y(object_dict["bbox_2d"])
        gt_cat = get_category(object_dict["name"])

        best_iou = 0.0
        best_match = None
        for llm_obj in llm_output:
            # print("LLM Object: ", llm_obj)
            # print("GT Box: ", gt_box)
            if "bbox_2d" not in llm_obj:
                continue
            iou = compute_iou(gt_box, llm_obj["bbox_2d"])
            # print("IOU: ", iou)
            if iou > best_iou:
                best_iou = iou
                best_match = llm_obj

        if best_match and best_iou >= 0.1:
            llm_cat = best_match.get("label") or best_match.get("category")
            if llm_cat == gt_cat:
                filtered.append(object_dict)
                print("Filtered: ", filtered)

    return filtered

# --- Core Logic ---

def prepare_query(scene_path: str, camera_key: str, camera_visible_set: dict, client_name: str, model_name: str) -> dict:
    """Prepares a single LLM query for a given camera in a scene."""
    visible_list = camera_visible_set[camera_key]
    visible_list = [i for i in visible_list if i in allowed_categories]
    object_list = ", ".join(visible_list)

    # Construct image path dynamically from scene and camera key
    camera_index = camera_key.split('_')[1]
    image_path = scene_path + f"/frames/Image/camera_0/Image_{camera_index}_0_0048_0.png"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    chat_history = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(str(image_path))}", "detail": "high"}},
                {"type": "text", "text": f'''
                > Generate bounding boxes around all the salient objects in the image.
                > Locate every instance that belongs to the following categories: {object_list}.
                > If you see multiple objects of the same kind, generate a separate bounding box for each one.
                > IMPORTANT: For Plant Containers, if the container is not visible and only the plant is visible, then do not generate a bounding box for the container.
                > Report bbox coordinates in following JSON format and do not output anything other than the JSON.
                '''
                 +
                '''
                **IMPORTANT**:
                '''
                +
                '''
                > The bounding box coordinates must be normalized to a **0-1000 scale** for both the X and Y axes, where the bottom-left corner is (0, 0) and the top-right corner is (1000, 1000).
                > The format of the JSON should be like this: [{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "object_name"}, {"bbox_2d": [x_min, y_min, x_max, y_max], "label": "object_name"}, ...]
                '''
                +
                '''
                > Each bounding box must be unique (no duplicates of the same coordinates and label).
                > Stop once all salient objects are covered. By saliency, we mean objects that are clearly visible and identifiable, not just small or obscure details such as objects inside shelves or small items and decorations.
                '''}
            ]
        }
    ]

    assert 'qwen' in model_name.lower(), "Only Qwen models are supported for this task."

    if model_name.startswith("gpt"):
        return {"messages": chat_history, "max_completion_tokens": MAX_TOKENS, "n": GENERATIONS}
        # return {"messages": chat_history, "max_completion_tokens": 4096, "n": GENERATIONS, "temperature": TEMPERATURE}
    else:
        return {"messages": chat_history, "max_tokens": MAX_TOKENS, "n": GENERATIONS, "temperature": TEMPERATURE}

def run_visibility_check(scenes: List[Path], client_name: str, model_name:str, api_base:str, api_key:str):
    """
    Processes multiple scenes to detect visible objects using an LLM in a batched manner.
    """

    if client_name == "vllm":
        client = create_vllm_client(model_name=model_name, api_base=api_base, api_key=api_key)
    elif client_name == "openai":
        client = create_openai_client(model_name=model_name, api_base=api_base, api_key=api_key)
    else:
        raise ValueError(f"Invalid client specified: {client}")

    all_queries = []
    query_mapping = []
    scene_input_data = {}

    print(f"Preparing queries for {len(scenes)} scenes...")
    for scene in tqdm(scenes, desc="Preparing Scenes"):
        input_json_path = scene + "/visible_objects.json"
        if not os.path.exists(input_json_path):
            print(f"Warning: Input file not found at {input_json_path}. Skipping scene {scene}.")
            continue

        with open(input_json_path, "r") as f:
            objects = json.load(f)
        scene_input_data[scene] = objects

        camera_keys = list(objects.keys())
        camera_visible_set = {
            key: list(set(get_category(obj_name) for obj_name in objects[key]))
            for key in camera_keys
        }

        for camera_key in camera_keys:
            try:
                query = prepare_query(scene, camera_key, camera_visible_set, client_name, model_name)
                all_queries.append(query)
                query_mapping.append((scene, camera_key))
            except FileNotFoundError as e:
                print(f"Warning: {e}. Skipping camera {camera_key} in scene {scene}.")

    if not all_queries:
        print("No valid queries could be prepared. Exiting.")
        return

    # 3. Make a single batch API call
    print(f"\nSending {len(all_queries)} batched requests to the LLM...")
    all_model_outputs = client.call_chat(queries=all_queries, tqdm_desc="Detecting Objects", tqdm_enable=True)
    print("✅ Batch request completed.")
    # print(all_model_outputs)
    # 4. Process results using the mapping
    scene_outputs = {scene: {cam_key: {} for cam_key in data} for scene, data in scene_input_data.items()}

    print("\nProcessing LLM outputs...")
    for i, model_output in enumerate(tqdm(all_model_outputs, desc="Processing Outputs")):
        scene, camera_key = query_mapping[i]
        original_objects = scene_input_data[scene]

        for choice in model_output.choices:
            try:
                raw_output = choice.message.content
                # match = re.search(r"\[.*\]", raw_output, re.DOTALL)
                # print("Raw Output: ", raw_output)
                # print("Match: ", match)
                # if not match:
                    # continue

                # data = json.loads(match.group(0))
                clean_json = re.sub(r"^```json|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()

                data = json.loads(clean_json)
                # print("Data: ", data)

                # convert bboxes to image coordinates. currently it is in the range of [0, 1000]
                for item in data:
                    item['bbox_2d'] = [
                        item['bbox_2d'][0] / 1000 * WIDTH,
                        item['bbox_2d'][1] / 1000 * HEIGHT,
                        item['bbox_2d'][2] / 1000 * WIDTH,
                        item['bbox_2d'][3] / 1000 * HEIGHT
                    ]

                # from PIL import Image
                # from PIL import ImageDraw

                # # Load image
                # image = Image.open(scene + f"/frames/Image/camera_0/Image_{camera_key.split('_')[1]}_0_0048_0.png")

                # # Plot image and boxes, use ImageDraw to draw the boxes
                # draw = ImageDraw.Draw(image)
                # for box in data:
                #     draw.rectangle(
                #         [box['bbox_2d'][0], box['bbox_2d'][1], box['bbox_2d'][2], box['bbox_2d'][3]],
                #         outline='red', width=1
                #     )
                # image.save(f'./llm_detected_objects_{camera_key}.png')


                # Normalize bounding boxes
                for item in data:
                    bbox_value = None
                    if "coordinates" in item:
                        bbox_value = item["coordinates"]
                    elif "bbox" in item:
                        bbox_value = item["bbox"]
                    elif "bbox_2d" in item: # Keep for backward compatibility
                        bbox_value = item["bbox_2d"]

                    if bbox_value:
                        x_min, y_min, x_max, y_max = bbox_value
                        item["bbox_2d"] = [
                            x_min / WIDTH,
                            y_min / HEIGHT,
                            x_max / WIDTH,
                            y_max / HEIGHT
                        ]

                filtered = filter_visible_objects(camera_key, data, original_objects)

                # Add new, unique objects to the output
                current_ids = set(scene_outputs[scene][camera_key].keys())
                for new_object in filtered:
                    if new_object["name"] not in current_ids:
                        scene_outputs[scene][camera_key][new_object["name"]] = new_object
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Error processing a choice for scene {scene}, camera {camera_key}: {e}")

    # 5. Save results to each scene's directory
    print("\nSaving results...")
    for scene, output_data in tqdm(scene_outputs.items(), desc="Saving Files"):
        output_json_path = scene + "/llm_detected_objects.json"
        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=2)

    print(f"\n✅ Finished processing. Saved results to '{str(output_json_path)}' in each scene directory.")