import os
import json
import re
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np
from tqdm import tqdm
from question_generation_v2.utils import get_category
from question_generation_v2.llm_utils import create_vllm_client, create_openai_client, encode_image

# --- Constants ---
WIDTH, HEIGHT = 1280, 720
PADDING = 20
MAX_TOKENS = 4096

# --- Helper Functions ---

def encode_pil_image(image: Image.Image, format="PNG") -> str:
    """Encodes a PIL Image as base64."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def prepare_query(prompt: str, model_name: str, image: Image.Image = None) -> dict:
    """Prepares a query dictionary for text-only or multimodal API calls."""
    content = [{"type": "text", "text": prompt}]
    if image:
        content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_pil_image(image)}", "detail": "high"},
        })

    chat_history = [{"role": "user", "content": content}]

    if model_name.startswith("gpt"):
        return {"messages": chat_history, "max_completion_tokens": MAX_TOKENS}
    return {"messages": chat_history, "max_tokens": MAX_TOKENS}

def set_color_and_cache(obj_data, found_colors_cache, object_name, color, source, processed_objects_set, cam_key):
    """
    Updates the object's color, saves it to the scene-wide cache,
    and marks the object as processed for that specific camera.
    """
    if obj_data:
        obj_data["color"] = color
        obj_data["color_source"] = source

    if object_name not in found_colors_cache:
        found_colors_cache[object_name] = {"color": color, "color_source": source}

    if processed_objects_set is not None and cam_key:
        processed_objects_set.add((cam_key, object_name))

# --- Core Function ---

def run_color_detection(scenes, client_name, model_name, api_base, api_key):
    """
    Processes multiple scenes in batch to identify the dominant color of each detected object,
    ensuring consistency across cameras within the same scene.
    """

    # --- Client Initialization ---
    if client_name == "vllm":
        client = create_vllm_client(model_name=model_name, api_base=api_base, api_key=api_key)
    elif client_name == "openai":
        client = create_openai_client(model_name=model_name, api_base=api_base, api_key=api_key)
    else:
        raise ValueError(f"Invalid client specified: {client_name}")

    # --- Prompts ---
    base_prompt = """Identify the major color of the {} object in the provided image. Your answer should follow the following instructions:
> Focus on the {} object itself. Ignore other objects when identifying the color.
> Do not take the background color into account. Just focus on the object itself.
> Use one-word colors.
> Just output the color of the object in your response, and nothing else."""

    lamp_prompt = "\n> For lamps, always select the color of the lamp shade from the following list: [beige, pink, green, black], not the lamp base."
    desk_prompt = "\n> For desks, always select the color of the desk top from the following list: [white, black], not the desk legs."
    window_prompt = "\n> For windows, if the curtains are present, give the color of the curtains."
    bed_prompt = "\n> For beds, give the color of the bed frame which refers to the support structue on which the mattress is kept. Do not include the color of the mettress or bed sheet in your response."
    plant_container_prompt = "\n> For plant containers, give the color of the plant container. Do not include the color of the plant in your response."

    rgb_mapping_prompt_base = """You are given a list of RGB color values. Your task is to map each RGB value to the nearest human-perceptible color name.
Instructions:
1. Use simple, one-word color names only (e.g., "red", "blue", "green", "beige", "black", "white", "brown", "red", "yellow", etc.)
2. Do NOT use compound colors like "bluish-green", "reddish-brown", "cyan", "burgundy", "greenish", "maroon", "teal" etc. Use simple, common color names.
3. Map each RGB value to the closest matching color from common human-perceptible colors.
4. Return your response as a JSON object mapping each RGB array to its color name, in the format: {{"[R, G, B]": "color_name", ...}}
5. Example: If RGB [255, 0, 0] is given, respond with {{"[255, 0, 0]": "red"}}
6. IMPORTANT: If two colors in the list are very similar (both perceptually and RGB-wise), then assign them the same color name. For example, if the list contains rgb values like [33, 33, 32] and [67, 54, 43], then both should be mapped to "brown" because these two colors are peceptually similar to humans as well as close in RGB space in absolute terms.

Here are the RGB values to map:
"""

    # Stores the single source of truth for each scene
    scene_data_map = {}
    # { scene_path: { "data": {...}, "asset_json": {...}, "blender_colors": {...}, "found_colors": {}, "processed_objects": set() } }

    rgb_mapping_queries = []
    rgb_mapping_scene_data = [] # List of (scene, rgb_to_objects) tuples

    # ====================================================================
    # STAGE 1: Data Loading & 'blender_colors.json' Processing
    # ====================================================================
    print("=" * 60)
    print("STAGE 1: Loading data and processing 'blender_colors.json'")
    print("=" * 60)

    print(f"Preparing Stage 1 data from {len(scenes)} scenes...")
    for scene in tqdm(scenes, desc="Preparing Stage 1"):
        input_json_path = os.path.join(scene, "llm_detected_objects.json")
        if not os.path.exists(input_json_path):
            print(f"Warning: {input_json_path} not found. Skipping scene {scene}.")
            continue

        with open(input_json_path, "r") as f:
            data = json.load(f)

        asset_path = os.path.join(scene, "coarse/asset_parameters.json")
        asset_json = {}
        if os.path.exists(asset_path):
            with open(asset_path, "r") as f:
                asset_json = json.load(f)

        blender_colors_path = os.path.join(scene, "blender_colors.json")
        blender_colors = {}
        if os.path.exists(blender_colors_path):
            with open(blender_colors_path, "r") as f:
                blender_colors = json.load(f)

        # Initialize this scene's central data store
        scene_data_map[scene] = {
            "data": data,
            "asset_json": asset_json,
            "blender_colors": blender_colors,
            "found_colors": {}, # Scene-wide color cache
            "processed_objects": set() # Set of (cam_key, object_name)
        }

        scene_info = scene_data_map[scene]
        found_colors = scene_info["found_colors"]
        processed_objects = scene_info["processed_objects"]

        rgb_to_objects = {} # Maps RGB tuple to list of (cam_key, object_name, obj_data, window_type)

        for cam_key, cam_data in data.items():
            for object_name, obj_data in cam_data.items():

                # 1. Skip irrelevant categories
                skip_words = ("art", "monitor", "fridge", "oven")
                name = object_name.lower().strip()

                if any(w in name for w in skip_words) or name == "t v":
                    set_color_and_cache(obj_data, found_colors, object_name, "", "not_applicable", processed_objects, cam_key)
                    continue

                with open(os.path.join(scene, "run_pipeline.sh"), "r") as f:
                    text = f.read()

                if "Kitchen" in text and "sink" in name:
                    print(f"Skipping sink color in kitchen scene {scene}")
                    set_color_and_cache(obj_data, found_colors, object_name, "", "not_applicable", processed_objects, cam_key)
                    continue

                # 2. Check scene-wide cache (in case it was set by a previous camera)
                if object_name in found_colors:
                    cached = found_colors[object_name]
                    set_color_and_cache(obj_data, found_colors, object_name, cached["color"], cached["color_source"], processed_objects, cam_key)
                    continue

                # 3. Check blender_colors.json
                object_id = obj_data.get("id")
                if object_id and isinstance(object_id, str):
                    object_id = object_id.strip()

                if object_id and object_id in blender_colors:
                    color_entry = blender_colors[object_id].get("color")

                    window_type = None
                    if get_category(object_name) == "Window":
                        shutter = asset_json.get(object_id, {}).get("Shutter", False)
                        window_type = "shutter" if shutter else "curtain"

                    if isinstance(color_entry, str):
                        # Direct string color
                        color_final = color_entry
                        if window_type:
                            color_final = f"{color_final} {window_type}"

                        set_color_and_cache(obj_data, found_colors, object_name, color_final, "blender_colors_string", processed_objects, cam_key)
                        continue

                    elif isinstance(color_entry, list) and len(color_entry) == 3:
                        # RGB array - collect for batch mapping
                        rgb_tuple = tuple(color_entry)
                        if rgb_tuple not in rgb_to_objects:
                            rgb_to_objects[rgb_tuple] = []
                        rgb_to_objects[rgb_tuple].append((cam_key, object_name, obj_data, window_type))
                        continue

        # Create RGB mapping query for this scene
        if rgb_to_objects:
            rgb_list_str = "\n".join([f"[{rgb[0]}, {rgb[1]}, {rgb[2]}]" for rgb in rgb_to_objects.keys()])
            rgb_mapping_prompt = rgb_mapping_prompt_base + rgb_list_str
            rgb_query = prepare_query(rgb_mapping_prompt, model_name)
            rgb_mapping_queries.append(rgb_query)
            rgb_mapping_scene_data.append((scene, rgb_to_objects))

    # ====================================================================
    # STAGE 2: RGB Batch Mapping (1st API Call)
    # ====================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Running RGB-to-Color-Name Mapping")
    print("=" * 60)

    if rgb_mapping_queries:
        print(f"Sending {len(rgb_mapping_queries)} RGB mapping queries (one per scene)...")
        rgb_outputs = client.call_chat(queries=rgb_mapping_queries, tqdm_enable=True)
        print("✅ RGB mapping complete.")

        for i, response in enumerate(rgb_outputs):
            scene, rgb_to_objects = rgb_mapping_scene_data[i]
            scene_info = scene_data_map[scene]
            found_colors = scene_info["found_colors"]
            processed_objects = scene_info["processed_objects"]

            try:
                response_text = response.choices[0].message.content.strip()
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    rgb_mapping = json.loads(json_match.group(0))
                    for key_str, color_name in rgb_mapping.items():
                        rgb_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+)\]', key_str)
                        if rgb_match:
                            rgb_tuple = (int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3)))
                            color_name_clean = color_name.strip().lower()

                            if rgb_tuple in rgb_to_objects and color_name_clean:
                                for cam_key, object_name, obj_data, window_type in rgb_to_objects[rgb_tuple]:
                                    color_final = color_name_clean
                                    if window_type:
                                        color_final = f"{color_name_clean} {window_type}"

                                    set_color_and_cache(obj_data, found_colors, object_name, color_final, "blender_colors_rgb", processed_objects, cam_key)
                                    print(f"RGB mapping: {scene} | {cam_key} | {object_name}: RGB {rgb_tuple} -> {color_final}")

            except Exception as e:
                print(f"Warning: Failed to parse RGB mapping for scene {scene}: {e}")
    else:
        print("No RGB mapping queries to process.")

    # ====================================================================
    # STAGE 3: Asset Parameter Check & VLM Query Preparation
    # ====================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Checking 'asset_parameters.json' and preparing VLM queries")
    print("=" * 60)

    all_vlm_queries = []
    query_to_object_map = [] # List of (scene, object_name)
    queued_vlm_objects = set() # (scene, object_name) tuples to avoid duplicate queries

    print(f"Preparing Stage 3 queries from {len(scenes)} scenes...")
    for scene in tqdm(scenes, desc="Preparing Stage 3"):
        if scene not in scene_data_map:
            continue

        scene_info = scene_data_map[scene]
        data = scene_info["data"]
        asset_json = scene_info["asset_json"]
        found_colors = scene_info["found_colors"]
        processed_objects = scene_info["processed_objects"]

        camera_file_mapping = {
            "camera_0_0": os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png"),
            "camera_1_0": os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png")
        }

        # Load base images for this scene
        base_images = {}
        for cam_key, image_path in camera_file_mapping.items():
            if os.path.exists(image_path):
                base_images[cam_key] = Image.open(image_path)

        for cam_key, cam_data in data.items():
            if cam_key not in base_images:
                continue # Skip cameras with no image

            base_image = base_images[cam_key]

            for object_name, obj_data in cam_data.items():

                # 1. Skip if already processed
                if (cam_key, object_name) in processed_objects:
                    continue

                # 2. Check cache (propagation check)
                if object_name in found_colors:
                    cached = found_colors[object_name]
                    set_color_and_cache(obj_data, found_colors, object_name, cached["color"], cached["color_source"], processed_objects, cam_key)
                    continue

                # 3. Check asset_parameters.json
                category = get_category(object_name)
                object_id = obj_data.get("id")
                if object_id and isinstance(object_id, str):
                    object_id = object_id.strip()

                color_final = None
                source = "asset_parameters"

                if object_id:
                    if category in ("Cabinet", "Shelf"):
                        color_present = asset_json.get(object_id, {}).get("color", False)
                        if color_present and isinstance(color_present, str):
                            color_present = color_present.strip().lower()
                            if color_present == "wood":
                                color_final = "light brown"
                            elif color_present == "black_wood":
                                color_final = "dark brown"
                            elif color_present in ["white", "red", "yellow", "blue", "green"]:
                                color_final = color_present

                    elif category == "Desk":
                        top_material = asset_json.get(object_id, {}).get("top_material", False)
                        if "white" in str(top_material).lower(): color_final = "white"
                        elif "black" in str(top_material).lower(): color_final = "black"

                if color_final:
                    # Update the scene-wide cache first
                    if object_name not in found_colors:
                        found_colors[object_name] = {"color": color_final, "color_source": source}

                    # Propagate this color to all cameras in the scene
                    for other_cam_key, other_cam_data in data.items():
                        if object_name in other_cam_data and (other_cam_key, object_name) not in processed_objects:
                            obj_to_update = other_cam_data[object_name]
                            set_color_and_cache(obj_to_update, found_colors, object_name, color_final, source, processed_objects, other_cam_key)
                    continue

                # 4. Prepare for VLM query (if not found in assets)
                if (scene, object_name) in queued_vlm_objects:
                    continue # Already queued by another camera, will be propagated later

                bbox_2d = obj_data["bbox_2d"]
                x_min = max(0, (bbox_2d[0] * WIDTH) - PADDING)
                y_min = max(0, ((1 - bbox_2d[3]) * HEIGHT) - PADDING)
                x_max = min(WIDTH, (bbox_2d[2] * WIDTH) + PADDING)
                y_max = min(HEIGHT, ((1 - bbox_2d[1]) * HEIGHT) + PADDING)

                cropped_image = base_image.crop((x_min, y_min, x_max, y_max))

                prompt = base_prompt.format(category, category)
                if category == "Lamp": prompt += lamp_prompt
                elif category == "Desk": prompt += desk_prompt
                elif category == "Window": prompt += window_prompt
                elif category == "Bed": prompt += bed_prompt
                elif category == "Plant Container": prompt += plant_container_prompt

                query = prepare_query(prompt, model_name, image=cropped_image)
                all_vlm_queries.append(query)
                # We only need to map the query back to the scene and object_name
                query_to_object_map.append((scene, object_name))
                queued_vlm_objects.add((scene, object_name))

    # ====================================================================
    # STAGE 4: VLM Batch Processing & Propagation (2nd API Call)
    # ====================================================================
    print("\n" + "=" * 60)
    print("STAGE 4: Running VLM Image-Crop Queries")
    print("=" * 60)

    if not all_vlm_queries:
        print("No Stage 4 VLM queries needed.")
    else:
        print(f"Sending a batch of {len(all_vlm_queries)} VLM color queries...")
        all_outputs = client.call_chat(queries=all_vlm_queries, tqdm_enable=True)
        print("✅ VLM color detection complete.")

        print("Updating object colors and propagating results...")
        for i, response in enumerate(all_outputs):
            try:
                color = response.choices[0].message.content.strip().lower()
                if not color or color in ["none", "null", "empty"]:
                    color = "unknown"
            except Exception:
                color = "unknown"

            scene, object_name = query_to_object_map[i]
            source = "image_crop"

            scene_info = scene_data_map[scene]
            data = scene_info["data"]
            asset_json = scene_info["asset_json"]
            found_colors = scene_info["found_colors"]
            processed_objects = scene_info["processed_objects"]

            # For windows, append shutter/curtain information from asset_parameters
            window_type = None
            if get_category(object_name) == "Window":
                # Get object_id from any camera's data (they should all have the same id)
                object_id = None
                for cam_key, cam_data in data.items():
                    if object_name in cam_data:
                        object_id = cam_data[object_name].get("id")
                        if object_id and isinstance(object_id, str):
                            object_id = object_id.strip()
                        break

                if object_id:
                    shutter = asset_json.get(object_id, {}).get("Shutter", False)
                    window_type = "shutter" if shutter else "curtain"
                    if window_type:
                        color = f"{color} {window_type}"

            # Update the scene-wide cache first
            if object_name not in found_colors:
                found_colors[object_name] = {"color": color, "color_source": source}
                print(f"VLM result: {scene} | {object_name}: {color}")

            # Propagate this color to all cameras in the scene
            for cam_key, cam_data in data.items():
                if object_name in cam_data and (cam_key, object_name) not in processed_objects:
                    obj_to_update = cam_data[object_name]
                    set_color_and_cache(obj_to_update, found_colors, object_name, color, source, processed_objects, cam_key)

    # ====================================================================
    # STAGE 5: Save Results
    # ====================================================================
    print("\n" + "=" * 60)
    print("STAGE 5: Saving updated JSON files")
    print("=" * 60)

    for scene in tqdm(scenes, desc="Saving"):
        if scene not in scene_data_map:
            continue

        # Get the modified data (the single source of truth)
        data = scene_data_map[scene]["data"]

        # Apply final defaults for any object that was truly missed
        for cam_key, cam_objs in data.items():
            for obj_name, obj_data in cam_objs.items():
                obj_data.setdefault("color", "")
                obj_data.setdefault("color_source", "unknown")

        output_path = os.path.join(scene, "llm_detected_objects_colors.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    print("\n✅ Color detection finished for all scenes.")