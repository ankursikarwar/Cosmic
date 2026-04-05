#!/usr/bin/env python3
"""
Aggregate map questions from format1 and format2 JSON files.
This script reads map question outputs from map_godbless.py and creates
aggregated versions compatible with the aggregate_data stage in datagen_pipeline.py.
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional


def extract_scene_info(scene_path: str) -> tuple:
    """
    Extract room_part and scene_id from scene path.
    Assumes path structure: .../room_part/scene_id/...
    """
    parts = Path(scene_path).parts
    if len(parts) >= 2:
        scene_id = parts[-1]
        room_part = parts[-2]
        return room_part, scene_id
    # Fallback: try to parse from string
    scene_str = str(scene_path)
    if "/" in scene_str:
        parts = scene_str.rstrip("/").split("/")
        scene_id = parts[-1]
        room_part = parts[-2] if len(parts) >= 2 else "unknown"
        return room_part, scene_id
    return "unknown", scene_str


def create_aggregated_sample(
    question_f1: Dict[str, Any],
    scene_path: str,
    question_idx: int,
    answerer_goal: str,
    helper_goal: str,
    sample_id_prefix: str = "cognitive_mapping"
) -> Dict[str, Any]:
    """
    Create an aggregated sample dict from format1 questions.
    Includes all fields required by aggregate_data stage and preserves existing fields.
    """
    room_part, scene_id = extract_scene_info(scene_path)

    asking_to = question_f1.get("asking_to", "agent_1")
    is_correct = question_f1.get("is_correct", False)

    gt_idx = 0 if is_correct else 1
    gt_answer = "Yes" if is_correct else "No"

    # Create sample_id
    sample_id = f"{sample_id_prefix}_{question_idx:06d}"

    # Build image paths
    user_1_image_local_path = os.path.join(scene_path, "frames/Image/camera_0/Image_0_0_0048_0.png")
    user_2_image_local_path = os.path.join(scene_path, "frames/Image/camera_0/Image_1_0_0048_0.png")

    # Build image URLs (relative paths starting with /img/)
    user_1_image = f"/img/{room_part}/{scene_id}/frames/Image/camera_0/Image_0_0_0048_0.png"
    user_2_image = f"/img/{room_part}/{scene_id}/frames/Image/camera_0/Image_1_0_0048_0.png"

    # Get map image path if available
    map_image_path = question_f1.get("image_path", None)
    global_map_image = map_image_path if map_image_path and os.path.exists(map_image_path) else None

    # Build the aggregated sample dict
    sample_dict = {
        # Required fields from aggregate_data
        "sample_id": sample_id,
        "question_type": "cognitive_mapping",
        "room_part": room_part,
        "scene_id": scene_id,
        "global_map_image": global_map_image,
        "user_1_image_local_path": user_1_image_local_path,
        "user_2_image_local_path": user_2_image_local_path,
        "user_1_image": user_1_image,
        "user_2_image": user_2_image,
        "user_1_goal": answerer_goal if asking_to == "agent_1" else helper_goal,
        "user_2_goal": answerer_goal if asking_to == "agent_2" else helper_goal,
        "user_1_question": "Is this top-down map of the room correct?" if asking_to == "agent_1" else None,
        "user_2_question": "Is this top-down map of the room correct?" if asking_to == "agent_2" else None,
        "options_user_1": ["Yes", "No"] if asking_to == "agent_1" else None,
        "options_user_2": ["Yes", "No"] if asking_to == "agent_2" else None,
        "user_1_gt_answer_idx": gt_idx if asking_to == "agent_1" else None,
        "user_2_gt_answer_idx": gt_idx if asking_to == "agent_2" else None,
        "user_1_gt_answer_text": gt_answer if asking_to == "agent_1" else None,
        "user_2_gt_answer_text": gt_answer if asking_to == "agent_2" else None,
        "user_1_perception": os.path.join(scene_path, "agent_1_input.txt"),
        "user_2_perception": os.path.join(scene_path, "agent_2_input.txt"),

        # Fields from map output (preserve all existing fields)
        "question": "Is this top-down map of the room correct?",
        "question_both_views": "Is this top-down map of the room correct?",
        "asking_to": asking_to,
        "correct_index": gt_idx,
        "option_categories": ["Yes", "No"],
        "map_image_path": map_image_path,
    }

    return sample_dict


def aggregate_cognitive_mapping_for_scene(
    scene_path: str,
    answerer_goal: str,
    helper_goal: str,
    sample_id_counter: int,
    sample_id_prefix: str = "cognitive_mapping"
) -> tuple:
    """
    Aggregate map questions for a single scene.
    Returns (aggregated_list, next_sample_id_counter)
    """
    format1_path = os.path.join(scene_path, "cognitive_mapping.json")

    if not os.path.exists(format1_path):
        return [], sample_id_counter

    with open(format1_path, "r") as f:
        questions_f1 = json.load(f)

    aggregated_f1 = []
    current_counter = sample_id_counter

    # Process questions for each agent
    for agent_key in ["agent_1", "agent_2"]:
        if agent_key not in questions_f1:
            continue

        agent_questions_f1 = questions_f1[agent_key]
        num_questions = len(agent_questions_f1)

        for q_idx in range(num_questions):
            q_f1 = agent_questions_f1[q_idx]

            # Skip empty questions
            if not q_f1:
                continue

            # Create aggregated sample with unique sample_id
            sample_dict = create_aggregated_sample(
                q_f1, scene_path, current_counter,
                answerer_goal, helper_goal, sample_id_prefix
            )

            # Skip if sample_dict is empty or invalid
            if not sample_dict or not sample_dict.get("sample_id"):
                continue

            aggregated_f1.append(sample_dict)
            current_counter += 1

    return aggregated_f1, current_counter


def find_all_scenes(base_dir: str) -> List[str]:
    """
    Find all scene directories that contain cognitive_mapping subdirectories.
    """
    scenes = []
    base_path = Path(base_dir)

    # Look for directories containing cognitive_mapping subdirectory
    for scene_dir in tqdm(base_path.rglob("cognitive_mapping")):
        # Get parent directory (the scene directory)
        scene_path = scene_dir.parent
        if scene_path.is_dir():
            scenes.append(str(scene_path))

    return sorted(list(set(scenes)))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate map questions from format1 and format2 JSON files"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing scene directories with cognitive_mapping subdirectories"
    )
    parser.add_argument(
        "--output_format1_json",
        type=str,
        default="./dataset_cognitive_mapping_format1.json",
        help="Output path for aggregated format1 JSON"
    )
    parser.add_argument(
        "--output_format2_json",
        type=str,
        default="./dataset_cognitive_mapping_format2.json",
        help="Output path for aggregated format2 JSON"
    )
    parser.add_argument(
        "--sample_id_prefix",
        type=str,
        default="map",
        help="Prefix for sample IDs (default: 'map')"
    )
    parser.add_argument(
        "--answerer_goal",
        type=str,
        default="Communicate with your partner to answer the following question correctly.",
        help="Goal text for the answerer agent"
    )
    parser.add_argument(
        "--helper_goal",
        type=str,
        default="Communicate with your partner to help them answer their question correctly.",
        help="Goal text for the helper agent"
    )

    args = parser.parse_args()

    # Find all scenes with map questions
    print(f"Searching for scenes with map questions in: {args.base_dir}")
    scenes = find_all_scenes(args.base_dir)
    print(f"Found {len(scenes)} scenes with map questions")

    if not scenes:
        print("No scenes found with map questions. Exiting.")
        return

    # Aggregate questions from all scenes
    all_aggregated_f1 = []
    sample_id_counter = 0
    skipped_scenes = 0
    skipped_questions = 0

    for scene_path in tqdm(scenes, desc="Aggregating map questions"):
        try:
            aggregated_f1, sample_id_counter = aggregate_cognitive_mapping_for_scene(
                scene_path, args.answerer_goal, args.helper_goal, sample_id_counter, args.sample_id_prefix
            )
            print(f"Aggregated {len(aggregated_f1)} questions from scene: {scene_path}")

            if aggregated_f1:
                # Filter out any remaining empty entries
                filtered_f1 = [q for q in aggregated_f1 if q and q.get("sample_id") and q.get("question")]

                if filtered_f1:
                    all_aggregated_f1.extend(filtered_f1)
                else:
                    skipped_questions += len(aggregated_f1)
            else:
                skipped_scenes += 1

        except Exception as e:
            print(f"Error processing scene {scene_path}: {e}")
            import traceback
            traceback.print_exc()
            skipped_scenes += 1
            continue

    # Save aggregated results
    print(f"\nSaving aggregated to: {args.output_format1_json}")
    os.makedirs(os.path.dirname(args.output_format1_json) if os.path.dirname(args.output_format1_json) else ".", exist_ok=True)
    with open(args.output_format1_json, "w") as f:
        json.dump(all_aggregated_f1, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total aggregated questions: {len(all_aggregated_f1)}")
    print(f"Processed {len(scenes)} scenes")
    if skipped_scenes > 0:
        print(f"Skipped {skipped_scenes} scenes (no valid questions)")
    if skipped_questions > 0:
        print(f"Skipped {skipped_questions} empty/invalid questions")


if __name__ == "__main__":
    main()
