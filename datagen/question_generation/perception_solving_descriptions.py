import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import numpy as np
from utils import get_category

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Solve perspective inputs."
    )
    parser.add_argument(
        "--visible_objects_json",
        type=str,
        default="./llm_detected_objects.json",
        help="Path to the visible objects JSON file.",
    )
    parser.add_argument(
        "--agent_1_file",
        type=str,
        default="./agent_1_input.txt",
        help="Path to the agent 1 output JSON file."
    )
    parser.add_argument(
        "--agent_2_file",
        type=str,
        default="./agent_2_input.txt",
        help="Path to the agent 2 output JSON file."
    )
    args = parser.parse_args()

    with open(args.visible_objects_json,  "r") as f:
        visible_data = json.load(f)

    final_strings = {}
    for cam in visible_data:
        cam_str = ""
        for obj in visible_data[cam]:
            obj_category = get_category(obj)
            bbox_2d = visible_data[cam][obj]["bbox_2d"]
            bbox_2d = [round(min(max(item, 0), 1), 2) for item in bbox_2d]
            strv = f"{obj_category}: {bbox_2d}".lower()
            cleaned = ' '.join(strv.split())
            cam_str += cleaned + "\n"
        final_strings[cam] = cam_str

    with open(args.agent_1_file, "w") as f:
        f.write(final_strings["camera_0_0"])
    with open(args.agent_2_file, "w") as f:
        f.write(final_strings["camera_1_0"])