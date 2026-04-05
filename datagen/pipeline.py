import os
import sys
import subprocess
import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import time
import json
from tqdm import tqdm
from PIL import Image
import wandb
from dotenv import load_dotenv
load_dotenv()

from datagen.scene_filtering import filter_scenes
from datagen.question_generation.llm_visible_objects import run_visibility_check
from datagen.question_generation.get_color_info import run_color_detection
from datagen.question_generation.paraphrase_questions import paraphrase_across_scenes
from datagen.question_generation.aggregate_map_questions import aggregate_cognitive_mapping_for_scene

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'datagen_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatagenPipeline:
    def __init__(
        self,
        base_dir: str,
        scene_datafile: str,
        dry_run: bool = False,
        stages_to_run: List[str] = None,
        max_scenes: int = None,
        seed: int = 42,
        overwrite_files: bool = False,
        client_scene_filtering: str = "openai",
        model_name_scene_filtering: str = "gpt-4.1-mini",
        api_base_scene_filtering: str = "https://api.openai.com/v1",
        client_color: str = "openai",
        model_name_color: str = "gpt-4.1-mini",
        api_base_color: str = "https://api.openai.com/v1",
        client_vis_objects: str = "vllm",
        model_name_vis_objects: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        api_base_vis_objects: str = "http://localhost:4877/v1",
        api_key: str = "openai_key.txt",
        log_wandb: bool = False,
        client_paraphrase: str = "openai",
        model_name_paraphrase: str = "gpt-4o-mini",
        api_base_paraphrase: str = "https://api.openai.com/v1",
        question_version: str = "V4",
    ):
        self.base_dir = Path(base_dir)
        self.scene_datafile = scene_datafile
        self.dry_run = dry_run
        self.stages_to_run = stages_to_run
        self.max_scenes = max_scenes
        self.seed = seed
        self.client_scene_filtering = client_scene_filtering
        self.model_name_scene_filtering = model_name_scene_filtering
        self.api_base_scene_filtering = api_base_scene_filtering
        self.client_color = client_color
        self.model_name_color = model_name_color
        self.api_base_color = api_base_color
        self.client_vis_objects = client_vis_objects
        self.model_name_vis_objects = model_name_vis_objects
        self.api_base_vis_objects = api_base_vis_objects
        self.api_key = api_key
        self.log_wandb = log_wandb
        self.client_paraphrase = client_paraphrase
        self.model_name_paraphrase = model_name_paraphrase
        self.api_base_paraphrase = api_base_paraphrase
        self.overwrite_files = overwrite_files
        self.question_version = question_version
        self.wandb_dict = {
            "num_global_counting_questions": 0,
            "num_anchor_recognition_questions": 0,
            "num_relative_distance_questions": 0,
            "num_relative_direction_questions": 0,
            "num_total_questions": 0,
            "global_counting_scenes": [],
            "anchor_recognition_scenes": [],
            "relative_distance_scenes": [],
            "relative_direction_scenes": [],
            "no_questions_scenes": []
        }

        self.processed_scenes = 0
        self.failed_scenes = {}
        self.successful_scenes = []
        self.accepted_scenes = 0
        self.rejected_scenes = 0

        self.script_dir = Path(__file__).parent
        self.question_generation_dir = Path(__file__).parent / "question_generation"
        import shutil
        local_blender = self.question_generation_dir / "blender-4.5.3-linux-x64" / "blender"
        self.blender_executable = str(local_blender) if local_blender.exists() else shutil.which("blender")
        self.get_object_info_script = self.question_generation_dir / "get_object_info.py"
        self.get_camera_info_script = self.question_generation_dir / "get_camera_info.py"
        self.get_blender_color_info_script = self.question_generation_dir / "get_blender_color.py"
        self.llm_visible_objects_script = self.question_generation_dir / "llm_visible_objects.py"
        self.bound_objects_script = self.question_generation_dir / "bound_objects.py"
        self.obj_color_info_script = self.question_generation_dir / "get_color_info.py"
        self.generate_descriptions_script = self.question_generation_dir / "generate_descriptions.py"
        self.generate_questions_script = self.question_generation_dir / "generate_questions.py"
        self.generate_maps_script = self.question_generation_dir / "map_gen.py"
        self.generate_paraphrase_script = self.question_generation_dir / "paraphrase_questions.py"
        self.solve_perception_script = self.question_generation_dir / "perception_solving_descriptions.py"

        self._check_scripts()

    def find_all_scenes(self):
        scenes_list = []

        if self._check_dir_type(self.base_dir) == "scene":
            logger.info(f"Processing single scene directory: {self.base_dir}")

            if self._check_scene_complete(self.base_dir):
                logger.info(f"Scene {self.base_dir} is complete")
                scenes_list.append(str(self.base_dir))
            else:
                logger.info(f"Scene {self.base_dir} is not complete")

        elif self._check_dir_type(self.base_dir) == "multiple_scenes_one_roomtype":
            logger.info(f"Processing multiple scenes one roomtype directory: {self.base_dir}")

            for item in self.base_dir.iterdir():
                if self._check_scene_complete(item):
                    logger.info(f"Scene {item} is complete")
                    scenes_list.append(str(item))

        elif self._check_dir_type(self.base_dir) == "multiple_scenes_multiple_roomtypes":
            logger.info(f"Processing multiple scenes multiple roomtypes directory: {self.base_dir}")

            for item in self.base_dir.iterdir():
                for subitem in item.iterdir():
                    if self._check_scene_complete(subitem):
                        logger.info(f"Scene {subitem} is complete")
                        scenes_list.append(str(subitem))

        elif self._check_dir_type(self.base_dir) == "multiple_folders_multiple_scenes_multiple_roomtypes":
            logger.info(f"Processing multiple folders multiple scenes multiple roomtypes directory: {self.base_dir}")

            for item in self.base_dir.iterdir():
                for subitem in item.iterdir():
                    for subsubitem in subitem.iterdir():
                        if self._check_scene_complete(subsubitem):
                            logger.info(f"Scene {subsubitem} is complete")
                            scenes_list.append(str(subsubitem))

        logger.info(f"Found {len(scenes_list)} complete scenes")
        return scenes_list

    def _check_dir_type(self, path: Path) -> str:
        logger.info(f"Single scene: Checking directory type for {path}")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        frame_path = os.path.join(path, "frames", "Image", "camera_0")
        if os.path.exists(os.path.join(frame_path, "Image_0_0_0048_0.png")) and os.path.exists(os.path.join(frame_path, "Image_1_0_0048_0.png")):
            return "scene"

        for item in path.iterdir():
            frame_path = os.path.join(item, "frames", "Image", "camera_0")
            if os.path.exists(os.path.join(frame_path, "Image_0_0_0048_0.png")) and os.path.exists(os.path.join(frame_path, "Image_1_0_0048_0.png")):
                return "multiple_scenes_one_roomtype"

        for item in path.iterdir():
            for subitem in item.iterdir():
                frame_path = os.path.join(subitem, "frames", "Image", "camera_0")
                if os.path.exists(os.path.join(frame_path, "Image_0_0_0048_0.png")) and os.path.exists(os.path.join(frame_path, "Image_1_0_0048_0.png")):
                    return "multiple_scenes_multiple_roomtypes"

        for item in path.iterdir():
            for subitem in item.iterdir():
                for subsubitem in subitem.iterdir():
                    frame_path = os.path.join(subsubitem, "frames", "Image", "camera_0")
                    if os.path.exists(os.path.join(frame_path, "Image_0_0_0048_0.png")) and os.path.exists(os.path.join(frame_path, "Image_1_0_0048_0.png")):
                        return "multiple_folders_multiple_scenes_multiple_roomtypes"

        raise ValueError(f"Path {path} is not a valid directory")

    def _check_scene_complete(self, path: Path) -> bool:
        required_files_frames = [
            "Image_0_0_0048_0.png",
            "Image_1_0_0048_0.png"
        ]

        for file in required_files_frames:
            if not os.path.exists(os.path.join(path, "frames", "Image", "camera_0", file)):
                return False

        # required_files_blend = [
        #     "scene.blend.zip", "scene.blend", "scene.blend.gz"
        # ]
        # # if any of the files is present, return True
        # present = False
        # for file in required_files_blend:
        #     if os.path.exists(os.path.join(path, "coarse", file)):
        #         present = True
        #         break
        # if not present:
        #     print(f"Scene {path} does not have any of the required blend files")
        #     return False

        return True

    def _check_scripts(self):
        required_scripts = [
            # self.blender_executable,
            # self.get_object_info_script,
            # self.get_camera_info_script,
            # self.llm_visible_objects_script,
        # sef.bound_objects_script,
            # self.obj_color_info_script,
            # self.generate_descriptions_script,
            # self.generate_questions_script,
            self.generate_maps_script,
            # self.generate_paraphrase_script,
        ]

        for script in required_scripts:
            if not os.path.exists(script):
                print(f"Required script not found: {script}")
                raise FileNotFoundError(f"Required script not found: {script}")

    def run_command(self, command: List[str], description: str, cwd: Path = None) -> bool:
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(command)}")

        if self.dry_run:
            logger.info("DRYRUN - Above command would be executed")
            return True

        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per command
            )

            logger.info(result.stdout)
            logger.info(result.stderr)

            if result.returncode == 0:
                logger.info(f"✓ {description} completed successfully")
                return True
            else:
                logger.error(f"✗ {description} failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {description} timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"✗ {description} failed with exception: {e}")
            return False

    def scene_filtering(self, scenes: List[Path], log_wandb: bool) -> List[Path]:
        filtered_scenes = filter_scenes(
            scenes,
            client=self.client_scene_filtering,
            model_name=self.model_name_scene_filtering,
            api_base=self.api_base_scene_filtering,
            api_key=self.api_key,
            )
        logger.info(f"Filtered {len(filtered_scenes)} scenes")
        logger.info(f"Accepted {len(scenes) - len(filtered_scenes)} scenes")
        logger.info(f"Writing filter results to file")

        self.accepted_scenes = len(scenes) - len(filtered_scenes)
        self.rejected_scenes = len(filtered_scenes)

        filter_results = {}
        if log_wandb:
            for scene in scenes:
                filter_results[str(scene)] = {
                    "result": "REJECT" if scene in filtered_scenes else "ACCEPT",
                    "reason": open(os.path.join(scene, "REJECT.txt" if scene in filtered_scenes else "ACCEPT.txt")).read()
                }

                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["scene_filter_result"] = "REJECT" if scene in filtered_scenes else "ACCEPT"
                    self.wandb_dict[str(scene)]["scene_filter_reason"] = open(os.path.join(scene, "REJECT.txt" if scene in filtered_scenes else "ACCEPT.txt")).read()
                else:
                    self.wandb_dict[str(scene)] = {
                        "scene_filter_result": "REJECT" if scene in filtered_scenes else "ACCEPT",
                        "scene_filter_reason": open(os.path.join(scene, "REJECT.txt" if scene in filtered_scenes else "ACCEPT.txt")).read()
                    }

        with open(f"metadata/scene_filtering_{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(filter_results, f, indent=4)

        return filtered_scenes

    def scene_object_info(self, scenes: List[Path], log_wandb: bool):
        for scene in tqdm(scenes, desc="Getting object information"):
            scene_blend_path = os.path.join(scene, "coarse", "scene.blend")
            visible_objects_path = os.path.join(scene, "visible_objects.json")
            logger.info(f"Getting object information for scene: {scene}")

            if not os.path.exists(visible_objects_path):
                if not self.run_command(
                    [
                        str(self.blender_executable), "-b", str(scene_blend_path), "-P",
                        str(self.get_object_info_script), "--", "--output_json", str(visible_objects_path), "--scene_dir", str(scene)
                    ],
                    description="Getting object information"
                ):
                    logger.error(f"Failed to get object information for scene: {scene}")
                    return False
            else:
                logger.info(f"Object information for scene: {scene} already exists")

            if log_wandb:
                visible_objects = json.load(open(visible_objects_path))
                visible_objects = json.dumps(visible_objects, indent=4)
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["blender_visible_objects"] = visible_objects
                else:
                    self.wandb_dict[str(scene)] = {
                        "blender_visible_objects": visible_objects
                    }
            logger.info(f"Object information for scene: {scene} saved to {visible_objects_path}")
        return True

    def scene_camera_info(self, scenes: List[Path], log_wandb: bool = True):
        for scene in tqdm(scenes, desc="Getting camera information"):
            scene_blend_path = os.path.join(scene, "coarse", "scene.blend")

            #########################################################
            # if not os.path.exists(scene_blend_path):
            #     logger.info(f"Scene blend file not found for scene: {scene}, unzipping it")
            #     # unzip the scene blend file
            #     scene_blend_path_zip = os.path.join(scene, "coarse", "scene.blend.zip")
            #     extracted_scene_blend_path = os.path.join(scene, "coarse")
            #     import zipfile
            #     with zipfile.ZipFile(scene_blend_path_zip, 'r') as zip_ref:
            #         zip_ref.extractall(extracted_scene_blend_path)
            #     logger.info(f"Unzipped the scene blend file: {scene_blend_path_zip} to {extracted_scene_blend_path}")
            #########################################################

            cameras_path = os.path.join(scene, "cameras.json")
            logger.info(f"Getting camera information for scene: {scene}")
            if not os.path.exists(cameras_path):
                if not self.run_command(
                    [
                        str(self.blender_executable), "-b", str(scene_blend_path), "-P",
                        str(self.get_camera_info_script), "--", "--output_json", str(cameras_path)
                    ],
                    description="Getting camera information"
                ):
                    logger.error(f"Failed to get camera information for scene: {scene}")
                    return False
            else:
                logger.info(f"Camera information for scene: {scene} already exists")

            #########################################################
            #Delete the scene blend file
            # if os.path.exists(scene_blend_path_zip):
            #     os.remove(os.path.join(scene, "coarse", "scene.blend"))
            #     logger.info(f"Deleted the extracted scene blend file: {os.path.join(scene, 'coarse', 'scene.blend')}")
            #########################################################

            if log_wandb:
                cameras = json.load(open(cameras_path))
                cameras = json.dumps(cameras, indent=4)
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["blender_cameras"] = cameras
                else:
                    self.wandb_dict[str(scene)] = {
                        "blender_cameras": cameras
                    }
            logger.info(f"Camera information for scene: {scene} saved to {cameras_path}")
        return True

    def scene_blender_color_info(self, scenes: List[Path], log_wandb: bool = True):
        for scene in tqdm(scenes, desc="Getting blender color information"):
            scene_blend_path = os.path.join(scene, "coarse", "scene.blend")
            blender_colors_path = os.path.join(scene, "blender_colors.json")
            scene_blend_path_zip = os.path.join(scene, "coarse", "scene.blend.zip")

            if not os.path.exists(blender_colors_path) or self.overwrite_files:
                #########################################################
                if not os.path.exists(scene_blend_path):
                    logger.info(f"Scene blend file not found for scene: {scene}, unzipping it")
                    # unzip the scene blend file
                    # scene_blend_path_zip = os.path.join(scene, "coarse", "scene.blend.zip")
                    extracted_scene_blend_path = os.path.join(scene, "coarse")
                    import zipfile
                    with zipfile.ZipFile(scene_blend_path_zip, 'r') as zip_ref:
                        zip_ref.extractall(extracted_scene_blend_path)
                    logger.info(f"Unzipped the scene blend file: {scene_blend_path_zip} to {extracted_scene_blend_path}")
                #########################################################


            logger.info(f"Getting color information for scene: {scene}")
            if not os.path.exists(blender_colors_path) or self.overwrite_files:
                if not self.run_command(
                    [
                        str(self.blender_executable), "-b", str(scene_blend_path), "-P",
                        str(self.get_blender_color_info_script), "--", "--output_json", str(blender_colors_path)
                    ],
                    description="Getting blender color information"
                ):
                    logger.error(f"Failed to get blender color information for scene: {scene}")
                    return False
            else:
                logger.info(f"Blender color information for scene: {scene} already exists")

            #########################################################
            #Delete the scene blend file
            # if os.path.exists(scene_blend_path_zip):
            #     if os.path.exists(os.path.join(scene, "coarse", "scene.blend")):
            #         os.remove(os.path.join(scene, "coarse", "scene.blend"))
            #         logger.info(f"Deleted the extracted scene blend file: {os.path.join(scene, 'coarse', 'scene.blend')}")
            #########################################################

            if log_wandb:
                blender_colors = json.load(open(blender_colors_path))
                blender_colors = json.dumps(blender_colors, indent=4)
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["blender_colors"] = blender_colors
                else:
                    self.wandb_dict[str(scene)] = {
                        "blender_colors": blender_colors
                    }
            logger.info(f"Blender color information for scene: {scene} saved to {blender_colors_path}")
        return True

    def scene_llm_visible_objects(self, scenes: List[Path], log_wandb: bool = True):
        all_scenes = scenes
        if not self.overwrite_files:
            incomplete_scenes = []

            for scene in scenes:
                if not os.path.exists(os.path.join(scene, "llm_detected_objects.json")):
                    incomplete_scenes.append(scene)
            scenes = incomplete_scenes

        if len(scenes) > 0:
            run_visibility_check(
                scenes,
                # incomplete_scenes,
                client_name=self.client_vis_objects,
                model_name=self.model_name_vis_objects,
                api_base=self.api_base_vis_objects,
                api_key=self.api_key
            )
        else:
            logger.info(f"All scenes for LLM visible objects are complete")

        if log_wandb:
            for scene in all_scenes:
                if os.path.exists(os.path.join(scene, "llm_detected_objects.json")):
                    llm_detected_objects_path = os.path.join(scene, "llm_detected_objects.json")
                    llm_detected_objects = json.load(open(llm_detected_objects_path))
                    llm_detected_objects = json.dumps(llm_detected_objects, indent=4)
                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["llm_visible_objects"] = llm_detected_objects
                    else:
                        self.wandb_dict[str(scene)] = {
                            "llm_visible_objects": llm_detected_objects
                        }

    def scene_bound_objects(self, scenes: List[Path], log_wandb: bool = True):
        for scene in tqdm(scenes, desc="Bounding objects"):
            llm_detected_objects_path = os.path.join(scene, "llm_detected_objects.json")
            frames_camera_0_path = os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png")
            frames_camera_1_path = os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png")
            bound_objects_camera_0_path = os.path.join(scene, "bounds/camera_0_0")
            bound_objects_camera_1_path = os.path.join(scene, "bounds/camera_1_0")

            if not os.path.exists(bound_objects_camera_0_path) or self.overwrite_files:
                if not self.run_command(
                    [
                        "python", str(self.bound_objects_script),
                        "--image", frames_camera_0_path,
                        "--llm_detected_objects_json", llm_detected_objects_path,
                        "--camera_key", "camera_0_0",
                        "--output_dir", bound_objects_camera_0_path
                    ],
                    description="Bounding objects for camera 0"
                ):
                    logger.error(f"Failed to bound objects for scene: {scene}")
                    return False
            else:
                logger.info(f"Bound objects for camera 0 for scene: {scene} already exists")

            if not os.path.exists(bound_objects_camera_1_path) or self.overwrite_files:
                if not self.run_command(
                    [
                        "python", str(self.bound_objects_script),
                        "--image", frames_camera_1_path,
                        "--llm_detected_objects_json", llm_detected_objects_path,
                        "--camera_key", "camera_1_0",
                        "--output_dir", bound_objects_camera_1_path
                    ],
                    description="Bounding objects for camera 1"
                ):
                    logger.error(f"Failed to bound objects for scene: {scene}")
                    return False
            else:
                logger.info(f"Bound objects for camera 1 for scene: {scene} already exists")

            if log_wandb:
                bound_objects_camera_0_img = wandb.Image(
                    os.path.join(bound_objects_camera_0_path, "camera_0_0_all_boxes.png")
                    )
                bound_objects_camera_1_img = wandb.Image(
                    os.path.join(bound_objects_camera_1_path, "camera_1_0_all_boxes.png")
                    )
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["bound_objects_camera_0_img"] = bound_objects_camera_0_img
                    self.wandb_dict[str(scene)]["bound_objects_camera_1_img"] = bound_objects_camera_1_img
                else:
                    self.wandb_dict[str(scene)] = {
                        "bound_objects_camera_0_img": bound_objects_camera_0_img,
                        "bound_objects_camera_1_img": bound_objects_camera_1_img
                    }
            logger.info(f"Bound objects for scene: {scene} saved to {bound_objects_camera_0_path} and {bound_objects_camera_1_path}")
        return True

    def scene_obj_color_info(self, scenes: List[Path], log_wandb: bool = True):
        all_scenes = scenes
        if not self.overwrite_files:
            incomplete_scenes = []
            for scene in scenes:
                if not os.path.exists(os.path.join(scene, "llm_detected_objects_colors.json")):
                    incomplete_scenes.append(scene)
            scenes = incomplete_scenes

        if len(scenes) > 0:
            run_color_detection(
                scenes,
                # incomplete_scenes,
                client_name=self.client_color,
                model_name=self.model_name_color,
                api_base=self.api_base_color,
                api_key=self.api_key
            )
        else:
            logger.info(f"All scenes for object color info are complete")

        if log_wandb:
            for scene in all_scenes:
                llm_detected_objects_color_path = os.path.join(scene, "llm_detected_objects_colors.json")
                llm_detected_objects_color = json.load(open(llm_detected_objects_color_path))
                llm_detected_objects_color = json.dumps(llm_detected_objects_color, indent=4)
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["llm_visible_objects_color"] = llm_detected_objects_color
                else:
                    self.wandb_dict[str(scene)] = {
                        "llm_visible_objects_color": llm_detected_objects_color
                    }

    def scene_generate_descriptions(self, scenes: List[Path], log_wandb: bool = True):
        for scene in tqdm(scenes, desc="Generating descriptions"):
            llm_detected_objects_color_path = os.path.join(scene, "llm_detected_objects_colors.json")
            visible_objects_path = os.path.join(scene, "visible_objects.json")
            visible_objects_with_descriptions_path = os.path.join(scene, "visible_objects_with_descriptions.json")
            full_description_json_path = os.path.join(scene, "full_description.json")
            if not os.path.exists(visible_objects_with_descriptions_path) or self.overwrite_files:
                if not self.run_command(
                    [
                        "python", str(self.generate_descriptions_script),
                        "--input_json", str(llm_detected_objects_color_path),
                        "--ground_truth_json", str(visible_objects_path),
                        "--output_json", str(visible_objects_with_descriptions_path),
                        "--full_description_json", str(full_description_json_path)
                    ],
                    description="Generating descriptions"
                ):
                    logger.error(f"Failed to generate descriptions for scene: {scene}")
                    continue
                    # return False
            else:
                logger.info(f"Descriptions for scene: {scene} already exists")

            if log_wandb:
                if os.path.exists(visible_objects_with_descriptions_path):
                    visible_objects_with_descriptions = json.load(open(visible_objects_with_descriptions_path))
                    visible_objects_with_descriptions = json.dumps(visible_objects_with_descriptions, indent=4)
                    full_descriptions = json.load(open(full_description_json_path))
                    full_descriptions = json.dumps(full_descriptions, indent=4)
                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["llm_visible_objects_descriptions"] = visible_objects_with_descriptions
                        self.wandb_dict[str(scene)]["full_descriptions"] = full_descriptions
                    else:
                        self.wandb_dict[str(scene)] = {
                            "llm_visible_objects_descriptions": visible_objects_with_descriptions,
                            "full_descriptions": full_descriptions
                        }
            logger.info(f"Descriptions for scene: {scene} saved to {visible_objects_with_descriptions_path}")
        return True

    def scene_solve_perception(self, scenes: List[Path], log_wandb: bool = True):
        for scene in tqdm(scenes, desc="Generating perception"):
            visible_objects_path = os.path.join(scene, "llm_detected_objects.json")
            output_1_file = os.path.join(scene, "agent_1_input.txt")
            output_2_file = os.path.join(scene, "agent_2_input.txt")
            # pass --ground_description_json if you want to use blender objects for perception solving
            if not os.path.exists(output_1_file) or not os.path.exists(output_2_file) or self.overwrite_files:
                if not self.run_command(
                    [
                        "python", str(self.solve_perception_script),
                        "--visible_objects_json", str(visible_objects_path),
                        "--agent_1_file", str(output_1_file),
                        "--agent_2_file", str(output_2_file),
                    ],
                    description="Solving perception"
                ):
                    logger.error(f"Failed to solve perception for scene: {scene}")
                    continue
                    # return False
            else:
                logger.info(f"Perception for scene: {scene} already exists")

            if log_wandb:
                output_1_file = open(output_1_file, "r").read()
                output_2_file = open(output_2_file, "r").read()
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["agent_1_file"] = output_1_file
                    self.wandb_dict[str(scene)]["agent_2_file"] = output_2_file
                else:
                    self.wandb_dict[str(scene)] = {
                        "agent_1_file": output_1_file,
                        "agent_2_file": output_2_file
                    }
            logger.info(f"Descriptions for scene: {scene} saved to {output_1_file} and {output_2_file}")
        return True

    def scene_generate_questions(self, scenes: List[Path], log_wandb: bool = True):
        for scene in tqdm(scenes, desc="Generating questions"):
            visible_objects_with_descriptions_path = os.path.join(scene, "visible_objects_with_descriptions.json")
            questions_path = os.path.join(scene, "questions.json")
            visible_objects_path = os.path.join(scene, "visible_objects.json")
            cameras_path = os.path.join(scene, "cameras.json")

            if not os.path.exists(questions_path) or self.overwrite_files:
                if not self.run_command(
                    [
                        "python", str(self.generate_questions_script),
                        "--input_json", str(visible_objects_with_descriptions_path),
                        "--output_json", str(questions_path),
                        "--ground_truth_json", str(visible_objects_path),
                        "--cam_data_file", str(cameras_path),
                    ],
                    description="Generating questions"
                ):
                    logger.error(f"Failed to generate questions for scene: {scene}")
                    continue
                    # return False
            else:
                logger.info(f"Questions for scene: {scene} already exists")

            if log_wandb:
                if os.path.exists(questions_path):
                    questions = json.load(open(questions_path))
                    self.wandb_dict["num_global_counting_questions"] += len(questions.get("global_counting_questions", []))
                    self.wandb_dict["num_anchor_recognition_questions"] += len(questions.get("anchor_recognition_questions", []))
                    self.wandb_dict["num_relative_distance_questions"] += len(questions.get("relative_distance_questions", []))
                    self.wandb_dict["num_relative_direction_questions"] += len(questions.get("relative_direction_questions", []))
                    self.wandb_dict["num_total_questions"] += (len(questions.get("global_counting_questions", [])) +
                                                                len(questions.get("anchor_recognition_questions", [])) +
                                                                len(questions.get("relative_distance_questions", [])) +
                                                                len(questions.get("relative_direction_questions", [])))
                    if len(questions.get("global_counting_questions", [])) > 0:
                        self.wandb_dict["global_counting_scenes"].append(str(scene))
                    if len(questions.get("anchor_recognition_questions", [])) > 0:
                        self.wandb_dict["anchor_recognition_scenes"].append(str(scene))
                    if len(questions.get("relative_distance_questions", [])) > 0:
                        self.wandb_dict["relative_distance_scenes"].append(str(scene))
                    if len(questions.get("relative_direction_questions", [])) > 0:
                        self.wandb_dict["relative_direction_scenes"].append(str(scene))
                    if (len(questions.get("global_counting_questions", [])) == 0 and
                        len(questions.get("anchor_recognition_questions", [])) == 0 and
                        len(questions.get("relative_distance_questions", [])) == 0 and
                        len(questions.get("relative_direction_questions", [])) == 0):
                        self.wandb_dict["no_questions_scenes"].append(str(scene))

                    questions = json.dumps(questions, indent=4)

                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["questions"] = questions
                    else:
                        self.wandb_dict[str(scene)] = {
                            "questions": questions
                        }
            logger.info(f"Questions for scene: {scene} saved to {questions_path}")
        return True

    def scene_generate_maps(self, scenes: List[Path], log_wandb: bool = True):
        """
        Generate map questions using map_godbless.py script.
        This stage can be run post-hoc after other stages are complete.
        """
        for scene in tqdm(scenes, desc="Generating map questions"):
            # Input files
            visible_objects_path = os.path.join(scene, "visible_objects.json")
            cameras_path = os.path.join(scene, "cameras.json")

            # Output files and directories
            output_dir_path = os.path.join(scene, "cognitive_mapping")
            output_format1_path = os.path.join(output_dir_path, "cognitive_mapping_format1.json")
            output_format2_path = os.path.join(output_dir_path, "cognitive_mapping_format2.json")

            output_json_path = os.path.join(scene, "cognitive_mapping.json")

            # Check if required input files exist
            if not os.path.exists(visible_objects_path):
                logger.warning(f"visible_objects.json not found for scene: {scene}, skipping")
                if log_wandb:
                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["cognitive_mapping"] = None
                    else:
                        self.wandb_dict[str(scene)] = {"cognitive_mapping": None}
                continue

            if not os.path.exists(cameras_path):
                logger.warning(f"cameras.json not found for scene: {scene}, skipping")
                if log_wandb:
                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["cognitive_mapping"] = None
                    else:
                        self.wandb_dict[str(scene)] = {"cognitive_mapping": None}
                continue

            # Check if output already exists
            if (os.path.exists(output_format1_path) and os.path.exists(output_format2_path)) and not self.overwrite_files:
                logger.info(f"Map questions for scene: {scene} already exist, skipping")
                # Still log to wandb if files exist
                continue

            # Delete existing cognitive_mapping folder if overwrite_files is True
            if self.overwrite_files and os.path.exists(output_dir_path):
                logger.info(f"Deleting existing cognitive_mapping folder for scene: {scene} (overwrite_files=True)")
                try:
                    shutil.rmtree(output_dir_path)
                    logger.info(f"Successfully deleted cognitive_mapping folder for scene: {scene}")
                except Exception as e:
                    logger.error(f"Failed to delete cognitive_mapping folder for scene: {scene}: {e}")
                    # Continue anyway - the script might handle it or fail later

            # Run map generation script
            # Use absolute paths to ensure correct file resolution
            if not self.run_command(
                [
                    "python", str(self.generate_maps_script),
                    "--input_json", str(os.path.abspath(visible_objects_path)),
                    "--cam_data_json", str(os.path.abspath(cameras_path)),
                    "--output_dir", str(os.path.abspath(output_dir_path)),
                    "--output_json", str(os.path.abspath(output_json_path)),
                    # "--output_format1_json", str(os.path.abspath(output_format1_path)),
                    # "--output_format2_json", str(os.path.abspath(output_format2_path))
                ],
                description="Generating map questions"
            ):
                logger.error(f"Failed to generate map questions for scene: {scene}")
                # Check if directory exists after failed script run
                if log_wandb:
                    if not os.path.exists(output_dir_path):
                        logger.warning(f"cognitive_mapping directory not found for scene: {scene} after script failure")
                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["cognitive_mapping"] = None
                    else:
                        self.wandb_dict[str(scene)] = {"cognitive_mapping": None}
                continue

            # Log to wandb if enabled
            if log_wandb:
                try:
                    # Check if cognitive_mapping directory exists
                    if not os.path.exists(output_dir_path):
                        logger.warning(f"cognitive_mapping directory not found for scene: {scene}, skipping wandb logging")
                        if str(scene) in self.wandb_dict.keys():
                            self.wandb_dict[str(scene)]["cognitive_mapping"] = None
                        else:
                            self.wandb_dict[str(scene)] = {"cognitive_mapping": None}
                        continue

                    # Load both format files
                    if os.path.exists(output_format1_path) and os.path.exists(output_format2_path):
                        questions_f1 = json.load(open(output_format1_path))
                        questions_f2 = json.load(open(output_format2_path))

                        # Structure to store map questions data
                        cognitive_mapping_data = {
                            "agent_1": [],
                            "agent_2": [],
                            "total_questions_agent_1": 0,
                            "total_questions_agent_2": 0
                        }

                        # Process questions for each agent
                        for agent_key in ["agent_1", "agent_2"]:
                            if agent_key in questions_f1 and agent_key in questions_f2:
                                agent_questions_f1 = questions_f1[agent_key]
                                agent_questions_f2 = questions_f2[agent_key]

                                # Ensure both lists have same length
                                num_questions = min(len(agent_questions_f1), len(agent_questions_f2))

                                for q_idx in range(num_questions):
                                    q_f1 = agent_questions_f1[q_idx]
                                    q_f2 = agent_questions_f2[q_idx]

                                    # Load the map image if it exists
                                    map_image = None
                                    if "map_image_path" in q_f1 and os.path.exists(q_f1["map_image_path"]):
                                        map_image = wandb.Image(q_f1["map_image_path"])

                                    # Extract correct answer for format2 (list of dicts)
                                    correct_idx = q_f1.get("correct_index", -1)
                                    correct_answer_f2 = None
                                    if correct_idx >= 0 and q_f2.get("options") and len(q_f2.get("options", [])) > correct_idx:
                                        correct_answer_f2 = q_f2.get("options", [])[correct_idx]

                                    # Create structured question data
                                    question_data = {
                                        "question_index": q_idx,
                                        "question_text": q_f1.get("question", ""),
                                        "question_both_views": q_f1.get("question_both_views", ""),
                                        "asking_to": q_f1.get("asking_to", agent_key),
                                        "options_format1": q_f1.get("options", []),  # ["A", "B", "C", "D"]
                                        "options_format2": q_f2.get("options", []),  # List of dicts with category coordinates
                                        "correct_index": correct_idx,
                                        "correct_answer_format1": q_f1.get("options", [])[correct_idx] if q_f1.get("options") and correct_idx >= 0 and len(q_f1.get("options", [])) > correct_idx else None,
                                        "correct_answer_format2": correct_answer_f2,  # Dict with category coordinates
                                        "num_objects": q_f1.get("num_objects", 0),
                                        "option_categories": q_f1.get("option_categories", []),  # ["correct", "type2", "type3", "counting"]
                                        "map_image": map_image,
                                        "map_image_path": q_f1.get("map_image_path", ""),
                                        "format1_full": q_f1,
                                        "format2_full": q_f2
                                    }

                                    cognitive_mapping_data[agent_key].append(question_data)

                                cognitive_mapping_data[f"total_questions_{agent_key}"] = num_questions

                        # Store in wandb_dict
                        if str(scene) in self.wandb_dict.keys():
                            self.wandb_dict[str(scene)]["cognitive_mapping"] = cognitive_mapping_data
                            self.wandb_dict[str(scene)]["cognitive_mapping_json_f1"] = json.dumps(questions_f1, indent=4)
                            self.wandb_dict[str(scene)]["cognitive_mapping_json_f2"] = json.dumps(questions_f2, indent=4)
                        else:
                            self.wandb_dict[str(scene)] = {
                                "cognitive_mapping": cognitive_mapping_data,
                                "cognitive_mapping_json_f1": json.dumps(questions_f1, indent=4),
                                "cognitive_mapping_json_f2": json.dumps(questions_f2, indent=4)
                            }

                        # Update global map question stats
                        if "num_cognitive_mapping_agent_1" not in self.wandb_dict:
                            self.wandb_dict["num_cognitive_mapping_agent_1"] = 0
                            self.wandb_dict["num_cognitive_mapping_agent_2"] = 0

                        self.wandb_dict["num_cognitive_mapping_agent_1"] += cognitive_mapping_data["total_questions_agent_1"]
                        self.wandb_dict["num_cognitive_mapping_agent_2"] += cognitive_mapping_data["total_questions_agent_2"]

                        logger.info(f"Logged {cognitive_mapping_data['total_questions_agent_1']} map questions for agent_1 and {cognitive_mapping_data['total_questions_agent_2']} for agent_2")

                    else:
                        logger.warning(f"Map question files not found for scene: {scene}")
                        if str(scene) in self.wandb_dict.keys():
                            self.wandb_dict[str(scene)]["cognitive_mapping"] = None
                        else:
                            self.wandb_dict[str(scene)] = {"cognitive_mapping": None}

                except Exception as e:
                    logger.warning(f"Failed to load map questions for wandb logging: {e}")
                    import traceback
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                    if str(scene) in self.wandb_dict.keys():
                        self.wandb_dict[str(scene)]["cognitive_mapping"] = None
                    else:
                        self.wandb_dict[str(scene)] = {"cognitive_mapping": None}

            logger.info(f"Map questions for scene: {scene} saved to {output_dir_path}")
        return True

    def scene_generate_paraphrase(self, scenes: List[Path], log_wandb: bool = True):
        all_scenes = scenes
        incomplete_scenes = []
        if not self.overwrite_files:
            for scene in scenes:
                if not os.path.exists(os.path.join(scene, "questions_paraphrased.json")):
                    incomplete_scenes.append(scene)
            scenes = incomplete_scenes

        if len(scenes) > 0:
            paraphrase_across_scenes(
                scenes,
                # incomplete_scenes,
                client_name=self.client_paraphrase,
                model_name=self.model_name_paraphrase,
                api_base=self.api_base_paraphrase,
                api_key=self.api_key
            )
        else:
            logger.info(f"All scenes for paraphrase are complete")

        if log_wandb:
            for scene in all_scenes:
                paraphrase_path = os.path.join(scene, "questions_paraphrased.json")
                if os.path.exists(paraphrase_path):
                    questions_paraphrase = json.load(open(paraphrase_path))
                    questions_paraphrase = json.dumps(questions_paraphrase, indent=4)
                else: # Check with Debangan
                    questions_paraphrase = None
                    logger.warning(f"Paraphrased questions for scene: {scene} not found")
                if str(scene) in self.wandb_dict.keys():
                    self.wandb_dict[str(scene)]["questions_paraphrase"] = questions_paraphrase
                else:
                    self.wandb_dict[str(scene)] = {
                        "questions_paraphrase": questions_paraphrase
                    }
            logger.info(f"Paraphrased questions for scene: {scene} saved to {paraphrase_path}")

    def aggregate_data(self, scenes: List[Path]):
        answerer_goal = "Communicate with your partner to answer the following question correctly."
        helper_goal = "Communicate with your partner to help them answer their question correctly."

        self.successful_scenes = []
        dataset_global_counting = []
        dataset_anchor_recognition = []
        dataset_relative_distance = []
        dataset_relative_direction = []

        global_counting_idx = 0
        anchor_recognition_idx = 0
        relative_distance_idx = 0
        relative_direction_idx = 0

        for scene in tqdm(scenes, desc="Aggregating data"):
            cameras_path = os.path.join(scene, "cameras.json")
            visible_objects_path = os.path.join(scene, "visible_objects.json")
            llm_detected_objects_path = os.path.join(scene, "llm_detected_objects.json")
            bound_objects_camera_0_path = os.path.join(scene, "bounds/camera_0_0/camera_0_0_all_boxes.png")
            bound_objects_camera_1_path = os.path.join(scene, "bounds/camera_1_0/camera_1_0_all_boxes.png")
            llm_detected_objects_color_path = os.path.join(scene, "llm_detected_objects_colors.json")
            visible_objects_with_descriptions_path = os.path.join(scene, "visible_objects_with_descriptions.json")
            questions_path = os.path.join(scene, "questions.json")
            paraphrase_path = os.path.join(scene, "questions_paraphrased.json")

            if (os.path.exists(cameras_path) and
                os.path.exists(visible_objects_path) and
                os.path.exists(llm_detected_objects_path) and
                os.path.exists(bound_objects_camera_0_path) and
                os.path.exists(bound_objects_camera_1_path) and
                os.path.exists(llm_detected_objects_color_path) and
                os.path.exists(visible_objects_with_descriptions_path) and
                os.path.exists(questions_path) and
                os.path.exists(paraphrase_path)):

                self.successful_scenes.append(str(scene))

                paraphrase_file = json.load(open(paraphrase_path))

                counting_questions = paraphrase_file.get("global_counting_questions", None)
                anchor_questions = paraphrase_file.get("anchor_recognition_questions", None)
                relative_distance_questions = paraphrase_file.get("relative_distance_questions", None)
                spatial_orientation_questions = paraphrase_file.get("relative_direction_questions", None)

                if counting_questions:
                    for question in counting_questions:
                        sample_dict = {
                            "sample_id": "global_counting_"+str(global_counting_idx).zfill(6),
                            "question_type": "global_counting",
                            "room_part": str(scene).split("/")[-2],
                            "scene_id": str(scene).split("/")[-1],
                            "global_map_image": None,
                            "user_1_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png"),
                            "user_2_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png"),
                            "user_1_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_0_0_0048_0.png",
                            "user_2_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_1_0_0048_0.png",
                            "user_1_goal": answerer_goal if question["asking_to"] == "agent_1" else helper_goal,
                            "user_2_goal": answerer_goal if question["asking_to"] == "agent_2" else helper_goal,
                            "user_1_question": question["question"] if question["asking_to"] == "agent_1" else None,
                            "user_2_question": question["question"] if question["asking_to"] == "agent_2" else None,
                            "options_user_1": question["options"] if question["asking_to"] == "agent_1" else None,
                            "options_user_2": question["options"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_2" else None,
                            "difficulty_sum": question["difficulty_sum"],
                            "difficulty_int": question["difficulty_int"],
                            "question_object": question["question_object"],
                            "correct_answer": question["correct_answer"],
                            "question_both_views": question["question_both_views"],
                            "scene_intersection": question["scene_intersection"],
                            "scene_union": question["scene_union"],
                            "user_1_perception": str(scene)+"/agent_1_input.txt",
                            "user_2_perception": str(scene)+"/agent_2_input.txt",
                        }
                        dataset_global_counting.append(sample_dict)
                        global_counting_idx += 1

                if anchor_questions:
                    for question in anchor_questions:
                        sample_dict = {
                            "sample_id": "anchor_recognition_"+str(anchor_recognition_idx).zfill(6),
                            "question_type": "anchor_recognition",
                            "room_part": str(scene).split("/")[-2],
                            "scene_id": str(scene).split("/")[-1],
                            "global_map_image": None,
                            "user_1_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png"),
                            "user_2_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png"),
                            "user_1_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_0_0_0048_0.png",
                            "user_2_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_1_0_0048_0.png",
                            "user_1_goal": answerer_goal if question["asking_to"] == "agent_1" else helper_goal,
                            "user_2_goal": answerer_goal if question["asking_to"] == "agent_2" else helper_goal,
                            "user_1_question": question["question"] if question["asking_to"] == "agent_1" else None,
                            "user_2_question": question["question"] if question["asking_to"] == "agent_2" else None,
                            "options_user_1": question["options"] if question["asking_to"] == "agent_1" else None,
                            "options_user_2": question["options"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_2" else None,
                            "difficulty": question["difficulty"],
                            "description_difficulty": question["description_difficulty"],
                            "distractor_difficulty": question["distractor_difficulty"],
                            "asking_to": question["asking_to"],
                            "correct_answer": question["correct_answer"],
                            "option_categories": question["option_categories"],
                            "question_both_views": question["question_both_views"],
                            "scene_intersection": question["scene_intersection"],
                            "scene_union": question["scene_union"],
                            "user_1_perception": str(scene)+"/agent_1_input.txt",
                            "user_2_perception": str(scene)+"/agent_2_input.txt",
                        }
                        dataset_anchor_recognition.append(sample_dict)
                        anchor_recognition_idx += 1

                if relative_distance_questions:
                    for question in relative_distance_questions:
                        sample_dict = {
                            "sample_id": "relative_distance_"+str(relative_distance_idx).zfill(6),
                            "question_type": "relative_distance",
                            "room_part": str(scene).split("/")[-2],
                            "scene_id": str(scene).split("/")[-1],
                            "global_map_image": None,
                            "user_1_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png"),
                            "user_2_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png"),
                            "user_1_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_0_0_0048_0.png",
                            "user_2_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_1_0_0048_0.png",
                            "user_1_goal": answerer_goal if question["asking_to"] == "agent_1" else helper_goal,
                            "user_2_goal": answerer_goal if question["asking_to"] == "agent_2" else helper_goal,
                            "user_1_question": question["question"] if question["asking_to"] == "agent_1" else None,
                            "user_2_question": question["question"] if question["asking_to"] == "agent_2" else None,
                            "options_user_1": question["options"] if question["asking_to"] == "agent_1" else None,
                            "options_user_2": question["options"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_2" else None,
                            "difficulty": question["difficulty"],
                            "description_difficulty": question["description_difficulty"],
                            "question_object": question["question_object"],
                            "correct_answer": question["correct_answer"],
                            "question_both_views": question["question_both_views"],
                            "option_categories": question["option_categories"],
                            "option_distances": question["option_distances"],
                            "ans_present_in_view": question["ans_present_in_view"],
                            "question_type": question["question_type"],
                            "agent_distribution": question["agent_distribution"],
                            "scene_intersection": question["scene_intersection"],
                            "scene_union": question["scene_union"],
                            "user_1_perception": str(scene)+"/agent_1_input.txt",
                            "user_2_perception": str(scene)+"/agent_2_input.txt",
                        }
                        dataset_relative_distance.append(sample_dict)
                        relative_distance_idx += 1

                if spatial_orientation_questions:
                    for question in spatial_orientation_questions:
                        sample_dict = {
                            "sample_id": "relative_direction_"+str(relative_direction_idx).zfill(6),
                            "question_type": "relative_direction",
                            "room_part": str(scene).split("/")[-2],
                            "scene_id": str(scene).split("/")[-1],
                            "global_map_image": None,
                            "user_1_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png"),
                            "user_2_image_local_path": os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png"),
                            "user_1_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_0_0_0048_0.png",
                            "user_2_image": "/img/"+str(scene).split("/")[-2]+"/"+str(scene).split("/")[-1]+"frames/Image/camera_0/Image_1_0_0048_0.png",
                            "user_1_goal": answerer_goal if question["asking_to"] == "agent_1" else helper_goal,
                            "user_2_goal": answerer_goal if question["asking_to"] == "agent_2" else helper_goal,
                            "user_1_question": question["question"] if question["asking_to"] == "agent_1" else None,
                            "user_2_question": question["question"] if question["asking_to"] == "agent_2" else None,
                            "options_user_1": question["options"] if question["asking_to"] == "agent_1" else None,
                            "options_user_2": question["options"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_idx": question["correct_index"] if question["asking_to"] == "agent_2" else None,
                            "user_1_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_1" else None,
                            "user_2_gt_answer_text": question["correct_answer"] if question["asking_to"] == "agent_2" else None,
                            "difficulty": question["difficulty"],
                            "description_difficulty": question["description_difficulty"],
                            "angle": question["angle"],
                            "distance": question["distance"],
                            "question_object": question["question_object"],
                            "correct_answer": question["correct_answer"],
                            "question_both_views": question["question_both_views"],
                            "other_agent_angle": question["other_agent_angle"],
                            "other_agent_distance": question["other_agent_distance"],
                            "scene_intersection": question["scene_intersection"],
                            "scene_union": question["scene_union"],
                            "user_1_perception": str(scene)+"/agent_1_input.txt",
                            "user_2_perception": str(scene)+"/agent_2_input.txt",
                        }
                        dataset_relative_direction.append(sample_dict)
                        relative_direction_idx += 1
            else:
                if not os.path.exists(cameras_path):
                    error = "cameras file is missing"
                elif not os.path.exists(visible_objects_path):
                    error = "visible objects file is missing"
                elif not os.path.exists(llm_detected_objects_path):
                    error = "llm detected objects file is missing"
                elif not os.path.exists(bound_objects_camera_0_path):
                    error = "bound objects camera 0 file is missing"
                elif not os.path.exists(bound_objects_camera_1_path):
                    error = "bound objects camera 1 file is missing"
                elif not os.path.exists(llm_detected_objects_color_path):
                    error = "llm detected objects color file is missing"
                elif not os.path.exists(visible_objects_with_descriptions_path):
                    error = "visible objects with descriptions file is missing"
                elif not os.path.exists(questions_path):
                    error = "questions file is missing"
                elif not os.path.exists(paraphrase_path):
                    error = "paraphrase file is missing"
                else:
                    error = "unknown error"

                self.failed_scenes[scene] = error

        logger.info(f"Saving dataset_global_counting to {DATA_DIR / f'dataset_global_counting.json'}")
        json.dump(dataset_global_counting, open(str(DATA_DIR / f"dataset_global_counting.json"), "w"), indent=4)
        logger.info(f"Saving dataset_anchor_recognition to {DATA_DIR / f'dataset_anchor_recognition.json'}")
        json.dump(dataset_anchor_recognition, open(str(DATA_DIR / f"dataset_anchor_recognition.json"), "w"), indent=4)
        logger.info(f"Saving dataset_relative_distance to {DATA_DIR / f'dataset_relative_distance.json'}")
        json.dump(dataset_relative_distance, open(str(DATA_DIR / f"dataset_relative_distance.json"), "w"), indent=4)
        logger.info(f"Saving dataset_relative_direction to {DATA_DIR / f'dataset_relative_direction.json'}")
        json.dump(dataset_relative_direction, open(str(DATA_DIR / f"dataset_relative_direction.json"), "w"), indent=4)

    def question_filtering(self):
        with open(str(DATA_DIR / f"dataset_global_counting.json"), "r") as f:
            counting_questions = json.load(f)

        with open(str(DATA_DIR / f"dataset_anchor_recognition.json"), "r") as f:
            anchor_questions = json.load(f)

        with open(str(DATA_DIR / f"dataset_relative_distance.json"), "r") as f:
            relative_distance_questions = json.load(f)

        with open(str(DATA_DIR / f"dataset_relative_direction.json"), "r") as f:
            spatial_questions = json.load(f)

        filtered_counting_questions = []
        filtered_anchor_questions = []
        filtered_relative_distance_questions = []
        filtered_spatial_questions = []

        for question in counting_questions:
            question_object = question["question_object"]
            room_part = question["room_part"]
            answer_value = question["user_1_gt_answer_text"] if question["user_1_gt_answer_text"] is not None else question["user_2_gt_answer_text"]
            try:
                answer_value = int(answer_value)
            except:
                continue

            difficulty_sum = question["difficulty_sum"]

            # Filter if the following holds true:
            if question_object in {"Sink", "Bathtub", "Toilet", "Mirror"} and "Bathroom" in room_part:
                continue
            if question_object in {"Shelf", "Cabinet", "Oven", "Beverage Fridge", "Dishwasher", "Sink"} and "Kitchen" in room_part:
                continue
            if question_object in {"Table Dining", "Chair"} and "DiningRoom" in room_part:
                continue
            if difficulty_sum == answer_value:
                continue
            else:
                filtered_counting_questions.append(question)

        for question in anchor_questions:
            print(question)
            filtered_anchor_questions.append(question)

        for question in spatial_questions:
            print(question)
            distance = question["distance"]
            if distance <  1.5:
                continue
            else:
                filtered_spatial_questions.append(question)

        for question in relative_distance_questions:
            print(question)
            distance = question["difficulty"]
            if distance <  1.5:
                continue
            else:
                filtered_relative_distance_questions.append(question)

        json.dump(filtered_counting_questions, open(str(DATA_DIR / f"dataset_global_counting_filtered.json"), "w"), indent=4)
        json.dump(filtered_anchor_questions, open(str(DATA_DIR / f"dataset_anchor_recognition_filtered.json"), "w"), indent=4)
        json.dump(filtered_relative_distance_questions, open(str(DATA_DIR / f"dataset_relative_distance_filtered.json"), "w"), indent=4)
        json.dump(filtered_spatial_questions, open(str(DATA_DIR / f"dataset_relative_direction_filtered.json"), "w"), indent=4)


    def aggregate_cognitive_mapping(self, scenes: List[Path]):
        answerer_goal = "Communicate with your partner to answer the following question correctly."
        helper_goal = "Communicate with your partner to help them answer their question correctly."

        dataset_cognitive_mapping = []
        sample_id_counter = 0

        for scene in tqdm(scenes, desc="Aggregating cognitive mapping data"):
            try:
                aggregated, sample_id_counter = aggregate_cognitive_mapping_for_scene(
                    str(scene), answerer_goal, helper_goal, sample_id_counter,
                    sample_id_prefix="cognitive_mapping"
                )
                filtered = [q for q in aggregated if q and q.get("sample_id") and q.get("question")]
                dataset_cognitive_mapping.extend(filtered)
            except Exception as e:
                logger.error(f"Error aggregating cognitive mapping for scene {scene}: {e}")
                continue

        output_path = str(DATA_DIR / f"dataset_cognitive_mapping.json")
        logger.info(f"Saving dataset_cognitive_mapping to {output_path}")
        json.dump(dataset_cognitive_mapping, open(output_path, "w"), indent=4)
        logger.info(f"Total cognitive mapping questions: {len(dataset_cognitive_mapping)}")

    def run_pipeline(self):
        logger.info("Starting Datagen Pipeline")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Max scenes: {self.max_scenes}")

        scenes = self.find_all_scenes()

        self.successful_scenes = scenes

        logger.info(f"Scene data file: {self.scene_datafile}")
        current_dataset_scenes = {}
        scene_ids_list = []
        for scene in scenes:
            scene_detailed_name = str(scene).split("/")[-2] + "_" + str(scene).split("/")[-1]
            current_dataset_scenes[scene_detailed_name] = {
                "scene_id": str(scene).split("/")[-1],
                "room_name": str(scene).split("/")[-2],
                "frame_1_path": os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png"),
                "frame_2_path": os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png")
            }
            scene_ids_list.append(str(scene).split("/")[-1])

        # print scene ids that are not unique and print their paths
        for scene_id in scene_ids_list:
            if scene_ids_list.count(scene_id) > 1:
                print(f"Scene ID {scene_id} is not unique and its path is {scene_id}")
        assert len(scene_ids_list) == len(set(scene_ids_list)), "Scene IDs are not unique"
        json.dump(current_dataset_scenes, open(self.scene_datafile, "w"), indent=4)

        if not scenes:
            logger.error("No valid scenes found to process")
            return

        if self.max_scenes and len(scenes) > self.max_scenes:
            total_scenes = len(scenes)
            random.seed(self.seed)
            scenes = random.sample(scenes, self.max_scenes)
            logger.info(f"Randomly sampled {self.max_scenes} scenes from {total_scenes} total scenes")
        elif self.max_scenes:
            logger.info(f"Processing all {len(scenes)} scenes (max_scenes >= total scenes)")

        scene_filtering_total_time = 0
        scene_object_info_total_time = 0
        scene_camera_info_total_time = 0
        scene_blender_color_info_total_time = 0
        scene_llm_visible_objects_total_time = 0
        scene_bound_objects_total_time = 0
        scene_obj_color_info_total_time = 0
        scene_generate_descriptions_total_time = 0
        scene_solve_perception_total_time = 0
        scene_generate_questions_total_time = 0
        scene_generate_maps_total_time = 0
        scene_generate_paraphrase_total_time = 0

        start_time = time.time()

        if self.log_wandb:
            for scene in tqdm(scenes, desc="Logging WandB", total=len(scenes)):
                scene_image_0 = os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png")
                scene_image_1 = os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png")
                scene_image_0 = wandb.Image(scene_image_0)
                scene_image_1 = wandb.Image(scene_image_1)
                asset_parameters = json.load(open(os.path.join(scene, "coarse/asset_parameters.json")))
                asset_parameters = json.dumps(asset_parameters, indent=4)
                try:
                    coarse_log = open(os.path.join(scene, "logs/coarse.out")).read()
                except:
                    logger.warning(f"Coarse log file not found for scene {scene}")
                    coarse_log = "Coarse log file not found"
                self.wandb_dict[str(scene)] = {
                    "scene_image_0": scene_image_0,
                    "scene_image_1": scene_image_1,
                    "asset_parameters": asset_parameters,
                    "coarse_log": coarse_log,
                }


        if "scene_filtering" in self.stages_to_run:
            scene_filtering_start_time = time.time()
            filtered_scenes = self.scene_filtering(scenes, log_wandb=self.log_wandb)
            scene_filtering_total_time = time.time() - scene_filtering_start_time
            logger.info(f"Scene filtering took {scene_filtering_total_time:.2f} seconds")

        if "scene_object_info" in self.stages_to_run:
            scene_object_info_start_time = time.time()
            self.scene_object_info(scenes, log_wandb=self.log_wandb)
            scene_object_info_total_time = time.time() - scene_object_info_start_time
            logger.info(f"Scene object info took {scene_object_info_total_time:.2f} seconds")

        if "scene_camera_info" in self.stages_to_run:
            scene_camera_info_start_time = time.time()
            self.scene_camera_info(scenes, log_wandb=self.log_wandb)
            scene_camera_info_total_time = time.time() - scene_camera_info_start_time
            logger.info(f"Scene camera info took {scene_camera_info_total_time:.2f} seconds")

        if "scene_blender_color_info" in self.stages_to_run:
            scene_blender_color_info_start_time = time.time()
            self.scene_blender_color_info(scenes, log_wandb=self.log_wandb)
            scene_blender_color_info_total_time = time.time() - scene_blender_color_info_start_time
            logger.info(f"Scene blender color info took {scene_blender_color_info_total_time:.2f} seconds")

        if "scene_llm_visible_objects" in self.stages_to_run:
            scene_llm_visible_objects_start_time = time.time()
            self.scene_llm_visible_objects(scenes, log_wandb=self.log_wandb)
            scene_llm_visible_objects_total_time = time.time() - scene_llm_visible_objects_start_time
            logger.info(f"Scene LLM visible objects took {scene_llm_visible_objects_total_time:.2f} seconds")

        if "scene_bound_objects" in self.stages_to_run:
            scene_bound_objects_start_time = time.time()
            self.scene_bound_objects(scenes, log_wandb=self.log_wandb)
            scene_bound_objects_total_time = time.time() - scene_bound_objects_start_time
            logger.info(f"Scene bound objects took {scene_bound_objects_total_time:.2f} seconds")

        if "scene_obj_color_info" in self.stages_to_run:
            scene_obj_color_info_start_time = time.time()
            self.scene_obj_color_info(scenes, log_wandb=self.log_wandb)
            scene_obj_color_info_total_time = time.time() - scene_obj_color_info_start_time
            logger.info(f"Scene object color info took {time.time() - scene_obj_color_info_start_time:.2f} seconds")

        if "scene_generate_descriptions" in self.stages_to_run:
            scene_generate_descriptions_start_time = time.time()
            self.scene_generate_descriptions(scenes, log_wandb=self.log_wandb)
            scene_generate_descriptions_total_time = time.time() - scene_generate_descriptions_start_time
            logger.info(f"Scene generate descriptions took {time.time() - scene_generate_descriptions_start_time:.2f} seconds")

        if "scene_solve_perception" in self.stages_to_run:
            scene_solve_perception_start_time = time.time()
            self.scene_solve_perception(scenes, log_wandb=self.log_wandb)
            scene_solve_perception_total_time = time.time() - scene_solve_perception_start_time
            logger.info(f"Scene solve perception took {time.time() - scene_solve_perception_start_time:.2f}")

        if "scene_generate_questions" in self.stages_to_run:
            scene_generate_questions_start_time = time.time()
            self.scene_generate_questions(scenes, log_wandb=self.log_wandb)
            scene_generate_questions_total_time = time.time() - scene_generate_questions_start_time
            logger.info(f"Scene generate questions took {scene_generate_questions_total_time:.2f} seconds")

        if "scene_generate_maps" in self.stages_to_run:
            scene_generate_maps_start_time = time.time()
            self.scene_generate_maps(scenes, log_wandb=self.log_wandb)
            scene_generate_maps_total_time = time.time() - scene_generate_maps_start_time
            logger.info(f"Scene generate map questions took {scene_generate_maps_total_time:.2f} seconds")

        if "scene_generate_paraphrase" in self.stages_to_run:
            scene_generate_paraphrase_start_time = time.time()
            self.scene_generate_paraphrase(scenes, log_wandb=self.log_wandb)
            scene_generate_paraphrase_total_time = time.time() - scene_generate_paraphrase_start_time
            logger.info(f"Scene generate paraphrase took {scene_generate_paraphrase_total_time:.2f} seconds")

        if "aggregate_data" in self.stages_to_run:
            aggregate_data_start_time = time.time()
            self.aggregate_data(scenes)
            aggregate_data_total_time = time.time() - aggregate_data_start_time
            logger.info(f"Aggregate data took {aggregate_data_total_time:.2f} seconds")

        if "aggregate_cognitive_mapping" in self.stages_to_run:
            aggregate_cognitive_mapping_start_time = time.time()
            self.aggregate_cognitive_mapping(scenes)
            aggregate_cognitive_mapping_total_time = time.time() - aggregate_cognitive_mapping_start_time
            logger.info(f"Aggregate cognitive mapping took {aggregate_cognitive_mapping_total_time:.2f} seconds")

        if "filter_questions" in self.stages_to_run:
            filter_data_start_time = time.time()
            self.question_filtering()
            filter_data_total_time = time.time() - filter_data_start_time
            logger.info(f"Aggregate data took {filter_data_total_time:.2f} seconds")

        question_stats = None
        scene_stats = None

        if self.log_wandb:
            run_name = str(self.base_dir).split("/")[-1]
            wandb.init(name=run_name, project="Cosmic_Datagen", entity="collaborative-spatial-intelligence")
            table = wandb.Table(columns=[
                "Scene_Image_0", "Scene_Image_1", "Questions", "Map_Questions", "Questions_Paraphrase",
                "Scene_Filter_Result", "Scene_Filter_Reason",
                "Bound_Objects_Camera_0", "Bound_Objects_Camera_1",
                "Blender_Visible_Objects",
                "LLM_Visible_Objects",
                "LLM_Visible_Objects_Color", "LLM_Visible_Objects_Descriptions", "Full_Descriptions", "agent_1_file", "agent_2_file",
                "Scene_Path", "Asset_Parameters", "Coarse_Log", "Blender_Cameras", "Blender_Colors"])

            question_stats = {
                "total_questions": 0,
                "global_counting_questions": 0,
                "anchor_recognition_questions": 0,
                "relative_distance_questions": 0,
                "relative_direction_questions": 0
            }
            scene_stats = {
                "living": 0,
                "dining": 0,
                "kitchen": 0,
                "bedroom": 0,
                "bathroom": 0,
                "total_scenes": 0,
                "accepted_scenes": self.accepted_scenes,
                "rejected_scenes": self.rejected_scenes
            }


            for idx, (scene, data) in enumerate(self.wandb_dict.items()):
                if scene in self.successful_scenes:
                    scene_path = str(scene)
                    scene_image_0 = data["scene_image_0"] if "scene_image_0" in data.keys() else None
                    scene_image_1 = data["scene_image_1"] if "scene_image_1" in data.keys() else None
                    scene_filter_result = data["scene_filter_result"] if "scene_filter_result" in data.keys() else None
                    scene_filter_reason = data["scene_filter_reason"] if "scene_filter_reason" in data.keys() else None
                    blender_visible_objects = data["blender_visible_objects"] if "blender_visible_objects" in data.keys() else None
                    blender_cameras = data["blender_cameras"] if "blender_cameras" in data.keys() else None
                    blender_colors = data["blender_colors"] if "blender_colors" in data.keys() else None
                    llm_visible_objects = data["llm_visible_objects"] if "llm_visible_objects" in data.keys() else None
                    bound_objects_camera_0_img = data["bound_objects_camera_0_img"] if "bound_objects_camera_0_img" in data.keys() else None
                    bound_objects_camera_1_img = data["bound_objects_camera_1_img"] if "bound_objects_camera_1_img" in data.keys() else None
                    llm_visible_objects_color = data["llm_visible_objects_color"] if "llm_visible_objects_color" in data.keys() else None
                    llm_visible_objects_descriptions = data["llm_visible_objects_descriptions"] if "llm_visible_objects_descriptions" in data.keys() else None
                    full_descriptions = data["full_descriptions"] if "full_descriptions" in data.keys() else None
                    agent_1_file = data["agent_1_file"] if "agent_1_file" in data.keys() else None
                    agent_2_file = data["agent_2_file"] if "agent_2_file" in data.keys() else None
                    questions = data["questions"] if "questions" in data.keys() else None

                    # Handle map questions - can be structured data or JSON string
                    # Only process if scene_generate_maps stage was run
                    cognitive_mapping_display = None
                    if "scene_generate_maps" in self.stages_to_run:
                        cognitive_mapping_data = data.get("cognitive_mapping", None)
                        cognitive_mapping_json_f1 = data.get("cognitive_mapping_json_f1", None)
                        cognitive_mapping_json_f2 = data.get("cognitive_mapping_json_f2", None)

                        # Create a formatted string representation of map questions for the table
                        if cognitive_mapping_data is not None:
                            try:
                                # Create a summary string with images and question details
                                map_summary_parts = []
                                for agent_key in ["agent_1", "agent_2"]:
                                    if agent_key in cognitive_mapping_data and cognitive_mapping_data[agent_key]:
                                        agent_questions = cognitive_mapping_data[agent_key]
                                        map_summary_parts.append(f"{agent_key.upper()}: {len(agent_questions)} questions")
                                        for q_data in agent_questions:
                                            q_idx = q_data.get("question_index", -1)
                                            correct_ans = q_data.get("correct_answer_format1", "N/A")
                                            option_cats = q_data.get("option_categories", [])
                                            map_summary_parts.append(
                                                f"  Q{q_idx}: Correct={correct_ans}, Categories={option_cats}"
                                            )
                                cognitive_mapping_display = "\n".join(map_summary_parts) if map_summary_parts else "No map questions"
                            except Exception as e:
                                logger.warning(f"Failed to format map questions display: {e}")
                                cognitive_mapping_display = cognitive_mapping_json_f1 if cognitive_mapping_json_f1 else None
                        else:
                            cognitive_mapping_display = cognitive_mapping_json_f1 if cognitive_mapping_json_f1 else None

                    questions_paraphrase = data["questions_paraphrase"] if "questions_paraphrase" in data.keys() else None
                    print("data: ", data.keys())
                    print("questions_paraphrase: ", questions_paraphrase)

                    asset_parameters = data["asset_parameters"] if "asset_parameters" in data.keys() else None
                    coarse_log = data["coarse_log"] if "coarse_log" in data.keys() else None

                    if "living" in scene_path.lower():
                        scene_stats["living"] += 1
                    elif "kitchen" in scene_path.lower():
                        scene_stats["kitchen"] += 1
                    elif "bedroom" in scene_path.lower():
                        scene_stats["bedroom"] += 1
                    elif "bathroom" in scene_path.lower():
                        scene_stats["bathroom"] += 1
                    elif "dining" in scene_path.lower():
                        scene_stats["dining"] += 1
                    scene_stats["total_scenes"] += 1

                    if questions_paraphrase:
                        questions_paraphrase_dict = json.loads(questions_paraphrase)
                        if "global_counting_questions" in questions_paraphrase_dict.keys():
                            question_stats["global_counting_questions"] += len(questions_paraphrase_dict["global_counting_questions"])
                            question_stats["total_questions"] += len(questions_paraphrase_dict["global_counting_questions"])
                        if "anchor_recognition_questions" in questions_paraphrase_dict.keys():
                            question_stats["anchor_recognition_questions"] += len(questions_paraphrase_dict["anchor_recognition_questions"])
                            question_stats["total_questions"] += len(questions_paraphrase_dict["anchor_recognition_questions"])
                        if "relative_distance_questions" in questions_paraphrase_dict.keys():
                            question_stats["relative_distance_questions"] += len(questions_paraphrase_dict["relative_distance_questions"])
                            question_stats["total_questions"] += len(questions_paraphrase_dict["relative_distance_questions"])
                        if "relative_direction_questions" in questions_paraphrase_dict.keys():
                            question_stats["relative_direction_questions"] += len(questions_paraphrase_dict["relative_direction_questions"])
                            question_stats["total_questions"] += len(questions_paraphrase_dict["relative_direction_questions"])
                    else:
                        print("Questions paraphrase file error")
                        # print("Scene: ", scene)
                        # raise ValueError("Questions paraphrase file is missing")

                    # Log individual map question images to wandb
                    # Only process if scene_generate_maps stage was run
                    map_question_images_agent_1 = []
                    map_question_images_agent_2 = []
                    if "scene_generate_maps" in self.stages_to_run:
                        cognitive_mapping_data = data.get("cognitive_mapping", None)
                        if cognitive_mapping_data is not None:
                            try:
                                for agent_key in ["agent_1", "agent_2"]:
                                    if agent_key in cognitive_mapping_data and cognitive_mapping_data[agent_key]:
                                        for q_data in cognitive_mapping_data[agent_key]:
                                            map_img = q_data.get("map_image")
                                            if map_img is not None:
                                                if agent_key == "agent_1":
                                                    map_question_images_agent_1.append(map_img)
                                                else:
                                                    map_question_images_agent_2.append(map_img)
                            except Exception as e:
                                logger.warning(f"Failed to extract map question images: {e}")

                    table.add_data(
                        scene_image_0,
                        scene_image_1,
                        questions,
                        cognitive_mapping_display,
                        questions_paraphrase,
                        scene_filter_result,
                        scene_filter_reason,
                        bound_objects_camera_0_img,
                        bound_objects_camera_1_img,
                        blender_visible_objects,
                        llm_visible_objects,
                        llm_visible_objects_color,
                        llm_visible_objects_descriptions,
                        full_descriptions,
                        agent_1_file,
                        agent_2_file,
                        scene_path,
                        asset_parameters,
                        coarse_log,
                        blender_cameras,
                        blender_colors
                        )

                    # Log map question images separately for better visualization
                    # Note: Individual image logging is now handled in the dedicated Map_Questions_Detailed table
                    # This section can be used for additional per-scene summaries if needed
                else:
                    print("Scene not found in successful scenes")

            print("len(self.wandb_dict): ", len(self.wandb_dict))
            print("len(self.successful_scenes): ", len(self.successful_scenes))


            wandb.log({"DatagenPipeline": table})

            # Create a dedicated table for map questions with images and metadata
            # Only create if scene_generate_maps stage was run
            if "scene_generate_maps" in self.stages_to_run:
                cognitive_mapping_table = wandb.Table(columns=[
                    "Scene_Path", "Agent", "Question_Index", "Question_Text",
                    "Map_Image", "Options_Format1", "Options_Format2",
                    "Correct_Index", "Correct_Answer_Format1", "Correct_Answer_Format2",
                    "Num_Objects", "Option_Categories", "Map_Image_Path"
                ])

                for scene, data in self.wandb_dict.items():
                    if scene in self.successful_scenes and "cognitive_mapping" in data:
                        cognitive_mapping_data = data.get("cognitive_mapping")
                        if cognitive_mapping_data is not None:
                            scene_path = str(scene)
                            for agent_key in ["agent_1", "agent_2"]:
                                if agent_key in cognitive_mapping_data and cognitive_mapping_data[agent_key]:
                                    for q_data in cognitive_mapping_data[agent_key]:
                                        # Format correct_answer_format2 for display (it's a dict)
                                        correct_ans_f2 = q_data.get("correct_answer_format2")
                                        correct_ans_f2_str = json.dumps(correct_ans_f2, indent=2) if correct_ans_f2 else "N/A"

                                        # Handle map_image - always create fresh wandb.Image from path
                                        # Don't use stored wandb.Image objects as they may not serialize properly
                                        map_image_path = q_data.get("map_image_path", "")
                                        map_image = None

                                        # Always create fresh wandb.Image from path if it exists
                                        if map_image_path and os.path.exists(map_image_path):
                                            try:
                                                map_image = wandb.Image(map_image_path)
                                            except Exception as e:
                                                logger.warning(f"Failed to create wandb.Image from {map_image_path}: {e}")
                                                map_image = None

                                        try:
                                            cognitive_mapping_table.add_data(
                                                scene_path,
                                                agent_key,
                                                q_data.get("question_index", -1),
                                                q_data.get("question_text", ""),
                                                map_image,  # Can be None - wandb should handle it, but we'll catch errors
                                                json.dumps(q_data.get("options_format1", []), indent=2),
                                                json.dumps(q_data.get("options_format2", []), indent=2),
                                                q_data.get("correct_index", -1),
                                                q_data.get("correct_answer_format1", "N/A"),
                                                correct_ans_f2_str,
                                                q_data.get("num_objects", 0),
                                                json.dumps(q_data.get("option_categories", []), indent=2),
                                                map_image_path
                                            )
                                        except Exception as e:
                                            logger.warning(f"Failed to add map question data to wandb table for scene {scene_path}, agent {agent_key}, question {q_data.get('question_index', -1)}: {e}")
                                            # Try again without the map_image (use None explicitly)
                                            try:
                                                cognitive_mapping_table.add_data(
                                                    scene_path,
                                                    agent_key,
                                                    q_data.get("question_index", -1),
                                                    q_data.get("question_text", ""),
                                                    None,  # Explicitly use None
                                                    json.dumps(q_data.get("options_format1", []), indent=2),
                                                    json.dumps(q_data.get("options_format2", []), indent=2),
                                                    q_data.get("correct_index", -1),
                                                    q_data.get("correct_answer_format1", "N/A"),
                                                    correct_ans_f2_str,
                                                    q_data.get("num_objects", 0),
                                                    json.dumps(q_data.get("option_categories", []), indent=2),
                                                    map_image_path
                                                )
                                            except Exception as e2:
                                                logger.error(f"Failed to add map question data even with None image: {e2}")
                                                # Skip this row if we can't add it
                                                continue

                # Only log the table if it has data
                if cognitive_mapping_table.data and len(cognitive_mapping_table.data) > 0:
                    try:
                        wandb.log({"Map_Questions_Detailed": cognitive_mapping_table})
                        logger.info(f"Successfully logged {len(cognitive_mapping_table.data)} map question rows to wandb")
                    except Exception as e:
                        logger.error(f"Failed to log Map_Questions_Detailed table to wandb: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.warning("Map_Questions_Detailed table is empty, skipping wandb logging")
            wandb.log({"Total Scenes": len(scenes)})
            wandb.log({"Successful Scenes": len(self.successful_scenes)})
            wandb.log({"Failed Scenes": len(self.failed_scenes)})
            wandb.log({"Total Execution Time": time.time() - start_time})
            wandb.log({"Scene Filtering Time": scene_filtering_total_time})
            wandb.log({"Scene Object Info Time": scene_object_info_total_time})
            wandb.log({"Scene Camera Info Time": scene_camera_info_total_time})
            wandb.log({"Scene Blender Color Info Time": scene_blender_color_info_total_time})
            wandb.log({"Scene LLM Visible Objects Time": scene_llm_visible_objects_total_time})
            wandb.log({"Scene Bound Objects Time": scene_bound_objects_total_time})
            wandb.log({"Scene Object Color Info Time": scene_obj_color_info_total_time})
            wandb.log({"Scene Generate Descriptions Time": scene_generate_descriptions_total_time})
            wandb.log({"Scene Solve Perception Time": scene_solve_perception_total_time})
            wandb.log({"Scene Generate Questions Time": scene_generate_questions_total_time})
            if "scene_generate_maps" in self.stages_to_run:
                wandb.log({"Scene Generate Maps Time": scene_generate_maps_total_time})
            wandb.log({"Scene Generate Paraphrase Time": scene_generate_paraphrase_total_time})
            wandb.log({"Num Global Counting Questions": self.wandb_dict["num_global_counting_questions"]})
            wandb.log({"Num Anchor Recognition Questions": self.wandb_dict["num_anchor_recognition_questions"]})
            wandb.log({"Num Relative Distance Questions": self.wandb_dict["num_relative_distance_questions"]})
            wandb.log({"Num Relative Direction Questions": self.wandb_dict["num_relative_direction_questions"]})
            if "scene_generate_maps" in self.stages_to_run:
                wandb.log({"Num Map Questions Agent 1": self.wandb_dict.get("num_cognitive_mapping_agent_1", 0)})
                wandb.log({"Num Map Questions Agent 2": self.wandb_dict.get("num_cognitive_mapping_agent_2", 0)})
                wandb.log({"Total Map Questions": self.wandb_dict.get("num_cognitive_mapping_agent_1", 0) + self.wandb_dict.get("num_cognitive_mapping_agent_2", 0)})
            wandb.log({"Global Counting Scenes": len(list(set(self.wandb_dict["global_counting_scenes"])))})
            wandb.log({"Anchor Recognition Scenes": len(list(set(self.wandb_dict["anchor_recognition_scenes"])))})
            wandb.log({"Relative Distance Scenes": len(list(set(self.wandb_dict["relative_distance_scenes"])))})
            wandb.log({"Relative Direction Scenes": len(list(set(self.wandb_dict["relative_direction_scenes"])))})
            wandb.log({"No Questions Scenes": len(list(set(self.wandb_dict["no_questions_scenes"])))})

            q_stats_data = [
                ["Total Questions", question_stats["total_questions"]],
                ["Global Counting Questions", question_stats["global_counting_questions"]],
                ["Anchor Recognition Questions", question_stats["anchor_recognition_questions"]],
                ["Relative Distance Questions", question_stats["relative_distance_questions"]],
                ["Relative Direction Questions", question_stats["relative_direction_questions"]],
            ]
            # Only add map questions to stats if scene_generate_maps stage was run
            if "scene_generate_maps" in self.stages_to_run:
                total_cognitive_mapping = self.wandb_dict.get("num_cognitive_mapping_agent_1", 0) + self.wandb_dict.get("num_cognitive_mapping_agent_2", 0)
                q_stats_data.append(["Map Questions", total_cognitive_mapping])
            q_stats_table = wandb.Table(data=q_stats_data, columns=["Question_Type", "Number_of_Questions"])
            wandb.log({"Question Stats":
                wandb.plot.bar(q_stats_table,
                    label="Question_Type",
                    value="Number_of_Questions",
                    title="Question Stats Bar Chart",
                    )})

            scene_stats_data = [
                ["Living Rooms", scene_stats["living"]],
                ["Kitchens", scene_stats["kitchen"]],
                ["Bedrooms", scene_stats["bedroom"]],
                ["Bathrooms", scene_stats["bathroom"]],
                ["Dining Rooms", scene_stats["dining"]],
                ["Total Scenes", scene_stats["total_scenes"]],
                ["Accepted Scenes", scene_stats["accepted_scenes"]],
                ["Rejected Scenes", scene_stats["rejected_scenes"]],
            ]
            scene_stats_table = wandb.Table(data=scene_stats_data, columns=["Scene_Type", "Number_of_Scenes"])
            wandb.log({"Scene Stats":
                wandb.plot.bar(scene_stats_table,
                    label="Scene_Type",
                    value="Number_of_Scenes",
                    title="Scene Stats Bar Chart",
                    )})
            wandb.finish()

        self.print_summary(
            start_time,
            scene_filtering_total_time,
            scene_object_info_total_time,
            scene_camera_info_total_time,
            scene_blender_color_info_total_time,
            scene_llm_visible_objects_total_time,
            scene_bound_objects_total_time,
            scene_obj_color_info_total_time,
            scene_generate_descriptions_total_time,
            scene_solve_perception_total_time,
            scene_generate_questions_total_time,
            scene_generate_maps_total_time,
            scene_generate_paraphrase_total_time,
            question_stats
            )


    def print_summary(
        self,
        start_time: float,
        scene_filtering_total_time: float,
        scene_object_info_total_time: float,
        scene_camera_info_total_time: float,
        scene_blender_color_info_total_time: float,
        scene_llm_visible_objects_total_time: float,
        scene_bound_objects_total_time: float,
        scene_obj_color_info_total_time: float,
        scene_generate_descriptions_total_time: float,
        scene_solve_perception_total_time: float,
        scene_generate_questions_total_time: float,
        scene_generate_maps_total_time: float,
        scene_generate_paraphrase_total_time: float,
        question_stats: dict = None
        ):
        """Print pipeline execution summary."""
        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total scenes processed: {self.processed_scenes}")
        logger.info(f"Successful scenes: {len(self.successful_scenes)}")
        logger.info(f"Failed scenes: {len(self.failed_scenes)}")
        logger.info(f"Total execution time: {duration:.2f} seconds")
        logger.info(f"Average time per scene: {duration/max(1, self.processed_scenes):.2f} seconds")
        logger.info(f"Scene Filtering Time: {scene_filtering_total_time:.2f} seconds")
        logger.info(f"Scene Object Info Time: {scene_object_info_total_time:.2f} seconds")
        logger.info(f"Scene Camera Info Time: {scene_camera_info_total_time:.2f} seconds")
        logger.info(f"Scene Blender Color Info Time: {scene_blender_color_info_total_time:.2f} seconds")
        logger.info(f"Scene LLM Visible Objects Time: {scene_llm_visible_objects_total_time:.2f} seconds")
        logger.info(f"Scene Bound Objects Time: {scene_bound_objects_total_time:.2f} seconds")
        logger.info(f"Scene Object Color Info Time: {scene_obj_color_info_total_time:.2f} seconds")
        logger.info(f"Scene Generate Descriptions Time: {scene_generate_descriptions_total_time:.2f} seconds")
        logger.info(f"Scene Solve Perception Time: {scene_solve_perception_total_time:.2f}")
        logger.info(f"Scene Generate Questions Time: {scene_generate_questions_total_time:.2f} seconds")
        if "scene_generate_maps" in self.stages_to_run:
            logger.info(f"Scene Generate Maps Time: {scene_generate_maps_total_time:.2f} seconds")
        logger.info(f"Scene Generate Paraphrase Time: {scene_generate_paraphrase_total_time:.2f} seconds")
        logger.info(f"Question Stats: {question_stats}")

        if self.failed_scenes:
            logger.info(f"\nFailed scenes:")
            for scene_name, error in self.failed_scenes.items():
                logger.info(f"  - {scene_name}")
                logger.info(f"    - Error: {error}")


def main():
    parser = argparse.ArgumentParser(description="Datagen Pipeline")
    parser.add_argument(
        "--base_dir",
        default="./scenes",
        help="Base directory containing all scenes or a specific room directory"
    )
    parser.add_argument(
        "--scene_datafile",
        default=str(Path(__file__).parent / "data" / "dataset_scenes.json"),
        help="Scene data file"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--stages_to_run",
        nargs="+",
        default=["scene_object_info", "scene_camera_info", "scene_llm_visible_objects", "scene_solve_perception", "scene_bound_objects", "scene_blender_color_info", "scene_obj_color_info", "scene_generate_descriptions", "scene_generate_questions", "scene_generate_maps", "scene_generate_paraphrase", "aggregate_data", "aggregate_cognitive_mapping"],
        help="Stages to run"
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (randomly sampled for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--client_scene_filtering",
        type=str,
        default="openai",
        help="Client name"
    )
    parser.add_argument(
        "--model_name_scene_filtering",
        type=str,
        default="gpt-4o-mini",
        help="Model name"
    )
    parser.add_argument(
        "--api_base_scene_filtering",
        type=str,
        default="https://api.openai.com/v1",
        help="API base"
    )
    parser.add_argument(
        "--client_color",
        type=str,
        default="openai",
        help="Client name"
    )
    parser.add_argument(
        "--model_name_color",
        type=str,
        default="gpt-4o-mini",
        help="Model name"
    )
    parser.add_argument(
        "--api_base_color",
        type=str,
        default="https://api.openai.com/v1",
        help="API base"
    )
    parser.add_argument(
        "--client_vis_objects",
        type=str,
        default="openai",
        help="Client name"
    )
    parser.add_argument(
        "--model_name_vis_objects",
        type=str,
        default="gpt-4o-mini",
        help="Model name"
    )
    parser.add_argument(
        "--api_base_vis_objects",
        type=str,
        default="https://api.openai.com/v1",
        help="API base"
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Log to wandb"
    )
    parser.add_argument(
        "--client_paraphrase",
        type=str,
        default="openai",
        help="Client name"
    )
    parser.add_argument(
        "--model_name_paraphrase",
        type=str,
        default="gpt-4o-mini",
        help="Model name for paraphrasing"
    )
    parser.add_argument(
        "--api_base_paraphrase",
        type=str,
        default="https://api.openai.com/v1",
        help="API base for paraphrasing"
    )
    parser.add_argument(
        "--overwrite_files",
        action="store_true",
        help="Overwrite files"
    )
    parser.add_argument(
        "--question_version",
        type=str,
        default="V1",
        help="Question version"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")

    pipeline = DatagenPipeline(
        base_dir=args.base_dir,
        scene_datafile=args.scene_datafile,
        dry_run=args.dry_run,
        stages_to_run=args.stages_to_run,
        max_scenes=args.max_scenes,
        seed=args.seed,
        client_scene_filtering=args.client_scene_filtering,
        model_name_scene_filtering=args.model_name_scene_filtering,
        api_base_scene_filtering=args.api_base_scene_filtering,
        client_color=args.client_color,
        model_name_color=args.model_name_color,
        api_base_color=args.api_base_color,
        client_vis_objects=args.client_vis_objects,
        model_name_vis_objects=args.model_name_vis_objects,
        api_base_vis_objects=args.api_base_vis_objects,
        api_key=api_key,
        log_wandb=args.log_wandb,
        client_paraphrase = args.client_paraphrase,
        model_name_paraphrase = args.model_name_paraphrase,
        api_base_paraphrase= args.api_base_paraphrase,
        overwrite_files=args.overwrite_files,
        question_version=args.question_version,
    )
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
