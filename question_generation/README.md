<h1 align="center">Question Generation Pipeline:<br>From Rendered Views to COSMIC Questions</h1>

<div align="justify">

> COSMIC requires a large, consistent set of question-answer pairs that are grounded in two partial (egocentric) views of the same 3D environment.
> This folder provides the Blender scripts and Python orchestration used to convert rendered scene data into object-level descriptions, multiple-choice questions, and paraphrased variants.

## Overview
The pipeline follows a simple sequence:

1. Use Blender to export per-camera object geometry (2D bounding boxes and 3D bounding-box corners).
2. Use an LLM/VLM to detect visible objects and normalize their 2D bounding boxes to a fixed JSON format.
3. Detect dominant colors and generate unique natural-language descriptions per object.
4. Generate task-specific multiple-choice questions from the resulting structured representation.
5. (Optional) Paraphrase questions to improve linguistic variety while preserving semantics.
6. (Optional) Visualize object bounds on the input images for debugging and sanity checks.
7. `../datagen_pipeline` is a script that automates the process of question generation

## Generation Pipeline
To generate the full question set for a single scene:

1. `cd` into the directory that contains the Blender file and the two camera images (or use `../datagen_pipeline` for automation).
   - The Blender file must be named `scene.blend`
   - The camera image filenames must be:
     - `Image_0_0_0048_0.png` for camera 0
     - `Image_1_0_0048_0.png` for camera 1
2. Ensure `utils.py` is available one directory up (`..`) so the Python scripts can access it to create the vLLM server client (and other shared utilities).
3. Export visible objects from Blender:
```bash
blender -b ./scene.blend -P ../get_object_info.py --output_json ./visible_objects.json
blender -b ./scene.blend -P ../get_camera_info.py --output_json ./cameras.json
```
4. Detect visible objects (LLM/VLM):
```bash
python ../llm_visible_objects.py --input_json ./visible_objects.json --output_json ./llm_detected_objects.json
```
5. Detect dominant colors for each detected object:
```bash
python ../get_color_info.py \
  --input_json ./llm_detected_objects.json \
  --output_json ./llm_detected_objects_color.json \
  --api_base https://api.openai.com/v1 \
  --client openai \
  --model_name gpt-5-mini \
  --api_key <key> \
  --scene ""
```
6. Generate object-level descriptions:
```bash
python ../generate_descriptions.py \
  --input_json ./llm_detected_objects_color.json \
  --output_json ./visible_objects_with_descriptions.json
```
7. Generate questions:
```bash
python ../generate_questions.py \
  --input_json ./visible_objects_with_descriptions.json \
  --output_json ./questions.json
```
8. Paraphrase questions (optional):
```bash
python ../paraphrase_questions.py \
  --input_json ./questions.json \
  --output_json ./questions_paraphrased.json \
  --api_base "https://api.openai.com/v1" \
  --client openai \
  --model_name "gpt-4o-mini" \
  --api_key ""
```
9. Visualize bounds on the input image(s) (optional; for debugging):
```bash
python ../bound_objects.py ./Image_0_0_0048_0.png ./llm_detected_objects.json camera_0_0 bounds/camera_0_0
python ../bound_objects.py --image ./Image_0_0_0048_0.png --llm_detected_objects_json ./visible_objects.json --camera_key camera_0_0 --output_dir bounds/camera_0_0
```

## File Layout
The files in this directory fall into three roles: shared utilities (`utils.py` / `llm_utils.py`), LLM/VLM-based stages (`llm_visible_objects.py`, `get_color_info.py`, `generate_descriptions.py`, `generate_questions.py`, `paraphrase_questions.py`), and Blender-export / rendering helpers (`get_object_info.py`, `get_camera_info.py`, `get_top_view.py`, `get_isometric_view.py`, `get_blender_color*.py`).

| File | Description |
|---|---|
| `utils.py` | Shared category logic (e.g., mapping raw object names to canonical categories) and the canonical `allowed_categories` list. |
| `llm_utils.py` | Helper code for calling OpenAI or a local vLLM server (async inference wrapper, client creation, API-base handling, and shared encoding utilities). |
| `llm_visible_objects.py` | Uses an LLM/VLM to propose 2D bounding boxes for visible objects given Blender-exported `visible_objects.json`. Writes `llm_detected_objects.json`. |
| `get_color_info.py` | Runs color detection for each object (dominant color prompts, optional caching via Blender-exported colors), producing `llm_detected_objects_color.json`. |
| `generate_descriptions.py` | Generates discriminative natural-language descriptions for objects from structured color + detection outputs, writing `visible_objects_with_descriptions.json`. |
| `generate_questions.py` | Builds task-specific multiple-choice questions from `visible_objects_with_descriptions.json` and writes `questions.json`. |
| `paraphrase_questions.py` | Uses an LLM to paraphrase questions/options while preserving focus and semantics, producing `questions_paraphrased.json`. |
| `bound_objects.py` | Debug visualization: overlays predicted/structured bounding boxes onto input images and saves annotated outputs (e.g., into `bounds/<camera_key>/`). |
| `get_object_info.py` | Blender script: exports per-camera visible objects, including normalized 2D bounding boxes and 3D bounding-box corners. Writes `visible_objects.json`. |
| `get_camera_info.py` | Blender script: exports camera intrinsics/extrinsics for camera objects. Writes `cameras.json`. |
| `get_top_view.py` | Blender rendering helper: renders a top-view style orthographic camera image (and related scene modifications like hiding exterior/ceiling). |
| `get_isometric_view.py` | Blender rendering helper: renders an isometric/tilted orthographic camera image (with similar render configuration and scene handling to `get_top_view.py`). |
| `get_blender_color.py` | Blender color-extraction script (material-based sampling) that exports a `blender_colors.json` cache for consistent color grounding. |
| `get_blender_color_v2.py` | Improved/alternative Blender color-extraction logic (more robust node tracing and color retrieval) for producing `blender_colors.json`. |
| `perception_solving_descriptions.py` | Utility stage that converts per-camera visible-object JSON into concise 'category + bbox' text inputs (intended for perception/agent reasoning style pipelines). |

</div>