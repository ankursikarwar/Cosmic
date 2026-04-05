# COSMIC Data Generation Pipeline

This directory contains the pipeline for generating the COSMIC benchmark dataset from raw 3D scenes.

## Overview

The pipeline processes indoor 3D scenes through a sequence of stages to produce multi-choice QA pairs across five tasks: Anchor Recognition, Global Counting, Relative Distance, Relative Direction, and Cognitive Mapping.

**Entry point:** `datagen/pipeline.py`  
**Output:** `datagen/data/dataset_<task>_<version>.json`

---

## Directory Structure

```
datagen/
├── pipeline.py                        # Main pipeline script
├── scene_filtering.py                 # LLM-based scene quality filtering
├── data/                              # Output directory (auto-created)
│   ├── dataset_scenes.json            # Registry of all processed scenes
│   ├── dataset_global_counting.json
│   ├── dataset_anchor_recognition.json
│   ├── dataset_relative_distance.json
│   ├── dataset_relative_direction.json
│   ├── dataset_cognitive_mapping.json
│   └── datagen_pipeline.log
└── question_generation/               # Per-scene processing scripts
    ├── get_object_info.py             # Blender: export visible objects + bboxes
    ├── get_camera_info.py             # Blender: export camera intrinsics/extrinsics
    ├── get_blender_color.py        # Blender: extract material colors
    ├── llm_visible_objects.py         # VLM: detect visible objects in images
    ├── get_color_info.py              # VLM: detect dominant object colors
    ├── generate_descriptions.py       # Generate natural language object descriptions
    ├── generate_questions.py          # Generate MCQs for 4 tasks
    ├── map_gen.py                     # Generate Cognitive Mapping questions
    ├── paraphrase_questions.py        # LLM paraphrasing of questions/options
    ├── aggregate_map_questions.py     # Aggregate cognitive mapping data
    ├── bound_objects.py               # Debug: bbox visualization overlay
    ├── perception_solving_descriptions.py  # Convert visible-object JSON to agent text inputs
    ├── utils.py                       # Category mapping, allowed_categories
    ├── llm_utils.py                   # OpenAI/vLLM async inference wrapper
    └── consistent_color_mapping.py    # Canonical color name mapping
```

---

## Scene Data Format

Each scene directory must have the following structure:

```
<scene_dir>/
├── frames/Image/camera_0/
│   ├── Image_0_0_0048_0.png    # Camera 0 image
│   └── Image_1_0_0048_0.png    # Camera 1 image
└── coarse/
    └── scene.blend              # Blender scene file (required for Blender stages)
```

The `--base_dir` argument accepts any of the following layouts:

| Layout | Description |
|---|---|
| Single scene | `base_dir/` is itself one scene |
| One room type | `base_dir/<scene>/` — multiple scenes, one room type |
| Multiple room types | `base_dir/<room_type>/<scene>/` |
| Multiple folders | `base_dir/<folder>/<room_type>/<scene>/` |

---

## Pipeline Stages

The pipeline runs the following stages in sequence. Each stage is optional and can be selected via `--stages_to_run`.

| Stage | Description | Output per scene |
|---|---|---|
| `scene_object_info` | Blender: exports all visible objects with bboxes | `visible_objects.json` |
| `scene_camera_info` | Blender: exports camera intrinsics/extrinsics | `cameras.json` |
| `scene_blender_color_info` | Blender: extracts material colors for objects | `blender_colors.json` |
| `scene_llm_visible_objects` | VLM: detects visible objects in each camera image | `llm_detected_objects.json` |
| `scene_bound_objects` | Overlays bboxes on images for debugging | `bounds/camera_0_0/`, `bounds/camera_1_0/` |
| `scene_obj_color_info` | VLM: detects dominant color of each detected object | `llm_detected_objects_colors.json` |
| `scene_generate_descriptions` | Generates natural language descriptions for each object | `visible_objects_with_descriptions.json` |
| `scene_solve_perception` | Converts object data to agent text inputs | `agent_1_input.txt`, `agent_2_input.txt` |
| `scene_generate_questions` | Generates MCQs for Anchor, Counting, Distance, Direction | `questions.json` |
| `scene_generate_maps` | Generates Cognitive Mapping questions | `cognitive_mapping.json`, `cognitive_mapping/` |
| `scene_generate_paraphrase` | LLM paraphrasing of questions and answer options | `questions_paraphrased.json` |
| `aggregate_data` | Aggregates per-scene questions into dataset JSON files | `datagen/data/dataset_<task>_<version>.json` |
| `aggregate_cognitive_mapping` | Aggregates cognitive mapping data into dataset JSON | `datagen/data/dataset_cognitive_mapping_<version>.json` |

---

## Setup

### Scene Generation

The 3D indoor scenes used as input to this pipeline are generated using a modified version of Infinigen, available at [ankursikarwar/infinigen (cosmic branch)](https://github.com/ankursikarwar/infinigen/tree/cosmic). Refer to that repository for instructions on generating scenes.

### Blender

Several stages (`scene_object_info`, `scene_camera_info`, `scene_blender_color_info`) require Blender. The pipeline looks for a local Blender installation first, then falls back to the system `blender` command.

To set up a local Blender installation:

```bash
cd datagen/question_generation
wget https://download.blender.org/release/Blender4.5/blender-4.5.3-linux-x64.tar.xz
tar -xf blender-4.5.3-linux-x64.tar.xz
rm blender-4.5.3-linux-x64.tar.xz
```

The pipeline will automatically use `datagen/question_generation/blender-4.5.3-linux-x64/blender` if it exists.

### VLM for `scene_llm_visible_objects`

The visible object detection stage can use either an OpenAI model or an open-source model via vLLM.

**Option 1 — OpenAI model (default):**

```bash
--client_vis_objects openai \
--model_name_vis_objects gpt-4o-mini \
--api_base_vis_objects https://api.openai.com/v1
```

Requires `OPENAI_API_KEY` set in `.env`.

**Option 2 — Open-source model via vLLM:**

First launch a vLLM server:

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 4877 --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

Then pass:

```bash
--client_vis_objects vllm \
--model_name_vis_objects Qwen/Qwen2.5-VL-72B-Instruct \
--api_base_vis_objects http://<hostname>:4877/v1
```

### API Keys

Set `OPENAI_API_KEY` in your `.env` file (at the project root). This is used by the scene filtering, color detection, and paraphrasing stages which default to OpenAI models.

---

## Running the Pipeline

Run from the **project root**:

```bash
python -m datagen.pipeline \
    --base_dir /path/to/scenes \
    --stages_to_run scene_object_info scene_camera_info scene_llm_visible_objects \
        scene_bound_objects scene_blender_color_info scene_obj_color_info \
        scene_generate_descriptions scene_solve_perception scene_generate_questions \
        scene_generate_maps scene_generate_paraphrase \
        aggregate_data aggregate_cognitive_mapping
```

### Run only aggregation (if per-scene files already exist)

```bash
python -m datagen.pipeline \
    --base_dir /path/to/scenes \
    --stages_to_run aggregate_data aggregate_cognitive_mapping
```

### Dry run (print commands without executing)

```bash
python -m datagen.pipeline \
    --base_dir /path/to/scenes \
    --dry_run
```

### Process a limited number of scenes (for testing)

```bash
python -m datagen.pipeline \
    --base_dir /path/to/scenes \
    --max_scenes 10 \
    --seed 42
```

### Overwrite existing per-scene files

```bash
python -m datagen.pipeline \
    --base_dir /path/to/scenes \
    --overwrite_files
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--base_dir` | `./scenes` | Root directory containing scene folders |
| `--scene_datafile` | `datagen/data/dataset_scenes.json` | Path to write the scene registry JSON |
| `--stages_to_run` | all stages | Space-separated list of stages to execute |
| `--max_scenes` | `None` | Randomly sample N scenes (useful for testing) |
| `--seed` | `42` | Random seed for scene sampling |
| `--overwrite_files` | `False` | Re-run stages even if output files already exist |
| `--dry_run` | `False` | Print commands without executing |
| `--log_wandb` | `False` | Log pipeline run and stats to W&B |

### Model arguments

Each of the three model-backed stages (scene filtering, color detection, paraphrasing) has its own client/model/api_base arguments:

| Argument | Default | Stage |
|---|---|---|
| `--client_color` | `openai` | Color detection |
| `--model_name_color` | `gpt-4o-mini` | Color detection |
| `--api_base_color` | `https://api.openai.com/v1` | Color detection |
| `--client_vis_objects` | `openai` | VLM visible object detection |
| `--model_name_vis_objects` | `gpt-4o-mini` | VLM visible object detection |
| `--api_base_vis_objects` | `https://api.openai.com/v1` | VLM visible object detection |
| `--client_paraphrase` | `openai` | Paraphrasing |
| `--model_name_paraphrase` | `gpt-4o-mini` | Paraphrasing |
| `--api_base_paraphrase` | `https://api.openai.com/v1` | Paraphrasing |

---

## Output Files

All aggregated dataset files are written to `datagen/data/`:

| File | Task |
|---|---|
| `dataset_global_counting.json` | Global Counting |
| `dataset_anchor_recognition.json` | Anchor Recognition |
| `dataset_relative_distance.json` | Relative Distance |
| `dataset_relative_direction.json` | Relative Direction |
| `dataset_cognitive_mapping.json` | Cognitive Mapping |
| `dataset_scenes.json` | Registry of all scenes |
| `datagen_pipeline.log` | Full pipeline log |

Each sample in the dataset JSON contains image paths, the question, answer options, ground truth answer index, and metadata fields specific to the task.

---

## W&B Logging

Pass `--log_wandb` to log pipeline stats, scene images, generated questions, and timings to Weights & Biases. Configure your W&B credentials in `.env`:

```ini
WANDB_ENTITY=your_wandb_entity
WANDB_DIR=/path/to/wandb/cache
```
