import os
import re
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple
from pathlib import Path # Import Path for type hinting
from datagen.question_generation.llm_utils import create_vllm_client, create_openai_client
from tqdm import tqdm

# -------------------
# Paraphrasing Prompt Builder (Modified)
# -------------------
def build_paraphrase_prompt(question_obj: Dict[str, Any], category: str) -> Tuple[str, Dict[str, Any]]:
    """
    Builds the prompt and the minimal JSON object to be paraphrased.

    Returns:
        A tuple of (prompt_string, json_to_paraphrase)
    """
    base_instruction = """
You are given a JSON object containing a 'question' and/or 'options' for a multiple-choice question.
Your task is to paraphrase the text values in this JSON.
•⁠ Fix grammar and phrasing so it is natural English.
•⁠ When rewriting object options (if 'options' is provided), there may be descriptors for color, size and neighboring objects.
•⁠ The format of the object in the descriptions is always "color" + "size" + "object category" + "neighboring object color" + "neighboring object category". Some of these fields may be empty but the order will always be the same. The object is the focus, and the neighboring object is merely an additional descriptor to uniquely identify the object.
•⁠ Make sure to keep in mind the focus object when paraphrasing questions. For example if the question is "Which direction is the white shelf located near a sofa?", the focus object is "white shelf" and the neighboring object is "sofa". Under no circumstances should you paraphrase the question to "Which direction is the sofa located near a white shelf?" as this changes the focus object with the neighboring object. Keep this in mind when paraphrasing any question and object - THE FOCUS OBJECT SHOULD NOT SWAP WITH THE NEIGHBORING OBJECT.
•⁠ Neighboring objects will have their own color, be sure to not get confused when paraphrasing
• Convert names like:
   "White Desk next to a white shelf" → "white desk located next to a white shelf".
   "small Wall Art" → "small wall art".
   "Monitor" → "Computer Monitor" (Always paraphrase a monitor as a computer monitor, and do not do this for any other objects)
• Neighboring relations can be 'next', 'near', or 'on' or 'on which'. When the relation is 'on'. A 'on' B means "A is kept on B", and A 'on which' B means "A on which B is kept".
   "Black Desk on which a green lamp" → "black desk on which a green lamp is kept"
   "beige Lamp on a black Desk" → "beige lamp on a black desk"
   "Black Shelf on which a TV" → "black shelf on which a TV is kept"
• When parphrasing objects like lamp, sizes are not 'big' or 'small' but are 'floor' or 'desk'. So convert below:
   "beige, Desk Lamp on a black Shelf" → "beige desk lamp kept on a black shelf"
   "beige, Desk Lamp on a white Desk" → "beige desk lamp kept on a white desk"
   "yellow Sofa near a brown shutter window" → "yellow sofa near a window with brown shutters"
• When parphrasing objects like bed, colors refer to the color of the bedframe and not the mattress, bedsheet, etc. So be explicit and convert below:
   "beige Bed near a white Shelf" → "bed with beige bedframe near a white shelf"
   "yellow Sofa near a green bed" → "yellow sofa near a bed with green bedframe"
• When paraphrasing windows, the color is either "color + shutter" or a "color + curtain" which refers to shutter and curtain respectively. So convert below:
   "brown shutter Window near a white Shelf" → "window with brown shutters near a white shelf"
   "purple curtain Window → "window with the purple curtains"
•⁠ IMPORTANT: Do not eliminate any descriptors from any object. Also when referring to neighbouring objects, use "a" and not "the", as it is not guaranteed to be unique.
• Make every option as natural sounding as possible while retaining the unique descriptors compared to the other options
•⁠ Return *only* the paraphrased JSON object that you were given.
•⁠ Output valid JSON only, no extra text, no explanations.
"""

    # This is the minimal JSON object the model will see and paraphrase
    json_to_paraphrase = {}

    if category == "global_counting_questions":
        json_to_paraphrase["question"] = question_obj["question"]
        json_to_paraphrase["question_both_views"] = question_obj["question_both_views"]
        prompt = base_instruction + f"""
Paraphrase the 'question' and 'question_both_views' fields. Do not remove 'total' from the question.
An example paraphrase would be "How many total shelf are there in the room" → "How many total shelves are there in the room?"
When you are paraphrasing questions related to counting of lamps you are to specify that they will count both floor and desk lamps in the room. For example, "What is the total number of Lamp in the room?" → "What is the total number of floor and desk lamps in the room?"

Here is the input JSON to paraphrase:
"""
    elif category == "relative_direction_questions":
        json_to_paraphrase["question"] = question_obj["question"]
        json_to_paraphrase["question_both_views"] = question_obj["question_both_views"]
        prompt = base_instruction + f"""
Paraphrase the 'question' and 'question_both_views' fields.

Begin the question with these phrases or a paraphrase of the below (whichever fits best):
- "From your perspective,"
- "Relative to you,"
- "With respect to you,"
- any other similar phrase as above

When the main object includes descriptors like "near", "next to", or "on", ensure:
- The relationship between objects remains clear and grammatical.
- Use "that is" if needed for clarity (e.g., "the white shelf that is near the sofa").
- Avoid awkward phrasing or descriptor clashes.

Ensure the paraphrase is fluent, unambiguous, and fully preserves the original meaning.

Examples:
Original: "Which direction is the white shelf located near a sofa?"
→ "From your perspective, which direction is the white shelf that is near a sofa?"
→ "From your perspective, in which direction is the white shelf that stands near a sofa?"

IMPORTANT NOTE: Do NOT use words related to seeing or visibility (e.g., see, visible, view, spot, look at, observe) in your paraphrase. The object is NOT directly visible to the agent. Paraphrases like "In which direction do you see the white shelf (near the sofa)?" are INVALID.

Here is the input JSON to paraphrase:
"""
    elif category == "relative_distance_questions":
        json_to_paraphrase["question"] = question_obj["question"]
        json_to_paraphrase["question_both_views"] = question_obj["question_both_views"]
        json_to_paraphrase["options"] = question_obj["options"]
        prompt = base_instruction + f"""
Paraphrase ALL the 'question', 'question_both_views' fields and all strings in the 'options' list.
Example paraphrase: "Which of the following objects is closest to the Door (brown)?" → "Which of the following object is nearest to a brown door?".
Feel free to use vocabulary such as "closest", "nearest", and anything else appropriate, but make sure that your paraphrase is unambiguous.
Here is the input JSON to paraphrase:
"""
    elif category == "cognitive_mapping_questions":
        json_to_paraphrase["question"] = question_obj["question"]
        json_to_paraphrase["question_both_views"] = question_obj["question_both_views"]
        json_to_paraphrase["options"] = question_obj["options"]
        prompt = base_instruction + f"""
Paraphrase ALL the 'question', 'question_both_views' fields and all strings in the 'options' list.
The options will be a list of four JSON dictionaries. The *keys* of these dictionaries are object descriptions that must be paraphrased.
These object keys will be the same across all four option dictionaries, but their *values* (which are arrays of integers) will be different.
Ensure that you paraphrase all object keys in each option dictionary while keeping the integer array values unchanged.
Example key paraphrase: "Desk (black, on the Monitor)" → "Black desk on which the monitor is kept"
Example key paraphrase: "Sofa (beige)" → "Beige sofa"
Make sure that the paraphrasing of keys is *consistent* across all option dictionaries.
Be very careful to ensure that the returned option structure (a list of 4 dictionaries) is valid.

Here is the input JSON to paraphrase:
"""
    else: # Default/other categories
        json_to_paraphrase["question"] = question_obj["question"]
        json_to_paraphrase["question_both_views"] = question_obj["question_both_views"]
        json_to_paraphrase["options"] = question_obj["options"]
        prompt = base_instruction + f"""
Paraphrase ALL the 'question', 'question_both_views' fields and all strings in the 'options' list.
Example paraphrase: "Which of the following objects is visible in both your and your partner's views of the room?" → "Which of the following objects is present in both your and your partner's views of the room?".
Feel free to use vocabulary such as "present", "visible", "perspective", "view", and anything else appropriate, but make sure that your paraphrase is unambiguous.

Here is the input JSON to paraphrase:
"""

    return prompt, json_to_paraphrase


# -------------------
# JSON extractor (safe) (Unchanged)
# -------------------
def extract_json_from_output(text: str) -> Dict[str, Any]:
    try:
        # Try to load the whole text first
        return json.loads(text)
    except json.JSONDecodeError:
        # If it fails, find the first valid JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                 print(f"Warning: Found JSON-like block, but failed to parse: {match.group(0)}")
                 return {}
        else:
            print(f"Warning: No JSON object found in output: {text}")
            return {}

# -------------------
# Main processing (Modified)
# -------------------
def paraphrase_across_scenes(
    scenes: List[Path], # Added type hint
    client_name: str,
    model_name: str,
    api_base: str,
    api_key: str
):
    # --- Initialize client ---
    if client_name == "vllm":
        client = create_vllm_client(model_name=model_name, api_base=api_base, api_key=api_key)
    elif client_name == "openai":
        client = create_openai_client(model_name=model_name, api_base=api_base, api_key=api_key)
    else:
        raise ValueError(f"Invalid client: {client_name}")

    # --- Collect queries from all scenes ---
    all_queries = []
    # This map now stores the original question object for merging later
    query_to_scene_map = []

    print(f"Preparing paraphrase queries from {len(scenes)} scenes...")
    for scene in tqdm(scenes, desc="Scenes"):
        scene = Path(scene)
        input_path = scene / "questions.json"

        if not input_path.exists():
            print(f"⚠️ {scene.name}: {input_path.name} not found, skipping.")
            continue

        with open(input_path, "r") as f:
            data = json.load(f)

        for category, questions in data.items():
            if not questions: # Skip empty question categories
                continue

            for idx, q in enumerate(questions):
                # Get the prompt and the minimal JSON to be paraphrased
                try:
                    # Validate required fields exist
                    if "question" not in q:
                        print(f"Warning: Missing 'question' field in {scene.name} {category}[{idx}]. Skipping.")
                        continue
                    if "question_both_views" not in q:
                        print(f"Warning: Missing 'question_both_views' field in {scene.name} {category}[{idx}]. Skipping.")
                        continue
                    # For categories that need options, check they exist
                    if category in ["anchor_recognition_questions", "relative_distance_questions",
                                   "relative_direction_questions", "cognitive_mapping_questions"]:
                        if "options" not in q:
                            print(f"Warning: Missing 'options' field in {scene.name} {category}[{idx}]. Skipping.")
                            continue
                    prompt, json_to_paraphrase = build_paraphrase_prompt(q, category)
                except KeyError as e:
                    print(f"KeyError building prompt for {scene.name} {category} {idx}: {e}. Skipping q.")
                    continue
                except Exception as e:
                    print(f"Error building prompt for {scene.name} {category} {idx}: {e}. Skipping q.")
                    continue

                # The full prompt content includes the instructions and the minimal JSON
                full_prompt_text = prompt + f"\n{json.dumps(json_to_paraphrase, indent=2)}"

                chat_history = [
                    {"role": "user", "content": [{"type": "text", "text": full_prompt_text}]}
                ]

                # Increased token limit for potentially large cognitive_mapping_questions
                max_tokens = 2048
                if model_name == "gpt-5-mini":
                    query = {"messages": chat_history, "max_completion_tokens": max_tokens}
                else:
                    query = {"messages": chat_history, "max_tokens": max_tokens}

                all_queries.append(query)
                # Store the original question 'q' for merging
                query_to_scene_map.append((scene, category, idx, q))

    if not all_queries:
        print("No paraphrasing queries found. Exiting.")
        return

    print(f"\nTotal queries to send: {len(all_queries)}")
    all_outputs = client.call_chat(queries=all_queries, tqdm_enable=True)

    print("\nUpdating paraphrased outputs...")
    scene_results = {}

    for out, (scene, category, idx, original_q) in zip(all_outputs, query_to_scene_map):
        try:
            text = out.choices[0].message.content
        except Exception:
            text = ""
        paraphrased_fields = extract_json_from_output(text)

        # Handle empty JSON (e.g., from relative_distance_questions rule)
        if not paraphrased_fields:
            print(f"Skipping {scene.name}/{category}[{idx}] due to empty/invalid JSON response.")
            continue

        final_paraphrased_obj = original_q.copy()
        final_paraphrased_obj.update(paraphrased_fields)

        # Categories that should have options (with exactly 4 options)
        categories_with_options = ["anchor_recognition_questions", "relative_distance_questions",
                                   "relative_direction_questions", "cognitive_mapping_questions"]

        # Validate options for categories that should have them
        if category in categories_with_options:
            if "options" not in final_paraphrased_obj:
                print(f"Skipping {scene.name}/{category}[{idx}]: Missing options field.")
                continue
            options = final_paraphrased_obj.get("options")
            if not isinstance(options, list) or len(options) != 4:
                print(f"Skipping {scene.name}/{category}[{idx}]: Invalid options (expected list of 4, got {type(options).__name__} of length {len(options) if isinstance(options, list) else 'N/A'}).")
                continue

        # Validate paraphrased options if they were paraphrased
        if category == "cognitive_mapping_questions":
            try:
                options = final_paraphrased_obj.get("options")
                if not all(isinstance(opt, dict) for opt in options):
                        print(f"Skipping {scene.name}/{category}[{idx}]: Map options are not dictionaries.")
                        continue
            except Exception as e:
                print(f"Error validating paraphrased options for {scene.name}/{category}[{idx}]: {e}")
                continue

        # Always update correct_answer based on options[correct_index] for all questions with options
        try:
            correct_index = final_paraphrased_obj.get("correct_index")
            options = final_paraphrased_obj.get("options")

            if correct_index is None:
                print(f"Skipping {scene.name}/{category}[{idx}]: Missing correct_index.")
                continue
            if options is None:
                print(f"Skipping {scene.name}/{category}[{idx}]: Missing options.")
                continue
            if not isinstance(options, list) or len(options) < 4:
                print(f"Skipping {scene.name}/{category}[{idx}]: Invalid options list.")
                continue
            if not (0 <= correct_index < len(options)):
                print(f"Skipping {scene.name}/{category}[{idx}]: correct_index out of bounds.")
                continue

            # Update the correct answer based on the final options (paraphrased or original)
            final_paraphrased_obj["correct_answer"] = options[correct_index]

        except Exception as e:
            print(f"Error updating correct_answer for {scene.name}/{category}[{idx}]: {e}")
            continue # Skip this question if updating failed

        if scene not in scene_results:
            scene_results[scene] = {}
        if category not in scene_results[scene]:
            scene_results[scene][category] = []
        scene_results[scene][category].append(final_paraphrased_obj)

    for scene, categories in tqdm(scene_results.items(), desc="Saving paraphrased scenes"):
        output_path = scene / "questions_paraphrased.json"
        with open(output_path, "w") as f:
            json.dump(categories, f, indent=2)
        print(f"✅ Saved paraphrased questions for {scene.name}")

    print("\n🎯 All scenes paraphrased successfully.")