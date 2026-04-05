import os
from pathlib import Path
from typing import List

from src.utils import create_openai_client
from src.utils import create_vllm_client
from src.utils import encode_image

def filter_scenes(
    scenes: List[Path], 
    client: str, 
    model_name: str, 
    api_base: str, 
    api_key: str,
    num_runs: int = 1) -> List[Path]:
    
    filtered_scenes = []
    
    if client == "openai":
        client = create_openai_client(api_base=api_base, api_key=api_key, model_name=model_name)
    elif client == "vllm":
        client = create_vllm_client(api_base=api_base, api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Invalid client: {client}")
    
    # Prepare all queries at once (num_runs * num_scenes)
    all_queries = []
    scene_query_mapping = []  # Maps query index to (scene, run_number)
    
    print(f"Preparing {len(scenes)} scenes with {num_runs} runs each = {len(scenes) * num_runs} total queries")
    
    for scene_idx, scene in enumerate(scenes):
        # Clean up previous results
        if os.path.exists(os.path.join(scene, "ACCEPT.txt")):
            os.remove(os.path.join(scene, "ACCEPT.txt"))
        if os.path.exists(os.path.join(scene, "REJECT.txt")):
            os.remove(os.path.join(scene, "REJECT.txt"))
        
        scene_image_0 = os.path.join(scene, "frames/Image/camera_0/Image_0_0_0048_0.png")
        scene_image_1 = os.path.join(scene, "frames/Image/camera_0/Image_1_0_0048_0.png")
        
        for run in range(num_runs):
            if 'gpt-5' in model_name.lower():
                query = {
                    "messages": [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(scene_image_0)}",
                                "detail": "high"
                            }}, 
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(scene_image_1)}",
                                "detail": "high"
                            }}, 
                            {"type": "text", "text": filter_prompt()}
                        ]}
                    ],
                    "max_completion_tokens": 2000,
                }
            elif 'thinking' in model_name.lower() and 'qwen' in model_name.lower():
                query = {
                    "messages": [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(scene_image_0)}",
                                "detail": "high"
                            }}, 
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(scene_image_1)}",
                                "detail": "high"
                            }}, 
                            {"type": "text", "text": filter_prompt()}
                        ]}
                    ],
                    "max_completion_tokens": 8192,
                    "seed": 1234,
                    "top_p": 0.95,
                    "presence_penalty": 0.0,
                    "temperature": 0.6
                }
            else:
                query = {
                    "messages": [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(scene_image_0)}",
                                "detail": "high"
                            }}, 
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{encode_image(scene_image_1)}",
                                "detail": "high"
                            }}, 
                            {"type": "text", "text": filter_prompt()}
                        ]}
                    ],
                    "max_tokens": 2000,
                }
            all_queries.append(query)
            scene_query_mapping.append((scene, run))
    
    # Make single batch API call
    print("Making batch API call...")
    responses = client.call_chat(
        all_queries, 
        tqdm_desc="Processing all scenes", 
        tqdm_enable=True
    )
    
    # Process results
    scene_results = {}  # scene -> list of responses
    
    for query_idx, response in enumerate(responses):
        scene, run = scene_query_mapping[query_idx]
        
        if scene not in scene_results:
            scene_results[scene] = []
        
        resp_content = response.choices[0].message.content
        scene_results[scene].append((run, resp_content))
    
    # Make final decisions for each scene
    for scene in scenes:
        print(f"Processing results for scene: {scene}")
        
        scene_responses = scene_results[scene]
        scene_rejected = False
        
        # Check if any run rejected the scene
        for run, resp_content in scene_responses:
            if "<result>yes</result>" in resp_content:
                print(f"  Scene rejected in run {run + 1}")
                scene_rejected = True
                break
            elif "</result>" not in resp_content:
                print(f"  Scene rejected in run {run + 1}")
                print(f"  Too much thinking so rejecting the scene")
                scene_rejected = True
                break

        
        # Final decision
        if scene_rejected:
            filtered_scenes.append(scene)
            print(f"Scene {scene} FINAL DECISION: REJECTED")
            # Save all responses for debugging
            with open(os.path.join(scene, "REJECT.txt"), "w") as f:
                f.write(f"Rejected after {num_runs} runs:\n\n")
                for run, resp in scene_responses:
                    f.write(f"Run {run + 1}:\n{resp}\n\n")
        else:
            print(f"Scene {scene} FINAL DECISION: ACCEPTED")
            # Save all responses for debugging
            with open(os.path.join(scene, "ACCEPT.txt"), "w") as f:
                f.write(f"Accepted after {num_runs} runs:\n\n")
                for run, resp in scene_responses:
                    f.write(f"Run {run + 1}:\n{resp}\n\n")

    return filtered_scenes
    
def system_prompt() -> str:
    sys_prompt = """
    You are a helpful assistant that filters scenes based on the images and the prompt.
    """
    return sys_prompt

def filter_prompt() -> str:
    filter_prompt = """
    Consider the two images from a 3D scene and the rules below to determine if the scene should be filtered.
    Both images are from the same scene, just from different viewpoints. Apply the rules to both the images.
    Don't overthink the scene.
    
    Rules:
    1. If the camera view is blocked by an object or an artifact, then the scene should be filtered.
    2. If any shelf in the scene is completely empty, then the scene should be filtered.
    3. If in any scene, objects appear very close to the camera lens, causing visible blur or blocking key elements (e.g., table, chairs, walls), then the scene should be filtered.
    4. In bathroom scenes, if there are any objects visible in the mirror, then the scene should be filtered
    5. Be VERY VERY STRICT with the filtering, and look carefully into the scene.
    6. Return "yes" in <result> tag if the scene should be filtered, otherwise return "no". For example, if the scene should be filtered, return "<result>yes</result>", otherwise return "<result>no</result>".
    7. Also return the reason for the filtering in <reason> tag. Be specific and not very verbose.
    """
    return filter_prompt