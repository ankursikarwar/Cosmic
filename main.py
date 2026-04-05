import os
import time
import json
import wandb
import uuid
import argparse
import random
import pyarrow.parquet as pq
from tqdm import tqdm
from src.utils import image_to_pil
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from src.conversation import TwoAgentConv
from src.conversation import SingleBothViews
from src.conversation import SingleOneView
from src.conversation import SingleNoView


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks_qa_file",
        type=str,
        default="counting_dataset_V_Final_2000.json",
        help="Path to tasks QA file"
    )
    parser.add_argument(
        "--experiment_variant",
        type=str,
        default="two_agent+parallel",
        choices=["two_agent+parallel", "single_agent+both_views", "single_agent+one_view", "single_agent+no_view"],
        help="Experiment variant"
    )
    parser.add_argument(
        "--max_num_turns",
        type=int,
        default=10,
        help="Maximum number of turns"
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Terminate conversation"
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Confidence"
    )
    parser.add_argument(
        "--bbox_provided",
        action="store_true",
        help="BBOX provided"
    )
    parser.add_argument(
        "--sg_communication",
        action="store_true",
        help="Scene graph communication"
    )
    parser.add_argument(
        "--answerer_model_name",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Answerer model name"
    )
    parser.add_argument(
        "--helper_model_name",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Helper model name"
    )
    parser.add_argument(
        "--answerer_client_name",
        type=str,
        default="vllm",
        help="Answerer client name"
    )
    parser.add_argument(
        "--helper_client_name",
        type=str,
        default="vllm",
        help="Helper client name"
    )
    parser.add_argument(
        "--answerer_api_base",
        type=str,
        default="http://localhost:4877/v1",
        help="Answerer API base"
    )
    parser.add_argument(
        "--helper_api_base",
        type=str,
        default="http://localhost:4877/v1",
        help="Helper API base"
    )
    parser.add_argument(
        "--single_agent_model_name",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Single agent model name"
    )
    parser.add_argument(
        "--single_agent_client_name",
        type=str,
        default="vllm",
        help="Single agent client name"
    )
    parser.add_argument(
        "--single_agent_api_base",
        type=str,
        default="http://localhost:4877/v1",
        help="Single agent API base"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="Maximum completion tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature"
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=["low", "medium", "high", "minimal", "none"],
        help="Reasoning effort for GPT and Gemini models"
    )
    parser.add_argument(
        "--bbox_tracking",
        action="store_true",
        help="BBOX tracking"
    )
    parser.add_argument(
        "--enable_logging",
        action="store_true",
        help="Enable logging"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Maximum number of questions to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default=os.environ.get("EVAL_DIR", "./eval_results"),
        help="Evaluation results path"
    )
    return parser.parse_args()


def detect_task(tasks_qa_file: str) -> str:
    """Detect task type from the parent folder name of the QA file."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(tasks_qa_file)))
    if parent == "anchor_recognition":
        return "anchor_recognition"
    elif parent == "global_counting":
        return "global_counting"
    elif parent == "relative_distance":
        return "relative_distance"
    elif parent == "relative_direction":
        return "relative_direction"
    elif parent == "cognitive_mapping":
        return "cognitive_mapping"
    else:
        raise NotImplementedError(f"Task not implemented for {tasks_qa_file}")


def is_map_task(tasks_qa_file: str) -> bool:
    """Auto-detect if map evaluation should be used based on parent folder name."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(tasks_qa_file)))
    return parent == "cognitive_mapping"


def load_tasks_qa(tasks_qa_file: str) -> list:
    """Load tasks from parquet or JSON file, returning a list of dicts."""
    if tasks_qa_file.endswith(".parquet"):
        table = pq.read_table(tasks_qa_file)
        return [
            {col: table[col][i].as_py() for col in table.column_names}
            for i in range(len(table))
        ]
    with open(tasks_qa_file, "r") as f:
        return json.load(f)


def strip_images(qa: dict) -> dict:
    """Return a copy of qa with image byte fields removed for JSON serialization."""
    return {k: v for k, v in qa.items()
            if not (isinstance(v, dict) and "bytes" in v)}


def get_two_agent_task_descriptions(tasks_qa_file: str):
    """Get answerer and helper task descriptions for two-agent mode."""
    task = detect_task(tasks_qa_file)

    if task == "global_counting":
        answerer_task_description = '''
                1. The task is to find the count of a given object.
                2. You and your partner must make sure that you are counting the total number of unique instances of that object in the room while preventing overcounting or undercounting, as there may be some common objects in both views.
                3. Note: Cabinets and Shelves refer to the entire furniture, not different compartments within a specific piece of furniture.
                '''
        helper_task_description = '''
                1. The task is to find the count of a given object.
                2. You and your partner must make sure that you are counting the total number of unique instances of that object in the room while preventing overcounting or undercounting, as there may be some common objects in both views.
                3. Note: Cabinets and Shelves refer to the entire furniture, not different compartments within a specific piece of furniture.
                '''
    elif task == "anchor_recognition":
        answerer_task_description = '''
                1. The task is to find the object that is common in both your and your partner's views.
                2. Only one of the objects in the options will be common to both views, while the other objects in the options will be present in only one of the views of the room - either the answerer's or the helper's.
                '''
        helper_task_description = '''
                1. The task is to find the object that is common in both your and your partner's views.
                2. Only one of the objects in the options will be common to both views, while the other objects in the options will be present in only one of the views of the room - either the answerer's or the helper's.
                '''
    elif task == "relative_direction":
        answerer_task_description = '''
                1. In this task, the Answerer must determine the direction of a target object from their own viewpoint.
                2. The Answerer cannot see the object directly — it is visible only to the Helper.
                3. Since the Answerer cannot see the object directly, to identify where the object is located, the Answerer must communicate with the Helper and use the information obtained to infer its direction relative to themselves.
                4. Note: Here, the directions are relative to the Answerer's orientation, i.e., their egocentric viewpoint.
                5. Directions (like front, front-left, front-right, etc.) describe where something is based on the Answerer's facing direction, not on what they can currently see. Even if the object is outside the view, it can still be called front-left or front-right if it lies in that direction relative to the Answerer.

                '''
        helper_task_description = '''
                1. In this task, the Answerer must determine the direction of a target object from their own viewpoint.
                2. The Answerer cannot see the object directly — it is visible only to the Helper.
                3. Since the Answerer cannot see the object directly, the Helper must communicate with the Answerer to provide the information needed to answer the question.
                4. Note: Here, the directions are relative to the Answerer's orientation, i.e., their egocentric viewpoint.
                5. Directions (like front, front-left, front-right, etc.) describe where something is based on the Answerer's facing direction, not on what they can currently see. Even if the object is outside the view, it can still be called front-left or front-right if it lies in that direction relative to the Answerer.
                '''
    elif task == "relative_distance":
        answerer_task_description = '''
                1. The task is to find which of the objects in the options is either the farthest or the closest to the object mentioned in the question.
                2. The object mentioned in the question is visible to both you and your partner.
                3. The objects in the options are visible either in only your view or only in your partner's view but not in both views.
                4. Important: The correct answer is based on both views combined. An object that looks closest or farthest from your perspective may not be correct, and the right choice might be an object you cannot see at all.
                '''
        helper_task_description = '''
                1. The task is to find which of the objects in the options is either the farthest or the closest to the object mentioned in the question.
                2. The object mentioned in the question is visible to both you and your partner.
                3. The objects in the options are visible either in only your view or only in your partner's view but not in both views.
                4. Important: The correct answer is based on both views combined. An object that looks closest or farthest from your perspective may not be correct, and the right choice might be an object you cannot see at all.
                '''
    elif task == "cognitive_mapping":
        answerer_task_description = '''
                1. The task is to identify if the provided map accurately depicts the top-down layout of the room.
                2. You and the Helper observe different, partial views of the room, and neither view is complete on its own. The full layout can only be inferred by communicating and combining information from both views.
                3. Evaluate the map only by the spatial arrangement of the objects it shows. Focus exclusively on the object categories listed in the legend, ignore any other items, and do not consider objects placed on top of other objects in your judgment.
                '''
        helper_task_description = '''
                1. The task is to identify if the provided map accurately depicts the top-down layout of the room.
                2. The map is only provided to the answerer agent.
                '''
    else:
        raise NotImplementedError(f"Task description not implemented for {tasks_qa_file}")

    return answerer_task_description, helper_task_description


def get_single_agent_task_description(tasks_qa_file: str) -> str:
    """Get task description for single-agent mode."""
    task = detect_task(tasks_qa_file)

    if task == "global_counting":
        return '''
                1. The task is to find the count of a given object.
                2. You must make sure that you are counting the total number of unique instances of that object in the room while preventing overcounting or undercounting, as there may be some common objects in both views.
                3. Note: Cabinets and Shelves refer to the entire furniture, not different compartments within a specific piece of furniture.
                '''
    elif task == "anchor_recognition":
        return '''
                1. The task is to find the object that is common in both the views.
                2. Only one of the objects in the options will be common to both views, while the other objects in the options will be present in only one of the views of the room.
                '''
    elif task == "relative_direction":
        return '''
                1. The task is to determine the direction of a target object from the perspective of one of the images.
                2. The target object appears in the other view, not in the one you are reasoning from. Use information from both views to infer its direction relative to the chosen image's viewpoint.
                3. Note: Directions are defined with respect to the image's orientation, i.e., its egocentric viewpoint.
                4. Directions (like front, front-left, front-right, etc.) describe where something is based on the direction the image is facing, not on what is currently visible from that view. Even if the target object is outside the frame, it can still be front-left or front-right if it lies in that direction relative to that view.
                '''
    elif task == "relative_distance":
        return '''
                1. The task is to find which of the objects in the options is either the farthest or the closest to the object mentioned in the question.
                2. The object mentioned in the question is visible in both views.
                3. The objects in the options are visible either in only first view or only in second view but not in both views.
                4. Important: The correct answer is based on both views combined.
                '''
    elif task == "cognitive_mapping":
        return '''
                1. The task is to identify if the provided map accurately depicts the top-down layout of the room.
                2. You observe two partial views of the room, and neither view is complete on its own. The full layout can only be inferred by combining information from both views.
                3. Evaluate the map only by the spatial arrangement of the objects it shows. Focus exclusively on the object categories listed in the legend, ignore any other items, and do not consider objects placed on top of other objects in your judgment.
                '''
    else:
        raise NotImplementedError(f"Task description not implemented for {tasks_qa_file}")


def get_wandb_project(experiment_variant: str) -> str:
    """Get wandb project name based on experiment variant."""
    if experiment_variant.startswith("two_agent"):
        return "Cosmic_TwoAgent"
    else:
        return "Cosmic_SingleAgent"


def main():
    args = parse_arguments()

    tasks_qa = load_tasks_qa(args.tasks_qa_file)

    task = detect_task(args.tasks_qa_file)
    use_map = is_map_task(args.tasks_qa_file)

    sg_comm = "_SG_COMM" if args.sg_communication else ""

    variant = args.experiment_variant
    if variant == "two_agent+parallel":
        run_two_agent_parallel(args, tasks_qa, task, use_map, sg_comm)

    elif variant == "single_agent+both_views":
        run_single_agent_both_views(args, tasks_qa, task, use_map, sg_comm)
    elif variant == "single_agent+one_view":
        run_single_agent_one_view(args, tasks_qa, task, use_map, sg_comm)
    elif variant == "single_agent+no_view":
        run_single_agent_no_view(args, tasks_qa, task, use_map, sg_comm)


def run_two_agent_parallel(args, tasks_qa, task, use_map, sg_comm):
    correct_count = 0
    incorrect_count = 0
    total_count = 0
    eval_results = []

    wandb_project = get_wandb_project(args.experiment_variant)
    if args.max_questions is not None:
        wandb.init(
            name=args.answerer_model_name.split("/")[-1]+"_"+task+sg_comm+"_"+str(args.max_questions)+"_parallel",
            project=wandb_project, entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    else:
        wandb.init(
            name=args.answerer_model_name.split("/")[-1]+"_"+task+sg_comm+"_parallel",
            project=wandb_project, entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))

    eval_results_path = os.path.join(args.eval_results_dir, args.experiment_variant, task, args.answerer_model_name)
    run_id = str(uuid.uuid4())[:8]
    eval_results_path = os.path.join(eval_results_path, run_id)
    print(eval_results_path)
    os.makedirs(eval_results_path, exist_ok=True)

    if args.max_questions is not None:
        tasks_qa = tasks_qa[:args.max_questions]
        random.seed(args.seed)
        random.shuffle(tasks_qa)
        print(f"Processing {len(tasks_qa)} questions (randomly sampled)")
    else:
        print(f"Processing {len(tasks_qa)} questions (all)")

    checkpoint_file = os.path.join(eval_results_path, "checkpoint.json")
    accuracy_checkpoint_file = os.path.join(eval_results_path, "accuracy_checkpoint.json")

    eval_results = []
    evaluated_indices = set()

    if os.path.exists(checkpoint_file):
        print(f"Loading existing checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "r") as f:
            eval_results = json.load(f)
        evaluated_indices = {result.get("q_idx", i) for i, result in enumerate(eval_results) if "q_idx" in result}
        if not evaluated_indices and len(eval_results) > 0:
            evaluated_indices = set(range(len(eval_results)))
        correct_count = sum(1 for result in eval_results if result.get("is_correct", False))
        incorrect_count = sum(1 for result in eval_results if not result.get("is_correct", False) and not result.get("parse_error", False))
        total_count = len(eval_results)
        print(f"Resuming: {total_count} conversations already evaluated")

    # Initialize all conversations
    conversations = {}
    question_data = {}

    print("Initializing all conversations...")
    for q_idx, qa in tqdm(enumerate(tasks_qa), total=len(tasks_qa), desc="Initializing conversations"):
        if q_idx in evaluated_indices:
            print(f"Skipping question {q_idx} (already evaluated)")
            continue

        q_type = qa["question_type"]
        room_part = qa.get("room_part", qa.get("sample_id", ""))
        scene_id = qa["scene_id"]

        if qa['user_1_question'] is None:
            helper_agent_view = qa['user_1_image']
            helper_agent_goal = qa['user_1_goal']
            answerer_agent_view = qa['user_2_image']
            answerer_agent_goal = qa['user_2_goal']
            answerer_agent_correct_answer_idx = qa['user_2_gt_answer_idx']
            answerer_agent_correct_answer_text = qa['user_2_gt_answer_text']
            if args.bbox_provided:
                helper_agent_bbox = qa['user_1_bbox']
                answerer_agent_bbox = qa['user_2_bbox']
            answerer_agent_question = qa['user_2_question']
            if use_map and "pilot" in str(args.tasks_qa_file).lower():
                answerer_agent_options = qa['options_format2']
            else:
                answerer_agent_options = qa['options_user_2']
        else:
            answerer_agent_view = qa['user_1_image']
            answerer_agent_goal = qa['user_1_goal']
            answerer_agent_correct_answer_idx = qa['user_1_gt_answer_idx']
            answerer_agent_correct_answer_text = qa['user_1_gt_answer_text']
            helper_agent_view = qa['user_2_image']
            helper_agent_goal = qa['user_2_goal']
            if args.bbox_provided:
                answerer_agent_bbox = qa['user_1_bbox']
                helper_agent_bbox = qa['user_2_bbox']
            answerer_agent_question = qa['user_1_question']
            if use_map and "pilot" in str(args.tasks_qa_file).lower():
                answerer_agent_options = qa['options_format2']
            else:
                answerer_agent_options = qa['options_user_1']

        question_dict = {
            "question_type": q_type,
            "question": answerer_agent_question,
            "options": answerer_agent_options,
            "answerer_goal": answerer_agent_goal,
            "helper_goal": helper_agent_goal,
        }

        # Add global_map_image to question_dict if map task
        if use_map and 'global_map_image' in qa:
            question_dict["global_map_image"] = qa['global_map_image']

        answerer_task_description, helper_task_description = get_two_agent_task_descriptions(args.tasks_qa_file)

        two_agent_conv = TwoAgentConv(
            question=question_dict,
            terminate=args.terminate,
            confidence=args.confidence,
            sg_communication=args.sg_communication,
            bbox_tracking=args.bbox_tracking,
            max_num_turns=args.max_num_turns,
            bbox_provided=args.bbox_provided,
            answerer_task_description=answerer_task_description,
            helper_task_description=helper_task_description,
            answerer_model_name=args.answerer_model_name,
            helper_model_name=args.helper_model_name,
            answerer_client_name=args.answerer_client_name,
            helper_client_name=args.helper_client_name,
            answerer_api_base=args.answerer_api_base,
            helper_api_base=args.helper_api_base,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            enable_logging=args.enable_logging,
        )

        two_agent_conv.initialize_conversation(
            answerer_images=[answerer_agent_view],
            helper_images=[helper_agent_view],
            answerer_bbox=answerer_agent_bbox if args.bbox_provided else None,
            helper_bbox=helper_agent_bbox if args.bbox_provided else None,
        )

        conversations[q_idx] = two_agent_conv
        question_data[q_idx] = {
            "qa": qa,
            "question_dict": question_dict,
            "answerer_agent_view": answerer_agent_view,
            "helper_agent_view": helper_agent_view,
            "answerer_agent_correct_answer_idx": answerer_agent_correct_answer_idx,
            "answerer_agent_correct_answer_text": answerer_agent_correct_answer_text,
            "answerer_agent_question": answerer_agent_question,
            "answerer_agent_options": answerer_agent_options,
            "room_part": room_part,
            "scene_id": scene_id,
            "q_type": q_type,
        }

    # Execute turn 1 for all conversations - BATCH PROCESSING
    print("Executing turn 1 (answerer messages) for all conversations in batch...")
    turn_1_queries = []
    turn_1_q_indices = []
    for q_idx in tqdm(conversations.keys(), desc="Preparing turn 1 queries"):
        query = conversations[q_idx].prepare_turn_1_query()
        if query is not None:
            turn_1_queries.append(query)
            turn_1_q_indices.append(q_idx)

    if turn_1_queries:
        print(f"  Sending batch of {len(turn_1_queries)} queries to VLLM server...")
        client = conversations[turn_1_q_indices[0]].answerer_agent.client
        responses = client.call_chat(turn_1_queries, tqdm_desc="Turn 1 - Answerer Batch", tqdm_enable=True)

        print(f"  Processing {len(responses)} responses...")
        for q_idx, response_obj in zip(turn_1_q_indices, responses):
            response = response_obj.choices[0].message.content if (response_obj and response_obj.choices) else None
            if response:
                conversations[q_idx].process_turn_1_response(response)

    # Execute turn 1 helper responses - BATCH PROCESSING
    print("Executing turn 1 (helper responses) for all conversations in batch...")
    turn_1_helper_queries = []
    turn_1_helper_q_indices = []
    for q_idx in tqdm(conversations.keys(), desc="Preparing turn 1 helper queries"):
        if not conversations[q_idx].is_terminated:
            query = conversations[q_idx].prepare_turn_1_helper_query()
            if query is not None:
                turn_1_helper_queries.append(query)
                turn_1_helper_q_indices.append(q_idx)

    if turn_1_helper_queries:
        print(f"  Sending batch of {len(turn_1_helper_queries)} queries to VLLM server...")
        client = conversations[turn_1_helper_q_indices[0]].helper_agent.client
        responses = client.call_chat(turn_1_helper_queries, tqdm_desc="Turn 1 - Helper Batch", tqdm_enable=True)

        print(f"  Processing {len(responses)} responses...")
        for q_idx, response_obj in zip(turn_1_helper_q_indices, responses):
            response = response_obj.choices[0].message.content if (response_obj and response_obj.choices) else None
            if response:
                conversations[q_idx].process_turn_1_helper_response(response)

    # Execute turns 2 to max_num_turns - BATCH PROCESSING
    for turn in tqdm(range(2, args.max_num_turns + 1), desc="Executing turns 2 to max_num_turns"):
        active_conversations = [q_idx for q_idx in conversations.keys() if not conversations[q_idx].is_terminated]
        if not active_conversations:
            print(f"All conversations terminated before turn {turn}")
            break

        print(f"Executing turn {turn} for {len(active_conversations)} active conversations in batch...")

        answerer_queries = []
        answerer_q_indices = []
        for q_idx in tqdm(active_conversations, desc=f"Preparing turn {turn} queries"):
            query, _ = conversations[q_idx].prepare_turn_query(turn)
            if query is not None:
                answerer_queries.append(query)
                answerer_q_indices.append(q_idx)

        if answerer_queries:
            print(f"  Sending batch of {len(answerer_queries)} answerer queries to VLLM server...")
            client = conversations[answerer_q_indices[0]].answerer_agent.client
            responses = client.call_chat(answerer_queries, tqdm_desc=f"Turn {turn} - Answerer Batch", tqdm_enable=True)

            print(f"  Processing {len(responses)} answerer responses...")
            for q_idx, response_obj in zip(answerer_q_indices, responses):
                response = response_obj.choices[0].message.content if (response_obj and response_obj.choices) else None
                if response:
                    conversations[q_idx].process_turn_answerer_response(response, turn)

        helper_queries = []
        helper_q_indices = []
        for q_idx in tqdm(active_conversations, desc=f"Preparing turn {turn} helper queries"):
            if not conversations[q_idx].is_terminated:
                answerer_message = conversations[q_idx].answerer_agent.chat_history[-1]['content']
                query = conversations[q_idx].prepare_turn_helper_query(answerer_message)
                if query is not None:
                    helper_queries.append(query)
                    helper_q_indices.append(q_idx)

        if helper_queries:
            print(f"  Sending batch of {len(helper_queries)} helper queries to VLLM server...")
            client = conversations[helper_q_indices[0]].helper_agent.client
            responses = client.call_chat(helper_queries, tqdm_desc=f"Turn {turn} - Helper Batch", tqdm_enable=True)

            print(f"  Processing {len(responses)} helper responses...")
            for q_idx, response_obj in zip(helper_q_indices, responses):
                response = response_obj.choices[0].message.content if (response_obj and response_obj.choices) else None
                if response:
                    conversations[q_idx].process_turn_helper_response(response, turn)

    # Finalize all conversations and query answerer agents
    print("Finalizing conversations and getting answers...")

    table_columns = [
        "Answerer_Agent_View", "Helper_Agent_View", "Question", "Options", "Conversation_Text", "Conversation_Dict", "Turns_Completed",
        "Is_Correct", "Parse_Error", "Response", "Agent_Answer_Text", "Agent_Confidence", "GT_Answer_Text", "Agent_Answer_Idx", "GT_Answer_Idx",
        "Answerer_Chat_History", "Helper_Chat_History",
        "Room_Part", "Scene_Id", "Question_Type",
        "Difficulty", "Description_Difficulty", "Distractor_Difficulty",
        "difficulty_sum", "Difficulty_Int", "Question_Object",
        "Angle", "Distance", "Composition",
    ]
    if use_map:
        table_columns.append("Map_Image")
    table = wandb.Table(columns=table_columns)

    # Batch prepare all query queries
    print("Preparing all answerer agent queries in batch...")
    query_queries = []
    query_q_indices = []
    for q_idx in tqdm(conversations.keys(), desc="Preparing query queries"):
        conv = conversations[q_idx]
        qdata = question_data[q_idx]
        query = conv.prepare_query_answerer_agent(qdata["question_dict"])
        if query is not None:
            query_queries.append(query)
            query_q_indices.append(q_idx)

    answers = {}
    if query_queries:
        print(f"  Sending batch of {len(query_queries)} query queries to VLLM server...")
        client = conversations[query_q_indices[0]].answerer_agent.client
        responses = client.call_chat(query_queries, tqdm_desc="Query - Answerer Batch", tqdm_enable=True)

        print(f"  Processing {len(responses)} query responses...")
        for q_idx, response_obj in zip(query_q_indices, responses):
            response = response_obj.choices[0].message.content if (response_obj and response_obj.choices) else None
            conv = conversations[q_idx]
            qdata = question_data[q_idx]
            answer = conv.process_query_answerer_agent_response(response, qdata["question_dict"])
            answers[q_idx] = answer

    # Process all results
    for q_idx in tqdm(conversations.keys(), desc="Processing results"):
        conv = conversations[q_idx]
        qdata = question_data[q_idx]
        qa = qdata["qa"]

        conversation_result = conv.finalize_conversation()
        answer = answers.get(q_idx)

        if answer is None:
            answer = conv.query_answerer_agent(qdata["question_dict"])

        is_correct = False
        parse_error = False

        if answer['answer_idx'] is not None:
            if answer['answer_idx'] == qdata["answerer_agent_correct_answer_idx"]:
                correct_count += 1
                is_correct = True
                print(f"Q{q_idx} Correct idx: {answer['answer_idx']}")
            else:
                incorrect_count += 1
                print(f"Q{q_idx} Incorrect idx: {answer['answer_idx']}")
        else:
            print(f"Q{q_idx} Failed to parse answer from response")
            incorrect_count += 1
            parse_error = True
        total_count += 1

        conv_text_filename = os.path.join(eval_results_path, f"conversation_{q_idx}.txt")
        with open(conv_text_filename, "w", encoding="utf-8") as f:
            f.write(conversation_result['conv_text'])

        conv_dict_filename = os.path.join(eval_results_path, f"conversation_dict_{q_idx}.json")
        with open(conv_dict_filename, "w", encoding="utf-8") as f:
            json.dump(conversation_result['conv_dict'], f, indent=4)

        eval_results.append({
            "q_idx": q_idx,
            "q_data": strip_images(qa),
            "agent_answer_idx": answer['answer_idx'],
            "agent_answer_letter": answer['answer_letter'],
            "agent_answer_text": answer['answer_text'],
            "agent_confidence": answer['confidence'],
            "is_correct": is_correct,
            "parse_error": parse_error,
            "response": answer['response'],
            "conv_text_filename": conv_text_filename,
            "conv_dict_filename": conv_dict_filename,
            "conv_text": conversation_result['conv_text'],
            "conv_dict": conversation_result['conv_dict'],
            "turns_completed": conversation_result['turns_completed'],
            "eval_args": vars(args),
            "answerer_chat_history": conv.answerer_agent.chat_history_no_image,
            "helper_chat_history": conv.helper_agent.chat_history_no_image,
        })

        row_data = [
            wandb.Image(image_to_pil(qdata["answerer_agent_view"])),
            wandb.Image(image_to_pil(qdata["helper_agent_view"])),
            qdata["answerer_agent_question"],
            "\n".join([f"{idx}) {option}" for idx, option in zip(["A", "B", "C", "D"], qdata["answerer_agent_options"])]),
            conversation_result['conv_text'],
            json.dumps(conversation_result['conv_dict'], indent=4),
            conversation_result['turns_completed'],
            is_correct,
            parse_error,
            answer['response'],
            answer['answer_text'],
            answer['confidence'],
            qdata["answerer_agent_correct_answer_text"] if not use_map else None,
            answer['answer_idx'],
            qdata["answerer_agent_correct_answer_idx"],
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in conv.answerer_agent.chat_history_no_image]),
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in conv.helper_agent.chat_history_no_image]),
            qdata["room_part"],
            qdata["scene_id"],
            qdata["q_type"],
            qa.get('difficulty'),
            (str(qa['description_difficulty'])
                if isinstance(qa.get('description_difficulty'), int)
                else ', '.join(str(n) for n in qa['description_difficulty'])
            ) if 'description_difficulty' in qa else None,
            qa.get('distractor_difficulty'),
            qa.get('difficulty_sum'),
            qa.get('difficulty_int'),
            qa.get('question_object'),
            qa.get('angle'),
            qa.get('distance'),
            qa.get('composition'),
        ]
        if use_map:
            map_img_path = qdata["question_dict"].get("global_map_image")
            row_data.append(wandb.Image(image_to_pil(map_img_path)) if map_img_path is not None else None)
        table.add_data(*row_data)

    # Save checkpoint
    with open(checkpoint_file, "w") as f:
        json.dump(eval_results, f, indent=4)

    accuracy = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "total_count": total_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0.0
    }
    with open(accuracy_checkpoint_file, "w") as f:
        json.dump(accuracy, f, indent=4)

    with open(os.path.join(eval_results_path, f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Correct count: {correct_count}")
    print(f"Incorrect count: {incorrect_count}")
    print(f"Total count: {total_count}")
    print(f"Accuracy: {correct_count / total_count if total_count > 0 else 0.0}")

    with open(os.path.join(eval_results_path, f"accuracy_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(accuracy, f, indent=4)

    wandb.log({"TwoAgentConv": table})
    wandb.log({"Correct Count": correct_count})
    wandb.log({"Incorrect Count": incorrect_count})
    wandb.log({"Total Count": total_count})
    wandb.log({"Accuracy": correct_count / total_count if total_count > 0 else 0.0})
    wandb.finish()


def run_single_agent_both_views(args, tasks_qa, task, use_map, sg_comm):
    correct_count = 0
    incorrect_count = 0
    total_count = 0
    eval_results = []

    wandb_project = get_wandb_project(args.experiment_variant)
    if args.max_questions is not None:
        wandb.init(
            name=args.single_agent_model_name.split("/")[-1]+"_"+task+"_"+str(args.max_questions),
            project=wandb_project, entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    else:
        wandb.init(
            name=args.single_agent_model_name.split("/")[-1]+"_"+task,
            project=wandb_project, entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))

    table_columns = [
        "First_View", "Second_View",
    ]
    if use_map:
        table_columns.append("Map")
    table_columns += [
        "Question", "Options", "Response",
        "Is_Correct", "Parse_Error", "Response", "Agent_Answer_Text", "Agent_Confidence", "GT_Answer_Text", "Agent_Answer_Idx", "GT_Answer_Idx",
        "Agent_Chat_History",
        "Room_Part", "Scene_Id", "Question_Type",
        "Difficulty", "Description_Difficulty", "Distractor_Difficulty",
        "difficulty_sum", "Difficulty_Int", "Question_Object",
        "Angle", "Distance", "Composition"
    ]
    table = wandb.Table(columns=table_columns)

    eval_results_path = os.path.join(args.eval_results_dir, args.experiment_variant, task, args.single_agent_model_name)
    run_id = str(uuid.uuid4())[:8]
    eval_results_path = os.path.join(eval_results_path, run_id)
    print(eval_results_path)
    os.makedirs(eval_results_path, exist_ok=True)

    if args.max_questions is not None:
        tasks_qa = tasks_qa[:args.max_questions]
        random.seed(args.seed)
        random.shuffle(tasks_qa)
        print(f"Processing {len(tasks_qa)} questions (randomly sampled)")
    else:
        print(f"Processing {len(tasks_qa)} questions (all)")

    task_description = get_single_agent_task_description(args.tasks_qa_file)

    for q_idx, qa in tqdm(enumerate(tasks_qa), total=len(tasks_qa), desc="Processing questions"):
        is_correct = False
        parse_error = False

        q_type = qa["question_type"]
        room_part = qa.get("room_part", qa.get("sample_id", ""))
        scene_id = qa["scene_id"]

        first_view = qa['user_1_image']
        second_view = qa['user_2_image']
        map_view = qa.get('global_map_image') if use_map else None

        if args.bbox_provided:
            first_view_bbox = qa['user_1_bbox']
            second_view_bbox = qa['user_2_bbox']

        if qa['user_1_question'] is None:
            correct_answer_idx = qa['user_2_gt_answer_idx']
            correct_answer_text = qa['user_2_gt_answer_text']
            question = qa['user_2_question']
            options = qa['options_user_2']
        else:
            correct_answer_idx = qa['user_1_gt_answer_idx']
            correct_answer_text = qa['user_1_gt_answer_text']
            question = qa['user_1_question']
            options = qa['options_user_1']

        question_dict = {
            "question_type": q_type,
            "question": question,
            "options": options,
        }

        single_agent_both_views_no_conversation = SingleBothViews(
            confidence=args.confidence,
            model_name=args.single_agent_model_name,
            client_name=args.single_agent_client_name,
            api_base=args.single_agent_api_base,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            enable_logging=args.enable_logging,
        )

        answer = single_agent_both_views_no_conversation.query_agent(
            images=[first_view, second_view],
            question=question_dict,
            task_description=task_description,
            bbox=[first_view_bbox, second_view_bbox] if args.bbox_provided else None,
            map_image=map_view,
        )

        if answer['answer_idx'] is not None:
            if answer['answer_idx'] == correct_answer_idx:
                correct_count += 1
                is_correct = True
                print(f"Correct idx: {answer['answer_idx']}")
            else:
                incorrect_count += 1
                print(f"Incorrect idx: {answer['answer_idx']}")
        else:
            print(f"Failed to parse answer from response")
            incorrect_count += 1
            parse_error = True
        total_count += 1

        response_filename = os.path.join(eval_results_path, f"response_{q_idx}.txt")
        with open(response_filename, "w", encoding="utf-8") as f:
            f.write(answer['response'])

        eval_results.append({
            "q_data": strip_images(qa),
            "agent_answer_idx": answer['answer_idx'],
            "agent_answer_letter": answer['answer_letter'],
            "agent_answer_text": answer['answer_text'],
            "agent_confidence": answer['confidence'],
            "is_correct": is_correct,
            "parse_error": parse_error,
            "response_filename": response_filename,
            "response": answer['response'],
            "eval_args": vars(args),
            "agent_chat_history": single_agent_both_views_no_conversation.agent.chat_history_no_image
        })

        row_data = [
            wandb.Image(image_to_pil(first_view)),
            wandb.Image(image_to_pil(second_view)),
        ]
        if use_map:
            row_data.append(wandb.Image(image_to_pil(map_view)) if map_view else None)
        row_data += [
            question,
            "\n".join([f"{idx}) {option}" for idx, option in zip(["A", "B", "C", "D"], options)]),
            answer['response'],
            is_correct,
            parse_error,
            answer['response'],
            answer['answer_text'] if not use_map else None,
            answer['confidence'],
            correct_answer_text if not use_map else None,
            answer['answer_idx'],
            correct_answer_idx,
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in single_agent_both_views_no_conversation.agent.chat_history_no_image]),
            room_part,
            scene_id,
            q_type,
            qa.get('difficulty'),
            (str(qa['description_difficulty'])
                if isinstance(qa.get('description_difficulty'), int)
                else ', '.join(str(n) for n in qa['description_difficulty'])
            ) if 'description_difficulty' in qa else None,
            qa.get('distractor_difficulty'),
            qa.get('difficulty_sum'),
            qa.get('difficulty_int'),
            qa.get('question_object'),
            qa.get('angle'),
            qa.get('distance'),
            qa.get('composition'),
        ]
        table.add_data(*row_data)

    with open(os.path.join(eval_results_path, f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Correct count: {correct_count}")
    print(f"Incorrect count: {incorrect_count}")
    print(f"Total count: {total_count}")
    print(f"Accuracy: {correct_count / total_count if total_count > 0 else 0.0}")

    accuracy = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "total_count": total_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0.0
    }
    with open(os.path.join(eval_results_path, f"accuracy_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(accuracy, f, indent=4)

    wandb.log({"SingleAgentBothViews": table})
    wandb.log({"Correct Count": correct_count})
    wandb.log({"Incorrect Count": incorrect_count})
    wandb.log({"Total Count": total_count})
    wandb.log({"Accuracy": correct_count / total_count if total_count > 0 else 0.0})
    wandb.finish()


def run_single_agent_one_view(args, tasks_qa, task, use_map, sg_comm):
    correct_count = 0
    incorrect_count = 0
    total_count = 0
    eval_results = []

    wandb_project = get_wandb_project(args.experiment_variant)
    if args.max_questions is not None:
        wandb.init(
            name=args.single_agent_model_name.split("/")[-1]+"_"+task+"_"+str(args.max_questions),
            project="Cosmic_SingleAgent_OneView", entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    else:
        wandb.init(
            name=args.single_agent_model_name.split("/")[-1]+"_"+task,
            project="Cosmic_SingleAgent_OneView", entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    table = wandb.Table(columns=[
        "Agent_View", "Other_View", "Question", "Options", "Response",
        "Is_Correct", "Parse_Error", "Agent_Answer_Text", "Agent_Confidence", "GT_Answer_Text", "Agent_Answer_Idx", "GT_Answer_Idx",
        "Agent_Chat_History",
        "Room_Part", "Scene_Id", "Question_Type",
        "Difficulty", "Description_Difficulty", "Distractor_Difficulty",
        "difficulty_sum", "Difficulty_Int", "Question_Object",
        "Angle", "Distance", "Composition"
    ])

    eval_results_path = os.path.join(args.eval_results_dir, args.experiment_variant, task, args.single_agent_model_name)
    print(eval_results_path)
    os.makedirs(eval_results_path, exist_ok=True)

    if args.max_questions is not None:
        tasks_qa = tasks_qa[:args.max_questions]
        random.seed(args.seed)
        random.shuffle(tasks_qa)
        print(f"Processing {len(tasks_qa)} questions (randomly sampled)")
    else:
        print(f"Processing {len(tasks_qa)} questions (all)")

    task_description = get_single_agent_task_description(args.tasks_qa_file)

    for q_idx, qa in tqdm(enumerate(tasks_qa), total=len(tasks_qa), desc="Processing questions"):
        is_correct = False
        parse_error = False

        q_type = qa["question_type"]
        room_part = qa.get("room_part", qa.get("sample_id", ""))
        scene_id = qa["scene_id"]

        if qa['user_1_question'] is None:
            agent_view = qa['user_2_image']
            other_view = qa['user_1_image']
            correct_answer_idx = qa['user_2_gt_answer_idx']
            correct_answer_text = qa['user_2_gt_answer_text']
            question = qa['user_2_question']
            options = qa['options_user_2']
            if args.bbox_provided:
                agent_view_bbox = qa['user_2_bbox']
        else:
            agent_view = qa['user_1_image']
            other_view = qa['user_2_image']
            correct_answer_idx = qa['user_1_gt_answer_idx']
            correct_answer_text = qa['user_1_gt_answer_text']
            question = qa['user_1_question']
            options = qa['options_user_1']
            if args.bbox_provided:
                agent_view_bbox = qa['user_1_bbox']

        question_dict = {
            "question_type": q_type,
            "question": question,
            "options": options,
        }

        single_agent_one_view_no_conversation = SingleOneView(
            confidence=args.confidence,
            model_name=args.single_agent_model_name,
            client_name=args.single_agent_client_name,
            api_base=args.single_agent_api_base,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            enable_logging=args.enable_logging,
        )

        answer = single_agent_one_view_no_conversation.query_agent(
            images=[agent_view],
            question=question_dict,
            task_description=task_description,
            bbox=[agent_view_bbox] if args.bbox_provided else None
        )

        if answer['answer_idx'] is not None:
            if answer['answer_idx'] == correct_answer_idx:
                correct_count += 1
                is_correct = True
            else:
                incorrect_count += 1
        else:
            incorrect_count += 1
            parse_error = True
        total_count += 1

        response_filename = os.path.join(eval_results_path, f"response_{q_idx}.txt")
        with open(response_filename, "w", encoding="utf-8") as f:
            f.write(answer['response'])

        eval_results.append({
            "q_data": strip_images(qa),
            "agent_answer_idx": answer['answer_idx'],
            "agent_answer_letter": answer['answer_letter'],
            "agent_answer_text": answer['answer_text'],
            "agent_confidence": answer['confidence'],
            "is_correct": is_correct,
            "parse_error": parse_error,
            "response_filename": response_filename,
            "response": answer['response'],
            "eval_args": vars(args),
            "agent_chat_history": single_agent_one_view_no_conversation.agent.chat_history_no_image
        })

        table.add_data(
            wandb.Image(image_to_pil(agent_view)),
            wandb.Image(image_to_pil(other_view)),
            question,
            "\n".join([f"{idx}) {option}" for idx, option in zip(["A", "B", "C", "D"], options)]),
            answer['response'],
            is_correct,
            parse_error,
            answer['answer_text'],
            answer['confidence'],
            correct_answer_text if not use_map else None,
            answer['answer_idx'],
            correct_answer_idx,
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in single_agent_one_view_no_conversation.agent.chat_history_no_image]),
            room_part,
            scene_id,
            q_type,
            qa.get('difficulty'),
            (str(qa['description_difficulty'])
                if isinstance(qa.get('description_difficulty'), int)
                else ', '.join(str(n) for n in qa['description_difficulty'])
            ) if 'description_difficulty' in qa else None,
            qa.get('distractor_difficulty'),
            qa.get('difficulty_sum'),
            qa.get('difficulty_int'),
            qa.get('question_object'),
            qa.get('angle'),
            qa.get('distance'),
            qa.get('composition'),
        )

    with open(os.path.join(eval_results_path, f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Correct count: {correct_count}")
    print(f"Incorrect count: {incorrect_count}")
    print(f"Total count: {total_count}")
    print(f"Accuracy: {correct_count / total_count if total_count > 0 else 0.0}")

    accuracy = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "total_count": total_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0.0
    }
    with open(os.path.join(eval_results_path, f"accuracy_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(accuracy, f, indent=4)

    wandb.log({"SingleAgentOneView": table})
    wandb.log({"Correct Count": correct_count})
    wandb.log({"Incorrect Count": incorrect_count})
    wandb.log({"Total Count": total_count})
    wandb.log({"Accuracy": correct_count / total_count if total_count > 0 else 0.0})
    wandb.finish()


def run_single_agent_no_view(args, tasks_qa, task, use_map, sg_comm):
    correct_count = 0
    incorrect_count = 0
    total_count = 0
    eval_results = []

    wandb_project = get_wandb_project(args.experiment_variant)
    if args.max_questions is not None:
        wandb.init(
            name=args.single_agent_model_name.split("/")[-1]+"_"+task+"_"+str(args.max_questions),
            project="Cosmic_SingleAgent_NoView", entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    else:
        wandb.init(
            name=args.single_agent_model_name.split("/")[-1]+"_"+task,
            project="Cosmic_SingleAgent_NoView", entity=os.environ.get("WANDB_ENTITY", "your-wandb-entity"))
    table = wandb.Table(columns=[
        "First_View", "Second_View", "Question", "Options", "Response",
        "Is_Correct", "Parse_Error", "Agent_Answer_Text", "Agent_Confidence", "GT_Answer_Text", "Agent_Answer_Idx", "GT_Answer_Idx",
        "Agent_Chat_History",
        "Room_Part", "Scene_Id", "Question_Type",
        "Difficulty", "Description_Difficulty", "Distractor_Difficulty",
        "difficulty_sum", "Difficulty_Int", "Question_Object",
        "Angle", "Distance", "Composition"
    ])

    eval_results_path = os.path.join(args.eval_results_dir, args.experiment_variant, task, args.single_agent_model_name)
    print(eval_results_path)
    os.makedirs(eval_results_path, exist_ok=True)

    if args.max_questions is not None:
        tasks_qa = tasks_qa[:args.max_questions]
        random.seed(args.seed)
        random.shuffle(tasks_qa)
        print(f"Processing {len(tasks_qa)} questions (randomly sampled)")
    else:
        print(f"Processing {len(tasks_qa)} questions (all)")

    task_description = get_single_agent_task_description(args.tasks_qa_file)

    for q_idx, qa in tqdm(enumerate(tasks_qa), total=len(tasks_qa), desc="Processing questions"):
        is_correct = False
        parse_error = False

        q_type = qa["question_type"]
        room_part = qa.get("room_part", qa.get("sample_id", ""))
        scene_id = qa["scene_id"]

        first_view = qa['user_1_image']
        second_view = qa['user_2_image']

        if qa['user_1_question'] is None:
            correct_answer_idx = qa['user_2_gt_answer_idx']
            correct_answer_text = qa['user_2_gt_answer_text']
            question = qa['user_2_question']
            options = qa['options_user_2']
        else:
            correct_answer_idx = qa['user_1_gt_answer_idx']
            correct_answer_text = qa['user_1_gt_answer_text']
            question = qa['user_1_question']
            options = qa['options_user_1']

        question_dict = {
            "question_type": q_type,
            "question": question,
            "options": options,
        }

        single_agent_no_view = SingleNoView(
            confidence=args.confidence,
            model_name=args.single_agent_model_name,
            client_name=args.single_agent_client_name,
            api_base=args.single_agent_api_base,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            enable_logging=args.enable_logging,
        )

        answer = single_agent_no_view.query_agent(
            question=question_dict,
            task_description=task_description,
        )

        if answer['answer_idx'] is not None:
            if answer['answer_idx'] == correct_answer_idx:
                correct_count += 1
                is_correct = True
            else:
                incorrect_count += 1
        else:
            incorrect_count += 1
            parse_error = True
        total_count += 1

        response_filename = os.path.join(eval_results_path, f"response_{q_idx}.txt")
        with open(response_filename, "w", encoding="utf-8") as f:
            f.write(answer['response'])

        eval_results.append({
            "q_data": strip_images(qa),
            "agent_answer_idx": answer['answer_idx'],
            "agent_answer_letter": answer['answer_letter'],
            "agent_answer_text": answer['answer_text'],
            "agent_confidence": answer['confidence'],
            "is_correct": is_correct,
            "parse_error": parse_error,
            "response_filename": response_filename,
            "response": answer['response'],
            "eval_args": vars(args),
            "agent_chat_history": single_agent_no_view.agent.chat_history_no_image
        })

        table.add_data(
            wandb.Image(image_to_pil(first_view)),
            wandb.Image(image_to_pil(second_view)),
            question,
            "\n".join([f"{idx}) {option}" for idx, option in zip(["A", "B", "C", "D"], options)]),
            answer['response'],
            is_correct,
            parse_error,
            answer['answer_text'],
            answer['confidence'],
            correct_answer_text if not use_map else None,
            answer['answer_idx'],
            correct_answer_idx,
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in single_agent_no_view.agent.chat_history_no_image]),
            room_part,
            scene_id,
            q_type,
            qa.get('difficulty'),
            (str(qa['description_difficulty'])
                if isinstance(qa.get('description_difficulty'), int)
                else ', '.join(str(n) for n in qa['description_difficulty'])
            ) if 'description_difficulty' in qa else None,
            qa.get('distractor_difficulty'),
            qa.get('difficulty_sum'),
            qa.get('difficulty_int'),
            qa.get('question_object'),
            qa.get('angle'),
            qa.get('distance'),
            qa.get('composition'),
        )

    with open(os.path.join(eval_results_path, f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Correct count: {correct_count}")
    print(f"Incorrect count: {incorrect_count}")
    print(f"Total count: {total_count}")
    print(f"Accuracy: {correct_count / total_count if total_count > 0 else 0.0}")

    accuracy = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "total_count": total_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0.0
    }
    with open(os.path.join(eval_results_path, f"accuracy_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(accuracy, f, indent=4)

    wandb.log({"SingleNoView": table})
    wandb.log({"Correct Count": correct_count})
    wandb.log({"Incorrect Count": incorrect_count})
    wandb.log({"Total Count": total_count})
    wandb.log({"Accuracy": correct_count / total_count if total_count > 0 else 0.0})
    wandb.finish()


if __name__ == "__main__":
    main()
