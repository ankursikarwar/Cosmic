import os
import re

from .utils import encode_image
from .utils import create_vllm_client, create_openai_client

from typing import List, Dict, Any

import logging
logger = logging.getLogger(__name__)


class ConvAgent:
    def __init__(
        self,
        agent_id: str,
        agent_role: str,
        model_name: str,
        client_name: str,
        api_base: str,
        max_completion_tokens: int,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.agent_id = agent_id
        self.agent_role = agent_role

        self.agent_view = None
        self.global_map = None

        if agent_role not in ["answerer", "helper"]:
            raise ValueError(f"Invalid agent role: {agent_role}")

        self.model_name = model_name
        self.client_name = client_name
        self.api_base = api_base
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.enable_logging = enable_logging

        self.chat_history: List[Dict[str, Any]] = []
        self.chat_history_no_image: List[Dict[str, Any]] = []

        self.bbox_tracking = None
        self.bbox_tracking_list: List[str] = []

        if client_name == "vllm":
            self.client = create_vllm_client(
                api_base=api_base, model_name=model_name)
        elif client_name in ["openai", "gemini", "claude"]:
            print(f"Client name: {client_name}")
            api_key = os.environ.get(client_name.upper() + "_API_KEY", "")
            print(f"API key: {api_key}")
            self.client = create_openai_client(
                api_base=api_base, api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Invalid client name: {client_name}")


    def initialize_views(self, images: List[str]):
        assert images is not None, "Images must be provided"
        self.agent_view = images
        assert len(images) == 1, "Single view agent must have only one view"


    def initialize_map(self, image: str):
        assert image is not None, "Images must be provided"
        self.global_map = image


    def init_conversation(
        self,
        question: dict = None,
        max_num_turns: int = 5,
        bbox: str = None,
        task_description: str = None,
        message_from_answerer_agent: str = None,
        terminate: bool = False,
        confidence: bool = False,
        sg_communication: bool = False,
        bbox_tracking: bool = False,
    ):
        self.bbox_tracking = bbox_tracking

        if self.agent_role == "answerer":
            system_prompt = self.prep_answerer_agent_system_prompt()

            content, content_no_image = self.prepare_init_query(
                views=self.agent_view,
                question=question,
                max_num_turns=max_num_turns,
                task_description=task_description,
                bbox=bbox,
                terminate=terminate,
                confidence=confidence,
                sg_communication=sg_communication,
                bbox_tracking=bbox_tracking,
            )

        elif self.agent_role == "helper":
            assert message_from_answerer_agent is not None, "answerer agent starts the conversation"
            system_prompt = self.prep_helper_agent_system_prompt()

            content, content_no_image = self.prepare_init_query(
                views=self.agent_view,
                question=question,
                max_num_turns=max_num_turns,
                task_description=task_description,
                bbox=bbox,
                message_from_answerer_agent=message_from_answerer_agent,
                terminate=terminate,
                confidence=confidence,
                sg_communication=sg_communication,
                bbox_tracking=bbox_tracking,
            )

        self.chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        self.chat_history_no_image = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_no_image}
        ]

        logger.info(f"Agent {self.agent_id} initialized conversation")

    def prepare_init_query(
        self,
        views: List[str],
        question: dict,
        max_num_turns: int,
        task_description: str,
        bbox: str = None,
        message_from_answerer_agent: str = None,
        terminate: bool = False,
        confidence: bool = False,
        sg_communication: bool = False,
        bbox_tracking: bool = False,
    ):

        if self.agent_role == "helper":
            assert message_from_answerer_agent is not None, "answerer agent starts the conversation"

        content = []
        content_no_image = []

        for view in views:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(view)}"}
            })
            content_no_image.append({
                "type": "image_url",
                "image_url": {"url": "IMAGE PLACEHOLDER"}
            })

        # Conditionally add map image for answerer on map questions
        if self.agent_role == "answerer" and question.get('question_type') == 'map' and self.global_map is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(self.global_map)}"}
            })
            content_no_image.append({
                "type": "image_url",
                "image_url": {"url": "IMAGE PLACEHOLDER"}
            })

        if self.agent_role == "answerer":
            agent_prompt = self.prep_answerer_agent_prompt(
                question=question,
                max_num_turns=max_num_turns,
                task_description=task_description,
                bbox=bbox,
                terminate=terminate,
                confidence=confidence,
                sg_communication=sg_communication,
                bbox_tracking=bbox_tracking,
            )
            content.append({
                "type": "text",
                "text": agent_prompt
            })
            content_no_image.append({
                "type": "text",
                "text": agent_prompt
            })

        elif self.agent_role == "helper":
            agent_prompt = self.prep_helper_agent_prompt(
                question=question,
                max_num_turns=max_num_turns,
                task_description=task_description,
                message_from_answerer_agent=message_from_answerer_agent,
                bbox=bbox,
                sg_communication=sg_communication,
                bbox_tracking=bbox_tracking,
            )
            content.append({
                "type": "text",
                "text": agent_prompt
            })
            content_no_image.append({
                "type": "text",
                "text": agent_prompt
            })

        return content, content_no_image

    def prepare_query(self):
        """Prepare the query dict without sending it. Used for batch processing."""
        if "gpt" in self.model_name or "gemini" in self.model_name:
            query = {
                "messages": self.chat_history,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature,
                "reasoning_effort": "high"
            }
        else:
            query = {
                "messages": self.chat_history,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature,
            }
        return query

    def process_response(self, response):
        """Process a response from the batch call and update chat history."""
        if self.bbox_tracking:
            matches = list(re.findall(r"<BBOX>(.*?)</BBOX>", response, re.DOTALL))
            if matches:
                print("Matches: ", matches)
                bboxes = [match.strip() for match in matches]
                self.bbox_tracking_list.append(bboxes)
                for match in matches:
                    response = response.replace(
                        "<BBOX>"+match.strip()+"</BBOX>",
                        ""
                    )

        self.chat_history.append({"role": "assistant", "content": response})
        self.chat_history_no_image.append({"role": "assistant", "content": response})
        logger.info(f"Agent {self.agent_id} generated response: {response}")

        return response

    def send_message(self):

        if "gpt" in self.model_name or "gemini" in self.model_name:
            query = [{
                "messages": self.chat_history,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature,
                "reasoning_effort": "high"
                }]
        else:
            query = [{
            "messages": self.chat_history,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
        }]

        terminate = False
        response = ""

        if self.model_name == "Qwen/Qwen3-VL-8B-Thinking" or self.model_name == "Qwen/Qwen3-VL-32B-Thinking":
            raise ValueError("Qwen Thinking model is not supported for this agent")

        elif self.model_name == "gemini-2.5-pro" or self.model_name == "gemini-2.5-flash":
            responses = self.client.call_chat(
                query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
            )
            response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
            print("Current response: ", response)
            retry_count = 0
            max_retries = 10
            while response is None and retry_count < max_retries:
                retry_count += 1
                print(f"Response is None, retrying (attempt {retry_count}/{max_retries})...")
                responses = self.client.call_chat(
                    query, tqdm_desc="Generating response (retry)", tqdm_enable=self.enable_logging
                )
                response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
                print("Current response: ", response)

            if response is None:
                print(f"Failed to get response after {max_retries} retries. Returning None.")

        else:
            responses = self.client.call_chat(
                query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
            )
            response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
            print("Current response: ", response)

        if self.bbox_tracking:
            matches = list(re.findall(r"<BBOX>(.*?)</BBOX>", response, re.DOTALL))
            if matches:
                print("Matches: ", matches)
                bboxes = [match.strip() for match in matches]
                self.bbox_tracking_list.append(bboxes)
                for match in matches:
                    response = response.replace(
                        "<BBOX>"+match.strip()+"</BBOX>",
                        ""
                    )

        self.chat_history.append({"role": "assistant", "content": response})
        self.chat_history_no_image.append({"role": "assistant", "content": response})
        logger.info(f"Agent {self.agent_id} generated response: {response}")

        return response

    def receive_message(self, message: str):
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history_no_image.append({"role": "user", "content": message})
        logger.info(f"Agent {self.agent_id} received message: {message}")
        return self.chat_history

    def query_agent(self,
        question: dict,
        confidence: bool,
    ):
        ask_question_prompt = self.prep_ask_question_prompt(
            question=question,
            confidence=confidence,
        )
        self.chat_history.append({"role": "user", "content": ask_question_prompt})
        self.chat_history_no_image.append({"role": "user", "content": ask_question_prompt})

        if "gpt" in self.model_name or "gemini" in self.model_name:
            query = [{
                "messages": self.chat_history,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature,
                "reasoning_effort": "high"
                }]
        else:
            query = [{
            "messages": self.chat_history,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
        }]

        responses = self.client.call_chat(
            query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
        )
        response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
        print("Query current response: ", response)

        retry_count = 0
        max_retries = 10
        while response is None and retry_count < max_retries:
            retry_count += 1
            print(f"Response is None in query_agent, retrying (attempt {retry_count}/{max_retries})...")
            responses = self.client.call_chat(
                query, tqdm_desc="Generating response (retry)", tqdm_enable=self.enable_logging
            )
            response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
            print("Query current response: ", response)

        if response is None:
            print(f"Failed to get response after {max_retries} retries. Returning None.")

        self.chat_history.append({"role": "assistant", "content": response})
        self.chat_history_no_image.append({"role": "assistant", "content": response})

        return response

    def prep_answerer_agent_prompt(
        self,
        question: dict,
        max_num_turns: int,
        task_description: str,
        bbox: str,
        terminate: bool,
        confidence: bool,
        sg_communication: bool = False,
        bbox_tracking: bool = False,
    ):
        option_idxs = ['A', 'B', 'C', 'D']
        options = question['options']
        options_str = "\n".join([f"{option_idxs[i]}) {option}" for i, option in enumerate(options)])

        prompt = f'''
        1. The provided image is your view of the room.
        2. HELPER AGENT also receives one image that shows a different view of the same room.
        3. You will be given a multiple-choice question with different options. Only one of the options is correct.
        4. You can send only one message at a time. You cannot send consecutive messages. You have to wait for the HELPER AGENT to respond before you can send your next message.
        5. You can send a maximum of {max_num_turns} messages to the HELPER AGENT.
        6. After the conversation is over, you will be asked to provide the answer.
        '''

        if confidence:
            prompt += "\n7. You will also be asked to provide a confidence score for your answer on a scale of 1 to 10."

        prompt += task_description

        if sg_communication:
            prompt += '''
            IMPORTANT:
            1. In your first message, you must use a detailed scene graph to describe your view of the room.
            2. You will also receive a scene graph from the HELPER AGENT in their first message.
            3. Reason over the scene graphs and the conversation history to answer the question.
            4. Mention the scene graph in this format <SCENE_GRAPH>scene_graph</SCENE_GRAPH> in your response.
            5. The scene graph should contain all objects and thier spatial relationships with each other.
            '''

        if bbox_tracking:
            prompt += '''
            IMPORTANT:
            1. Whenever you mention an object in your response, you must also mention its bounding box just next to the object name in the format <BBOX>bounding_box</BBOX>.
            2. The bounding box should be in the format of <BBOX>x1,y1,x2,y2</BBOX>.
            3. For example, "Hi! I see a side table<BBOX>263,379,284,419</BBOX> next to the couch."
            4. This bounding box is just for analysis. It wont be shown to the user.
            '''

        if bbox:
            if terminate:
                prompt += "\nNote: When you are ready to answer the question, you can terminate the conversation early by saying 'TERMINATE'. Use exact word 'TERMINATE' in your response."
                prompt += f"\nNote: Here is the list of objects and their bounding boxes present in your view of the room: {bbox}"
                prompt += "\nThese bounding boxes are provided to you for your reference. You can use them to answer the question if needed."
            else:
                prompt += f"\nNote: Here is the list of objects and their bounding boxes present in your view of the room: {bbox}"
                prompt += "\nThese bounding boxes are provided to you for your reference. You can use them to answer the question if needed."
        else:
            if terminate:
                prompt += "\nNote: When you are ready to answer the question, you can terminate the conversation early by saying 'TERMINATE'. Use exact word 'TERMINATE' in your response."

        prompt += f"\nGoal: {question['answerer_goal']}"
        prompt += f"\n\nQUESTION: {question['question']}"
        prompt += f"\nOPTIONS: {options_str}"
        prompt += '''
        Begin the conversation with the HELPER AGENT. You MUST generate all your messages in this format ANSWERER AGENT: <RESPONSE>.
        Do not deviate from this format.
        '''

        return prompt

    def prep_helper_agent_prompt(
        self,
        question: dict,
        max_num_turns: int,
        task_description: str,
        message_from_answerer_agent: str,
        bbox: str,
        sg_communication: bool = False,
        bbox_tracking: bool = False,
    ):
        prompt = f'''
        1. The provided image is your view of the room.
        2. ANSWERER AGENT also receives one image that shows a different view of the same room.
        3. You can send only one message at a time. You cannot send consecutive messages. You have to wait for the ANSWERER AGENT to respond before you can send your next message.
        4. You can send a maximum of {max_num_turns} messages to the ANSWERER AGENT.
        '''

        prompt += task_description

        if sg_communication:
            prompt += '''
            IMPORTANT:
            1. In your first message, you must use a detailed scene graph to describe your view of the room.
            2. You will also receive a scene graph from the ANSWERER AGENT in their first message.
            3. Reason over the scene graphs and the conversation history to answer the question.
            4. Mention the scene graph in this format <SCENE_GRAPH>scene_graph</SCENE_GRAPH> in your response.
            5. The scene graph should contain all objects and thier spatial relationships with each other.
            '''

        if bbox_tracking:
            prompt += '''
            IMPORTANT:
            1. Whenever you mention an object in your response, you must also mention its bounding box just next to the object name in the format <BBOX>bounding_box</BBOX>.
            2. The bounding box should be in the format of <BBOX>x1,y1,x2,y2</BBOX>.
            3. For example, "Hi! I see a side table<BBOX>263,379,284,419</BBOX> next to the couch."
            4. This bounding box is just for analysis. It wont be shown to the user.
            '''

        if bbox:
            prompt += "\nNote: Here is the list of objects and their bounding boxes present in your view of the room: {bbox}"
            prompt += "\nThese bounding boxes are provided to you for your reference. You can use them to answer the question if needed."

        prompt += f"\nGoal: {question['helper_goal']}"
        prompt += '''
        Begin the conversation with the ANSWERER AGENT by responding to their first message. You MUST generate all your messages in this format HELPER AGENT: <RESPONSE>.
        Do not deviate from this format.
        '''
        prompt += f"\n\n{message_from_answerer_agent}"

        return prompt

    def prep_ask_question_prompt(
        self,
        question: dict,
        confidence: bool,
    ):
        option_idxs = ['A', 'B', 'C', 'D']
        options = question['options']
        options_str = "\n".join([f"{option_idxs[i]}) {option}" for i, option in enumerate(options)])

        if question.get('question_type') != 'map':
            prompt = f'''
            Now you need to answer the multiple-choice question based on your view and the conversation with the HELPER AGENT.

            QUESTION: {question['question']}
            OPTIONS: {options_str}

            Instructions:
            1. Select the correct answer from the given options. Make sure to select only one of the options from the given options A, B, C, or D.
            2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER> or <ANSWER>C</ANSWER> or <ANSWER>D</ANSWER>.
            '''
        else:
            prompt = f'''
            Now you need to answer the multiple-choice question based on your view and the conversation with the HELPER AGENT.

            QUESTION: {question['question']}
            OPTIONS: {options_str}

            Instructions:
            1. Select the correct answer from the given options. Make sure to select only one of the options from the given options A or B.
            2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER>.
            '''

        if confidence:
            prompt += "\n3. Also provide a confidence score for your answer on a scale of 1 to 10."
            prompt += "\n4. Format the confidence score like <CONFIDENCE>confidence_score</CONFIDENCE>."

        return prompt


    def prep_answerer_agent_system_prompt(self):
        prompt = '''
        1. You will be participating in a COLLABORATIVE TASK to answer a question.
        2. You are the ANSWERER AGENT.
        3. You will be connected to a HELPER AGENT.
        4. In this task, you and the HELPER AGENT will receive one image each that shows different views of the same room.
        5. You have to chat and collaborate with the HELPER AGENT to answer your question correctly.
        6. Overall, your role is to answer your question correctly by having a conversation with the HELPER AGENT.
        '''
        return prompt

    def prep_helper_agent_system_prompt(self):
        prompt = '''
        1. You will be participating in a COLLABORATIVE TASK.
        2. You are the HELPER AGENT.
        3. You will be connected to an ANSWERER AGENT.
        4. In this task, you and the ANSWERER AGENT will receive one image each that shows different views of the same room.
        5. You have to chat and collaborate with the ANSWERER AGENT to help them answer their question correctly.
        6. Overall, your role is to help the ANSWERER AGENT by having a conversation with them.
        '''
        return prompt



class BothViewsAgent:
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        client_name: str,
        api_base: str,
        max_completion_tokens: int,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.agent_views = None
        self.agent_id = agent_id
        self.has_map = False

        self.model_name = model_name
        self.client_name = client_name
        self.api_base = api_base
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.enable_logging = enable_logging

        self.chat_history: List[Dict[str, Any]] = []
        self.chat_history_no_image: List[Dict[str, Any]] = []

        if client_name == "vllm":
            self.client = create_vllm_client(
                api_base=api_base, model_name=model_name)
        elif client_name in ["openai", "gemini"]:
            api_key = os.environ.get(client_name.upper() + "_API_KEY", "")
            self.client = create_openai_client(
                api_base=api_base, api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Invalid client name: {client_name}")


    def intialize_and_query_agent(
        self,
        images: List[str],
        question: dict,
        task_description: str,
        bbox: List[str] = None,
        confidence: bool = False,
        map_image: str = None,
    ):
        assert images is not None, "Images must be provided"
        assert len(images) == 2, "Both views agent must have two views"

        if bbox is not None:
            assert len(bbox) == 2, "Both views agent must have two bboxes"

        self.has_map = map_image is not None

        system_prompt = self.prep_agent_system_prompt(is_map=self.has_map)

        content, content_no_image = self.prepare_init_query(
            views=images,
            question=question,
            task_description=task_description,
            bbox=bbox,
            confidence=confidence,
            map_image=map_image,
        )

        self.chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        self.chat_history_no_image = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_no_image}
        ]

        if "gpt" in self.model_name or "gemini" in self.model_name:
            query = [{
                "messages": self.chat_history,
                "temperature": self.temperature,
                "reasoning_effort": "high"
            }]
        else:
            query = [{
                "messages": self.chat_history,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature,
            }]

        responses = self.client.call_chat(
            query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
        )
        response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
        print("Query current response: ", response)

        retry_count = 0
        max_retries = 10
        while response is None and retry_count < max_retries:
            retry_count += 1
            print(f"Response is None in query_agent, retrying (attempt {retry_count}/{max_retries})...")
            responses = self.client.call_chat(
                query, tqdm_desc="Generating response (retry)", tqdm_enable=self.enable_logging
            )
            response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
            print("Query current response: ", response)

        if response is None:
            print(f"Failed to get response after {max_retries} retries. Returning None.")

        self.chat_history.append({"role": "assistant", "content": response})
        self.chat_history_no_image.append({"role": "assistant", "content": response})

        return response

    def prepare_init_query(
        self,
        views: List[str],
        question: dict,
        task_description: str,
        bbox: List[str] = None,
        confidence: bool = False,
        map_image: str = None,
    ):

        content = []
        content_no_image = []

        for view in views:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(view)}"}
            })
            content_no_image.append({
                "type": "image_url",
                "image_url": {"url": "IMAGE PLACEHOLDER"}
            })

        if map_image is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(map_image)}"}
            })
            content_no_image.append({
                "type": "image_url",
                "image_url": {"url": "IMAGE PLACEHOLDER"}
            })

        agent_prompt = self.prep_agent_prompt(
            question=question,
            task_description=task_description,
            bbox=bbox,
            confidence=confidence,
            is_map=(map_image is not None),
        )
        content.append({
            "type": "text",
            "text": agent_prompt
        })
        content_no_image.append({
            "type": "text",
            "text": agent_prompt
        })

        return content, content_no_image

    def prep_agent_prompt(
        self,
        question: dict,
        task_description: str,
        bbox: List[str],
        confidence: bool = False,
        is_map: bool = False,
    ):
        if is_map:
            option_idxs = ['A', 'B']
        else:
            option_idxs = ['A', 'B', 'C', 'D']
        options = question['options']
        options_str = "\n".join([f"{option_idxs[i]}) {option}" for i, option in enumerate(options)])

        if is_map:
            prompt = '''
            1. The provided images are two different views of the same room along with a top-down map of the room.
            2. You will be given a multiple-choice question with different options. Only one of the options is correct.
            '''
        else:
            prompt = '''
            1. The provided images are two different views of the same room.
            2. You will be given a multiple-choice question with different options. Only one of the options is correct.
            '''

        prompt += task_description

        if bbox:
            prompt += f"\nNote: Here is the list of objects and their bounding boxes present in the first view of the room: {bbox[0]}"
            prompt += f"\nNote: Here is the list of objects and their bounding boxes present in the second view of the room: {bbox[1]}"
            prompt += "\nThese bounding boxes are provided to you for your reference. You can use them to answer the question if needed."

        if is_map:
            prompt += f'''
            Now you need to answer the multiple-choice question based on your two views of the room and the top-down map of the room.

            QUESTION: {question['question']}
            OPTIONS: {options_str}

            Instructions:
            1. Select the correct answer from the given options.
            2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER>.'''
        else:
            prompt += f'''
            Now you need to answer the multiple-choice question based on your two views of the room.

            QUESTION: {question['question']}
            OPTIONS: {options_str}

            Instructions:
            1. Select the correct answer from the given options.
            2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER> or <ANSWER>C</ANSWER> or <ANSWER>D</ANSWER>.'''

        if confidence:
            prompt += "\n3. Also provide a confidence score for your answer on a scale of 1 to 10."
            prompt += "\n4. Format the confidence score like <CONFIDENCE>confidence_score</CONFIDENCE>."

        return prompt

    def prep_agent_system_prompt(self, is_map: bool = False):
        if is_map:
            return '''
            1. You will be participating in a QUESTION ANSWERING TASK to answer a question.
            2. In this task, you will receive two images that shows different views of the same room along with a top-down map of the room.
            3. You have to answer the multiple-choice question based on your two views of the room and the top-down map of the room.
            4. Overall, your role is to answer the question correctly.
            '''
        else:
            return '''
            1. You will be participating in a QUESTION ANSWERING TASK to answer a question.
            2. In this task, you will receive two images that shows different views of the same room.
            3. You have to answer the multiple-choice question based on your two views of the room.
            4. Overall, your role is to answer the question correctly.
            '''

class OneViewAgent:
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        client_name: str,
        api_base: str,
        max_completion_tokens: int,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.agent_views = None
        self.agent_id = agent_id

        self.model_name = model_name
        self.client_name = client_name
        self.api_base = api_base
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.enable_logging = enable_logging

        self.chat_history: List[Dict[str, Any]] = []
        self.chat_history_no_image: List[Dict[str, Any]] = []

        if client_name == "vllm":
            self.client = create_vllm_client(
                api_base=api_base, model_name=model_name)
        elif client_name in ["openai", "gemini"]:
            api_key = os.environ.get(client_name.upper() + "_API_KEY", "")
            self.client = create_openai_client(
                api_base=api_base, api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Invalid client name: {client_name}")


    def intialize_and_query_agent(
        self,
        images: List[str],
        question: dict,
        task_description: str,
        bbox: List[str] = None,
        confidence: bool = False,
    ):
        assert images is not None, "Images must be provided"
        assert len(images) == 1, "One view agent must have one view"

        if bbox is not None:
            assert len(bbox) == 1, "One view agent must have one bbox"

        system_prompt = self.prep_agent_system_prompt()

        content, content_no_image = self.prepare_init_query(
            views=images,
            question=question,
            task_description=task_description,
            bbox=bbox,
            confidence=confidence,
        )

        self.chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        self.chat_history_no_image = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_no_image}
        ]

        query = [{
            "messages": self.chat_history,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature
        }]

        response = ""
        if confidence:
            while re.search(r"<ANSWER>.*?</ANSWER>\s*<CONFIDENCE>.*?</CONFIDENCE>", response, re.DOTALL) is None:
                responses = self.client.call_chat(
                    query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
                )
                response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
                print("Query current response: ", response)
                if response is None:
                    print("Response is None, will retry in next iteration...")
                    continue
        else:
            while re.search(r"<ANSWER>.*?</ANSWER>", response, re.DOTALL) is None:
                responses = self.client.call_chat(
                    query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
                )
                response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
                print("Query current response: ", response)
                if response is None:
                    print("Response is None, will retry in next iteration...")
                    continue
        self.chat_history.append({"role": "assistant", "content": response})
        self.chat_history_no_image.append({"role": "assistant", "content": response})

        return response

    def prepare_init_query(
        self,
        views: List[str],
        question: dict,
        task_description: str,
        bbox: List[str] = None,
        confidence: bool = False,
    ):
        content = []
        content_no_image = []

        for view in views:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(view)}"}
            })
            content_no_image.append({
                "type": "image_url",
                "image_url": {"url": "IMAGE PLACEHOLDER"}
            })

        agent_prompt = self.prep_agent_prompt(
            question=question,
            task_description=task_description,
            bbox=bbox,
            confidence=confidence,
        )
        content.append({
            "type": "text",
            "text": agent_prompt
        })
        content_no_image.append({
            "type": "text",
            "text": agent_prompt
        })

        return content, content_no_image

    def prep_agent_prompt(
        self,
        question: dict,
        task_description: str,
        bbox: List[str],
        confidence: bool = False,
    ):
        option_idxs = ['A', 'B', 'C', 'D']
        options = question['options']
        options_str = "\n".join([f"{option_idxs[i]}) {option}" for i, option in enumerate(options)])

        prompt = '''
        1. The provided image is a view of the room.
        2. You will be given a multiple-choice question with different options. Only one of the options is correct.
        '''

        prompt += task_description

        if bbox:
            prompt += f"\nNote: Here is the list of objects and their bounding boxes present in the view of the room: {bbox[0]}"
            prompt += "\nThese bounding boxes are provided to you for your reference. You can use them to answer the question if needed."

        prompt += f'''
        Now you need to answer the multiple-choice question based on your view of the room.

        QUESTION: {question['question']}
        OPTIONS: {options_str}

        Instructions:
        1. Select the correct answer from the given options.
        2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER> or <ANSWER>C</ANSWER> or <ANSWER>D</ANSWER>.'''

        if confidence:
            prompt += "\n3. Also provide a confidence score for your answer on a scale of 1 to 10."
            prompt += "\n4. Format the confidence score like <CONFIDENCE>confidence_score</CONFIDENCE>."

        return prompt

    def prep_agent_system_prompt(self):
        return '''
        1. You will be participating in a QUESTION ANSWERING TASK to answer a question.
        2. You are a ANSWERER AGENT.
        3. In this task, you will receive one image that shows a view of a room.
        4. You have to answer the multiple-choice question based on your view of the room.
        5. Overall, your role is to answer the question correctly.
        '''


class NoViewAgent:
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        client_name: str,
        api_base: str,
        max_completion_tokens: int,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.agent_id = agent_id

        self.model_name = model_name
        self.client_name = client_name
        self.api_base = api_base
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.enable_logging = enable_logging

        self.chat_history: List[Dict[str, Any]] = []
        self.chat_history_no_image: List[Dict[str, Any]] = []

        if client_name == "vllm":
            self.client = create_vllm_client(
                api_base=api_base, model_name=model_name)
        elif client_name in ["openai", "gemini"]:
            api_key = os.environ.get(client_name.upper() + "_API_KEY", "")
            self.client = create_openai_client(
                api_base=api_base, api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Invalid client name: {client_name}")


    def intialize_and_query_agent(
        self,
        question: dict,
        task_description: str,
        confidence: bool = False,
    ):
        system_prompt = self.prep_agent_system_prompt()

        content, content_no_image = self.prepare_init_query(
            question=question,
            task_description=task_description,
            confidence=confidence,
        )

        self.chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        self.chat_history_no_image = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_no_image}
        ]

        query = [{
            "messages": self.chat_history,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature
        }]

        response = ""
        if confidence:
            while re.search(r"<ANSWER>.*?</ANSWER>\s*<CONFIDENCE>.*?</CONFIDENCE>", response, re.DOTALL) is None:
                responses = self.client.call_chat(
                    query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
                )
                response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
                print("Query current response: ", response)
                if response is None:
                    print("Response is None, will retry in next iteration...")
                    continue
        else:
            while re.search(r"<ANSWER>.*?</ANSWER>", response, re.DOTALL) is None:
                responses = self.client.call_chat(
                    query, tqdm_desc="Generating response", tqdm_enable=self.enable_logging
                )
                response = responses[0].choices[0].message.content if (responses and responses[0] and responses[0].choices) else None
                print("Query current response: ", response)
                if response is None:
                    print("Response is None, will retry in next iteration...")
                    continue
        self.chat_history.append({"role": "assistant", "content": response})
        self.chat_history_no_image.append({"role": "assistant", "content": response})

        return response

    def prepare_init_query(
        self,
        question: dict,
        task_description: str,
        confidence: bool = False,
    ):
        content = []
        content_no_image = []

        agent_prompt = self.prep_agent_prompt(
            question=question,
            task_description=task_description,
            confidence=confidence,
        )
        content.append({
            "type": "text",
            "text": agent_prompt
        })
        content_no_image.append({
            "type": "text",
            "text": agent_prompt
        })

        return content, content_no_image

    def prep_agent_prompt(
        self,
        question: dict,
        task_description: str,
        confidence: bool = False,
    ):
        option_idxs = ['A', 'B', 'C', 'D']
        options = question['options']
        options_str = "\n".join([f"{option_idxs[i]}) {option}" for i, option in enumerate(options)])

        prompt = '''
        1. You will be given a multiple-choice question with different options. Only one of the options is correct.
        '''
        prompt += task_description

        prompt += f'''
        Now you need to answer the multiple-choice question based on common-sense knowledge.

        QUESTION: {question['question']}
        OPTIONS: {options_str}

        Instructions:
        1. Select the correct answer from the given options.
        2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER> or <ANSWER>C</ANSWER> or <ANSWER>D</ANSWER>.'''

        if confidence:
            prompt += "\n3. Also provide a confidence score for your answer on a scale of 1 to 10."
            prompt += "\n4. Format the confidence score like <CONFIDENCE>confidence_score</CONFIDENCE>."

        return prompt

    def prep_agent_system_prompt(self):
        return '''
        1. You will be participating in a QUESTION ANSWERING TASK to answer a question.
        2. You are a ANSWERER AGENT.
        3. In this task, you have to answer the multiple-choice question based on common-sense knowledge.
        4. Overall, your role is to answer the question correctly.
        '''
