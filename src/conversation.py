import re
from typing import List

from .agent import ConvAgent
from .agent import BothViewsAgent
from .agent import OneViewAgent
from .agent import NoViewAgent

from .conv_utils import print_conv_beginning
from .conv_utils import print_agent_message
from .conv_utils import print_conv_terminated
from .conv_utils import print_conv_completed


class TwoAgentConv:
    def __init__(
        self,
        question: dict,
        terminate: bool,
        confidence: bool,
        sg_communication: bool,
        bbox_tracking: bool,
        max_num_turns: int,
        bbox_provided: bool,
        answerer_task_description: str,
        helper_task_description: str,
        answerer_model_name: str = "gpt-4o-mini",
        helper_model_name: str = "gpt-4o-mini",
        answerer_client_name: str = "openai",
        helper_client_name: str = "openai",
        answerer_api_base: str = "https://api.openai.com/v1",
        helper_api_base: str = "https://api.openai.com/v1",
        max_completion_tokens: int = 1000,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        assert max_num_turns > 0, "Number of turns must be greater than 0"

        self.max_num_turns = max_num_turns
        self.question = question
        self.bbox_provided = bbox_provided
        self.terminate = terminate
        self.confidence = confidence
        self.sg_communication = sg_communication
        self.bbox_tracking = bbox_tracking
        self.answerer_task_description = answerer_task_description
        self.helper_task_description = helper_task_description

        self.answerer_model_name = answerer_model_name
        self.helper_model_name = helper_model_name

        self.answerer_agent = ConvAgent(
            agent_id=1,
            agent_role="answerer",
            model_name=answerer_model_name,
            client_name=answerer_client_name,
            api_base=answerer_api_base,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            enable_logging=enable_logging
        )
        self.helper_agent = ConvAgent(
            agent_id=2,
            agent_role="helper",
            model_name=helper_model_name,
            client_name=helper_client_name,
            api_base=helper_api_base,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            enable_logging=enable_logging
        )

    def initialize_conversation(
        self,
        answerer_images: List[str],
        helper_images: List[str],
        answerer_bbox: str = None,
        helper_bbox: str = None,
    ):
        """Initialize the conversation but don't run it yet."""
        self.conversation = ""
        self.conversation_dict = []
        self.current_turn = 0
        self.is_terminated = False
        self.last_message = None
        self.answerer_bbox = answerer_bbox
        self.helper_bbox = helper_bbox

        self.conversation += print_conv_beginning(
            self.answerer_agent, self.helper_agent, self.max_num_turns)

        self.answerer_agent.initialize_views(answerer_images)
        # Conditionally initialize map for map questions
        if self.question.get('question_type') == 'map' and 'global_map_image' in self.question:
            self.answerer_agent.initialize_map(self.question['global_map_image'])
        self.helper_agent.initialize_views(helper_images)

    def prepare_turn_1_query(self):
        """Prepare the query for turn 1 (answerer) without sending it."""
        if self.is_terminated:
            return None

        self.answerer_agent.init_conversation(
            question=self.question,
            max_num_turns=self.max_num_turns,
            bbox=self.answerer_bbox if self.bbox_provided else None,
            task_description=self.answerer_task_description,
            terminate=self.terminate,
            confidence=self.confidence,
            sg_communication=self.sg_communication,
            bbox_tracking=self.bbox_tracking,
        )
        return self.answerer_agent.prepare_query()

    def process_turn_1_response(self, response):
        """Process the response from turn 1 (answerer)."""
        if self.is_terminated:
            return

        message = self.answerer_agent.process_response(response)
        self.last_message = message

        self.conversation_dict.append({
            "agent_role": self.answerer_agent.agent_role,
            "message": message,
            "turn": 1,
            "message_number": 1,
        })
        self.conversation += print_agent_message(
            self.answerer_agent.agent_role, message, 1, 1)

        self.current_turn = 1

    def execute_turn_1(self):
        """Execute turn 1 of the conversation."""
        if self.is_terminated:
            return

        self.answerer_agent.init_conversation(
            question=self.question,
            max_num_turns=self.max_num_turns,
            bbox=self.answerer_bbox if self.bbox_provided else None,
            task_description=self.answerer_task_description,
            terminate=self.terminate,
            confidence=self.confidence,
            sg_communication=self.sg_communication,
            bbox_tracking=self.bbox_tracking,
        )
        message = self.answerer_agent.send_message()
        self.last_message = message

        self.conversation_dict.append({
            "agent_role": self.answerer_agent.agent_role,
            "message": message,
            "turn": 1,
            "message_number": 1,
        })
        self.conversation += print_agent_message(
            self.answerer_agent.agent_role, message, 1, 1)

        self.current_turn = 1

    def prepare_turn_1_helper_query(self):
        """Prepare the query for turn 1 (helper) without sending it."""
        if self.is_terminated:
            return None

        self.helper_agent.init_conversation(
            question=self.question,
            max_num_turns=self.max_num_turns,
            bbox=self.helper_bbox if self.bbox_provided else None,
            task_description=self.helper_task_description,
            message_from_answerer_agent=self.last_message,
            terminate=self.terminate,
            sg_communication=self.sg_communication,
            bbox_tracking=self.bbox_tracking,
        )
        return self.helper_agent.prepare_query()

    def process_turn_1_helper_response(self, response):
        """Process the response from turn 1 (helper)."""
        if self.is_terminated:
            return

        message = self.helper_agent.process_response(response)
        self.last_message = message

        self.conversation_dict.append({
            "agent_role": self.helper_agent.agent_role,
            "message": message,
            "turn": 1,
            "message_number": 2,
        })
        self.conversation += print_agent_message(
            self.helper_agent.agent_role, message, 1, 2)

    def execute_turn_1_helper_response(self):
        """Execute helper's response in turn 1."""
        if self.is_terminated:
            return

        self.helper_agent.init_conversation(
            question=self.question,
            max_num_turns=self.max_num_turns,
            bbox=self.helper_bbox if self.bbox_provided else None,
            task_description=self.helper_task_description,
            message_from_answerer_agent=self.last_message,
            terminate=self.terminate,
            sg_communication=self.sg_communication,
            bbox_tracking=self.bbox_tracking,
        )
        message = self.helper_agent.send_message()
        self.last_message = message

        self.conversation_dict.append({
            "agent_role": self.helper_agent.agent_role,
            "message": message,
            "turn": 1,
            "message_number": 2,
        })
        self.conversation += print_agent_message(
            self.helper_agent.agent_role, message, 1, 2)

    def prepare_turn_query(self, turn: int):
        """Prepare queries for a specific turn (turn >= 2) without sending them."""
        if self.is_terminated or turn < 2:
            return None, False

        self.answerer_agent.receive_message(self.last_message)
        answerer_query = self.answerer_agent.prepare_query()

        return answerer_query, True

    def process_turn_answerer_response(self, response, turn: int):
        """Process answerer's response for a specific turn and check for termination."""
        if self.is_terminated:
            return False

        message = self.answerer_agent.process_response(response)

        self.conversation_dict.append({
            "agent_role": self.answerer_agent.agent_role,
            "message": message,
            "turn": turn,
            "message_number": 1,
        })
        self.conversation += print_agent_message(
            self.answerer_agent.agent_role, message, turn, 1)

        if self.terminate:
            if message is not None:
                if ("terminate" in message.lower() or "termiante" in message.lower()
                    or 'termitnate' in message.lower() or 'termitate' in message.lower()):
                    self.conversation += print_conv_terminated()
                    self.is_terminated = True
                    self.current_turn = turn
                    return False

        return True

    def prepare_turn_helper_query(self, answerer_message: str):
        """Prepare helper query after answerer has responded."""
        if self.is_terminated:
            return None

        self.helper_agent.receive_message(answerer_message)
        return self.helper_agent.prepare_query()

    def process_turn_helper_response(self, response, turn: int):
        """Process helper's response for a specific turn."""
        if self.is_terminated:
            return

        message = self.helper_agent.process_response(response)
        self.last_message = message

        self.conversation_dict.append({
            "agent_role": self.helper_agent.agent_role,
            "message": message,
            "turn": turn,
            "message_number": 2,
        })
        self.conversation += print_agent_message(
            self.helper_agent.agent_role, message, turn, 2)

        self.current_turn = turn

    def execute_turn(self, turn: int):
        """Execute a specific turn (turn >= 2)."""
        if self.is_terminated or turn < 2:
            return

        self.answerer_agent.receive_message(self.last_message)
        message = self.answerer_agent.send_message()

        self.conversation_dict.append({
            "agent_role": self.answerer_agent.agent_role,
            "message": message,
            "turn": turn,
            "message_number": 1,
        })
        self.conversation += print_agent_message(
            self.answerer_agent.agent_role, message, turn, 1)

        if self.terminate:
            if message is not None:
                if ("terminate" in message.lower() or "termiante" in message.lower()
                    or 'termitnate' in message.lower() or 'termitate' in message.lower()):
                    self.conversation += print_conv_terminated()
                    self.is_terminated = True
                    self.current_turn = turn
                    return

        self.helper_agent.receive_message(message)
        message = self.helper_agent.send_message()
        self.last_message = message

        self.conversation_dict.append({
            "agent_role": self.helper_agent.agent_role,
            "message": message,
            "turn": turn,
            "message_number": 2,
        })
        self.conversation += print_agent_message(
            self.helper_agent.agent_role, message, turn, 2)

        self.current_turn = turn

    def finalize_conversation(self):
        """Finalize the conversation and return results."""
        if not self.is_terminated:
            self.conversation += print_conv_completed()

        return {
            "conv_text": self.conversation,
            "conv_dict": self.conversation_dict,
            "turns_completed": self.current_turn,
        }

    def run_conversation(
        self,
        answerer_images: List[str],
        helper_images: List[str],
        answerer_bbox: str = None,
        helper_bbox: str = None,
    ):
        """Run the full conversation (legacy method for backward compatibility)."""
        self.initialize_conversation(answerer_images, helper_images, answerer_bbox, helper_bbox)
        self.execute_turn_1()
        self.execute_turn_1_helper_response()

        for turn in range(2, self.max_num_turns + 1):
            if self.is_terminated:
                break
            self.execute_turn(turn)

        return self.finalize_conversation()

    def prepare_query_answerer_agent(self, question: dict):
        """Prepare the query for answerer agent without sending it. Used for batch processing."""
        ask_question_prompt = self.answerer_agent.prep_ask_question_prompt(
            question=question,
            confidence=self.confidence,
        )
        self.answerer_agent.chat_history.append({"role": "user", "content": ask_question_prompt})
        self.answerer_agent.chat_history_no_image.append({"role": "user", "content": ask_question_prompt})

        if "gpt" in self.answerer_agent.model_name or "gemini" in self.answerer_agent.model_name:
            query = {
                "messages": self.answerer_agent.chat_history,
                "max_completion_tokens": self.answerer_agent.max_completion_tokens,
                "temperature": self.answerer_agent.temperature,
                "reasoning_effort": "high"
            }
        else:
            query = {
                "messages": self.answerer_agent.chat_history,
                "max_completion_tokens": self.answerer_agent.max_completion_tokens,
                "temperature": self.answerer_agent.temperature,
            }

        return query

    def process_query_answerer_agent_response(self, response: str, question: dict):
        """Process the response from query_answerer_agent and return parsed answer."""
        if response is None:
            print(f"Response is None in process_query_answerer_agent_response")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": None,
            }

        is_map = question.get('question_type') == 'map'

        if not is_map:
            match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", response.strip(), re.IGNORECASE)
        else:
            match = re.search(r"<ANSWER>\s*([A-B])\s*</ANSWER>", response.strip(), re.IGNORECASE)

        conf_match = None
        if self.confidence:
            conf_match = re.search(r"<CONFIDENCE>\s*([1-9]|10)\s*</CONFIDENCE>", response.strip(), re.IGNORECASE)
        if not match or (self.confidence and not conf_match):
            print(f"Parsing failure in answer response: {response}")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": response,
            }

        answer_letter = match.group(1).upper()
        confidence = conf_match.group(1) if self.confidence else None

        selected_option = None
        idx = None
        if not is_map:
            if question['options'] and answer_letter in ['A', 'B', 'C', 'D']:
                idx = ['A', 'B', 'C', 'D'].index(answer_letter)
                selected_option = question['options'][idx]
        else:
            if question['options'] and answer_letter in ['A', 'B']:
                idx = ['A', 'B'].index(answer_letter)
                selected_option = question['options'][idx]

        return {
            "answer_idx": idx,
            "answer_letter": answer_letter,
            "answer_text": selected_option,
            "confidence": confidence,
            "response": response,
        }

    def query_answerer_agent(self, question: dict):
        response = self.answerer_agent.query_agent(question, confidence=self.confidence)

        retry_count = 0
        max_retries = 10
        while response is None and retry_count < max_retries:
            retry_count += 1
            print(f"Response is None in query_answerer_agent, retrying (attempt {retry_count}/{max_retries})...")
            response = self.answerer_agent.query_agent(question, confidence=self.confidence)

        if response is None:
            print(f"Failed to get response after {max_retries} retries. Returning None response.")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": None,
            }

        is_map = question.get('question_type') == 'map'

        if not is_map:
            match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", response.strip(), re.IGNORECASE)
        else:
            match = re.search(r"<ANSWER>\s*([A-B])\s*</ANSWER>", response.strip(), re.IGNORECASE)

        conf_match = None
        if self.confidence:
            conf_match = re.search(r"<CONFIDENCE>\s*([1-9]|10)\s*</CONFIDENCE>", response.strip(), re.IGNORECASE)

        if not match or (self.confidence and not conf_match):
            print(f"Parsing failure in answer response: {response}")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": response,
            }

        answer_letter = match.group(1).upper()
        confidence = conf_match.group(1) if self.confidence else None

        selected_option = None
        idx = None
        if not is_map:
            if question['options'] and answer_letter in ['A', 'B', 'C', 'D']:
                idx = ['A', 'B', 'C', 'D'].index(answer_letter)
                selected_option = question['options'][idx]
        else:
            if question['options'] and answer_letter in ['A', 'B']:
                idx = ['A', 'B'].index(answer_letter)
                selected_option = question['options'][idx]

        return {
            "answer_idx": idx,
            "answer_letter": answer_letter,
            "answer_text": selected_option,
            "confidence": confidence,
            "response": response,
        }


class SingleBothViews:
    def __init__(
        self,
        confidence: bool,
        model_name: str = "gpt-4o-mini",
        client_name: str = "openai",
        api_base: str = "https://api.openai.com/v1",
        max_completion_tokens: int = 1000,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.confidence = confidence
        self.model_name = model_name

        self.agent = BothViewsAgent(
            agent_id=1,
            model_name=model_name,
            client_name=client_name,
            api_base=api_base,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            enable_logging=enable_logging
        )

    def query_agent(self, images: List[str], question: dict, task_description: str, bbox: List[str] = None, map_image: str = None):
        response = self.agent.intialize_and_query_agent(
            images=images,
            question=question,
            task_description=task_description,
            bbox=bbox,
            confidence=self.confidence,
            map_image=map_image,
        )

        is_map = question.get('question_type') == 'map'

        if not is_map:
            match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", response.strip(), re.IGNORECASE)
        else:
            match = re.search(r"<ANSWER>\s*([A-B])\s*</ANSWER>", response.strip(), re.IGNORECASE)

        conf_match = None
        if self.confidence:
            conf_match = re.search(r"<CONFIDENCE>\s*([1-9]|10)\s*</CONFIDENCE>", response.strip(), re.IGNORECASE)
        if not match or (self.confidence and not conf_match):
            print(f"Parsing failure in answer response: {response}")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": response,
            }

        answer_letter = match.group(1).upper()
        confidence = conf_match.group(1) if self.confidence else None
        selected_option = None
        idx = None
        if not is_map:
            if question['options'] and answer_letter in ['A', 'B', 'C', 'D']:
                idx = ['A', 'B', 'C', 'D'].index(answer_letter)
                selected_option = question['options'][idx]
        else:
            if question['options'] and answer_letter in ['A', 'B']:
                idx = ['A', 'B'].index(answer_letter)
                selected_option = question['options'][idx]

        return {
            "answer_idx": idx,
            "answer_letter": answer_letter,
            "answer_text": selected_option,
            "confidence": confidence,
            "response": response,
        }


class SingleOneView:
    def __init__(
        self,
        confidence: bool,
        model_name: str = "gpt-4o-mini",
        client_name: str = "openai",
        api_base: str = "https://api.openai.com/v1",
        max_completion_tokens: int = 1000,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.confidence = confidence
        self.model_name = model_name

        self.agent = OneViewAgent(
            agent_id=1,
            model_name=model_name,
            client_name=client_name,
            api_base=api_base,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            enable_logging=enable_logging
        )

    def query_agent(self, images: List[str], question: dict, task_description: str, bbox: List[str] = None):
        response = self.agent.intialize_and_query_agent(
            images=images,
            question=question,
            task_description=task_description,
            bbox=bbox,
            confidence=self.confidence,
        )

        match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", response.strip(), re.IGNORECASE)

        conf_match = None
        if self.confidence:
            conf_match = re.search(r"<CONFIDENCE>\s*([1-9]|10)\s*</CONFIDENCE>", response.strip(), re.IGNORECASE)
        if not match or (self.confidence and not conf_match):
            print(f"Parsing failure in answer response: {response}")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": response,
            }

        answer_letter = match.group(1).upper()
        confidence = conf_match.group(1) if self.confidence else None
        selected_option = None
        idx = None
        if question['options'] and answer_letter in ['A', 'B', 'C', 'D']:
            idx = ['A', 'B', 'C', 'D'].index(answer_letter)
            selected_option = question['options'][idx]

        return {
            "answer_idx": idx,
            "answer_letter": answer_letter,
            "answer_text": selected_option,
            "confidence": confidence,
            "response": response,
        }


class SingleNoView:
    def __init__(
        self,
        confidence: bool,
        model_name: str = "gpt-4o-mini",
        client_name: str = "openai",
        api_base: str = "https://api.openai.com/v1",
        max_completion_tokens: int = 1000,
        temperature: float = 1.0,
        enable_logging: bool = True,
    ):

        self.confidence = confidence
        self.model_name = model_name

        self.agent = NoViewAgent(
            agent_id=1,
            model_name=model_name,
            client_name=client_name,
            api_base=api_base,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            enable_logging=enable_logging
        )

    def query_agent(self, question: dict, task_description: str):
        response = self.agent.intialize_and_query_agent(
            question=question,
            task_description=task_description,
            confidence=self.confidence,
        )

        retry_count = 0
        max_retries = 10
        while response is None and retry_count < max_retries:
            retry_count += 1
            print(f"Response is None in query_agent, retrying (attempt {retry_count}/{max_retries})...")
            response = self.agent.intialize_and_query_agent(
                question=question,
                task_description=task_description,
                confidence=self.confidence,
            )

        if response is None:
            print(f"Failed to get response after {max_retries} retries. Returning None response.")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": None,
            }

        match = re.search(r"<ANSWER>\s*([A-D])\s*</ANSWER>", response.strip(), re.IGNORECASE)

        conf_match = None
        if self.confidence:
            conf_match = re.search(r"<CONFIDENCE>\s*([1-9]|10)\s*</CONFIDENCE>", response.strip(), re.IGNORECASE)

        if not match or (self.confidence and not conf_match):
            print(f"Parsing failure in answer response: {response}")
            return {
                "answer_idx": None,
                "answer_letter": None,
                "answer_text": None,
                "confidence": None,
                "response": response,
            }

        answer_letter = match.group(1).upper()
        confidence = conf_match.group(1) if self.confidence else None

        selected_option = None
        idx = None
        if question['options'] and answer_letter in ['A', 'B', 'C', 'D']:
            idx = ['A', 'B', 'C', 'D'].index(answer_letter)
            selected_option = question['options'][idx]

        return {
            "answer_idx": idx,
            "answer_letter": answer_letter,
            "answer_text": selected_option,
            "confidence": confidence,
            "response": response,
        }
