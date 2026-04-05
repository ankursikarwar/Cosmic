import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Union, Any, Generator
import base64
import os

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    stop_after_delay,
    wait_fixed,
)
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

class OpenAIApiInference:
    """
    A unified interface for making inference calls to either OpenAI API or VLLM server.

    This class provides async batch processing capabilities with retry logic and rate limiting.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str = "NONE",
        model_name: str = "default",
        max_workers: int = 32,
        timeout: int = 1000,
    ):
        """
        Initialize the OpenAIApiInference client.

        Args:
            api_base: Base URL for the API (OpenAI or VLLM server)
            api_key: API key for authentication
            model_name: Model name to use for inference
            max_workers: Maximum number of concurrent requests
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.max_workers = max_workers
        self._semaphore = None
        self._semaphore_loop_id = None
        self.api_base = api_base

        # Initialize client with optional api_base parameter
        client_kwargs = {
            "base_url": api_base,
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 3,  # OpenAI client has built-in retries
        }

        self.client = AsyncOpenAI(**client_kwargs)

    def _get_semaphore(self):
        """Get or create semaphore in the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            # No running loop, create a new one (shouldn't happen in normal usage)
            current_loop = asyncio.get_event_loop()
            current_loop_id = id(current_loop)

        # Recreate semaphore if it doesn't exist or if we're in a different event loop
        if self._semaphore is None or self._semaphore_loop_id != current_loop_id:
            self._semaphore = asyncio.Semaphore(self.max_workers)
            self._semaphore_loop_id = current_loop_id

        return self._semaphore

    @retry(
        retry=retry_if_exception_type(
            exception_types=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
            )
        ),
        stop=stop_after_delay(2560),
        wait=wait_fixed(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def call_openai_api(self, query_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call the OpenAI API with retry logic."""
        semaphore = self._get_semaphore()
        async with semaphore:
            logger.debug(f"Making API call for query {query_id}")
            try:
                response = await self.client.chat.completions.create(model=self.model_name, **query)
                return {
                    "query_id": query_id,
                    "response": response,
                }
            except Exception as e:
                logger.warning(f"API call failed for query {query_id}: {str(e)}")
                # Try to update API base from vllm file
                if "https" not in self.api_base:
                    new_api_base = get_api_base_from_vllm_file()
                    logger.info(f"New API base: {new_api_base}")
                    if new_api_base:
                        logger.info(f"Updating API base to: {new_api_base}")
                        self.client = AsyncOpenAI(
                            base_url=new_api_base,
                            api_key=self.client.api_key,
                            timeout=self.client.timeout,
                            max_retries=3
                        )
                raise

    async def call_chat_async(
        self, queries: List[Dict[str, Any]], tqdm_desc: str = "Performing API calls", tqdm_enable: bool = True
    ) -> List[ChatCompletion]:
        """Process multiple queries concurrently with progress tracking."""
        tasks = []
        for i, query in enumerate(queries):
            task = asyncio.create_task(self.call_openai_api(query_id=i, query=query))
            tasks.append(task)

        results = {}
        for task in tqdm_asyncio.as_completed(tasks, desc=tqdm_desc, disable=not tqdm_enable):
            res = await task
            results[res["query_id"]] = res["response"]

        # Put results in order of queries
        results = [results[i] for i in range(len(queries))]
        return results

    def call_chat(
        self, queries: List[Dict[str, Any]], tqdm_desc: str = "Performing API calls", tqdm_enable: bool = True
    ) -> List[ChatCompletion]:
        """Synchronous wrapper for call_chat_async."""
        return asyncio.run(self.call_chat_async(queries, tqdm_desc, tqdm_enable))


def get_api_base_from_vllm_file(file_path: str = "./vllm_server_node.txt") -> Optional[str]:
    """Get API base from vllm_server_node.txt file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                server_name = f.read().strip()
                return f"http://{server_name}:4877/v1"
    except:
        pass
    return None

def create_openai_client(
    api_base: str = "https://api.openai.com/v1",
    api_key: str = "NONE",
    model_name: str = "gpt-5-mini"
) -> OpenAIApiInference:
    """Create an OpenAI client."""
    return OpenAIApiInference(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name
    )

def create_vllm_client(
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "NONE",
    model_name: str = "default"
) -> OpenAIApiInference:
    """Create a VLLM client."""
    return OpenAIApiInference(
        api_base=api_base,
        api_key="NONE",
        model_name=model_name
    )

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_system_prompt(prompt_type: str, question: str, agent_id: int, other_agent_id: int) -> str:
    if question is not None:
        question_type = question["question_type"]
        prompt_type = f"{prompt_type}_{question_type}"
    prompt_file = f"prompts/{prompt_type}_sys_prompt.txt"

    if not os.path.exists(prompt_file):
        raise ValueError(f"Prompt file {prompt_file} not found")

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        if other_agent_id is not None:
            return prompt_template.format(agent_id=agent_id, other_agent_id=other_agent_id)
        else:
            return prompt_template.format(agent_id=agent_id)

    except Exception as e:
        raise ValueError(f"Error loading prompt file {prompt_file}: {e}")

def load_two_agent_conv_prompt(
    agent_id: int,
    other_agent_id: int,
    message_from_other_agent: str = None,
    global_map: str = None,
    question: dict = None,
    num_turns: int = 5
) -> str:
    if agent_id == 2:
        assert message_from_other_agent is not None, "Message from other agent is required for agent 2"

    if global_map is not None:
        if question is not None:
            question_type = question["question_type"]
            prompt_file = f"prompts/two_agent_conv_agent{agent_id}_gmap_q_conditioning_{question_type}_prompt.txt"
        else:
            prompt_file = f"prompts/two_agent_conv_agent{agent_id}_gmap_prompt.txt"
    else:
        if question is not None:
            question_type = question["question_type"]
            if question_type in ["spatial", "perspective_taking"]:
                if question["question"] is not None:
                    prompt_file = f"prompts/two_agent_conv_agent{agent_id}_q_conditioning_{question_type}_prompt.txt"
                else:
                    prompt_file = f"prompts/two_agent_conv_agent{agent_id}_q_conditioning_{question_type}_prompt_helper.txt"
            else:
                prompt_file = f"prompts/two_agent_conv_agent{agent_id}_q_conditioning_{question_type}_prompt.txt"
        else:
            prompt_file = f"prompts/two_agent_conv_agent{agent_id}_prompt.txt"

    if not os.path.exists(prompt_file):
        raise ValueError(f"Prompt file {prompt_file} not found")

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()


        if agent_id == 1:
            if question is not None:
                question_text = question["question"]
                if question_text is not None:
                    num_options = len(question['options'])
                    if num_options == 4:
                        option_letters = ['a', 'b', 'c', 'd']
                    elif num_options == 2:
                        option_letters = ['a', 'b']
                    options = "\n".join([f"{letter}. {option}" for letter, option in zip(option_letters, question['options'])])
                    agent_prompt = prompt_template.format(agent_id=agent_id,
                                                    other_agent_id=other_agent_id,
                                                    question=question_text,
                                                    options=options,
                                                    num_turns=num_turns)
                else:
                    agent_prompt = prompt_template.format(agent_id=agent_id,
                                                    other_agent_id=other_agent_id,
                                                    num_turns=num_turns)
            else:
                agent_prompt = prompt_template.format(agent_id=agent_id,
                                                other_agent_id=other_agent_id)
        elif agent_id == 2:
            if question is not None:
                question_text = question["question"]
                if question_text is not None:
                    num_options = len(question['options'])
                    if num_options == 4:
                        option_letters = ['a', 'b', 'c', 'd']
                    elif num_options == 2:
                        option_letters = ['a', 'b']
                    options = "\n".join([f"{letter}. {option}" for letter, option in zip(option_letters, question['options'])])
                    agent_prompt = prompt_template.format(agent_id=agent_id,
                                                    other_agent_id=other_agent_id,
                                                    message_from_other_agent=message_from_other_agent,
                                                    question=question_text,
                                                    options=options,
                                                    num_turns=num_turns)
                else:
                    agent_prompt = prompt_template.format(agent_id=agent_id,
                                                    other_agent_id=other_agent_id,
                                                    message_from_other_agent=message_from_other_agent,
                                                    num_turns=num_turns)
            else:
                agent_prompt = prompt_template.format(agent_id=agent_id,
                                                other_agent_id=other_agent_id,
                                                message_from_other_agent=message_from_other_agent)
        else:
            raise ValueError(f"Invalid agent id: {agent_id}")

        return agent_prompt
    except Exception as e:
        raise ValueError(f"Error loading prompt file {prompt_file}: {e}")

def load_single_agent_qa_prompt(agent_id: int, global_map: str = None, question: dict = None) -> str:
    if global_map is not None:
        if question is not None:
            question_type = question["question_type"]
            prompt_file = f"prompts/single_agent_qa_agent{agent_id}_gmap_q_conditioning_{question_type}_prompt.txt"
        else:
            prompt_file = f"prompts/single_agent_qa_agent{agent_id}_gmap_prompt.txt"
    else:
        if question is not None:
            question_type = question["question_type"]
            prompt_file = f"prompts/single_agent_qa_agent{agent_id}_q_conditioning_{question_type}_prompt.txt"
        else:
            prompt_file = f"prompts/single_agent_qa_agent{agent_id}_prompt.txt"

    if not os.path.exists(prompt_file):
        raise ValueError(f"Prompt file {prompt_file} not found")

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        agent_prompt = prompt_template.format(agent_id=agent_id)
        return agent_prompt
    except Exception as e:
        raise ValueError(f"Error loading prompt file {prompt_file}: {e}")

def load_ask_question_prompt(agent_id: int, question, multiple_choice: List[str] = None) -> str:
    if multiple_choice:
        prompt_file = f"prompts/ask_question_agent_multiple_choice_prompt.txt"
    else:
        prompt_file = f"prompts/ask_question_agent_prompt.txt"

    if not os.path.exists(prompt_file):
        raise ValueError(f"Prompt file {prompt_file} not found")

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        if multiple_choice:
            agent_prompt = prompt_template.format(question=question, multiple_choice=multiple_choice)
        else:
            agent_prompt = prompt_template.format(question=question)

        return agent_prompt
    except Exception as e:
        raise ValueError(f"Error loading prompt file {prompt_file}: {e}")

def load_get_answer_q_conditioned_prompt() -> str:
    prompt_file = f"prompts/get_answer_q_conditioned_prompt.txt"
    if not os.path.exists(prompt_file):
        raise ValueError(f"Prompt file {prompt_file} not found")
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template
    except Exception as e:
        raise ValueError(f"Error loading prompt file {prompt_file}: {e}")

def parse_answer_idx(answer: str) -> int:
    try:
        if not isinstance(answer, str) or not answer.strip():
            print(f"Invalid input: {answer}")
            return None
        print(f"Answer: {answer}")

        if "<Answer>" not in answer or "</Answer>" not in answer:
            print(f"Missing required tags in: {answer}")
            return None

        parts_after_start = answer.split("<Answer>")
        if len(parts_after_start) < 2:
            print(f"Invalid format after <Answer> tag: {answer}")
            return None

        content_with_end = parts_after_start[1]

        parts_before_end = content_with_end.split("</Answer>")
        if len(parts_before_end) < 2:
            print(f"Invalid format before </Answer> tag: {answer}")
            return None

        # Extract the answer content
        answer_content = parts_before_end[0]

        # Check if content is empty or whitespace only
        if not answer_content.strip():
            print(f"Empty answer content: {answer}")
            return None

        # Clean and normalize the answer
        cleaned_answer = answer_content.strip().lower()

        # Map letters to indices
        letter_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        # Check if the answer is a valid letter
        if cleaned_answer in letter_to_idx:
            idx = letter_to_idx[cleaned_answer]
            print(f"Successfully parsed: '{cleaned_answer}' -> index {idx}")
            return idx
        else:
            print(f"Invalid answer letter: '{cleaned_answer}'. Expected a, b, c, or d")
            return None

    except Exception as e:
        print(f"Error parsing answer: {e}")
        return None
