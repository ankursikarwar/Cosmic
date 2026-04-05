import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Union, Any, Generator
import base64
import os

import openai
import requests
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
        max_workers: int = 8,  # Reduced from 32 to avoid vLLM multimodal cache issues
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
        self.api_base = api_base
        self.max_workers = max_workers
        self._semaphore = None
        self._semaphore_loop_id = None
        
        # Initialize client with optional api_base parameter
        # Note: We handle retries ourselves, so set client retries to 0
        # and use a reasonable timeout that works with our asyncio.wait_for wrapper
        client_kwargs = {
            "base_url": api_base,
            "api_key": api_key,
            "timeout": 100,  # 100s timeout per request (less than our 120s wrapper)
            "max_retries": 0,  # We handle retries ourselves with tenacity
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
    
    def _is_model_not_found_error(self, error: Exception) -> bool:
        """Check if the error is specifically about model not found."""
        error_str = str(error).lower()
        return (
            isinstance(error, openai.NotFoundError) and 
            ("model" in error_str and ("does not exist" in error_str or "not found" in error_str))
        )
    
    def _is_model_overloaded_error(self, error: Exception) -> bool:
        """Check if the error is about model being overloaded (503)."""
        error_str = str(error).lower()
        return (
            isinstance(error, openai.InternalServerError) and 
            ("overloaded" in error_str or "503" in error_str or "unavailable" in error_str)
        )
    
    def _is_prompt_too_long_error(self, error: Exception) -> bool:
        """Check if the error is about prompt being too long."""
        error_str = str(error).lower()
        return (
            isinstance(error, openai.BadRequestError) and 
            ("decoder prompt" in error_str and "longer than the maximum" in error_str)
        )
    
    def _is_invalid_content_error(self, error: Exception) -> bool:
        """Check if the error is about invalid message content (e.g., content is None)."""
        error_str = str(error).lower()
        return (
            isinstance(error, openai.BadRequestError) and 
            (("input should be a valid string" in error_str or "input should be iterable" in error_str) and
             ("'content'" in error_str or "'input': none" in error_str or "content': none" in error_str or
              "content', 'str'" in error_str or "'content', 'str'" in error_str))
        )
    
    def _is_max_tokens_too_large_error(self, error: Exception) -> bool:
        """Check if the error is about max_tokens or max_completion_tokens being too large."""
        error_str = str(error).lower()
        return (
            isinstance(error, openai.BadRequestError) and 
            ("max_tokens" in error_str or "max_completion_tokens" in error_str) and
            "too large" in error_str
        )
    

    @retry(
        retry=retry_if_exception_type(
            exception_types=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
                openai.BadRequestError,
            )
        ),
        # stop=stop_after_attempt(10),
        # wait=wait_exponential(multiplier=5),
        stop=stop_after_delay(2560),
        wait=wait_fixed(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def call_openai_api(self, query_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Call the OpenAI API with retry logic."""
        print(f"[Query {query_id}] Entering call_openai_api, waiting for semaphore...")
        semaphore = self._get_semaphore()
        async with semaphore:
            print(f"[Query {query_id}] Acquired semaphore, making API call to {self.api_base} with model {self.model_name}")
            logger.debug(f"Making API call for query {query_id} to {self.api_base} with model {self.model_name}")
            # Validate and filter out messages with None content
            if "messages" in query:
                original_messages = query["messages"]
                filtered_messages = []
                for i, msg in enumerate(original_messages):
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if content is None:
                            logger.warning(
                                f"Filtering out message {i} with None content in query {query_id}. "
                                f"Message: {msg}"
                            )
                            continue
                    filtered_messages.append(msg)
                query["messages"] = filtered_messages
                
                # If all messages were filtered out, return None response
                if not filtered_messages:
                    logger.error(
                        f"All messages filtered out (all had None content) for query {query_id}. "
                        f"Returning None response."
                    )
                    return {
                        "query_id": query_id,
                        "response": None,
                    }
            
            try:
                print(f"[Query {query_id}] Calling client.chat.completions.create...")
                # Add an additional timeout wrapper to prevent indefinite hanging
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(model=self.model_name, **query),
                        timeout=120  # 2 minute hard timeout for the API call
                    )
                    print(f"[Query {query_id}] ✓ Received successful response")
                    return {
                        "query_id": query_id,
                        "response": response,
                    }
                except asyncio.TimeoutError:
                    print(f"[Query {query_id}] ✗ Hard timeout after 120s - API call took too long")
                    raise openai.APITimeoutError("Request exceeded 120 second hard timeout")
            except openai.NotFoundError as e:
                # Check if it's a model not found error
                if self._is_model_not_found_error(e):
                    print(f"[Query {query_id}] ✗ NotFoundError: Model '{self.model_name}' not found on {self.api_base}")
                    logger.error(
                        f"Model '{self.model_name}' not found on server {self.api_base}. "
                        f"Error: {str(e)}. Will retry a few times on the same server before switching."
                    )
                    # First, retry a few times on the same server
                    same_server_retries = 3
                    for same_server_attempt in range(1, same_server_retries + 1):
                        try:
                            logger.info(f"Retry {same_server_attempt}/{same_server_retries} on same server {self.api_base}")
                            response = await asyncio.wait_for(
                                self.client.chat.completions.create(model=self.model_name, **query),
                                timeout=120
                            )
                            logger.info(f"Successfully made API call on {self.api_base} after {same_server_attempt} retry(ies)")
                            return {
                                "query_id": query_id,
                                "response": response,
                            }
                        except asyncio.TimeoutError:
                            print(f"[Query {query_id}] ✗ Retry {same_server_attempt} timed out after 120s")
                            if same_server_attempt < same_server_retries:
                                await asyncio.sleep(2)
                                continue
                            else:
                                logger.warning(f"All {same_server_retries} retries timed out. Will switch servers.")
                                break
                        except openai.NotFoundError as retry_error:
                            if self._is_model_not_found_error(retry_error):
                                if same_server_attempt < same_server_retries:
                                    logger.warning(
                                        f"Retry {same_server_attempt}/{same_server_retries}: Model still not found on {self.api_base}. "
                                        f"Retrying on same server..."
                                    )
                                    await asyncio.sleep(2)  # Wait before next retry
                                    continue
                                else:
                                    logger.warning(
                                        f"All {same_server_retries} retries on {self.api_base} failed. "
                                        f"Will now switch servers from file."
                                    )
                                    break  # Exit loop to start switching servers
                            else:
                                # Other NotFoundError, re-raise
                                raise
                        except Exception as retry_error:
                            # Other error, log and continue retrying
                            if same_server_attempt < same_server_retries:
                                logger.warning(
                                    f"Retry {same_server_attempt}/{same_server_retries}: Error on {self.api_base}: {str(retry_error)}. "
                                    f"Retrying on same server..."
                                )
                                await asyncio.sleep(2)
                                continue
                            else:
                                logger.warning(
                                    f"All {same_server_retries} retries on {self.api_base} failed. "
                                    f"Will now switch servers from file."
                                )
                                break
                    
                    # After exhausting same-server retries, start switching servers
                    retry_count = 0
                    while True:
                        retry_count += 1
                        new_api_base = get_api_base_from_vllm_file(
                            self.model_name
                        )
                        if new_api_base:
                            # Switch to new server if it's different from current
                            if new_api_base != self.api_base:
                                logger.info(f"Attempt {retry_count}: Switching to new API base: {new_api_base}")
                                self.api_base = new_api_base
                                self.client = AsyncOpenAI(
                                    base_url=new_api_base,
                                    api_key=self.client.api_key,
                                    timeout=self.client.timeout,
                                    max_retries=3
                                )
                            
                            # Try the API call with the current server
                            try:
                                response = await asyncio.wait_for(
                                    self.client.chat.completions.create(model=self.model_name, **query),
                                    timeout=120
                                )
                                logger.info(f"Successfully made API call after switching to {self.api_base} (attempt {retry_count})")
                                return {
                                    "query_id": query_id,
                                    "response": response,
                                }
                            except asyncio.TimeoutError:
                                print(f"[Query {query_id}] ✗ Server switch attempt {retry_count} timed out after 120s")
                                logger.warning(f"Attempt {retry_count}: Timeout on {self.api_base}. Will retry...")
                                await asyncio.sleep(2)
                                continue
                            except openai.NotFoundError as retry_error:
                                if self._is_model_not_found_error(retry_error):
                                    logger.warning(
                                        f"Attempt {retry_count}: Model still not found on {self.api_base}. "
                                        f"Will continue retrying by checking file for new server..."
                                    )
                                    await asyncio.sleep(2)  # Wait before next retry to allow file to update
                                    continue
                                else:
                                    # Other NotFoundError, re-raise
                                    raise
                            except Exception as retry_error:
                                # Other error, log and continue retrying
                                logger.warning(
                                    f"Attempt {retry_count}: Error on {self.api_base}: {str(retry_error)}. "
                                    f"Will continue retrying..."
                                )
                                await asyncio.sleep(2)
                                continue
                        else:
                            # No API base from file, wait and retry
                            logger.warning(f"Attempt {retry_count}: No API base found in file. Waiting and retrying...")
                            await asyncio.sleep(5)
                            continue
                else:
                    # Other NotFoundError (e.g., endpoint not found) - retry
                    logger.warning(f"API call failed for query {query_id}: {str(e)}")
                    raise
            except openai.InternalServerError as e:
                # Check if it's a model overloaded error (503)
                if self._is_model_overloaded_error(e):
                    print(f"[Query {query_id}] ✗ InternalServerError: Model '{self.model_name}' is overloaded (503) on {self.api_base}")
                    logger.error(
                        f"Model '{self.model_name}' is overloaded on server {self.api_base}. "
                        f"Error: {str(e)}. Will keep retrying on the same server."
                    )
                    # Keep retrying on the same server indefinitely
                    retry_count = 0
                    while True:
                        retry_count += 1
                        try:
                            # Exponential backoff: 5s, 10s, 15s, 20s, 25s, then cap at 30s
                            wait_time = min(5 * retry_count, 30)
                            print(f"[Query {query_id}] Waiting {wait_time}s before retry {retry_count} (overloaded)...")
                            logger.info(f"Retry {retry_count} on same server {self.api_base} (overloaded), waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            print(f"[Query {query_id}] Retrying request (attempt {retry_count})...")
                            response = await asyncio.wait_for(
                                self.client.chat.completions.create(model=self.model_name, **query),
                                timeout=120
                            )
                            print(f"[Query {query_id}] ✓ Success after {retry_count} retries (was overloaded)")
                            logger.info(f"Successfully made API call on {self.api_base} after {retry_count} retry(ies) (overloaded)")
                            return {
                                "query_id": query_id,
                                "response": response,
                            }
                        except asyncio.TimeoutError:
                            print(f"[Query {query_id}] ✗ Overload retry {retry_count} timed out after 120s")
                            logger.warning(f"Retry {retry_count}: Timeout on {self.api_base} (overloaded). Will continue retrying...")
                            continue
                        except openai.InternalServerError as retry_error:
                            if self._is_model_overloaded_error(retry_error):
                                print(f"[Query {query_id}] ✗ Still overloaded after retry {retry_count}")
                                logger.warning(
                                    f"Retry {retry_count}: Model still overloaded on {self.api_base}. "
                                    f"Will continue retrying on same server..."
                                )
                                continue
                            else:
                                # Other InternalServerError, re-raise to let retry decorator handle it
                                print(f"[Query {query_id}] ✗ Different InternalServerError: {str(retry_error)}")
                                raise
                        except Exception as retry_error:
                            # Other error, log and continue retrying
                            print(f"[Query {query_id}] ✗ Error on retry {retry_count}: {type(retry_error).__name__}: {str(retry_error)}")
                            logger.warning(
                                f"Retry {retry_count}: Error on {self.api_base}: {str(retry_error)}. "
                                f"Will continue retrying on same server..."
                            )
                            continue
                else:
                    # Other InternalServerError - let retry decorator handle it
                    print(f"[Query {query_id}] ✗ InternalServerError (not overloaded): {str(e)}")
                    raise
            except openai.BadRequestError as e:
                # Check if it's a prompt too long error
                if self._is_prompt_too_long_error(e):
                    print(f"[Query {query_id}] ✗ BadRequestError: Prompt too long")
                    logger.error(
                        f"Prompt too long for query {query_id} on {self.api_base} with model {self.model_name}. "
                        f"Error: {str(e)}. Returning None response."
                    )
                    # Return None response instead of retrying (input problem, not transient)
                    return {
                        "query_id": query_id,
                        "response": None,
                    }
                # Check if it's an invalid content error (e.g., content is None)
                elif self._is_invalid_content_error(e):
                    print(f"[Query {query_id}] ✗ BadRequestError: Invalid message content")
                    logger.error(
                        f"Invalid message content (None) for query {query_id} on {self.api_base} with model {self.model_name}. "
                        f"Error: {str(e)}. Returning None response."
                    )
                    # Return None response instead of retrying (input problem, not transient)
                    return {
                        "query_id": query_id,
                        "response": None,
                    }
                # Check if it's a max_tokens too large error
                elif self._is_max_tokens_too_large_error(e):
                    print(f"[Query {query_id}] ✗ BadRequestError: Max tokens too large")
                    logger.error(
                        f"Max tokens/completion tokens too large for query {query_id} on {self.api_base} with model {self.model_name}. "
                        f"Error: {str(e)}. Returning None response."
                    )
                    # Return None response instead of retrying (input problem, not transient)
                    return {
                        "query_id": query_id,
                        "response": None,
                    }
                else:
                    # Other BadRequestError - let retry decorator handle it
                    print(f"[Query {query_id}] ✗ BadRequestError: {str(e)}")
                    raise
            except openai.APITimeoutError as e:
                print(f"[Query {query_id}] ✗ APITimeoutError: Request timed out after waiting. Error: {str(e)}")
                raise
            except openai.RateLimitError as e:
                print(f"[Query {query_id}] ✗ RateLimitError: Rate limit exceeded. Error: {str(e)}")
                raise
            except openai.APIConnectionError as e:
                print(f"[Query {query_id}] ✗ APIConnectionError: Connection failed. Error: {str(e)}")
                raise
            except Exception as e:
                print(f"[Query {query_id}] ✗ Unexpected error: {type(e).__name__}: {str(e)}")
                logger.warning(f"API call failed for query {query_id} on {self.api_base}: {str(e)} for {self.model_name}")
                # Try to update API base from vllm file for other errors
                new_api_base = get_api_base_from_vllm_file(self.model_name)
                if new_api_base and new_api_base != self.api_base:
                    print(f"[Query {query_id}] Switching to new API base: {new_api_base}")
                    logger.info(f"Switching to new API base: {new_api_base}")
                    self.api_base = new_api_base
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


def get_api_base_from_vllm_file(model_name: str, file_path: str = "./vllm_server_node.txt") -> Optional[str]:
    """Get API base from vllm_server_node.txt file."""
    model_file_path = f"./vllm_server_node_{model_name}.txt"
    try:
        if os.path.exists(model_file_path):
            with open(model_file_path, 'r') as f:
                server_name = f.read().strip()
                return f"http://{server_name}:4877/v1"
    except:
        pass
    return None

def get_api_base_from_github(repo_url: str = "https://github.com/username/api-base") -> Optional[str]:
    """
    Get the API base URL from the api.txt file in the GitHub repository.

    Args:
        repo_url (str): The GitHub repository URL (e.g., 'https://github.com/username/repo' or 'git@github.com:username/repo.git')

    Returns:
        Optional[str]: The API base URL from api.txt or None if not found
    """
    # Handle SSH format
    if repo_url.startswith("git@github.com:"):
        repo_url = repo_url.replace("git@github.com:", "https://github.com/")

    # Handle HTTPS format
    if not repo_url.startswith("https://github.com/"):
        return None

    # Remove .git suffix if present
    repo_url = repo_url.rstrip(".git")

    # Extract username and repo name
    match = re.match(r"https://github.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return None

    username, repo = match.groups()

    # Construct raw content URL for api.txt
    raw_url = f"https://raw.githubusercontent.com/{username}/{repo}/main/api.txt"

    try:
        response = requests.get(raw_url)
        if response.status_code == 200:
            api_base = response.text.strip()
            print(f"API base: {api_base}")
            api_base = api_base.strip()
            return api_base
        return None
    except Exception:
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

def create_client_from_github(
    repo_url: str = "https://github.com/username/api-base",
    api_key: str = "NONE",
    model_name: str = "default"
) -> Optional[OpenAIApiInference]:
    """Create a client using API base from GitHub repository."""
    api_base = get_api_base_from_github(repo_url)
    if api_base:
        return OpenAIApiInference(
            api_base=api_base,
            api_key=api_key,
            model_name=model_name
        )
    return None

def encode_image(image):
    """Accept a file path (str) or a HuggingFace image dict {'bytes': b'...'}."""
    if isinstance(image, dict) and "bytes" in image:
        return base64.b64encode(image["bytes"]).decode("utf-8")
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_pil(image, max_size: int = 512):
    """Convert a file path or HF image dict to a resized PIL Image (for wandb logging)."""
    from PIL import Image as PILImage
    import io
    if isinstance(image, dict) and "bytes" in image:
        img = PILImage.open(io.BytesIO(image["bytes"]))
    else:
        img = PILImage.open(image)
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    return img

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
            print(f"❌ Invalid input: {answer}")
            return None        
        print(f"Answer: {answer}")
        
        if "<Answer>" not in answer or "</Answer>" not in answer:
            print(f"❌ Missing required tags in: {answer}")
            return None
        
        parts_after_start = answer.split("<Answer>")
        if len(parts_after_start) < 2:
            print(f"❌ Invalid format after <Answer> tag: {answer}")
            return None
        
        content_with_end = parts_after_start[1]
        
        parts_before_end = content_with_end.split("</Answer>")
        if len(parts_before_end) < 2:
            print(f"❌ Invalid format before </Answer> tag: {answer}")
            return None
        
        # Extract the answer content
        answer_content = parts_before_end[0]
        
        # Check if content is empty or whitespace only
        if not answer_content.strip():
            print(f"❌ Empty answer content: {answer}")
            return None
        
        # Clean and normalize the answer
        cleaned_answer = answer_content.strip().lower()
        
        # Map letters to indices
        letter_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        
        # Check if the answer is a valid letter
        if cleaned_answer in letter_to_idx:
            idx = letter_to_idx[cleaned_answer]
            print(f"✅ Successfully parsed: '{cleaned_answer}' -> index {idx}")
            return idx
        else:
            print(f"❌ Invalid answer letter: '{cleaned_answer}'. Expected a, b, c, or d")
            return None
        
    except Exception as e:
        print(f"❌ Error parsing answer: {e}")
        return None