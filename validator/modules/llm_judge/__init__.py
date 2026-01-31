import os
import re
import random
import json
import httpx
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from loguru import logger
from huggingface_hub import HfApi
from typing import List, Dict, Any, Generator
from validator.modules.llm_judge.prompt import get_prompt
from validator.modules.llm_judge.utils import download_file
from validator.exceptions import LLMJudgeException, InvalidModelParametersException
from validator.modules.llm_judge.template import template_dict
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from validator.modules.base import (
    BaseValidationModule,
    BaseConfig,
    BaseInputData,
    BaseMetrics,
)

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

api = HfApi()
LOWEST_POSSIBLE_SCORE = -999


class LLMJudgeConfig(BaseConfig):
    gen_batch_size: int = 1
    eval_batch_size: int = 16
    gen_temperature: float = 0.1


class LLMJudgeMetrics(BaseMetrics):
    score: float  # LLM output score as the main metric


class LLMJudgeInputData(BaseInputData):
    """Input data for LLMJudge"""
    hg_repo_id: str
    revision: str
    context_length: int
    max_params: int
    validation_set_url: str
    base_model: str
    eval_args: dict


class LLMJudgeValidationModule(BaseValidationModule):
    config_schema = LLMJudgeConfig
    metrics_schema = LLMJudgeMetrics
    input_data_schema = LLMJudgeInputData
    task_type = "llm_evaluation"

    def __init__(self, config: LLMJudgeConfig, skip_llm_eval: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.client = None
        self.model_clients = {}  # model_name -> OpenAI client
        self.model_name_map = {}  # model_name -> actual API model name
        self.available_models = []
        self.hf_model = None
        self.hf_tokenizer = None
        self.skip_llm_eval = skip_llm_eval

        if not skip_llm_eval:
            # Initialize clients (default + per-model overrides)
            self._initialize_client()
            self._initialize_model_clients()
            self._fetch_available_models()

    def _initialize_client(self):
        """Initialize the default OpenAI client from OPENAI_BASE_URL / OPENAI_API_KEY."""
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url or not api_key:
            logger.warning("Default OPENAI_BASE_URL/OPENAI_API_KEY not set, skipping default client")
            return
        try:
            http_client = httpx.Client(
                headers={"Authorization": f"Bearer {api_key}"},
            )

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
            )

        except Exception as e:
            raise LLMJudgeException(f"Failed to initialize default OpenAI client: {e}") from e

    def _create_openai_client(self, base_url: str, api_key: str) -> OpenAI:
        """Create an OpenAI client for a specific endpoint."""
        http_client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def _initialize_model_clients(self):
        """
        Initialize per-model OpenAI clients from environment variables.

        Env var pattern (prefix-based):
            <PREFIX>_BASE_URL=https://api.example.com/v1
            <PREFIX>_API_KEY=sk-xxx
            <PREFIX>_MODELS=model-a,model-b
            <PREFIX>_MODEL_MAP=model-a:actual-api-name  (optional)

        Supported prefixes: KIMI, DEEPSEEK, MINIMAX (and any custom prefix).
        The module scans for all *_MODELS env vars and creates a client for
        each model listed, mapping it in self.model_clients.

        If the platform uses a different model name than the one used in
        eval_model_list, set <PREFIX>_MODEL_MAP to remap. Format is
        comma-separated pairs of "local_name:api_name".
        Example: DEEPSEEK_MODEL_MAP=deepseek-v3.2:deepseek-chat
        """
        # Scan env vars for *_MODELS pattern
        model_env_prefixes = set()
        for key in os.environ:
            if key.endswith("_MODELS") and key != "MODELS":
                prefix = key[: -len("_MODELS")]
                model_env_prefixes.add(prefix)

        for prefix in sorted(model_env_prefixes):
            base_url = os.getenv(f"{prefix}_BASE_URL")
            api_key = os.getenv(f"{prefix}_API_KEY")
            models_str = os.getenv(f"{prefix}_MODELS", "")

            if not base_url or not api_key:
                logger.warning(f"Skipping {prefix}: missing {prefix}_BASE_URL or {prefix}_API_KEY")
                continue

            models = [m.strip() for m in models_str.split(",") if m.strip()]
            if not models:
                continue

            # Parse optional model name mapping
            model_map_str = os.getenv(f"{prefix}_MODEL_MAP", "")
            if model_map_str:
                for mapping in model_map_str.split(","):
                    mapping = mapping.strip()
                    if ":" in mapping:
                        local_name, api_name = mapping.split(":", 1)
                        self.model_name_map[local_name.strip()] = api_name.strip()
                        logger.info(f"Model name mapping: '{local_name.strip()}' -> '{api_name.strip()}'")

            try:
                client = self._create_openai_client(base_url, api_key)
                for model_name in models:
                    self.model_clients[model_name] = client
                    logger.info(f"Registered model '{model_name}' -> {base_url} (prefix: {prefix})")
            except Exception as e:
                logger.error(f"Failed to create client for {prefix}: {e}")

    def _get_client_for_model(self, model_name: str) -> OpenAI:
        """Return the OpenAI client for a given model, falling back to the default client."""
        if model_name in self.model_clients:
            return self.model_clients[model_name]
        if self.client is not None:
            return self.client
        raise LLMJudgeException(
            f"No API client available for model '{model_name}'. "
            f"Set {model_name.upper().replace('-', '_')}_BASE_URL / _API_KEY / _MODELS in .env, "
            f"or set the default OPENAI_BASE_URL / OPENAI_API_KEY."
        )

    def _fetch_available_models(self):
        # Start with models that have dedicated clients
        self.available_models = list(self.model_clients.keys())

        # Also fetch models from the default client
        if self.client is not None:
            try:
                models_response = self.client.models.list()
                default_models = [model.id for model in models_response.data]
                # Merge, avoiding duplicates
                for m in default_models:
                    if m not in self.available_models:
                        self.available_models.append(m)
            except Exception as e:
                logger.error(
                    f"Warning: Failed to fetch models from default API ({e})"
                )

        if not self.available_models:
            logger.warning("No evaluation models available")
            self.available_models = ["gpt-4o"]

    def _download_lora_config(self, repo_id: str, revision: str) -> bool:
        try:
            api.hf_hub_download(
                repo_id=repo_id,
                filename="adapter_config.json",
                local_dir="judge",
                revision=revision,
            )
        except Exception as e:
            if "adapter_config.json" in str(e):
                logger.error("No adapter_config.json found in the repo, assuming full model")
                return False
            else:
                raise  # Re-raise the exception if it's not related to the missing file
        return True

    def _load_model(self, repo_id: str, revision: str = "main", max_params: int = None):

        is_lora = self._download_lora_config(repo_id, revision=revision)
        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_cache=False,
            device_map="auto",
        )
        if is_lora:
            api.snapshot_download(
                repo_id=repo_id,
                local_dir="judge",
                revision=revision,
            )
            with open("judge/adapter_config.json", "r") as f:
                adapter_config = json.load(f)
            base_model = adapter_config["base_model_name_or_path"]
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True, use_fast=True
            )
            base_hf_model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            hf_model = PeftModel.from_pretrained(
                base_hf_model,
                "judge",
                device_map="auto",
            )
            self.hf_model = hf_model.merge_and_unload()
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                repo_id, trust_remote_code=True, use_fast=True
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        total = sum(p.numel() for p in self.hf_model.parameters())
        if total > max_params:
            logger.info(
                f"Total model params: {total} exceeds the limit {max_params}, submitting validation result with a large loss"
            )
            raise InvalidModelParametersException(f"Model parameters {total} exceed limit {max_params}")

    def _load_local_model(self, model_path: str, is_lora: bool = False, max_params: int = None, base_model_path: str = None):
        """Load a model from a local directory path (no HuggingFace download).

        Args:
            model_path: Path to the model or LoRA adapter directory.
            is_lora: Whether the model is a LoRA adapter.
            max_params: Maximum allowed parameters (skip check if None).
            base_model_path: Local path to the base model for LoRA. If not provided,
                uses base_model_name_or_path from adapter_config.json.
        """
        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_cache=False,
            device_map="auto",
        )
        if is_lora:
            if base_model_path:
                base_model = base_model_path
            else:
                adapter_config_path = os.path.join(model_path, "adapter_config.json")
                if not os.path.exists(adapter_config_path):
                    raise LLMJudgeException(f"adapter_config.json not found at {adapter_config_path}")
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                base_model = adapter_config["base_model_name_or_path"]
            logger.info(f"Loading LoRA adapter from {model_path}, base model: {base_model}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True, use_fast=True
            )
            base_hf_model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            hf_model = PeftModel.from_pretrained(
                base_hf_model, model_path, device_map="auto"
            )
            self.hf_model = hf_model.merge_and_unload()
        else:
            logger.info(f"Loading model from local path: {model_path}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        total = sum(p.numel() for p in self.hf_model.parameters())
        logger.info(f"Model loaded. Total parameters: {total:,}")
        if max_params is not None and total > max_params:
            raise InvalidModelParametersException(
                f"Model parameters {total} exceed limit {max_params}"
            )

    def _construct_conversation_template(
            self, conversation: List[Dict[str, str]], base_model: str,
    ) -> str:
        try:
            if base_model not in template_dict:
                logger.info(f"Template {base_model} not found, using default")
                base_model = "default"

            template = template_dict[base_model]

            conversation_parts = []

            # Use provided system_text or fall back to template default
            if template.system_format:
                system_prompt = (
                    conversation["system"] if "system" in conversation else None
                )
                system_content = (
                    system_prompt if system_prompt else "You are a helpful assistant."
                )
                if system_content:
                    formatted_system = template.system_format.format(
                        content=system_content
                    )
                    conversation_parts.append(formatted_system)

                # Multi-turn conversation: format each message according to template
                for msg in conversation["conversations"]:
                    if msg["role"] == "user":
                        user_text = template.user_format.format(
                            content=msg["content"],
                            stop_token=self.hf_tokenizer.eos_token,
                        )
                        conversation_parts.append(user_text)
                    elif msg["role"] == "assistant":
                        assistant_text = template.assistant_format.format(
                            content=msg["content"],
                            stop_token=self.hf_tokenizer.eos_token,
                        )
                        conversation_parts.append(assistant_text)

            conversation_format = "".join(conversation_parts)
        except Exception as e:
            raise LLMJudgeException(
                f"Failed to construct conversation template: {e}"
            ) from e

        return conversation_format

    def _generate_response(
            self,
            context_length: int,
            user_input: list = list[
                list[dict[str, str]]
            ],  # list of conversations, each conversation is a list of messages
            base_model: str = "default",
            max_length: int = 2048,
            batch_size: int = 1,
            eval_args: dict = None,

    ) -> list:
        if self.hf_model is None or self.hf_tokenizer is None:
            raise LLMJudgeException("HuggingFace model not loaded")

        try:
            results = []
            total_batches = (len(user_input) + batch_size - 1) // batch_size
            for batch_idx, i in enumerate(range(0, len(user_input), batch_size), 1):
                batch_conversations = user_input[i: i + batch_size]

                # Apply chat template with fallback
                batch_conversation_templates = []
                for conversation in batch_conversations:
                    template = self._construct_conversation_template(
                        conversation, base_model=base_model,
                    )

                    batch_conversation_templates.append(template)

                # Tokenization with padding for batch processing
                model_inputs = self.hf_tokenizer(
                    batch_conversation_templates,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                )

                # Move to device if available
                if (
                        torch.cuda.is_available()
                        and next(self.hf_model.parameters()).is_cuda
                ):
                    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

                logger.info(f"Generating batch {batch_idx}/{total_batches} ({min(i + batch_size, len(user_input))}/{len(user_input)} conversations)...")
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **model_inputs,
                        max_new_tokens=max_length,
                        temperature=self.config.gen_temperature,
                        do_sample=True,
                        pad_token_id=self.hf_tokenizer.eos_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                    )

                # Decode responses
                for j, output in enumerate(outputs):
                    # Get the input length for this specific sequence
                    input_length = model_inputs["input_ids"][j].shape[0]

                    # Extract only the newly generated tokens
                    generated_ids = output[input_length:]
                    assistant_response = self.hf_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    results.append(assistant_response)
                    # print("assistant_response:", assistant_response)

            logger.info(f"Completed generating all {len(user_input)} conversations")
            return results

        except Exception as e:
            raise LLMJudgeException(f"Failed to generate response: {e}") from e

    def _generate_response_batched(
            self,
            context_length: int,
            user_input: list,
            base_model: str = "default",
            max_length: int = 2048,
            batch_size: int = 1,
            eval_args: dict = None,
    ) -> Generator[tuple, None, None]:
        """Generate responses in batches, yielding (start_index, batch_responses) per batch.

        This allows the caller to evaluate each batch immediately after generation,
        enabling early detection of API configuration issues.
        """
        if self.hf_model is None or self.hf_tokenizer is None:
            raise LLMJudgeException("HuggingFace model not loaded")

        try:
            total_batches = (len(user_input) + batch_size - 1) // batch_size
            for batch_idx, i in enumerate(range(0, len(user_input), batch_size), 1):
                batch_conversations = user_input[i: i + batch_size]

                # Apply chat template with fallback
                batch_conversation_templates = []
                for conversation in batch_conversations:
                    template = self._construct_conversation_template(
                        conversation, base_model=base_model,
                    )
                    batch_conversation_templates.append(template)

                # Tokenization with padding for batch processing
                model_inputs = self.hf_tokenizer(
                    batch_conversation_templates,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                )

                # Move to device if available
                if (
                        torch.cuda.is_available()
                        and next(self.hf_model.parameters()).is_cuda
                ):
                    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

                logger.info(f"Generating batch {batch_idx}/{total_batches} ({min(i + batch_size, len(user_input))}/{len(user_input)} conversations)...")
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **model_inputs,
                        max_new_tokens=max_length,
                        temperature=self.config.gen_temperature,
                        do_sample=True,
                        pad_token_id=self.hf_tokenizer.eos_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                    )

                # Decode responses
                batch_results = []
                for j, output in enumerate(outputs):
                    input_length = model_inputs["input_ids"][j].shape[0]
                    generated_ids = output[input_length:]
                    assistant_response = self.hf_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()
                    batch_results.append(assistant_response)

                yield (i, batch_results)

            logger.info(f"Completed generating all {len(user_input)} conversations")

        except Exception as e:
            raise LLMJudgeException(f"Failed to generate response: {e}") from e

    def _select_eval_model(self, eval_args: dict) -> str:
        """
        Select evaluation model based on eval_args configuration
        """
        eval_model_list = eval_args.get("eval_model_list", [])

        if eval_model_list:
            # Check if all models in eval_model_list are available
            available_eval_models = [
                model for model in eval_model_list if model in self.available_models
            ]

            if len(available_eval_models) == len(eval_model_list):
                selected_model = random.choice(eval_model_list)
                logger.info(f"Using eval_model_list: selected {selected_model}")
                return selected_model

        # random selection from available models
        selected_model = random.choice(self.available_models)
        return selected_model

    def _normalize_score(
            self, score: float, min_score: float = 0, max_score: float = 10.0
    ) -> float:
        """
        Normalize score to (0, 1) range

        Args:
            score: Original score
            min_score: Minimum possible score (default: 0.0)
            max_score: Maximum possible score (default: 10.0)

        Returns:
            Normalized score in (0, 1) range
        """

        # Normalize to [0, 1] range
        normalized = (score - min_score) / (max_score - min_score)

        # epsilon = 1e-8
        # normalized = max(epsilon, min(1.0 - epsilon, normalized))

        return normalized

    # Models whose APIs do not accept the "system" role
    _NO_SYSTEM_ROLE_MODELS = {"minimax-m2.1"}

    def _adapt_messages_for_model(
            self, model_name: str, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Adapt messages for models that don't support the 'system' role.
        Merges system messages into the first user message as a prefix.
        """
        if model_name not in self._NO_SYSTEM_ROLE_MODELS:
            return messages

        system_parts = []
        other_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                other_messages.append(msg)

        if not system_parts:
            return messages

        # Prepend system content to the first user message
        system_text = "\n".join(system_parts)
        adapted = []
        prepended = False
        for msg in other_messages:
            if msg["role"] == "user" and not prepended:
                adapted.append({
                    "role": "user",
                    "content": f"[System instruction]: {system_text}\n\n{msg['content']}",
                })
                prepended = True
            else:
                adapted.append(msg)

        # If no user message found, add system as user message
        if not prepended:
            adapted.insert(0, {"role": "user", "content": system_text})

        return adapted

    def _call_gpt(
            self, messages: List[Dict[str, str]], eval_args: dict
    ) -> tuple[str, str]:
        """
        Call GPT API with model and temperature from eval_args, with exponential backoff retry.

        Args:
            messages: Chat messages
            eval_args: Evaluation arguments containing model and temperature config

        Returns:
            Tuple of (API response content, selected model name)
        """
        # Check if a specific model is requested
        if "selected_model" in eval_args:
            selected_model = eval_args["selected_model"]
        else:
            selected_model = self._select_eval_model(eval_args)
        temperature = eval_args.get("temperature", 0.1)  # Default eval temperature

        # Patch: kimi-k2.5 requires temperature=1
        if selected_model == "kimi-k2.5":
            temperature = 1

        # Resolve the actual API model name (may differ from eval_model_list name)
        api_model_name = self.model_name_map.get(selected_model, selected_model)

        # Some APIs (e.g. MiniMax) don't support "system" role.
        # Merge system messages into the first user message.
        adapted_messages = self._adapt_messages_for_model(selected_model, messages)

        # Safety check: ensure no system role for models that don't support it
        if selected_model in self._NO_SYSTEM_ROLE_MODELS:
            has_system = any(m["role"] == "system" for m in adapted_messages)
            if has_system:
                logger.warning(
                    f"System role still present after adaptation for {selected_model}, force-removing"
                )
                adapted_messages = [m for m in adapted_messages if m["role"] != "system"]

        logger.debug(
            f"API call to '{selected_model}' (api_name='{api_model_name}'): "
            f"roles={[m['role'] for m in adapted_messages]}"
        )

        params = {
            "model": api_model_name,
            "messages": adapted_messages,
            "temperature": temperature,
            "seed": random.randint(0, 10000),
        }

        def log_retry(retry_state):
            logger.warning(
                f"API call failed (attempt {retry_state.attempt_number}/3), "
                f"retrying in {retry_state.next_action.sleep:.1f}s: {retry_state.outcome.exception()}"
            )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            before_sleep=log_retry,
            reraise=True,
        )
        def _make_api_call():
            client = self._get_client_for_model(selected_model)
            completion = client.chat.completions.create(**params)
            return completion.choices[0].message.content

        try:
            content = _make_api_call()
            return content, selected_model
        except Exception as e:
            raise LLMJudgeException(
                f"API call failed with model {selected_model} after retries: {e}"
            )

    def _parse_jsonl_conversations(self, test_file: str) -> List[Dict[str, Any]]:
        """Parse JSONL file into conversation structures without generating responses."""
        input_conversations = []

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]

        for line_num, json_data in enumerate(test_data):

            try:
                # Extract system information if available
                system_text = json_data.get("system", None)
                input_conversations_data = {
                    "system": system_text,
                    "conversations": [],
                }

                # Extract conversation history for multi-turn support
                conversation_to_process = []
                reference_response = None

                if "conversations" in json_data:
                    conversations = json_data["conversations"]
                    if isinstance(conversations, list) and conversations:
                        # Filter valid messages
                        for msg in conversations:
                            role = msg.get("role", "")
                            content = msg.get("content", "").strip()
                            if role in ["user", "assistant"] and content:
                                conversation_to_process.append(
                                    {"role": role, "content": content}
                                )

                        # Extract reference response (last assistant message) if it exists
                        reference_response = None
                        if (
                                conversation_to_process
                                and conversation_to_process[-1]["role"] == "assistant"
                        ):
                            reference_response = conversation_to_process[-1]["content"]
                            conversation_to_process = conversation_to_process[:-1]

                # If no conversations found, try to extract from "user" field
                if not conversation_to_process:
                    user_input = json_data.get("user", "").strip()
                    if user_input:
                        conversation_to_process = [
                            {"role": "user", "content": user_input}
                        ]

                if not conversation_to_process:
                    logger.warning(
                        f"Warning: No user input found in line {line_num}, skipping"
                    )
                    continue

                input_conversations_data["conversations"] = conversation_to_process

                input_conversations.append(
                    {
                        "conversation": input_conversations_data,
                        "line_num": line_num,
                        "reference": reference_response,
                    }
                )

            except json.JSONDecodeError:
                logger.warning(f"Warning: Invalid JSON on line {line_num}, skipping")
                continue

        if not input_conversations:
            raise LLMJudgeException("No valid conversations were found")

        return input_conversations

    def _build_conversation_result(
            self, input_item: Dict[str, Any], assistant_response: str,
            gen_try: int, max_gen_try: int
    ) -> Dict[str, Any]:
        """Build evaluation-ready conversation structure from input and generated response."""
        final_conversations = (
                [
                    {
                        "role": "system",
                        "content": input_item["conversation"]["system"],
                    }
                ]
                + input_item["conversation"]["conversations"]
                + [{"role": "assistant", "content": assistant_response}]
        )
        return {
            "conversations": final_conversations,
            "generation_index": gen_try,
            "total_generations": max_gen_try,
            "reference": input_item.get("reference"),
        }

    def _load_jsonl_conversations(
            self,
            base_model: str,
            test_file: str,
            eval_args: dict,
            context_length: int
    ) -> List[Dict[str, Any]]:
        """Parse JSONL and generate all responses. Used by validate() for backward compatibility."""
        max_gen_try = eval_args.get("gen_require", 1)
        input_conversations = self._parse_jsonl_conversations(test_file)

        generated_conversations = []

        for gen_try in range(max_gen_try):
            logger.info(f"Generation attempt {gen_try + 1}/{max_gen_try} for {len(input_conversations)} conversations...")
            batch_conversations = [item["conversation"] for item in input_conversations]

            assistant_responses = self._generate_response(
                user_input=batch_conversations,
                base_model=base_model,
                batch_size=self.config.gen_batch_size,
                eval_args=eval_args,
                context_length=context_length,
            )

            for input_item, assistant_response in zip(
                    input_conversations, assistant_responses
            ):
                conversation = self._build_conversation_result(
                    input_item, assistant_response, gen_try, max_gen_try
                )
                generated_conversations.append(conversation)

        if not generated_conversations:
            raise LLMJudgeException("No valid conversations were generated")

        return generated_conversations

    def _format_single_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Format a single conversation for evaluation"""
        conversations = conversation_data.get("conversations", [])

        if not conversations:
            return "No conversation found"

        # Format the conversation
        formatted_parts = []

        for msg in conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        return "\n\n".join(formatted_parts)

    def _construct_evaluation_prompt(
            self, conversation_context: str, prompt_id: int, reference: str = None
    ) -> List[Dict[str, str]]:
        """Construct evaluation prompt for a single conversation"""
        try:
            if reference and prompt_id == 2:
                # Use reference evaluation prompt
                user_message = get_prompt(prompt_id, conversation_context, reference)
            else:
                user_message = get_prompt(prompt_id, conversation_context)
        except ValueError as e:
            user_message = get_prompt(1, conversation_context)
            logger.warning(f"Warning: {e}. Using default prompt.")

        system_prompt = """You are a fair judge, please output the score, confidence, and reasoning for the given conversation."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _parse_llm_response(self, response: str, model_name: str = None) -> Dict[str, Any]:
        result = {"score": 5.0, "confidence": 0, "reasoning": None}

        try:
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)

                if "score" in parsed_json:
                    result["score"] = float(parsed_json["score"])
                if "confidence" in parsed_json:
                    result["confidence"] = float(parsed_json["confidence"])
                if "reasoning" in parsed_json:
                    result["reasoning"] = str(parsed_json["reasoning"])

                return result
            else:
                # No JSON found in response
                logger.error(
                    f"Model '{model_name}' did not return valid JSON format. "
                    f"Response: {response[:200]}..."
                )
                return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(
                f"Model '{model_name}' returned malformed JSON: {e}. "
                f"Response: {response[:200]}..."
            )
            return result

    def _evaluate_single_conversation(
            self,
            conversation_data: Dict[str, Any],
            eval_args: dict,
            max_eval_try: int,
            conv_idx: int,
    ) -> Dict[str, Any]:
        """
        Simplified version: evaluate a single conversation and return scores, confidences, and reasoning
        """
        conversation_context = self._format_single_conversation(conversation_data)
        reference = conversation_data.get("reference")
        messages = self._construct_evaluation_prompt(conversation_context, eval_args.get("prompt_id", 0), reference)

        conv_scores = []
        conv_confidences = []
        conv_reasoning = []

        # Get available models for evaluation
        eval_model_list = eval_args.get("eval_model_list", [])
        available_eval_models = [
            model for model in eval_model_list if model in self.available_models
        ]

        # If no models specified or available, use all available models
        if not available_eval_models:
            available_eval_models = self.available_models

        # Evaluate with each model for max_eval_try times
        for model_idx, model_name in enumerate(available_eval_models):
            for try_num in range(max_eval_try):
                # Create modified eval_args to specify the exact model to use
                model_eval_args = eval_args.copy()
                model_eval_args["selected_model"] = model_name

                response, selected_model = self._call_gpt(messages, model_eval_args)
                parsed_result = self._parse_llm_response(response, model_name=selected_model)

                conv_scores.append(parsed_result["score"])
                if parsed_result["confidence"] is not None:
                    conv_confidences.append(parsed_result["confidence"])
                if parsed_result["reasoning"]:
                    conv_reasoning.append(
                        f"Conv{conv_idx}-Model{model_idx + 1}({selected_model})-Try{try_num + 1}: {parsed_result['reasoning']}"
                    )

        return {
            "scores": conv_scores,
            "confidences": conv_confidences,
            "reasoning": conv_reasoning,
        }

    def validate(self, data: LLMJudgeInputData, **kwargs) -> LLMJudgeMetrics:
        eval_file = download_file(data.validation_set_url)

        try:
            self._load_model(data.hg_repo_id, data.revision, data.max_params)
        except InvalidModelParametersException as e:
            # lowest possible reward for invalid model parameters
            logger.info(f"Invalid model parameters: {e}")
            return LLMJudgeMetrics(score=LOWEST_POSSIBLE_SCORE)

        # Stage 1: Generate all responses
        logger.info("Stage 1: Generating all conversations for evaluation...")
        all_conversations = self._load_jsonl_conversations(
            data.base_model,
            eval_file,
            data.eval_args,
            data.context_length
        )

        # Load evaluation arguments

        max_eval_try = data.eval_args.get("eval_require", 3)  # Default max evaluation tries
        eval_batch_size = self.config.eval_batch_size

        # Calculate total evaluation calls
        eval_model_list = data.eval_args.get("eval_model_list", [])
        available_eval_models = [
            model for model in eval_model_list if model in self.available_models
        ]
        if not available_eval_models:
            available_eval_models = self.available_models

        total_eval_calls = (
                len(all_conversations) * len(available_eval_models) * max_eval_try
        )

        # Stage 2: Direct parallel evaluation of all conversations
        logger.info(
            f"Stage 2: Evaluating {len(all_conversations)} conversations with {len(available_eval_models)} models, "
            f"{max_eval_try} tries each = {total_eval_calls} total evaluations using {eval_batch_size} workers..."
        )

        # Prepare evaluation tasks for all conversations directly
        evaluation_tasks = []
        for idx, conversation_data in enumerate(all_conversations):
            evaluation_tasks.append(
                (
                    conversation_data,
                    data.eval_args,
                    max_eval_try,
                    idx,  # conversation index
                )
            )

        with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._evaluate_single_conversation, *task): task
                for task in evaluation_tasks
            }

            # Collect results with progress tracking
            evaluation_results = []
            completed_count = 0
            total_tasks = len(evaluation_tasks)
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    evaluation_results.append(result)
                    completed_count += 1
                    
                    # Calculate progress
                    evaluations_done = completed_count * len(available_eval_models) * max_eval_try
                    progress_pct = (evaluations_done / total_eval_calls) * 100
                    logger.info(f"Evaluation progress: {completed_count}/{total_tasks} conversations evaluated ({evaluations_done}/{total_eval_calls} LLM calls, {progress_pct:.1f}%)")
                except Exception as e:
                    logger.error(f"Evaluation task failed: {e}")
                    completed_count += 1
                    # Add default result for failed tasks
                    evaluation_results.append(
                        {
                            "scores": [5.0],
                            "confidences": [0.5],
                            "reasoning": ["Evaluation failed"],
                        }
                    )

        # Execute all evaluations in parallel
        logger.info(f"Completed all {total_eval_calls} evaluation calls across {len(all_conversations)} conversations")
        
        all_weighted_scores = []
        all_reasoning = []
        # Process all results
        for result in evaluation_results:
            scores = result.get("scores", [])
            confidences = result.get("confidences", [])

            if scores and confidences:
                for s, c in zip(scores, confidences):
                    all_weighted_scores.append(s * c)
            if result.get("reasoning"):
                all_reasoning.extend(result["reasoning"])

        # Calculate overall averages across all conversations
        raw_avg_score = sum(all_weighted_scores) / len(all_weighted_scores) if all_weighted_scores else 5.0
        combined_reasoning = "\n\n".join(all_reasoning) if all_reasoning else None

        # Normalize the final score to (0, 1) range
        score_finally = self._normalize_score(raw_avg_score)
        logger.info(f"Overall normalized score_finally (0-1 range): {score_finally:.4f}")
        return LLMJudgeMetrics(
            score=score_finally
        )

    def validate_local(
        self,
        model_path: str,
        validation_file: str,
        base_model: str = "default",
        context_length: int = 2048,
        max_params: int = None,
        eval_args: dict = None,
        is_lora: bool = False,
        base_model_path: str = None,
    ) -> dict:
        """
        Run validation locally without FLock API.
        Loads model from local path, generates responses, optionally evaluates via LLM judge.

        Returns:
            dict with keys: num_conversations, conversations, and optionally
            score, raw_score, reasoning.
        """
        if eval_args is None:
            eval_args = {"prompt_id": 1, "gen_require": 1, "eval_require": 1}

        # Stage 0: Load model from local path
        logger.info(f"Loading model from {model_path}...")
        try:
            self._load_local_model(model_path, is_lora=is_lora, max_params=max_params, base_model_path=base_model_path)
        except InvalidModelParametersException as e:
            logger.info(f"Invalid model parameters: {e}")
            return {"num_conversations": 0, "score": LOWEST_POSSIBLE_SCORE}

        # Parse conversations from file (no generation yet)
        input_conversations = self._parse_jsonl_conversations(validation_file)
        logger.info(f"Parsed {len(input_conversations)} conversations from {validation_file}")

        max_gen_try = eval_args.get("gen_require", 1)
        max_eval_try = eval_args.get("eval_require", 1)
        eval_batch_size = self.config.eval_batch_size

        # Resolve available eval models once
        if not self.skip_llm_eval:
            eval_model_list = eval_args.get("eval_model_list", [])
            available_eval_models = [
                m for m in eval_model_list if m in self.available_models
            ]
            if not available_eval_models:
                available_eval_models = self.available_models
            logger.info(f"Evaluation models: {available_eval_models}")

        all_conversations = []
        all_weighted_scores = []
        all_reasoning = []
        pipeline_batch_num = 0

        # Pipeline: generate 1 batch -> evaluate immediately -> next batch
        for gen_try in range(max_gen_try):
            logger.info(f"Generation attempt {gen_try + 1}/{max_gen_try} for {len(input_conversations)} conversations...")
            batch_conversations = [item["conversation"] for item in input_conversations]

            for start_idx, batch_responses in self._generate_response_batched(
                user_input=batch_conversations,
                base_model=base_model,
                batch_size=self.config.gen_batch_size,
                eval_args=eval_args,
                context_length=context_length,
            ):
                pipeline_batch_num += 1

                # Build conversation structures for this batch
                batch_conv_data = []
                for j, response in enumerate(batch_responses):
                    input_item = input_conversations[start_idx + j]
                    conv_data = self._build_conversation_result(
                        input_item, response, gen_try, max_gen_try
                    )
                    batch_conv_data.append(conv_data)
                    all_conversations.append(conv_data)

                # Evaluate this batch immediately (if LLM eval enabled)
                if not self.skip_llm_eval:
                    logger.info(
                        f"Pipeline batch {pipeline_batch_num}: evaluating "
                        f"{len(batch_conv_data)} conversations with {len(available_eval_models)} models..."
                    )

                    eval_tasks = [
                        (conv, eval_args, max_eval_try, start_idx + j)
                        for j, conv in enumerate(batch_conv_data)
                    ]

                    with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
                        futures = {
                            executor.submit(self._evaluate_single_conversation, *t): t
                            for t in eval_tasks
                        }
                        for future in as_completed(futures):
                            try:
                                eval_result = future.result()
                            except Exception as e:
                                logger.error(f"Evaluation failed: {e}")
                                eval_result = {
                                    "scores": [5.0], "confidences": [0.5],
                                    "reasoning": ["Evaluation failed"],
                                }

                            scores = eval_result.get("scores", [])
                            confidences = eval_result.get("confidences", [])
                            if scores and confidences:
                                for s, c in zip(scores, confidences):
                                    all_weighted_scores.append(s * c)
                            if eval_result.get("reasoning"):
                                all_reasoning.extend(eval_result["reasoning"])

        logger.info(f"Pipeline complete: {len(all_conversations)} conversations generated and evaluated")

        result = {
            "num_conversations": len(all_conversations),
            "conversations": all_conversations,
        }

        if not self.skip_llm_eval:
            raw_avg_score = (
                sum(all_weighted_scores) / len(all_weighted_scores)
                if all_weighted_scores else 5.0
            )
            combined_reasoning = "\n\n".join(all_reasoning) if all_reasoning else None
            score_finally = self._normalize_score(raw_avg_score)

            logger.info(f"Raw average score: {raw_avg_score:.4f}")
            logger.info(f"Normalized score (0-1): {score_finally:.4f}")

            result["score"] = score_finally
            result["raw_score"] = raw_avg_score
            result["reasoning"] = combined_reasoning
        else:
            logger.info("Skipping LLM-as-judge evaluation (evaluation disabled)")
            result["score"] = None
            result["note"] = "LLM evaluation skipped; only generation was performed"

        return result

    def cleanup(self):
        """Clean up resources"""
        if self.client and hasattr(self.client, "http_client"):
            try:
                self.client.http_client.close()
            except Exception:
                pass
        self.client = None

        # Close per-model clients
        closed = set()
        for model_name, client in self.model_clients.items():
            client_id = id(client)
            if client_id not in closed and hasattr(client, "http_client"):
                try:
                    client.http_client.close()
                except Exception:
                    pass
                closed.add(client_id)
        self.model_clients.clear()

        # Clean up HuggingFace model resources
        if self.hf_model is not None:
            try:
                if torch.cuda.is_available():
                    self.hf_model.cpu()
                    torch.cuda.empty_cache()
                del self.hf_model
            except Exception:
                pass
            self.hf_model = None

        if self.hf_tokenizer is not None:
            del self.hf_tokenizer
            self.hf_tokenizer = None


MODULE = LLMJudgeValidationModule
