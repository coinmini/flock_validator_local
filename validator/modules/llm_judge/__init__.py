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
from validator.modules.llm_judge.prompt import get_prompt, template_str
from validator.modules.llm_judge.utils import download_file
from validator.modules.llm_judge.constant import SUPPORTED_BASE_MODELS
from validator.exceptions import LLMJudgeException, InvalidModelParametersException
from validator.modules.llm_judge.template import template_dict
from peft import PeftModel
from jinja2 import Environment
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
    gen_temperature: float = 0.7


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

    # Models whose APIs do not accept the "system" role
    _NO_SYSTEM_ROLE_MODELS = {"minimax-m2.1"}
    _NO_SEED_MODELS = {"gemini"}

    def __init__(self, config: LLMJudgeConfig, skip_llm_eval: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.client = None
        self.model_clients: Dict[str, OpenAI] = {}
        self.model_name_map: Dict[str, str] = {}
        self.available_models: List[str] = []
        self.hf_model = None
        self.hf_tokenizer = None
        self.skip_llm_eval = skip_llm_eval

        if not skip_llm_eval:
            # Initialize client and get available models
            self._initialize_client()
            self._initialize_model_clients()
            self._fetch_available_models()

    def _initialize_client(self):
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url or not api_key:
            logger.warning("Default OPENAI_BASE_URL/OPENAI_API_KEY not set, skipping default client")
            return
        try:
            timeout = httpx.Timeout(
                connect=10.0,  # 10s to establish connection
                read=120.0,  # 120s max to receive response
                write=20.0,  # 20s to send request
                pool=10.0,  # 10s to acquire connection from pool
            )

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )

        except Exception as e:
            raise LLMJudgeException(
                f"Failed to initialize default OpenAI client: {e}"
            ) from e

    def _create_openai_client(self, base_url: str, api_key: str) -> OpenAI:
        timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,
            write=20.0,
            pool=10.0,
        )
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _initialize_model_clients(self):
        """
        Initialize per-model OpenAI clients from environment variables.

        Env var pattern (prefix-based):
            <PREFIX>_BASE_URL=https://api.example.com/v1
            <PREFIX>_API_KEY=sk-xxx
            <PREFIX>_MODELS=model-a,model-b
            <PREFIX>_MODEL_MAP=model-a:actual-api-name  (optional)
        """
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
        if model_name in self.model_clients:
            return self.model_clients[model_name]
        if self.client is not None:
            return self.client
        raise LLMJudgeException(
            f"No API client available for model '{model_name}'. "
            f"Set {model_name.upper().replace('-', '_')}_BASE_URL / _API_KEY / _MODELS in .env, "
            f"or set the default OPENAI_BASE_URL / OPENAI_API_KEY."
        )

    def _adapt_messages_for_model(self, model_name: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Adapt messages for models that don't support the 'system' role."""
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

        system_text = "\n".join(system_parts)
        adapted = []
        prepended = False
        for msg in other_messages:
            if msg["role"] == "user" and not prepended:
                adapted.append({"role": "user", "content": f"[System instruction]: {system_text}\n\n{msg['content']}"})
                prepended = True
            else:
                adapted.append(msg)

        if not prepended:
            adapted.insert(0, {"role": "user", "content": system_text})
        return adapted

    def _fetch_available_models(self):
        self.available_models = list(self.model_clients.keys())

        if self.client is not None:
            try:
                models_response = self.client.models.list()
                default_models = [model.id for model in models_response.data]
                for m in default_models:
                    if m not in self.available_models:
                        self.available_models.append(m)
            except Exception as e:
                logger.error(
                    f"Warning: Failed to fetch models from API ({e}), using fallback models"
                )

        if not self.available_models:
            logger.warning("No evaluation models available, using fallback")
            self.available_models = ["gpt-4o"]

    def _download_lora_config(self, repo_id: str, revision: str) -> bool:
        try:
            api.hf_hub_download(
                repo_id=repo_id,
                filename="adapter_config.json",
                local_dir="judge",
                revision=revision,
                force_download=True,  # Disable fallback to cached files
            )
        except Exception as e:
            if "adapter_config.json" in str(e):
                logger.error(
                    "No adapter_config.json found in the repo, assuming full model"
                )
                return False
            else:
                raise  # Re-raise the exception if it's not related to the missing file
        return True

    def _load_model(self, repo_id: str, revision: str = "main", max_params: int = None):

        is_lora = self._download_lora_config(repo_id, revision=revision)

        # Determine best dtype: prefer BF16 for numerical stability, fallback to FP16
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
                logger.info("Using bfloat16 for better numerical stability")
            else:
                compute_dtype = torch.float16
                logger.info("Using float16 (bfloat16 not supported on this GPU)")
        else:
            compute_dtype = torch.float32
            logger.info("Using float32 (CPU mode)")

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=compute_dtype,
            device_map="auto",
        )
        if is_lora:
            api.snapshot_download(
                repo_id=repo_id,
                local_dir="judge",
                revision=revision,
                force_download=True,  # Disable fallback to cached files
            )
            with open("judge/adapter_config.json", "r") as f:
                adapter_config = json.load(f)
            base_model = adapter_config["base_model_name_or_path"]
            if base_model in SUPPORTED_BASE_MODELS:
                logger.info(
                    f"LoRA's base model '{base_model}' is in SUPPORTED_BASE_MODELS. "
                    f"Using it for tokenizer."
                )
            else:
                logger.error(
                    f"LoRA's base model '{base_model}' is not in SUPPORTED_BASE_MODELS. "
                    f"Marking assignment as failed."
                )
                raise

            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True, use_fast=True, padding_side="left"
            )
            base_hf_model = AutoModelForCausalLM.from_pretrained(
                base_model, **model_kwargs
            )
            hf_model = PeftModel.from_pretrained(
                base_hf_model,
                "judge",
                device_map="auto",
            )
            self.hf_model = hf_model.merge_and_unload()
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                repo_id, trust_remote_code=True, use_fast=True, padding_side="left"
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=compute_dtype,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        total = sum(p.numel() for p in self.hf_model.parameters())
        if total > max_params:
            logger.info(
                f"Total model params: {total} exceeds the limit {max_params}, submitting validation result with a large loss"
            )
            raise InvalidModelParametersException(
                f"Model parameters {total} exceed limit {max_params}"
            )

    def _load_local_model(
        self,
        model_path: str,
        is_lora: bool = False,
        max_params: int = None,
        base_model_path: str = None,
    ):
        """Load a model from a local directory path (no HuggingFace download).

        Args:
            model_path: Path to the model or LoRA adapter directory.
            is_lora: Whether the model is a LoRA adapter.
            max_params: Maximum allowed parameters (skip check if None).
            base_model_path: Local path to the base model for LoRA. If not provided,
                uses base_model_name_or_path from adapter_config.json.
        """
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
                logger.info("Using bfloat16 for better numerical stability")
            else:
                compute_dtype = torch.float16
                logger.info("Using float16 (bfloat16 not supported on this GPU)")
        else:
            compute_dtype = torch.float32
            logger.info("Using float32 (CPU mode)")

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=compute_dtype,
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
                base_model, trust_remote_code=True, use_fast=True, padding_side="left"
            )
            base_hf_model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            hf_model = PeftModel.from_pretrained(
                base_hf_model, model_path, device_map="auto"
            )
            self.hf_model = hf_model.merge_and_unload()
        else:
            logger.info(f"Loading model from local path: {model_path}")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=True, padding_side="left"
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=compute_dtype,
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
        self,
        conversation: Dict[str, Any],
        base_model: str = "default",
    ) -> str:
        """Construct conversation template using tokenizer's apply_chat_template.
        This matches the official FLock-validator approach, ensuring correct
        formatting for each model architecture (ChatML for Qwen, etc.).
        """
        try:
            if not isinstance(conversation, dict):
                raise LLMJudgeException(
                    f"Conversation must be a dict, got {type(conversation)}"
                )
            if "conversations" not in conversation:
                raise LLMJudgeException(
                    f"Conversation dict must have 'conversations' key"
                )
            if not conversation["conversations"]:
                raise LLMJudgeException(f"Conversation 'conversations' list is empty")

            messages = []
            if conversation.get("system"):
                messages.append({"role": "system", "content": conversation["system"]})

            messages += conversation["conversations"]

            tools_for_template = conversation.get("tools", None)
            if tools_for_template is not None:
                try:
                    if isinstance(tools_for_template, str):
                        tools_for_template = json.loads(tools_for_template)
                except Exception:
                    pass

            # Disable thinking for Qwen3.5 models that support it
            try:
                template = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tools=tools_for_template,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # Fallback for tokenizers that don't support enable_thinking
                template = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tools=tools_for_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            if not template or not template.strip():
                raise LLMJudgeException(
                    f"Empty conversation template generated"
                )

        except LLMJudgeException:
            raise
        except Exception as e:
            raise LLMJudgeException(
                f"Failed to construct conversation template: {e}"
            ) from e

        return template

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
                batch_conversations = user_input[i : i + batch_size]

                # Apply chat template with fallback
                batch_conversation_templates = []
                for conversation in batch_conversations:

                    messages = []
                    if "system" in conversation:
                        messages.append({
                            "role": "system",
                            "content": conversation["system"]
                        })

                    messages += conversation["conversations"]
                    tools_for_template = conversation.get("tools", None)
                    try:
                        if isinstance(tools_for_template, str):
                            tools_for_template = json.loads(tools_for_template)
                    except Exception:
                        # leave tools_for_template as-is if parsing fails
                        pass
                    template = self.hf_tokenizer.apply_chat_template(
                        messages,
                        tools=tools_for_template,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )

                    # Validate template is not empty
                    if not template or not template.strip():
                        logger.error(
                            f"Empty template generated for conversation: {conversation}"
                        )
                        raise LLMJudgeException(
                            f"Empty conversation template generated"
                        )

                    batch_conversation_templates.append(template)

                # Simple tokenization
                # Only pad when batch_size > 1 to avoid unnecessary overhead
                model_inputs = self.hf_tokenizer(
                    batch_conversation_templates,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=(batch_size > 1),
                )

                # Validate tokenization results
                if (
                    "input_ids" not in model_inputs
                    or model_inputs["input_ids"].size(0) == 0
                ):
                    logger.error(
                        f"Tokenization failed for batch {batch_idx}. Templates: {batch_conversation_templates}"
                    )
                    raise LLMJudgeException(
                        f"Tokenization produced empty input_ids for batch {batch_idx}"
                    )

                if model_inputs["input_ids"].size(1) == 0:
                    logger.error(
                        f"Tokenization produced empty sequences for batch {batch_idx}. Templates: {batch_conversation_templates}"
                    )
                    raise LLMJudgeException(
                        f"Tokenization produced empty sequences for batch {batch_idx}"
                    )

                # Move to device if available
                if (
                    torch.cuda.is_available()
                    and next(self.hf_model.parameters()).is_cuda
                ):
                    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

                logger.info(
                    f"Generating batch {batch_idx}/{total_batches} ({min(i + batch_size, len(user_input))}/{len(user_input)} conversations)..."
                )
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **model_inputs,
                        max_new_tokens=max_length,
                        temperature=self.config.gen_temperature,  # Non thinking-General 0.7 ,Reasoning 1
                        do_sample=True,
                        top_p=0.8,  # Non thinking-General 0.8 ,Reasoning 0.95
                        top_k=20,    #
                        pad_token_id=self.hf_tokenizer.eos_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                    )

                # Validate generation outputs
                if outputs.size(0) == 0:
                    logger.error(
                        f"Model generation produced no outputs for batch {batch_idx}"
                    )
                    raise LLMJudgeException(
                        f"Model generation produced no outputs for batch {batch_idx}"
                    )

                # Decode responses
                for j, output in enumerate(outputs):
                    # Get the input length for this specific sequence
                    if j >= model_inputs["input_ids"].size(0):
                        logger.error(
                            f"Output index {j} out of bounds for input_ids size {model_inputs['input_ids'].size(0)}"
                        )
                        continue

                    input_length = model_inputs["input_ids"][j].size(0)

                    # Validate output length
                    if output.size(0) == 0:
                        logger.warning(f"Empty output at index {j}, skipping")
                        results.append("")
                        continue

                    if input_length > output.size(0):
                        logger.warning(
                            f"Input length {input_length} exceeds output length {output.size(0)} at index {j}, using full output"
                        )
                        generated_ids = output
                    else:
                        # Extract only the newly generated tokens
                        generated_ids = output[input_length:]

                    assistant_response = self.hf_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    results.append(assistant_response)

                    # Log sample generated texts for verification
                    global_idx = i + j
                    if global_idx < 3:  # Log first 3 generated responses
                        preview = (
                            assistant_response[:500] + "..."
                            if len(assistant_response) > 500
                            else assistant_response
                        )
                        logger.info(
                            f"[Sample Generation {global_idx + 1}] Response preview:\n{preview}"
                        )

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
        Uses local template_dict for conversation formatting (for local validation).
        """
        if self.hf_model is None or self.hf_tokenizer is None:
            raise LLMJudgeException("HuggingFace model not loaded")

        try:
            total_batches = (len(user_input) + batch_size - 1) // batch_size
            for batch_idx, i in enumerate(range(0, len(user_input), batch_size), 1):
                batch_conversations = user_input[i : i + batch_size]

                batch_conversation_templates = []
                for conversation in batch_conversations:
                    template = self._construct_conversation_template(conversation, base_model=base_model)
                    if not template or not template.strip():
                        raise LLMJudgeException("Empty conversation template generated")
                    batch_conversation_templates.append(template)

                for t_idx, tmpl in enumerate(batch_conversation_templates):
                    global_data_idx = i + t_idx
                    logger.info(
                        f"[Data {global_data_idx}] SFT Input template (first 500 chars):\n{tmpl[:500]}"
                    )

                model_inputs = self.hf_tokenizer(
                    batch_conversation_templates,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=(batch_size > 1),
                    truncation=True,
                )

                if "input_ids" not in model_inputs or model_inputs["input_ids"].size(0) == 0:
                    raise LLMJudgeException(f"Tokenization produced empty input_ids for batch {batch_idx}")
                if model_inputs["input_ids"].size(1) == 0:
                    raise LLMJudgeException(f"Tokenization produced empty sequences for batch {batch_idx}")

                if torch.cuda.is_available() and next(self.hf_model.parameters()).is_cuda:
                    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

                logger.info(f"Generating batch {batch_idx}/{total_batches} ({min(i + batch_size, len(user_input))}/{len(user_input)} conversations)...")
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **model_inputs,
                        max_new_tokens=max_length,
                        temperature=self.config.gen_temperature,
                        do_sample=True,
                        top_p=0.8,
                        top_k=20,
                        pad_token_id=self.hf_tokenizer.eos_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                    )

                if outputs.size(0) == 0:
                    raise LLMJudgeException(f"Model generation produced no outputs for batch {batch_idx}")

                batch_results = []
                for j, output in enumerate(outputs):
                    if j >= model_inputs["input_ids"].size(0):
                        continue
                    input_length = model_inputs["input_ids"][j].size(0)
                    if output.size(0) == 0:
                        batch_results.append("")
                        continue
                    if input_length > output.size(0):
                        generated_ids = output
                    else:
                        generated_ids = output[input_length:]

                    assistant_response = self.hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    batch_results.append(assistant_response)

                    global_idx = i + j
                    logger.info(
                        f"[Data {global_idx}] SFT Output (first 300 chars):\n{assistant_response[:300]}"
                    )

                yield (i, batch_results)

            logger.info(f"Completed generating all {len(user_input)} conversations")

        except Exception as e:
            raise LLMJudgeException(f"Failed to generate response: {e}") from e

    def _select_eval_model(self, eval_args: dict) -> str:
        """
        Select evaluation model based on eval_args configuration
        """
        eval_model_list = self._resolve_eval_models(eval_args)

        if eval_model_list:
            selected_model = random.choice(eval_model_list)
            logger.info(f"Using eval_model_list: selected {selected_model}")
            return selected_model

        # random selection from available models
        selected_model = random.choice(self.available_models)
        return selected_model

    def _resolve_eval_models(self, eval_args: dict) -> List[str]:
        """
        Resolve requested eval models against provider model list.
        Supports alias forms like "<model>-low/high" by matching their parsed base model.
        Returns requested model names (so extra params can still be derived later).
        """
        requested_models = eval_args.get("eval_model_list", []) if eval_args else []
        if not requested_models:
            return self.available_models

        resolved_models = []
        missing_models = []
        for requested_model in requested_models:
            if requested_model in self.available_models:
                resolved_models.append(requested_model)
                continue

            parsed_model_name, _ = self._parse_model_name_to_params(requested_model)
            if parsed_model_name in self.available_models:
                logger.info(
                    f"Resolved eval model '{requested_model}' to available model '{parsed_model_name}'."
                )
                resolved_models.append(requested_model)
            else:
                missing_models.append(requested_model)

        # De-duplicate while preserving order
        resolved_models = list(dict.fromkeys(resolved_models))

        if missing_models:
            logger.warning(
                f"Requested eval models not available and will be skipped: {missing_models}"
            )

        if not resolved_models:
            logger.warning(
                "No requested eval models matched provider list, falling back to all available models."
            )
            return self.available_models

        return resolved_models

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

    def _parse_model_name_to_params(self, model_name: str) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        model_parts: list[str] = []

        REASONING_EFFORTS = {"low", "high"}

        for part in model_name.split('-'):
            if part in REASONING_EFFORTS:
                params["reasoning_effort"] = part
            elif part == "thinking":
                extra["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                model_parts.append(part)

        cleaned_model_name = '-'.join(model_parts) if model_parts else model_name
        if cleaned_model_name == "kimi-k2.5":
            extra.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        params["extra_body"] = extra
        return cleaned_model_name, params

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
            eval_model = eval_args["selected_model"]
        else:
            eval_model = self._select_eval_model(eval_args)
        temperature = eval_args.get("temperature", 0.1)  # Default eval temperature

        selected_model, model_params = self._parse_model_name_to_params(eval_model)

        # Patch: kimi-k2.5 and kimi-k2.5-thinking both require temperature=1
        if eval_model in ("kimi-k2.5", "kimi-k2.5-thinking"):
            temperature = 1

        api_model_name = self.model_name_map.get(selected_model, selected_model)
        adapted_messages = self._adapt_messages_for_model(eval_model, messages)

        if eval_model in self._NO_SYSTEM_ROLE_MODELS:
            has_system = any(m["role"] == "system" for m in adapted_messages)
            if has_system:
                logger.warning(f"System role still present after adaptation for {eval_model}, force-removing")
                adapted_messages = [m for m in adapted_messages if m["role"] != "system"]

        params = {
            "model": api_model_name,
            "messages": adapted_messages,
            "temperature": temperature,
        }
        # Some providers (e.g. Gemini) don't support the seed parameter
        if not any(prefix in eval_model.lower() for prefix in self._NO_SEED_MODELS):
            params["seed"] = random.randint(0, 10000)
        params.update(model_params)

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
            client = self._get_client_for_model(eval_model)
            completion = client.chat.completions.create(**params)
            return completion.choices[0].message.content

        try:
            content = _make_api_call()
            return content, eval_model
        except Exception as e:
            raise LLMJudgeException(
                f"API call failed with model {eval_model} after retries: {e}"
            )

    def _parse_jsonl_conversations(self, test_file: str) -> List[Dict[str, Any]]:
        """Parse JSONL file into conversation structures without generating responses.
        Used by validate_local() for local validation.
        """
        input_conversations = []

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]

        for line_num, json_data in enumerate(test_data):
            try:
                system_text = json_data.get("system", None)
                input_conversations_data = {"system": system_text, "conversations": []}

                conversation_to_process = []
                reference_response = None
                tools_info = None
                pending_tool_call_ids = []
                tool_call_counter = 0

                if "conversations" in json_data:
                    conversations = json_data["conversations"]
                    if isinstance(conversations, list) and conversations:
                        for msg in conversations:
                            role = msg.get("role", "")
                            content = msg.get("content", "").strip()
                            if not content:
                                continue

                            if role == "function_call":
                                tool_call_counter += 1
                                tool_call_id = f"call_{tool_call_counter}"
                                try:
                                    call_data = json.loads(content)
                                    tool_call_msg = {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": call_data.get("name", ""),
                                                    "arguments": call_data.get("arguments", {}),
                                                },
                                            }
                                        ],
                                    }
                                    conversation_to_process.append(tool_call_msg)
                                except (json.JSONDecodeError, KeyError):
                                    tool_call_id = None
                                    conversation_to_process.append(
                                        {"role": "assistant", "content": content}
                                    )
                                if tool_call_id:
                                    pending_tool_call_ids.append(tool_call_id)

                            elif role == "observation":
                                if pending_tool_call_ids:
                                    obs_tool_call_id = pending_tool_call_ids.pop(0)
                                else:
                                    obs_tool_call_id = "call_unknown"
                                conversation_to_process.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": obs_tool_call_id,
                                        "content": content,
                                    }
                                )

                            elif role in ["user", "assistant"] and content:
                                conversation_to_process.append({"role": role, "content": content})

                        # Extract reference response (last assistant or function_call message)
                        # Use original conversations data to check role (matches official)
                        reference_response = None
                        if conversation_to_process:
                            last_msg = conversations[-1]
                            if last_msg["role"] in ["assistant"]:
                                reference_response = last_msg["content"]
                                conversation_to_process = conversation_to_process[:-1]
                            elif last_msg["role"] in ["function_call"]:
                                env = Environment(trim_blocks=True, lstrip_blocks=True)
                                conversation_template = env.from_string(template_str)
                                reference_response = conversation_template.render(
                                    messages=[conversation_to_process[-1]],
                                    trim_blocks=True, lstrip_blocks=True,
                                )
                                conversation_to_process = conversation_to_process[:-1]

                        if "tools" in json_data:
                            tools_info = json_data["tools"]

                if not conversation_to_process:
                    user_input = json_data.get("user", "").strip()
                    if user_input:
                        conversation_to_process = [{"role": "user", "content": user_input}]

                if not conversation_to_process:
                    logger.warning(f"Warning: No user input found in line {line_num}, skipping")
                    continue

                input_conversations_data["conversations"] = conversation_to_process
                if tools_info is not None:
                    input_conversations_data["tools"] = tools_info
                input_conversations.append({
                    "conversation": input_conversations_data,
                    "line_num": line_num,
                    "reference": reference_response,
                    "tools": tools_info,
                })
                logger.info(
                    f"[Data {line_num}] Loaded: {len(conversation_to_process)} messages, "
                    f"has_reference={reference_response is not None}, has_tools={tools_info is not None}, "
                    f"user_query={conversation_to_process[0]['content'][:150]}..."
                )

            except json.JSONDecodeError:
                logger.warning(f"Warning: Invalid JSON on line {line_num}, skipping")
                continue

        if not input_conversations:
            raise LLMJudgeException("No valid conversations were found")
        return input_conversations

    def _build_conversation_result(
        self, input_item: Dict[str, Any], assistant_response: str, gen_try: int, max_gen_try: int
    ) -> Dict[str, Any]:
        """Build evaluation-ready conversation structure from input and generated response."""
        final_conversations = (
            [{"role": "system", "content": input_item["conversation"]["system"]}]
            + input_item["conversation"]["conversations"]
            + [{"role": "assistant", "content": assistant_response}]
        )
        return {
            "conversations": final_conversations,
            "generation_index": gen_try,
            "total_generations": max_gen_try,
            "reference": input_item.get("reference"),
            "tools": input_item.get("tools"),
        }

    def _load_jsonl_conversations(
        self, base_model: str, test_file: str, eval_args: dict, context_length: int
    ) -> List[Dict[str, Any]]:

        # Extract parameters from eval_args
        max_gen_try = eval_args.get("gen_require", 1)  # Default max generation tries

        # Parse all input conversations first
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
                tools_info = None
                pending_tool_call_ids: list[str] = []
                tool_call_counter = 0
                if "conversations" in json_data:
                    conversations = json_data["conversations"]
                    if isinstance(conversations, list) and conversations:
                        # Filter valid messages
                        for msg in conversations:
                            role = msg.get("role", "")
                            content = msg.get("content", "").strip()
                            if not content:
                                continue
                            if role == "function_call":
                                tool_call_counter += 1
                                tool_call_id = f"call_{tool_call_counter}"

                                try:
                                    call_data = json.loads(content)
                                    tool_call_msg = {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": call_data.get("name", ""),
                                                    "arguments": call_data.get("arguments", {})
                                                },
                                            }
                                        ],
                                    }
                                    conversation_to_process.append(tool_call_msg)
                                except (json.JSONDecodeError, KeyError):
                                    tool_call_id = None
                                    conversation_to_process.append(
                                        {"role": "assistant", "content": content}
                                    )
                                if tool_call_id:
                                    pending_tool_call_ids.append(tool_call_id)

                            elif role == "observation":
                                if pending_tool_call_ids:
                                    tool_call_id = pending_tool_call_ids.pop(0)
                                else:
                                    tool_call_id = "call_unknown"

                                conversation_to_process.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": content,
                                    }
                                )
                            elif role in ["user", "assistant"]:
                                conversation_to_process.append(
                                    {"role": role, "content": content}
                                )

                        # Extract reference response (last assistant or function_call message)
                        reference_response = None

                        if conversation_to_process:
                            last_msg = conversations[-1]
                            if last_msg["role"] in ["assistant"]:
                                reference_response = last_msg["content"]
                                conversation_to_process = conversation_to_process[:-1]
                            elif last_msg["role"] in ["function_call"]:
                                env = Environment(trim_blocks=True, lstrip_blocks=True)
                                conversation_template = env.from_string(template_str)
                                reference_response = conversation_template.render(
                                    messages=[conversation_to_process[-1]], trim_blocks=True,
                                                         lstrip_blocks=True)
                                conversation_to_process = conversation_to_process[:-1]

                        # Extract tools information if available (for function_call evaluation)
                        if "tools" in json_data:
                            tools_info = json_data["tools"]

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
                if tools_info is not None:
                    input_conversations_data["tools"] = tools_info

                input_conversations.append(
                    {
                        "conversation": input_conversations_data,
                        "line_num": line_num,
                        "reference": reference_response,
                        "tools": tools_info,
                    }
                )

            except json.JSONDecodeError:
                logger.warning(f"Warning: Invalid JSON on line {line_num}, skipping")
                continue

        if not input_conversations:
            raise LLMJudgeException("No valid conversations were found")

        generated_conversations = []

        # Generate responses for each input conversation
        for gen_try in range(max_gen_try):
            logger.info(
                f"Generation attempt {gen_try + 1}/{max_gen_try} for {len(input_conversations)} conversations..."
            )
            # Prepare batch of conversations for generation
            batch_conversations = [item["conversation"] for item in input_conversations]

            # Generate responses using batch processing
            assistant_responses = self._generate_response(
                user_input=batch_conversations,
                base_model=base_model,
                batch_size=self.config.gen_batch_size,
                eval_args=eval_args,
                context_length=context_length,
            )

            # Create conversation structures for this generation
            for input_item, assistant_response in zip(
                input_conversations, assistant_responses
            ):
                # Create final conversation with assistant response
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

                # Create conversation structure for this generation
                conversation = {
                    "conversations": final_conversations,
                    "generation_index": gen_try,
                    "total_generations": max_gen_try,
                    "reference": input_item.get("reference"),
                    "tools": input_item.get("tools"),
                }
                generated_conversations.append(conversation)

        if not generated_conversations:
            raise LLMJudgeException("No valid conversations were generated")

        # Log generation summary
        non_empty_count = sum(
            1
            for conv in generated_conversations
            if conv["conversations"][-1]["content"].strip()
        )
        avg_length = sum(
            len(conv["conversations"][-1]["content"])
            for conv in generated_conversations
        ) / len(generated_conversations)
        logger.info(
            f"Generation summary: {len(generated_conversations)} total, "
            f"{non_empty_count} non-empty, avg response length: {avg_length:.0f} chars"
        )

        return generated_conversations

    def _format_single_conversation(
        self, conversation_data: Dict[str, Any]
    ) -> tuple[str, str]:
        """Format a single conversation for evaluation, returns (context, assistant_response)"""
        conversations = conversation_data.get("conversations", [])

        if not conversations:
            return "No conversation found", ""

        # Format the conversation, separating context from the last assistant response
        formatted_parts = []
        assistant_response = ""

        for i, msg in enumerate(conversations):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Check if this is the last message and it's an assistant/function_call response
            if i == len(conversations) - 1 and role in ["assistant", "function_call"]:
                assistant_response = content
                break

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            elif role == "function_call":
                formatted_parts.append(f"Function Call: {content}")
            elif role == "observation":
                formatted_parts.append(f"Observation: {content}")

        return "\n\n".join(formatted_parts), assistant_response

    def _construct_evaluation_prompt(
        self,
        conversation_context: str,
        prompt_id: int,
        reference: str = None,
        tools: str = None,
        assistant_response: str = None,
    ) -> List[Dict[str, str]]:
        """Construct evaluation prompt for a single conversation"""
        try:
            if prompt_id == 1:
                # Default evaluation prompt with assistant response
                user_message = get_prompt(
                    prompt_id,
                    conversation_context,
                    assistant_response=assistant_response,
                )
            elif reference and prompt_id == 2:
                # Use reference evaluation prompt with assistant response
                user_message = get_prompt(
                    prompt_id,
                    conversation_context,
                    reference,
                    assistant_response=assistant_response,
                )
            elif reference and prompt_id == 3 and tools:
                # Use function_call reference evaluation prompt with tools and assistant response
                user_message = get_prompt(
                    prompt_id,
                    conversation_context,
                    reference,
                    tools,
                    assistant_response,
                )
            else:
                user_message = get_prompt(
                    prompt_id,
                    conversation_context,
                    assistant_response=assistant_response,
                )
        except ValueError as e:
            user_message = get_prompt(
                1, conversation_context, assistant_response=assistant_response
            )
            logger.warning(f"Warning: {e}. Using default prompt.")

        system_prompt = """You are a fair judge, please output the score, confidence, and reasoning for the given conversation."""
        return [
            {"role": "user", "content": user_message},
            {"role": "system", "content": system_prompt},
        ]

    def _parse_llm_response(
        self, response: str, model_name: str = None, conv_idx: int = None
    ) -> Dict[str, Any]:
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
                    f"[Conv {conv_idx}] Model '{model_name}' did not return valid JSON format. "
                    f"Response: {response[:200]}..."
                )
                return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(
                f"[Conv {conv_idx}] Model '{model_name}' returned malformed JSON: {e}. "
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
        conversation_context, assistant_response = self._format_single_conversation(
            conversation_data
        )
        reference = conversation_data.get("reference")
        tools = conversation_data.get("tools")

        # Log evaluation entry with data identification
        last_user_msg = ""
        for msg in reversed(conversation_data.get("conversations", [])):
            if msg.get("role") == "user":
                last_user_msg = msg["content"][:150]
                break
        logger.info(
            f"[Conv {conv_idx}] === Starting evaluation ===\n"
            f"  User query: {last_user_msg}...\n"
            f"  Has reference: {reference is not None}\n"
            f"  Has tools: {tools is not None}"
        )

        messages = self._construct_evaluation_prompt(
            conversation_context,
            eval_args.get("prompt_id", 0),
            reference,
            tools,
            assistant_response,
        )

        # Log what is being sent to judge models
        logger.info(
            f"[Conv {conv_idx}] Evaluation input:\n"
            f"  prompt_id={eval_args.get('prompt_id', 0)}\n"
            f"  conversation_context (first 500 chars): {conversation_context[:500]}\n"
            f"  assistant_response (first 300 chars): {assistant_response[:300]}\n"
            f"  reference (first 300 chars): {str(reference)[:300] if reference else 'None'}"
        )

        conv_scores = []
        conv_confidences = []
        conv_reasoning = []

        # Get available models for evaluation
        available_eval_models = self._resolve_eval_models(eval_args)

        # Evaluate with each model for max_eval_try times
        for model_idx, model_name in enumerate(available_eval_models):
            for try_num in range(max_eval_try):
                # Create modified eval_args to specify the exact model to use
                model_eval_args = eval_args.copy()
                model_eval_args["selected_model"] = model_name

                response, selected_model = self._call_gpt(messages, model_eval_args)
                parsed_result = self._parse_llm_response(
                    response, model_name=selected_model, conv_idx=conv_idx
                )

                # Log each judge model's result
                logger.info(
                    f"[Conv {conv_idx}] Judge result: model={selected_model}, try={try_num+1}/{max_eval_try}, "
                    f"score={parsed_result['score']}, confidence={parsed_result['confidence']}, "
                    f"raw_response (first 200 chars): {str(response)[:200]}"
                )

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
            logger.error(f"Invalid model parameters: {e}")
            return LLMJudgeMetrics(score=LOWEST_POSSIBLE_SCORE)
        except Exception as e:
            logger.error(f"Exception when load model: {e}")
            return LLMJudgeMetrics(score=LOWEST_POSSIBLE_SCORE)

        # Stage 1: Generate all responses
        logger.info("Stage 1: Generating all conversations for evaluation...")
        all_conversations = self._load_jsonl_conversations(
            data.base_model, eval_file, data.eval_args, data.context_length
        )

        # Load evaluation arguments

        max_eval_try = data.eval_args.get(
            "eval_require", 3
        )  # Default max evaluation tries
        eval_batch_size = self.config.eval_batch_size

        # Calculate total evaluation calls
        available_eval_models = self._resolve_eval_models(data.eval_args)

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
                    evaluations_done = (
                        completed_count * len(available_eval_models) * max_eval_try
                    )
                    progress_pct = (evaluations_done / total_eval_calls) * 100
                    logger.info(
                        f"Evaluation progress: {completed_count}/{total_tasks} conversations evaluated ({evaluations_done}/{total_eval_calls} LLM calls, {progress_pct:.1f}%)"
                    )
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
        logger.info(
            f"Completed all {total_eval_calls} evaluation calls across {len(all_conversations)} conversations"
        )

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
        raw_avg_score = (
            sum(all_weighted_scores) / len(all_weighted_scores)
            if all_weighted_scores
            else 5.0
        )
        combined_reasoning = "\n\n".join(all_reasoning) if all_reasoning else None

        # Normalize the final score to (0, 1) range
        score_finally = self._normalize_score(raw_avg_score)
        logger.info(
            f"Overall normalized score_finally (0-1 range): {score_finally:.4f}"
        )
        return LLMJudgeMetrics(score=score_finally)

    def validate_local(
        self, model_path: str, validation_file: str, base_model: str = "default",
        context_length: int = 2048, max_params: int = None, eval_args: dict = None,
        is_lora: bool = False, base_model_path: str = None,
    ) -> dict:
        """
        Run validation locally without FLock API.
        Loads model from local path, generates responses, optionally evaluates via LLM judge.
        """
        if eval_args is None:
            eval_args = {"prompt_id": 1, "gen_require": 1, "eval_require": 1}

        logger.info(f"Loading model from {model_path}...")
        try:
            self._load_local_model(model_path, is_lora=is_lora, max_params=max_params, base_model_path=base_model_path)
        except InvalidModelParametersException as e:
            logger.info(f"Invalid model parameters: {e}")
            return {"num_conversations": 0, "score": LOWEST_POSSIBLE_SCORE}

        input_conversations = self._parse_jsonl_conversations(validation_file)
        logger.info(f"Parsed {len(input_conversations)} conversations from {validation_file}")

        max_gen_try = eval_args.get("gen_require", 1)
        max_eval_try = eval_args.get("eval_require", 1)
        eval_batch_size = self.config.eval_batch_size

        if not self.skip_llm_eval:
            available_eval_models = self._resolve_eval_models(eval_args)
            logger.info(f"Evaluation models: {available_eval_models}")

        all_conversations = []
        all_weighted_scores = []
        all_reasoning = []
        pipeline_batch_num = 0

        for gen_try in range(max_gen_try):
            logger.info(f"Generation attempt {gen_try + 1}/{max_gen_try} for {len(input_conversations)} conversations...")
            batch_conversations = []
            for item in input_conversations:
                conv = dict(item["conversation"])
                if item.get("tools"):
                    conv["tools"] = item["tools"]
                batch_conversations.append(conv)

            for start_idx, batch_responses in self._generate_response_batched(
                user_input=batch_conversations, base_model=base_model,
                batch_size=self.config.gen_batch_size, eval_args=eval_args, context_length=context_length,
            ):
                pipeline_batch_num += 1
                batch_conv_data = []
                for j, response in enumerate(batch_responses):
                    input_item = input_conversations[start_idx + j]
                    conv_data = self._build_conversation_result(input_item, response, gen_try, max_gen_try)
                    batch_conv_data.append(conv_data)
                    all_conversations.append(conv_data)

                if not self.skip_llm_eval:
                    logger.info(f"Pipeline batch {pipeline_batch_num}: evaluating {len(batch_conv_data)} conversations with {len(available_eval_models)} models...")
                    eval_tasks = [(conv, eval_args, max_eval_try, start_idx + j) for j, conv in enumerate(batch_conv_data)]

                    with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
                        futures = {executor.submit(self._evaluate_single_conversation, *t): t for t in eval_tasks}
                        for future in as_completed(futures):
                            try:
                                eval_result = future.result()
                            except Exception as e:
                                logger.error(f"Evaluation failed: {e}")
                                eval_result = {"scores": [5.0], "confidences": [0.5], "reasoning": ["Evaluation failed"]}

                            scores = eval_result.get("scores", [])
                            confidences = eval_result.get("confidences", [])
                            if scores and confidences:
                                for s, c in zip(scores, confidences):
                                    all_weighted_scores.append(s * c)
                            if eval_result.get("reasoning"):
                                all_reasoning.extend(eval_result["reasoning"])

        logger.info(f"Pipeline complete: {len(all_conversations)} conversations generated and evaluated")

        result = {"num_conversations": len(all_conversations), "conversations": all_conversations}

        if not self.skip_llm_eval:
            raw_avg_score = sum(all_weighted_scores) / len(all_weighted_scores) if all_weighted_scores else 5.0
            score_finally = self._normalize_score(raw_avg_score)
            logger.info(f"Raw average score: {raw_avg_score:.4f}")
            logger.info(f"Normalized score (0-1): {score_finally:.4f}")
            result["score"] = score_finally
            result["raw_score"] = raw_avg_score
            result["reasoning"] = "\n\n".join(all_reasoning) if all_reasoning else None
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
