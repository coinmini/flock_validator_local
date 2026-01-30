# LLM Judge Validation Module

## Overview

The LLM Judge module is a validation system that evaluates Large Language Model (LLM) submissions using automated judging. It downloads submitted models from HuggingFace, validates their parameters and specifications, generates responses using a validation dataset, and scores the responses using an LLM-as-a-judge evaluation approach.

## Prerequisites

- Conda package manager
- Python 3.12
- HuggingFace account and token
- FLock API key
- Access to OpenAI-compatible API endpoint (e.g. FLock API Platfrom)

## Environment Setup

### 1. Install Dependencies

The module uses a conda environment that is automatically created when you run the module for the first time. The environment is defined in `environment.yml` and includes:

- Python 3.12
- OpenAI API client
- HuggingFace Transformers
- PyTorch
- PEFT
- And other dependencies

The environment will be automatically created with the name `flock-validation-llm_judge`.

### 2. Configure Environment Variables

Create a `.env` file in the `validator/modules/llm_judge/` directory:

```bash
cp validator/modules/llm_judge/.env.example validator/modules/llm_judge/.env
```

Edit the `.env` file and set your API credentials:

```bash
OPENAI_BASE_URL=https://api.flock.io/v1
OPENAI_API_KEY=your_flock_api_key_here
```

**Note:** The `OPENAI_API_KEY` in the `.env` file should be set to your FLock API key, which is used for LLM-as-a-judge evaluation.

## Usage

### Basic Command

Run the validation module from the root directory of the repository:

```bash
python run.py llm_judge \
  --task_ids 312 \
  --flock-api-key YOUR_FLOCK_API_KEY \
  --hf-token YOUR_HUGGINGFACE_TOKEN
```

### Command Arguments

#### Required Arguments

- `llm_judge` - The module name (first positional argument)
- `--task_ids` - Comma-separated list of task IDs to validate (e.g., `312` or `312,313,314`)
- `--flock-api-key` - Your FLock API key for accessing the federated learning platform
- `--hf-token` - Your HuggingFace token for downloading models

#### Optional Arguments

- `--time-sleep` - Time to sleep between retries in seconds (default: 180)
- `--assignment-lookup-interval` - Assignment lookup interval in seconds (default: 180)
- `--debug` - Enable debug mode for verbose logging

### Example Commands

**Single task validation:**

```bash
python run.py llm_judge \
  --task_ids 312 \
  --flock-api-key fl0ck_4p1_k3y_3x4mpl3 \
  --hf-token hf_YourTokenHere
```

### Using Environment Variables

You can also set credentials as environment variables instead of command-line arguments:

```bash
export FLOCK_API_KEY=fl0ck_4p1_k3y_3x4mpl3
export HF_TOKEN=hf_YourTokenHere

python run.py llm_judge --task_ids 312
```

## Local Validation Mode

For locally trained models, you can run validation without FLock API or HuggingFace token using `local_validate.py`.

### Basic Command

```bash
python local_validate.py \
  --model-path /path/to/your/local/model \
  --validation-file /path/to/validation.jsonl
```

### Command Arguments

#### Required Arguments

- `--model-path` - Path to the local model directory
- `--validation-file` - Path to the local validation JSONL file

#### Optional Arguments

- `--base-model` - Model template name for conversation formatting (default: `default`). Options: `default`, `llama3`, `qwen1.5`, `phi3`, `phi4`, `llama2`, `gemma`, `mistral`, `mixtral`, `yi`, `zephyr`
- `--context-length` - Maximum context length (default: 2048)
- `--max-params` - Maximum model parameters limit (skip check if not set)
- `--eval-with-llm` / `--no-eval-with-llm` - Enable LLM-as-judge evaluation via OpenAI API (default: disabled)
- `--eval-model` - Evaluation model name (can be specified multiple times, e.g. `--eval-model gpt-4o --eval-model gpt-4`)
- `--prompt-id` - Evaluation prompt template ID: `1` = default, `2` = reference-based (default: 1)
- `--output-format` - Output format: `text` or `json` (default: text)
- `--gen-batch-size` - Generation batch size (default: 1)
- `--eval-batch-size` - Evaluation parallelism workers (default: 16)
- `--is-lora` - Flag indicating the model is a LoRA adapter

### Example Commands

**Generate responses only (no LLM evaluation):**

```bash
python local_validate.py \
  --model-path ./my-trained-model \
  --validation-file ./test_data/final_validation_set.jsonl \
  --base-model llama3
```

**With LLM-as-judge evaluation (requires `.env` config):**

```bash
python local_validate.py \
  --model-path ./my-trained-model \
  --validation-file ./test_data/final_validation_set.jsonl \
  --base-model llama3 \
  --eval-with-llm \
  --eval-model gpt-4o
```

**LoRA adapter model:**

```bash
python local_validate.py \
  --model-path ./my-lora-adapter \
  --validation-file ./test_data/final_validation_set.jsonl \
  --is-lora
```

**JSON output:**

```bash
python local_validate.py \
  --model-path ./my-trained-model \
  --validation-file ./test_data/final_validation_set.jsonl \
  --output-format json
```

### Notes

- If using `--eval-with-llm`, configure `OPENAI_BASE_URL` and `OPENAI_API_KEY` in `validator/modules/llm_judge/.env`
- When using `--is-lora`, the `adapter_config.json` in the model directory must contain `base_model_name_or_path` pointing to the base model (can be a local path or HuggingFace model ID)
