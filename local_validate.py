import sys
import json
import click
from loguru import logger
from validator.modules.llm_judge import LLMJudgeValidationModule, LLMJudgeConfig


@click.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to local model directory')
@click.option('--validation-file', type=click.Path(exists=True), required=True,
              help='Path to local validation JSONL file')
@click.option('--base-model', type=str, default='default',
              help='Base model template name (default, llama3, qwen1.5, phi3, phi4, etc.)')
@click.option('--context-length', type=int, default=2048,
              help='Maximum context length')
@click.option('--max-params', type=int, default=None,
              help='Maximum allowed model parameters (skip check if not set)')
@click.option('--eval-with-llm/--no-eval-with-llm', default=False,
              help='Enable LLM-as-judge evaluation via OpenAI API (requires .env config)')
@click.option('--eval-model', type=str, multiple=True,
              help='Evaluation model name(s), can be specified multiple times')
@click.option('--prompt-id', type=int, default=1,
              help='Evaluation prompt template ID (1=default, 2=reference-based)')
@click.option('--output-format', type=click.Choice(['text', 'json']), default='text',
              help='Output format for results')
@click.option('--gen-batch-size', type=int, default=1,
              help='Generation batch size')
@click.option('--eval-batch-size', type=int, default=16,
              help='Evaluation parallelism (number of workers)')
@click.option('--is-lora', is_flag=True, default=False,
              help='Model is a LoRA adapter (expects adapter_config.json in model-path)')
def main(model_path, validation_file, base_model, context_length, max_params,
         eval_with_llm, eval_model, prompt_id, output_format,
         gen_batch_size, eval_batch_size, is_lora):
    """
    Run LLM Judge validation locally against a model on disk.
    No FLock API or HuggingFace token required.
    """
    config = LLMJudgeConfig(
        gen_batch_size=gen_batch_size,
        eval_batch_size=eval_batch_size,
    )

    skip_llm_eval = not eval_with_llm
    module = LLMJudgeValidationModule(config=config, skip_llm_eval=skip_llm_eval)

    eval_args = {
        "prompt_id": prompt_id,
        "gen_require": 1,
        "eval_require": 1,
    }
    if eval_model:
        eval_args["eval_model_list"] = list(eval_model)

    try:
        result = module.validate_local(
            model_path=model_path,
            validation_file=validation_file,
            base_model=base_model,
            context_length=context_length,
            max_params=max_params,
            eval_args=eval_args,
            is_lora=is_lora,
        )

        if output_format == 'json':
            print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        else:
            print(f"\n{'='*60}")
            print("LOCAL VALIDATION RESULT")
            print(f"{'='*60}")
            print(f"Model path:       {model_path}")
            print(f"Validation file:  {validation_file}")
            print(f"Base model:       {base_model}")
            print(f"Conversations:    {result['num_conversations']}")
            if result.get('score') is not None:
                print(f"Normalized score: {result['score']:.4f}")
                if result.get('raw_score') is not None:
                    print(f"Raw score (0-10): {result['raw_score']:.4f}")
            else:
                print(f"Score:            {result.get('note', 'N/A')}")
            print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    finally:
        module.cleanup()


if __name__ == "__main__":
    main()
