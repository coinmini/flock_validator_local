import sys
import click
from validator.validation_runner import ValidationRunner

@click.command()
@click.argument(
    "module",
    type=str,
    required=True,
)
@click.option(
    "--task_ids",
    type=str,
    required=True,
    help="The ids of the task, separated by comma",
)
@click.option('--flock-api-key', envvar='FLOCK_API_KEY', required=True, help='Flock API key')
@click.option('--hf-token', envvar='HF_TOKEN', required=True, help='HuggingFace token')
@click.option('--time-sleep', envvar='TIME_SLEEP', default=60 * 3, type=int, show_default=True, help='Time to sleep between retries (seconds)')
@click.option('--assignment-lookup-interval', envvar='ASSIGNMENT_LOOKUP_INTERVAL', default=60 * 3, type=int, show_default=True, help='Assignment lookup interval (seconds)')
@click.option("--debug", is_flag=True)
def main(module: str, task_ids: str, flock_api_key: str, hf_token: str, time_sleep: int, assignment_lookup_interval: int, debug: bool):
    """
    CLI entrypoint for running the validation process.
    Delegates core logic to ValidationRunner.
    """
    runner = ValidationRunner(
        module=module,
        task_ids=task_ids.split(","),
        flock_api_key=flock_api_key,
        hf_token=hf_token,
        time_sleep=time_sleep,
        assignment_lookup_interval=assignment_lookup_interval,
        debug=debug,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        click.echo("\nValidation interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
