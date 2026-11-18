from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from pydantic_ai import Agent
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend
from pathlib import Path
from pydantic_ai.capabilities import Hooks


BEDROCK_REGION = 'eu-west-1'
BEDROCK_PROXY_URL = 'https://bedrock-runtime.aws-proxy.skillerwhale.com/'
BEDROCK_DEFAULT_MODEL = 'eu.anthropic.claude-haiku-5-5'


def get_attendance_id(path=Path(__file__).with_name('attendance_id')):
    """Read the attendance ID from the exercise checkout."""
    return Path(path).read_text().strip()


def get_model(model_name: str=BEDROCK_DEFAULT_MODEL) -> BedrockConverseModel:
    """Returns BedrockConverseModel setup with attendance_id credentials."""
    return BedrockConverseModel(
        model_name,
        provider=BedrockProvider(
            region_name=BEDROCK_REGION,
            base_url=BEDROCK_PROXY_URL,
            aws_access_key_id=get_attendance_id(),
            aws_secret_access_key='<unused>',
        ),
    )


async def log_model_turn(logger, ctx, *, request_context, response):
    """Log model response or tool call."""
    if response.text:
        logger.info('Model: %s', response.text)
    for call in response.tool_calls:
        logger.info('Tool call: %s %r', call.tool_name, call.args)
    return response


async def log_tool_result(logger, ctx, *, call, tool_def, args, result):
    """Log tool result."""
    logger.info('Tool result: %s %s', call.tool_name, result)
    return result


async def run_agent_loop(
        workspace_path,
        base_image,
        model_name,
        prompt,
        logger,
        capabilities,
        tool_retries=5
    ):
    work_dir = Path(workspace_path).resolve()
    work_dir.mkdir(exist_ok=True)

    # setup docker-based sandbox
    #   - image is the Docker base image used
    #   - docker_args binds a volume mount to the host
    workspace = DockerSandboxBackend(
        image=base_image,
        allow_internet_access=False,
        docker_args=['--volume', f'{work_dir}:/workspace'],
    )

    # setup agent
    #   model - the LLM, via Bedrock
    #   capabilities - Agent capabilities - tools
    #   hooks - a hook so we can see what the agent does
    agent = Agent(
        model=get_model(model_name),
        retries={'tools': tool_retries},
        capabilities=[
            *capabilities,
            Hooks(after_model_request=lambda *args, **kwargs: log_model_turn(logger, *args, **kwargs)),
        ]
    )

    # Run agent turns and clean-up
    try:
        result = await agent.run(
            user_prompt=prompt,
            workspace=workspace,
        )
        print(result.output)
    finally:
        if workspace.ref is not None:
            await DockerSandbox().destroy(workspace.ref)
