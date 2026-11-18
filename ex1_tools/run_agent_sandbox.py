from pydantic_ai_harness import FileSystem, Shell

from pathlib import Path
import asyncio
import logging

from common import run_agent_loop


# Agent sandbox setup
SANDBOX_HOST_ROOT = Path(__file__).resolve().parent
SANDBOX_HOST_WORKDIR = SANDBOX_HOST_ROOT / 'agent-workdir'

SANDBOX_BACKEND_IMAGE = 'python:3.13-slim'
SANDBOX_MODEL = 'eu.anthropic.claude-haiku-5-5'
SANDBOX_MODEL_CAPABILITIES = [
    FileSystem(read_only=True),
    Shell(tools=['run_command']),
]
SANDBOX_MODEL_PROMPT = """
    In /workspace, read file.yml/
    Fill README.md with human-readable login instructions.
"""

# Logging setup
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(message)s',
    filename=SANDBOX_HOST_ROOT / 'sandbox.log'
)
logger = logging.getLogger('agent_sandbox')


# Run agent in docker sandbox
if __name__ == '__main__':
    asyncio.run(run_agent_loop(
        SANDBOX_HOST_WORKDIR,
        SANDBOX_BACKEND_IMAGE,
        SANDBOX_MODEL,
        SANDBOX_MODEL_PROMPT,
        logger,
        SANDBOX_MODEL_CAPABILITIES,
    ))

