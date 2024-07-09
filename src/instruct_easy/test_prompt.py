
import pytest

from pydantic import BaseModel, Field
from rich.console import Console

from instruct_easy import prompt, UserMessage, SystemMessage, LLMModel

console = Console()


class PythonExample(BaseModel):
    content: str = Field(
        ..., description="Python code to be executed in Markdown format."
    )


@pytest.fixture(autouse=True, scope="session")
def config_options():
    console.log("Setup", style="bold italic purple")

    message = UserMessage(
        content="Hello Professor, how would you write the Y Combinator function in Python using the standard library?"
    )
    message2 = UserMessage(content="Hello Professor, does 1 + 1 = 2?")
    context = [
        SystemMessage(
            content="As a Software Engineering Professor, I am here to help you with your Python coding problems."
        )
    ]
    model = LLMModel.Claude3_Haiku
    yield context, model, message, message2
    console.log("Teardown", style="bold italic purple")


def test_prompt(config_options):
    context, model, message, message2 = config_options
    @prompt(
        context=context,
        model=model,
    )
    def do_prompt(_: str, input: PythonExample = None):
        content = input.content
        console.log(content)
        return content

    result = do_prompt(message, style="bold italic green")
    assert result is not None

    result2 = do_prompt(message2, style="bold italic green")
    assert result2 is not None


@pytest.mark.asyncio
async def test_async_prompt(config_options):
    context, model, message, message2 = config_options
    @prompt(
        context=context,
        model=model,
    )
    async def do_prompt2(_: str, input: PythonExample = None):
        content = input.content
        console.log(content)
        return content

    result = await do_prompt2(message, style="bold italic green")
    assert result is not None

    result2 = await do_prompt2(message2, style="bold italic green")
    assert result2 is not None
