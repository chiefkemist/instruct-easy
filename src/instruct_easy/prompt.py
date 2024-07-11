import asyncio
from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator, Iterable, List, Tuple, Union

from anthropic import Anthropic, AsyncAnthropic
from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletionMessageParam as GroqChatCompletionMessageParam
from instructor import Instructor, AsyncInstructor, Partial
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam as OpenAIChatCompletionMessageParam
from rich.console import Console

from instruct_easy.utils import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    LLMModel,
    gen_async_client,
    gen_client,
    get_type_info,
)

console = Console()


@contextmanager
def msg_ctx(
    message: Union[SystemMessage, UserMessage, AssistantMessage],
    context: List[Union[SystemMessage, UserMessage, AssistantMessage]] = [],
    model=LLMModel.GPT4_Omni,
    persist: bool = False,
) -> Generator[
    Tuple[
        Union[AsyncInstructor, AsyncOpenAI, AsyncAnthropic, AsyncGroq],
        Union[Instructor, OpenAI, Anthropic, Groq],
        Iterable[Union[OpenAIChatCompletionMessageParam, GroqChatCompletionMessageParam]],
    ],
    None,
    None,
]:
    console.log(f"Prompting: {message=}", style="bold italic cyan")
    if persist:
        history = []
        new_context = [*history, *context, message]
        console.log("Retrieving Message History...")
    else:
        new_context = [*context, message]
    messages = [m.model_dump() for m in new_context]
    async_client = gen_async_client(model=model)
    client = gen_client(model=model)
    yield async_client, client, messages


def prompt(
    context: List[Union[SystemMessage, UserMessage, AssistantMessage]] = [],
    model=LLMModel.GPT4_Omni,
    max_tokens=1024,
    max_retries=3,
    stream=False,
):
    def decorator_prompt(func: Callable):
        InputType = get_type_info(func, "input")

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrap_prompt_result(*args, **kwargs):
                console.log(
                    f"[{model}, {max_tokens}, {max_retries}]", style="bold italic green"
                )
                # result = func(*args, **kwargs)
                message = UserMessage(content=args[0].content)
                with msg_ctx(message, context=context, model=model) as (
                    aclient,
                    _,
                    messages,
                ):
                    if stream:
                        PartialInputType = Partial[InputType]
                        completion = await aclient.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            max_retries=max_retries,
                            response_model=PartialInputType,
                            messages=messages,
                            stream=stream,
                        )
                        result = await func(
                            message.content, input=completion, stream=completion
                        )
                        return result
                    else:
                        completion = await aclient.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            max_retries=max_retries,
                            response_model=InputType,
                            messages=messages,
                        )
                        result = await func(message.content, input=completion)
                        return result
            return async_wrap_prompt_result
        else:

            @wraps(func)
            def wrap_prompt_result(*args, **kwargs):
                console.log(
                    f"[{model}, {max_tokens}, {max_retries}]", style="bold italic green"
                )
                # result = func(*args, **kwargs)
                message = UserMessage(content=args[0].content)
                with msg_ctx(message, context=context, model=model) as (
                    _,
                    client,
                    messages,
                ):
                    completion = client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                        response_model=InputType,
                        messages=messages,
                    )
                    result = func(message.content, completion)
                    return result
            return wrap_prompt_result

    return decorator_prompt
