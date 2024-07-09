from enum import Enum
from typing import Callable, Literal, Union

import instructor
from anthropic import Anthropic, AsyncAnthropic
from groq import Groq, AsyncGroq
from instructor import Instructor, AsyncInstructor
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel


class LLMModel(str, Enum):
    Claude3_Opus = "claude-3-opus-20240229"
    Claude35_Sonnet = "claude-3-5-sonnet-20240620"
    Claude3_Haiku = "claude-3-haiku-20240307"
    GPT4_Omni = "gpt-4o"
    GPT35_Turbo = "gpt-3.5-turbo"
    LLAMA3_70b = "llama3-70b-8192"
    LLAMA3_8b = "llama3-8b-8192"


def gen_client(model=LLMModel.GPT4_Omni) -> Union[Instructor, OpenAI, Anthropic, Groq]:
    match model:
        case LLMModel.Claude3_Opus | LLMModel.Claude35_Sonnet | LLMModel.Claude3_Haiku:
            client = instructor.from_anthropic(Anthropic())
        case LLMModel.GPT4_Omni | LLMModel.GPT35_Turbo:
            client = instructor.patch(OpenAI())
        case LLMModel.LLAMA3_70b | LLMModel.LLAMA3_8b:
            client = instructor.from_groq(Groq())
    return client


def gen_async_client(
    model=LLMModel.GPT4_Omni,
) -> Union[AsyncInstructor, AsyncOpenAI, AsyncAnthropic, AsyncGroq]:
    match model:
        case LLMModel.Claude3_Opus | LLMModel.Claude35_Sonnet | LLMModel.Claude3_Haiku:
            client = instructor.from_anthropic(AsyncAnthropic())
        case LLMModel.GPT4_Omni | LLMModel.GPT35_Turbo:
            client = instructor.patch(AsyncOpenAI())
        case LLMModel.LLAMA3_70b | LLMModel.LLAMA3_8b:
            client = instructor.from_groq(AsyncGroq())
    return client


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


def get_type_info(func: Callable, prm_name: str):
    TypeClazz = func.__annotations__[prm_name]
    return TypeClazz
