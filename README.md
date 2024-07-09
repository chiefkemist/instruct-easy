# instruct-easy

Utilities to reduce the boilerplate code of Python projects using the [Instructor](https://python.useinstructor.com/) library.

Example:

```python

import asyncio

from pydantic import BaseModel, Field

from instruct_easy import prompt, UserMessage, SystemMessage, LLMModel

class PythonExample(BaseModel):
    content: str = Field(
        ..., description="Python code to be executed in Markdown format."
    )

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

@prompt(
    context=context,
    model=model,
)
def test_prompt(_: str, input: PythonExample = None):
    print(input.content)

@prompt(
    context=context,
    model=model,
)
async def test_prompt2(_: str, input: PythonExample = None):
    print(input.content)

test_prompt(message)

asyncio.get_event_loop().run_until_complete(test_prompt2(message2))

```