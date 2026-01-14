---
name: openai-api
description: Use when task involves OpenAI, GPT, LLM extraction, structured output parsing, async AI pipelines, or text generation with AI models. Load BEFORE planning or writing code.
---

# OpenAI API (January 2026)

> **Note:** When implementing OpenAI API code, ask the user to clarify requirements if needed (e.g., model selection, structured output schemas, error handling strategy, concurrency limits).

## DO NOT USE (Deprecated)

- `gpt-4o`, `gpt-4`, `gpt-3.5-turbo` - use GPT-5 family instead
- `client.chat.completions.create()` - use `client.responses.create()` or `client.responses.parse()`
- `response_format` parameter - use `text.format` (raw API) or `text_format` (Python SDK)
- prefer `openai.AsyncOpenAI()` when possible for better performance
- avoid `messages` parameter - use `input` for user content, `instructions` for system prompt
- Instructor library - use native Pydantic/Zod support

## Models

| Model | API Name | Use Case | Input/1M | Output/1M | Context |
|-------|----------|----------|----------|-----------|---------|
| GPT-5.2 | `gpt-5.2` | Complex reasoning, vision, planning | $1.75 | $14 | 196k |
| GPT-5.2-pro | `gpt-5.2-pro` | Highest quality, difficult questions | $21 | $168 | 400k |
| GPT-5-mini | `gpt-5-mini` | Cost-efficient default | $0.25 | $2 | 128k |
| GPT-5-nano | `gpt-5-nano` | High volume, simple tasks | $0.05 | $0.40 | 32k |

**Model selection:**
- `gpt-5-mini` - default for most tasks, preferred for normal extraction
- `gpt-5.2` - complex reasoning, vision, coding
- `gpt-5.2-pro` - critical tasks where quality > cost
- `gpt-5-nano` - simple bulk extraction, simple classification

## Basic Text Generation

### Python (Async)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate(prompt: str) -> str:
    response = await client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )
    return response.output_text
```

### Python (Sync)

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input="Write a haiku about programming.",
)
print(response.output_text)
```

### JavaScript/TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-5-mini",
    input: "Write a haiku about programming.",
});

console.log(response.output_text);
```

## Structured Outputs

Structured Outputs ensures model responses adhere to your JSON schema exactly.

### Python SDK - Using `responses.parse()` with `text_format`

The simplest approach - SDK handles schema conversion automatically:

```python
from openai import AsyncOpenAI
from pydantic import BaseModel

client = AsyncOpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

async def extract_event(text: str) -> CalendarEvent:
    response = await client.responses.parse(
        model="gpt-5-mini",
        instructions="Extract the event information.",
        input=text,
        text_format=CalendarEvent,
    )
    return response.output_parsed  # Already a CalendarEvent instance
```

### With Field Descriptions

Use `Field()` to add descriptions that guide the model:

```python
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

client = AsyncOpenAI()

class BudgetPeriod(BaseModel):
    """A single budget period entry."""
    start_date: str = Field(...,description="Period start date in ISO 8601 YYYY-MM format",examples=["2026-01", "2025-06"],)
    name: str = Field(description="Name of the budget period",)
    amount: float = Field(...,description="Budget amount in USD",ge=0,)

async def extract_budget(text: str) -> BudgetPeriod:
    response = await client.responses.parse(
        model="gpt-5-mini",
        input=text,
        text_format=BudgetPeriod,
    )
    return response.output_parsed
```

### JavaScript/TypeScript - Using Zod

```typescript
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const client = new OpenAI();

const CalendarEvent = z.object({
    name: z.string(),
    date: z.string(),
    participants: z.array(z.string()),
});

const response = await client.responses.parse({
    model: "gpt-5-mini",
    instructions: "Extract the event information.",
    input: "Alice and Bob are going to a science fair on Friday.",
    text: {
        format: zodTextFormat(CalendarEvent, "calendar_event"),
    },
});

const event = response.output_parsed;
```

### Nested Models with Lists

Root schema must be an object, so wrap lists in a container model:

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    addresses: list[Address]

class PeopleList(BaseModel):
    people: list[Person]

async def extract_people(text: str) -> list[Person]:
    response = await client.responses.parse(
        model="gpt-5-mini",
        input=text,
        text_format=PeopleList,
    )
    # Access the full parsed object or extract the list
    full_result = response.output_parsed        # PeopleList instance
    return response.output_parsed.people        # list[Person]
```

### Enums and Literals

```python
from enum import Enum
from typing import Literal

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class Analysis(BaseModel):
    sentiment: Sentiment
    confidence: Literal["low", "medium", "high"]
    summary: str | None = None
```

## Chain of Thought with Structured Output

Guide the model through step-by-step reasoning:

```python
class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

async def solve_math(problem: str) -> MathReasoning:
    response = await client.responses.parse(
        model="gpt-5-mini",
        instructions="You are a math tutor. Guide through the solution step by step.",
        input=problem,
        text_format=MathReasoning,
    )
    return response.output_parsed
```

## Reasoning Effort

Control thinking depth for reasoning models:

```python
# GPT-5.2 - supports none, low, medium, high
# GPT-5-mini - supports low, medium, high (no "none")
# GPT-5.2-pro - only supports high
response = await client.responses.create(
    model="gpt-5.2",
    input="Solve this complex problem...",
    reasoning={"effort": "high"},
)
```

## System Instructions

```python
response = await client.responses.create(
    model="gpt-5-mini",
    instructions="You are a helpful assistant. Be concise and accurate.",
    input=user_message,
)
```

## Multi-turn Conversations

### Using Conversations API (Recommended)

```python
# Create a persistent conversation object
conversation = await client.conversations.create()

# Use conversation ID across sessions/devices
response = await client.responses.create(
    model="gpt-5-mini",
    input="What are the 5 Ds of dodgeball?",
    conversation=conversation.id,
)

# Continue same conversation later
response2 = await client.responses.create(
    model="gpt-5-mini",
    input="Which one is the most important?",
    conversation=conversation.id,
)
```

### Using `previous_response_id`

```python
response1 = await client.responses.create(
    model="gpt-5-mini",
    input="Tell me a joke",
)

response2 = await client.responses.create(
    model="gpt-5-mini",
    previous_response_id=response1.id,
    input="Explain why this is funny",
)
```

Note: `instructions` only applies to the current request. When using `previous_response_id`, instructions from previous turns are not included in context.

### Manual Context Management

```python
# Build conversation history manually, the only exception to using role-based messages
response = await client.responses.create(
    model="gpt-5-mini",
    input=[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And its population?"},
    ],
)
```

## Web Search Tool

```python
response = await client.responses.create(
    model="gpt-5-mini",
    tools=[{"type": "web_search"}],
    input="What are the latest developments in AI?",
)
print(response.output_text)
```

### With Domain Filtering and With User Location

```python
response = await client.responses.create(
    model="gpt-5-mini",
    tools=[{
        "type": "web_search",
        "filters": {
            "allowed_domains": ["arxiv.org", "nature.com", "science.org"]
        },
        "user_location": {
            "type": "approximate",
            "country": "US",
            "city": "San Francisco",
            "region": "California",
        }
    }],
    input="Latest research on large language models",
)
```


## Streaming

### Python

```python
async def stream_response(prompt: str):
    async with client.responses.stream(
        model="gpt-5-mini",
        input=prompt,
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
```

### JavaScript Streaming

```typescript
const stream = client.responses
    .stream({
        model: "gpt-5-mini",
        input: "Tell me a story",
    })
    .on("response.output_text.delta", (event) => {
        process.stdout.write(event.delta);
    });

const result = await stream.finalResponse();
```

## Concurrent Async Extraction

Use a semaphore to limit concurrent API calls and avoid rate limits:

```python
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel

client = AsyncOpenAI()

class ExtractedData(BaseModel):
    title: str
    summary: str
    keywords: list[str]

async def extract_with_limit(
    texts: list[str],
    max_concurrent: int = 5,
) -> list[ExtractedData]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_one(text: str) -> ExtractedData:
        async with semaphore:
            response = await client.responses.parse(
                model="gpt-5-mini",
                input=text,
                text_format=ExtractedData,
            )
            return response.output_parsed

    return await asyncio.gather(*[extract_one(t) for t in texts])

# Usage
results = await extract_with_limit(documents, max_concurrent=10)
```

## Error Handling with Retry

```python
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

client = AsyncOpenAI()

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
)
async def robust_call(prompt: str) -> str:
    response = await client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )
    return response.output_text
```


## JSON Schema Constraints

Structured Outputs supports these JSON Schema features:

**Supported types:** String, Number, Boolean, Integer, Object, Array, Enum, anyOf

**String constraints:**
- `pattern` - regex pattern
- `format` - date-time, date, time, email, uuid, ipv4, ipv6, hostname, duration

**Number constraints:**
- `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`

**Array constraints:**
- `minItems`, `maxItems`

**Requirements:**
- All fields must be `required` (use `| None` for optional-like behavior)
- `additionalProperties: false` must be set on all objects
- Root must be an object type (not anyOf at root level)
- Max 5000 properties total, 10 levels of nesting
- Max 1000 enum values across all properties

## References

- [Extended Patterns](references/extended.md) - Vision/image input, function calling, built-in tools (web search, code interpreter, file search), Agents SDK, vector stores, batch API, embeddings, audio, and image generation
