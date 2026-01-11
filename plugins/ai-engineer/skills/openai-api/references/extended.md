# Extended OpenAI Patterns

## Vision / Image Input

### Single Image (URL)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def analyze_image(image_url: str, prompt: str) -> str:
    response = await client.responses.create(
        model="gpt-5.2",  # Use gpt-5.2 for best vision
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )
    return response.output_text
```

### Single Image (Base64)

```python
import base64
from pathlib import Path

async def analyze_local_image(image_path: str, prompt: str) -> str:
    image_data = Path(image_path).read_bytes()
    base64_image = base64.standard_b64encode(image_data).decode("utf-8")

    # Detect media type
    suffix = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = media_types.get(suffix, "image/png")

    response = await client.responses.create(
        model="gpt-5.2",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{base64_image}",
                    },
                ],
            }
        ],
    )
    return response.output_text
```

### Vision with Structured Output

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str
    total: float
    line_items: list[dict]

async def extract_invoice(image_path: str) -> Invoice:
    image_data = Path(image_path).read_bytes()
    base64_image = base64.standard_b64encode(image_data).decode("utf-8")

    response = await client.responses.parse(
        model="gpt-5.2",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract all invoice data from this image."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                ],
            }
        ],
        text_format=Invoice,
    )
    return response.output_parsed
```

### Multiple Images

```python
async def compare_images(image_urls: list[str], prompt: str) -> str:
    content = [{"type": "input_text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})

    response = await client.responses.create(
        model="gpt-5.2",
        input=[{"role": "user", "content": content}],
    )
    return response.output_text
```

### JavaScript Vision

```typescript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI();

const imageBuffer = fs.readFileSync("image.png");
const base64Image = imageBuffer.toString("base64");

const response = await client.responses.create({
    model: "gpt-5.2",
    input: [
        {
            role: "user",
            content: [
                { type: "input_text", text: "What's in this image?" },
                {
                    type: "input_image",
                    image_url: `data:image/png;base64,${base64Image}`,
                },
            ],
        },
    ],
});

console.log(response.output_text);
```

## Function Calling / Custom Tools

### Define Functions (Responses API Style)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
    }
]

async def call_with_tools(prompt: str):
    response = await client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        tools=tools,
    )
    return response
```

### Handle Function Calls

```python
import json

async def process_with_tools(prompt: str):
    response = await client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        tools=tools,
    )

    # Check for function calls in output
    for item in response.output:
        if item.type == "function_call":
            name = item.name
            args = json.loads(item.arguments)
            call_id = item.call_id

            # Execute your function
            if name == "get_weather":
                result = await get_weather(**args)

            # Continue conversation with function result
            response = await client.responses.create(
                model="gpt-5-mini",
                input=[
                    {"role": "user", "content": prompt},
                    item,  # Include the function_call item
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                ],
                tools=tools,
            )

    return response.output_text
```

### Multiple Functions

```python
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_flights",
        "description": "Search for flights between cities",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"},
            },
            "required": ["origin", "destination", "date"],
            "additionalProperties": False,
        },
    },
]
```

### Force Specific Function

```python
response = await client.responses.create(
    model="gpt-5-mini",
    input="What's the weather?",
    tools=tools,
    tool_choice={"type": "function", "name": "get_weather"},
)
```

### JavaScript Function Calling

```typescript
import OpenAI from "openai";

const client = new OpenAI();

const tools = [
    {
        type: "function" as const,
        name: "get_weather",
        description: "Get weather for a location",
        parameters: {
            type: "object",
            properties: {
                location: { type: "string" },
            },
            required: ["location"],
            additionalProperties: false,
        },
    },
];

const response = await client.responses.create({
    model: "gpt-5-mini",
    input: "What's the weather in Paris?",
    tools,
});

for (const item of response.output) {
    if (item.type === "function_call") {
        console.log(`Function: ${item.name}`);
        console.log(`Arguments: ${item.arguments}`);
    }
}
```

## Built-in Tools

### Web Search

```python
response = await client.responses.create(
    model="gpt-5-mini",
    tools=[{"type": "web_search"}],
    input="Latest AI news",
)

# Access citations from annotations
for item in response.output:
    if item.type == "message":
        for content in item.content:
            if content.type == "output_text":
                for annotation in content.annotations:
                    if annotation.type == "url_citation":
                        print(f"Source: {annotation.url}")
```

### Code Interpreter

```python
response = await client.responses.create(
    model="gpt-5-mini",
    input="Calculate the standard deviation of [1, 2, 3, 4, 5, 100]",
    tools=[{"type": "code_interpreter"}],
)
```

### File Search

```python
# First create a vector store and upload files
vector_store = await client.vector_stores.create(name="my-docs")

file = await client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants",
)

await client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id,
)

# Then use file search
response = await client.responses.create(
    model="gpt-5-mini",
    input="What does the document say about pricing?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id],
    }],
)
```

### Combining Tools

```python
response = await client.responses.create(
    model="gpt-5.2",
    input="Search the web for Python 3.13 features and write code demonstrating them",
    tools=[
        {"type": "web_search"},
        {"type": "code_interpreter"},
    ],
)
```

## Agents SDK

### Installation

```bash
pip install openai-agents
```

### Basic Agent

```python
from agents import Agent, Runner

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="gpt-5-mini",
)

async def run_agent(prompt: str) -> str:
    result = await Runner.run(agent, prompt)
    return result.final_output
```

### Agent with Custom Tools

```python
from agents import Agent, Runner, function_tool

@function_tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@function_tool
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

agent = Agent(
    name="calculator",
    instructions="You help with calculations and data fetching.",
    model="gpt-5-mini",
    tools=[calculate, fetch_data],
)
```

### Agent with Built-in Tools

```python
from agents import Agent, WebSearchTool, FileSearchTool

agent = Agent(
    name="researcher",
    instructions="You research topics using web search.",
    model="gpt-5-mini",
    tools=[WebSearchTool()],
)
```

### Multi-Agent Handoffs

```python
from agents import Agent, Runner

research_agent = Agent(
    name="researcher",
    instructions="Research topics and gather information.",
    model="gpt-5-mini",
    tools=[WebSearchTool()],
)

writer_agent = Agent(
    name="writer",
    instructions="Write content based on research.",
    model="gpt-5-mini",
    handoffs=[research_agent],
)

result = await Runner.run(writer_agent, "Write about quantum computing")
```

### Structured Output Agent

```python
from agents import Agent, Runner
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    summary: str
    key_points: list[str]

agent = Agent(
    name="reporter",
    instructions="Generate structured reports.",
    model="gpt-5-mini",
    output_type=Report,
)

result = await Runner.run(agent, "Report on AI trends")
report: Report = result.final_output
```

## Files and Vector Stores

### Create and Populate Vector Store

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Create vector store
vector_store = await client.vector_stores.create(
    name="my-documents",
)

# Upload and add file
file = await client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants",
)

await client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id,
)

# Wait for processing
await client.vector_stores.files.poll(
    vector_store_id=vector_store.id,
    file_id=file.id,
)
```

### File Search with Filters

```python
response = await client.responses.create(
    model="gpt-5-mini",
    input="Find Q4 revenue information",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id],
        "max_num_results": 10,
        "filters": {
            "type": "eq",
            "key": "category",
            "value": "financial",
        },
    }],
)
```

### Custom Chunking

```python
await client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id,
    chunking_strategy={
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 800,
            "chunk_overlap_tokens": 400,
        },
    },
)
```

### Batch Upload

```python
from pathlib import Path

async def upload_directory(directory: str, vector_store_id: str):
    for path in Path(directory).glob("**/*.pdf"):
        file = await client.files.create(
            file=open(path, "rb"),
            purpose="assistants",
        )
        await client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file.id,
        )
```

## Advanced Structured Outputs

### UI Generation (Recursive Schema)

```python
from enum import Enum
from typing import List
from pydantic import BaseModel

class UIType(str, Enum):
    div = "div"
    button = "button"
    header = "header"
    section = "section"
    field = "field"
    form = "form"

class Attribute(BaseModel):
    name: str
    value: str

class UI(BaseModel):
    type: UIType
    label: str
    children: List["UI"]
    attributes: List[Attribute]

UI.model_rebuild()  # Required for recursive types

class Response(BaseModel):
    ui: UI

async def generate_ui(description: str) -> UI:
    response = await client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "You are a UI generator. Convert descriptions into UI structures."},
            {"role": "user", "content": description},
        ],
        text_format=Response,
    )
    return response.output_parsed.ui
```

### Content Moderation

```python
from enum import Enum
from typing import Optional

class Category(str, Enum):
    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"

class ContentCompliance(BaseModel):
    is_violating: bool
    category: Optional[Category]
    explanation_if_violating: Optional[str]

async def moderate_content(text: str) -> ContentCompliance:
    response = await client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Determine if the content violates guidelines."},
            {"role": "user", "content": text},
        ],
        text_format=ContentCompliance,
    )
    return response.output_parsed
```

### Data Extraction

```python
class ResearchPaper(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    keywords: list[str]
    publication_date: str | None = None

async def extract_paper_info(text: str) -> ResearchPaper:
    response = await client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Extract research paper metadata from the text."},
            {"role": "user", "content": text},
        ],
        text_format=ResearchPaper,
    )
    return response.output_parsed
```

## Batch API

### Create Batch Request

```python
import json

# Prepare batch input file
requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5-mini",
            "input": f"Summarize: {text}",
        }
    }
    for i, text in enumerate(texts)
]

# Write to JSONL file
with open("batch_input.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Upload and create batch
batch_file = await client.files.create(
    file=open("batch_input.jsonl", "rb"),
    purpose="batch",
)

batch = await client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/responses",
    completion_window="24h",
)
```

### Check Batch Status

```python
batch = await client.batches.retrieve(batch.id)
print(f"Status: {batch.status}")
print(f"Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
```

### Retrieve Results

```python
if batch.status == "completed":
    output_file = await client.files.content(batch.output_file_id)
    results = [json.loads(line) for line in output_file.text.splitlines()]
```

## Embeddings

```python
response = await client.embeddings.create(
    model="text-embedding-3-large",
    input="Your text here",
    dimensions=1024,  # Optional: reduce dimensions
)

embedding = response.data[0].embedding
```

### Batch Embeddings

```python
response = await client.embeddings.create(
    model="text-embedding-3-large",
    input=["Text 1", "Text 2", "Text 3"],
)

embeddings = [item.embedding for item in response.data]
```

## Audio

### Speech to Text

```python
transcription = await client.audio.transcriptions.create(
    model="whisper-1",
    file=open("audio.mp3", "rb"),
)
print(transcription.text)
```

### Text to Speech

```python
response = await client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, how are you?",
)

with open("output.mp3", "wb") as f:
    f.write(response.content)
```

## Image Generation

```python
response = await client.images.generate(
    model="dall-e-3",
    prompt="A futuristic city at sunset",
    size="1024x1024",
    quality="hd",
    n=1,
)

image_url = response.data[0].url
```
