---
name: llm-tracing
description: Use when task involves LLM observability, tracing, Langfuse, OpenTelemetry for AI, monitoring LLM calls, or debugging AI pipelines. Load BEFORE planning or writing code.
---

# LLM Tracing with Langfuse (January 2026)

## Installation

```bash
# Python - base (SDK v3.12.0+)
pip install langfuse openai python-dotenv

# Python - with OpenTelemetry auto-instrumentation (for multi-provider apps)
pip install langfuse openinference-instrumentation-openai openinference-instrumentation-google-genai

# Node.js (SDK v4+)
npm install @langfuse/tracing @langfuse/otel @opentelemetry/sdk-node
```

## Environment Variables

```bash
# .env - Python uses LANGFUSE_HOST, TypeScript uses LANGFUSE_BASE_URL
LANGFUSE_PUBLIC_KEY='pk-lf-...'
LANGFUSE_SECRET_KEY='sk-lf-...'
LANGFUSE_HOST='https://us.cloud.langfuse.com'        # Python SDK
LANGFUSE_BASE_URL='https://us.cloud.langfuse.com'   # TypeScript SDK
```

## Python: Choosing an Instrumentation Approach

**Decision guide:**
- Simple app with OpenAI only → use **Langfuse wrapper** (fewest dependencies)
- Need custom metadata or specific control → use **Manual** instrumentation
- Multiple LLM providers (OpenAI + Google + others) → use **OpenTelemetry** (consistent pattern)

## Python Option 1: Langfuse OpenAI Wrapper (Simplest)

Best for simple apps using only OpenAI. Auto-captures model, tokens, and I/O:

```python
from langfuse import observe, get_client
from langfuse.openai import AsyncOpenAI  # Wrapped client - auto-captures everything

langfuse = get_client()
client = AsyncOpenAI()

async def summarize(text: str) -> str:
    response = await client.responses.create(
        model="gpt-5",
        input=f"Summarize this text:\n\n{text}",
    )
    return response.output_text

langfuse.flush()  # Always flush before process exits
```

## Python Option 2: Manual Tracing (Full Control)

With auto-instrumentation enabled (Options 1 and 3), it's unnecessary duplication.
Use when you need explicit control over what gets traced and custom metadata:

```python
from langfuse import observe, get_client
from openai import AsyncOpenAI  # Regular client (not wrapped)

langfuse = get_client()
client = AsyncOpenAI()

@observe(as_type="generation", name="openai-chat")
async def chat(prompt: str) -> str:
    response = await client.responses.create(
        model="gpt-5",
        input=prompt,
    )

    # Manually update the generation with token usage
    langfuse.update_current_generation(
        input=prompt,
        output=response.output_text,
        metadata={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    )
    return response.output_text

langfuse.flush()
```

## Python Option 3: OpenTelemetry Auto-Instrumentation (Multi-Provider)

Best for apps calling multiple LLM providers. OpenInference instrumentors work with Langfuse SDK v3 (it sets up OpenTelemetry internally):

```python
from dotenv import load_dotenv
load_dotenv()

from langfuse import observe, get_client
langfuse = get_client()

# Auto-instrument BOTH SDKs - one line each
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

OpenAIInstrumentor().instrument()
GoogleGenAIInstrumentor().instrument()

# Use native clients (NOT wrapped versions)
from openai import AsyncOpenAI
from google import genai

openai_client = AsyncOpenAI()
google_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@observe(name="my-pipeline")  # Parent span groups child LLM calls
async def run_pipeline():
    # Both calls auto-traced as children
    await openai_client.responses.create(...)
    google_client.models.generate_content(...)

langfuse.flush()  # Always flush before exit
```

**Key insight:** `@observe()` creates parent spans that automatically group child LLM calls from the instrumentors.

## Python SDK v3 Key Points

| Topic | Details |
|-------|---------|
| SDK Version | v3.12.0+ (OpenTelemetry-based internally) |
| Imports | `from langfuse import observe, get_client` (NOT `from langfuse.decorators`) |
| Client | `get_client()` returns singleton |
| Langfuse OpenAI wrapper | `from langfuse.openai import AsyncOpenAI` - auto-captures model, tokens, I/O |
| OpenTelemetry | `openinference-instrumentation-*` packages auto-connect to Langfuse's internal OTEL setup |
| Parent spans | Use `@observe(name="...")` on parent function to group child LLM calls |
| Token usage | Pass via `metadata={"input_tokens": ..., "output_tokens": ...}` |
| Flush | `langfuse.flush()` - required before process exits |
| Environment | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (not `_BASE_URL`) |

## TypeScript: Instrumentation Options

**Important:** Unlike Python, there's no `openinference-instrumentation-*` equivalent for Node.js. TypeScript options are:
- **`observeOpenAI()` wrapper** - auto-traces OpenAI calls
- **Manual `startObservation().update().end()`** - for other providers (Google GenAI, Anthropic, etc.)

## TypeScript: OpenTelemetry Setup (Required)

SDK v4 requires OpenTelemetry initialization. Create `instrumentation.ts`:

```typescript
// instrumentation.ts - import this FIRST in your app entry point
import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";

export const langfuseSpanProcessor = new LangfuseSpanProcessor();

export const sdk = new NodeSDK({
  spanProcessors: [langfuseSpanProcessor],
});

sdk.start();
```

## TypeScript: Wrapped OpenAI Client (Recommended)

The simplest approach - auto-captures model, tokens, and I/O:

```typescript
import "./instrumentation"; // Must be first!
import OpenAI from "openai";
import { observeOpenAI } from "@langfuse/openai";

const openai = observeOpenAI(new OpenAI());

async function main() {
  // Automatically traced - captures latency, tokens, errors
  const response = await openai.responses.create({
    model: "gpt-5",
    input: "Hello!",
  });

  console.log(response.output_text);
}

main();
```

## TypeScript: Manual Tracing

Use `observe` wrapper or `startActiveObservation` for explicit control:

```typescript
import "./instrumentation";
import { observe, startActiveObservation, startObservation } from "@langfuse/tracing";
import OpenAI from "openai";

const openai = new OpenAI();

// Option 1: observe wrapper (simplest for wrapping functions)
const fetchData = observe(
  async (source: string) => {
    const response = await openai.responses.create({
      model: "gpt-5",
      input: `Fetch data from ${source}`,
    });
    return response.output_text;
  },
  { name: "fetch-data", asType: "generation" }
);

// Option 2: startActiveObservation (nested spans with context)
async function processRequest(userId: string, query: string) {
  return await startActiveObservation("process-request", async (span) => {
    span.update({ input: { userId, query }, metadata: { source: "api" } });

    // Nested generation - automatically linked to parent span
    const generation = startObservation(
      "llm-call",
      { model: "gpt-5", input: query },
      { asType: "generation" }
    );

    try {
      const response = await openai.responses.create({
        model: "gpt-5",
        input: query,
      });

      const result = response.output_text;
      generation.update({
        output: { content: result },
        usageDetails: {
          input: response.usage?.input_tokens ?? 0,
          output: response.usage?.output_tokens ?? 0,
        },
      }).end();

      span.update({ output: { result } });
      return result;
    } catch (error) {
      generation.update({
        output: { error: error instanceof Error ? error.message : "Failed" },
      }).end();
      throw error;
    }
  });
}
```

## TypeScript SDK v4 Key Points

| Topic | Details |
|-------|---------|
| SDK Version | v4+ (OpenTelemetry-based) |
| Packages | `@langfuse/tracing`, `@langfuse/otel`, `@opentelemetry/sdk-node` |
| OpenAI wrapper | `@langfuse/openai` with `observeOpenAI(new OpenAI())` |
| Setup | Must initialize `NodeSDK` with `LangfuseSpanProcessor` before tracing |
| Tracing | `observe()`, `startActiveObservation()`, `startObservation()` from `@langfuse/tracing` |
| Token usage | `usageDetails: { input: N, output: M }` in `update()` |
| Flush | `langfuseSpanProcessor.forceFlush()` or `sdk.shutdown()` |
| Environment | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` (not `_HOST`) |

## Serverless: Singleton Pattern

For serverless environments (Vercel, AWS Lambda, Cloud Functions):

**Why serverless is different:**
- **No persistent process** - Function can be terminated immediately after response
- **Cold starts** - SDK must initialize quickly and reuse across invocations
- **Async flushing** - Background sends will be killed if you don't await flush

**Key differences from long-running servers:**
| Aspect | Long-running server | Serverless |
|--------|--------------------|-----------|
| Flush timing | End of process / periodic | **Every request** before response |
| SDK init | Once at startup | Module scope (reused across warm invocations) |
| Flush method | `flush()` (fire-and-forget OK) | `await flush()` or `forceFlush()` (**must await**) |

### Python

```python
from langfuse import observe, get_client
from langfuse.openai import AsyncOpenAI
import os

# Module scope: initialized once, reused across warm invocations
langfuse = get_client()
client = AsyncOpenAI()

@observe(name="handle-request")
async def handle_request(prompt: str):
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return  # Tracing disabled

    response = await client.responses.create(
        model="gpt-5",
        input=prompt,
    )

    # SERVERLESS DIFFERENCE: Must flush BEFORE returning response
    # In a regular server, you might flush periodically or at shutdown
    # In serverless, the function may be killed immediately after return
    langfuse.flush()
    return response.output_text
```

### TypeScript

```typescript
// instrumentation.ts - runs once at cold start, reused on warm invocations
import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";

export const langfuseSpanProcessor = new LangfuseSpanProcessor();
export const sdk = new NodeSDK({ spanProcessors: [langfuseSpanProcessor] });
sdk.start();
```

```typescript
// route.ts - API handler
import { langfuseSpanProcessor } from "./instrumentation";
import { startActiveObservation } from "@langfuse/tracing";

export async function POST(request: Request) {
  return await startActiveObservation("api-request", async (span) => {
    span.update({ input: { path: "/api/process" } });

    // ... do LLM calls with tracing ...

    span.update({ output: { success: true } });

    // SERVERLESS DIFFERENCE: Must await forceFlush() before returning
    // Unlike long-running servers where background sends complete eventually,
    // serverless functions can be frozen/killed immediately after response
    await langfuseSpanProcessor.forceFlush();

    return Response.json({ success: true });
  });
}
```

## Observation Types

```python
# Python SDK v3 - use @observe decorator
from langfuse import observe

@observe(as_type="generation", name="llm-call")  # LLM model calls (most common)
async def call_llm(): ...

@observe(name="fetch-context")  # Span - any operation (default type)
async def fetch_context(): ...

@observe(as_type="event", name="cache-hit")  # Event - point-in-time markers
def log_cache_hit(): ...
```

```typescript
// TypeScript SDK v4
import { startObservation, observe } from "@langfuse/tracing";

// Generation - LLM model calls (most common)
const gen = startObservation("llm-call", { model: "gpt-5" }, { asType: "generation" });
gen.update({ output: { content: "..." }, usageDetails: { input: 10, output: 5 } }).end();

// Span - any operation (default type)
const span = startObservation("fetch-context", { input: { query: "..." } });
span.update({ output: { result: "..." } }).end();

// Tool - external tool/function calls
const tool = startObservation("search", { input: { q: "..." } }, { asType: "tool" });
tool.update({ output: { results: [...] } }).end();
```

## Status Levels (Python)

```python
# Python SDK v3 - levels are set via update_current_observation
from langfuse import observe, get_client

langfuse = get_client()

@observe(as_type="generation", name="llm-call")
async def call_llm():
    try:
        # ... LLM call ...
        langfuse.update_current_observation(level="DEFAULT")  # Success
    except Exception:
        langfuse.update_current_observation(level="ERROR")    # Failure
        raise
```

## Capturing Token Usage

```python
langfuse.update_current_generation(
    input=prompt,
    output=response.output_text,
    metadata={
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
    },
)
```

## Privacy Best Practices

**DO track:**
- Truncated prompts (first 200-500 chars)
- Token usage statistics
- Error messages
- Metadata

**DON'T track:**
- Full prompts with PII
- Actual image/file content (track counts instead)


## Flush: Critical for Serverless

Always flush before the response ends:

```python
# Python SDK v3
from langfuse import get_client

langfuse = get_client()
langfuse.flush()  # Required before process exits
```

```typescript
// TypeScript SDK v4
import { langfuseSpanProcessor, sdk } from "./instrumentation";

// Option 1: Force flush (keeps SDK running - use in serverless)
await langfuseSpanProcessor.forceFlush();

// Option 2: Full shutdown (flushes and stops SDK - use for CLI tools)
await sdk.shutdown();
```