---
name: google-image
description: Use when task involves Google image generation, Gemini image models, Nano Banana, Nano Banana Pro, Imagen, or AI image creation with Google APIs. Load BEFORE planning or writing code.
---

# Google Image Generation API (January 2026)

## DO NOT USE (Deprecated)

- `imagen-3.0-generate-001` - use Nano Banana models instead
- `imagen-3.0-fast-generate-001` - use `gemini-2.5-flash-image` instead
- Tag-style prompts like "dog, park, 4k, realistic" - use natural sentences

## Models

| Model | API Name | Resolution | Ref Images | Tokens | Best For |
|-------|----------|------------|------------|--------|----------|
| Nano Banana | `gemini-2.5-flash-image` | 1024px fixed | Up to 3 | 1290 | Fast, bulk, simple prompts |
| Nano Banana Pro | `gemini-3-pro-image-preview` | 1K/2K/4K | Up to 14 | 1120-2000 | Quality, text, complex scenes |

**Model selection:**
- `gemini-2.5-flash-image` - speed priority, bulk generation, simple edits
- `gemini-3-pro-image-preview` - text rendering, character consistency, professional assets, complex prompts

## Installation

```bash
pip install google-genai        # Python
npm install @google/genai       # Node.js
```

## Basic Generation

### Python

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="A cinematic wide shot of a futuristic sports car speeding through a rainy Tokyo street at night",
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="2K",
        ),
    ),
)

# Extract and save image
for part in response.parts:
    if image := part.as_image():
        image.save("output.png")
```

### TypeScript

```typescript
import { GoogleGenAI } from "@google/genai";
import fs from "fs";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const response = await ai.models.generateContent({
  model: "gemini-3-pro-image-preview",
  contents: [{ text: "A cinematic wide shot of a futuristic sports car speeding through a rainy Tokyo street at night" }],
  config: {
    responseModalities: ["IMAGE"],
    imageConfig: {
      aspectRatio: "16:9",
      imageSize: "2K",
    },
  },
});

const part = response.candidates?.[0]?.content?.parts?.[0];
if (part?.inlineData?.data) {
  const buffer = Buffer.from(part.inlineData.data, "base64");
  fs.writeFileSync("output.png", buffer);
}
```

## Configuration Options

### Aspect Ratios

`1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

### Image Sizes (Nano Banana Pro only)

| Size | Max Resolution | Tokens | Use Case |
|------|----------------|--------|----------|
| `1K` | ~1344px | 1120 | Thumbnails, previews |
| `2K` | ~2688px | 1120 | Social media, web |
| `4K` | ~6336px (21:9) | 2000 | Print, professional |

### Response Modalities

- `["IMAGE"]` - Image only (faster)
- `["TEXT", "IMAGE"]` - Both text explanation and image

## Prompting Best Practices

Use natural language (not tag-style prompts). Be hyper-specific: describe context/intent, photography terms, textures/materials.

```python
prompt = """Close-up portrait of an elderly craftsman for a documentary magazine,
illuminated by soft golden hour light from a nearby window,
captured with 85mm portrait lens at f/1.8,
weathered hands with visible skin texture,
shallow depth of field with bokeh background"""
```

## Text Rendering

Nano Banana Pro excels at accurate text in images.

### Python

```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="""Create a modern, minimalist logo for "AURORA TECH"

    The text should be in a clean, bold sans-serif font.
    Include a subtle abstract aurora/northern lights motif.
    Color scheme: deep blue gradient with cyan accents.
    White background for versatility.""",
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="2K",
        ),
    ),
)
```

### Infographics

```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="""Create a polished editorial infographic about coffee brewing methods.

    Include 4 sections: Pour Over, French Press, Espresso, Cold Brew.
    Each section should have: an icon, brewing time, and grind size.
    Style: clean, modern, muted earth tones.
    Title at top: "The Art of Coffee Brewing"
    Include charts comparing flavor profiles.""",
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="9:16",
            image_size="2K",
        ),
    ),
)
```

## Reference Images

### Style Transfer

```python
from PIL import Image

style_ref = Image.open("impressionist_painting.jpg")

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        style_ref,
        """Transform this photograph into the artistic style shown.
        Preserve the original composition and subject placement.
        Render with visible brushstrokes and vibrant color palette.
        Maintain the dreamy, atmospheric quality of the reference.""",
        Image.open("photo_to_transform.jpg"),
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(image_size="2K"),
    ),
)
```

### Character Consistency

```python
# Up to 14 reference images, 5 humans with high fidelity
person_ref = Image.open("person_reference.jpg")

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        person_ref,
        """Create a studio portrait of this exact person.
        Keep their facial features exactly the same.
        New pose: profile view looking right.
        Expression: confident smile.
        Lighting: professional three-point studio setup.
        Background: clean white seamless.""",
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="3:4",
            image_size="2K",
        ),
    ),
)
```

### TypeScript Reference Images

```typescript
import fs from "fs";

const styleRef = fs.readFileSync("style_reference.png").toString("base64");
const photoRef = fs.readFileSync("photo.jpg").toString("base64");

const response = await ai.models.generateContent({
  model: "gemini-3-pro-image-preview",
  contents: [
    { inlineData: { mimeType: "image/png", data: styleRef } },
    { inlineData: { mimeType: "image/jpeg", data: photoRef } },
    { text: "Transform the second image into the artistic style of the first image" },
  ],
  config: {
    responseModalities: ["IMAGE"],
    imageConfig: { imageSize: "2K" },
  },
});
```

## Image Editing

Use semantic instructions - no manual masking needed.

### Python

```python
original = Image.open("room_photo.jpg")

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        original,
        """Edit this room:
        - Change the wall color to warm terracotta
        - Replace the sofa with a mid-century modern design in olive green
        - Add a large monstera plant in the corner
        - Keep the lighting and perspective exactly the same""",
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(image_size="2K"),
    ),
)
```

### Object Removal

```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        Image.open("beach_photo.jpg"),
        "Remove all people from this beach scene, fill naturally with sand and water",
    ],
    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
)
```

## Mixed Text and Image Output

```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="Design a logo for a sustainable coffee brand called 'Green Bean' and explain your design choices",
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="2K",
        ),
    ),
)

for part in response.parts:
    if part.text:
        print("Design rationale:", part.text)
    elif image := part.as_image():
        image.save("logo.png")
```

## Multi-Turn Conversations (Iterative Refinement)

Edit over re-rolling - request specific changes conversationally.

### Python

```python
chat = client.chats.create(
    model="gemini-3-pro-image-preview",
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="2K",
        ),
    ),
)

# Initial generation
response1 = chat.send_message("Create a cozy cabin interior with a fireplace, evening lighting")

# Iterative refinement - don't regenerate from scratch
response2 = chat.send_message("Add a sleeping cat curled up on the armchair")
response3 = chat.send_message("Make the fire brighter and add some snow visible through the window")
```

### TypeScript

```typescript
const chat = ai.chats.create({
  model: "gemini-3-pro-image-preview",
  config: {
    responseModalities: ["TEXT", "IMAGE"],
    imageConfig: { aspectRatio: "16:9", imageSize: "2K" },
  },
});

const response1 = await chat.sendMessage("Create a cozy cabin interior with a fireplace");
const response2 = await chat.sendMessage("Add a sleeping cat on the armchair");
```

## Google Search Grounding

Generate images based on real-time information.

```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="Create an infographic showing today's weather forecast for Tokyo",
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="9:16"),
        tools=[{"google_search": {}}],
    ),
)
```

## Layout Control with Wireframes

Upload sketches to define exact placement.

```python
wireframe = Image.open("ui_wireframe.png")

response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[
        wireframe,
        """Transform this wireframe into a high-fidelity mobile app UI.
        Style: Modern fintech app, dark mode.
        Follow the exact layout and element placement.
        Add realistic content, icons, and typography.""",
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="9:16",
            image_size="2K",
        ),
    ),
)
```

## Batch Generation with Concurrency

### Python

```python
import asyncio
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

async def generate_one(prompt: str, index: int) -> dict:
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config={"response_modalities": ["IMAGE"], "image_config": {"image_size": "2K"}},
        )
        image = response.parts[0].as_image()
        return {"index": index, "image": image}
    except Exception as e:
        return {"index": index, "error": str(e)}

async def generate_batch(prompts: list[str], max_concurrent: int = 4) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(prompt: str, idx: int):
        async with semaphore:
            return await generate_one(prompt, idx)

    return await asyncio.gather(*[limited(p, i) for i, p in enumerate(prompts)])

# Usage
results = asyncio.run(generate_batch([
    "A red sports car on a mountain road",
    "A blue sailboat at sunset",
    "A green forest path in autumn",
]))
```

### TypeScript

```typescript
async function generateBatch(prompts: string[], maxConcurrent = 4) {
  const semaphore = { count: maxConcurrent };

  const acquire = async () => {
    while (semaphore.count <= 0) await new Promise(r => setTimeout(r, 100));
    semaphore.count--;
  };
  const release = () => { semaphore.count++; };

  const generateOne = async (prompt: string, index: number) => {
    await acquire();
    try {
      const response = await ai.models.generateContent({
        model: "gemini-3-pro-image-preview",
        contents: [{ text: prompt }],
        config: { responseModalities: ["IMAGE"], imageConfig: { imageSize: "2K" } },
      });
      const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      return { index, data };
    } catch (error) {
      return { index, error: error instanceof Error ? error.message : "Failed" };
    } finally {
      release();
    }
  };

  return Promise.all(prompts.map((p, i) => generateOne(p, i)));
}
```

## Retry with Exponential Backoff

### Python

```python
import time

def generate_with_retry(prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=prompt,
                config={"response_modalities": ["IMAGE"], "image_config": {"image_size": "2K"}},
            )
            return response.parts[0].as_image()
        except Exception as e:
            err = str(e).lower()
            retryable = any(x in err for x in ["503", "rate", "overloaded", "resource_exhausted"])
            if not retryable or attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return None
```

## Usage Metadata

```python
response = client.models.generate_content(...)

usage = response.usage_metadata
print(f"Input tokens: {usage.prompt_token_count}")
print(f"Output tokens: {usage.candidates_token_count}")
print(f"Total: {usage.total_token_count}")

# Detailed breakdown by modality
for detail in usage.prompt_tokens_details or []:
    print(f"  {detail.modality}: {detail.token_count}")
```

## Common Use Cases

### Product Photography

```python
prompt = """High-resolution studio product photograph of a luxury watch.
Three-point softbox lighting setup on pure white background.
Sharp focus on dial details, subtle reflections on polished steel.
Professional e-commerce style, ultra-realistic."""
```

### YouTube Thumbnails

```python
prompt = """Eye-catching YouTube thumbnail for a cooking video.
Subject: Sizzling steak on cast iron pan with dramatic flame.
Text overlay: "PERFECT SEAR" in bold yellow Impact font with black outline.
High contrast, vibrant colors, appetizing food photography."""
```

### Social Media Graphics

```python
prompt = """Instagram story graphic for a fitness brand.
Bold geometric shapes in coral and teal.
Text: "30 DAY CHALLENGE" in modern sans-serif.
Clean, energetic, Gen-Z aesthetic.
Include space for profile tag at bottom."""
```

### Storyboard Sequence

```python
# Generate consistent character across multiple scenes
character_ref = Image.open("hero_character.png")

scenes = [
    "walking through a forest, wide shot",
    "discovering a hidden door, medium shot",
    "entering a magical realm, close-up of amazed expression",
]

for i, scene in enumerate(scenes):
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            character_ref,
            f"Scene {i+1}: This exact character {scene}. Maintain consistent appearance, clothing, and features.",
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", image_size="2K"),
        ),
    )
```

## Configuration Reference

| Parameter | Python | TypeScript | Values |
|-----------|--------|------------|--------|
| Output types | `response_modalities` | `responseModalities` | `["IMAGE"]`, `["TEXT", "IMAGE"]` |
| Aspect ratio | `image_config.aspect_ratio` | `imageConfig.aspectRatio` | 10 ratios |
| Resolution | `image_config.image_size` | `imageConfig.imageSize` | `1K`, `2K`, `4K` |
| Search grounding | `tools=[{"google_search": {}}]` | `tools: [{ googleSearch: {} }]` | - |

**Pricing (Dec 2025):** Nano Banana ~$0.04 | Pro 1K-2K ~$0.13 | Pro 4K ~$0.24

All outputs include invisible SynthID watermarks (cannot be disabled).
