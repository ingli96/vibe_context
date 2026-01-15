# Vibe Context

A Claude Code Skills marketplace containing plugins with domain-specific Skills for building mini SaaS products.

## What are Skills?

Skills are directories containing a `SKILL.md` file that teach Claude domain knowledge, workflows, and best practices. Claude automatically loads Skills when user requests match the skill's description.

## Available Plugins

### claude-skiller

Meta-skill for creating new Claude Code Skills with proper structure and best practices.

**Skills:**
- `skill-creator` - Scaffolds new Skills, explains structure, helps troubleshoot existing Skills

### ai-engineer

Skills for AI/LLM development including API patterns and observability.

**Skills:**
- `openai-api` - OpenAI Responses API patterns, structured output, async pipelines
- `google-image` - Google image generation with Nano Banana models
- `llm-tracing` - LLM observability with Langfuse and OpenTelemetry

### integrations

Third-party service integrations (coming soon): Supabase, Stripe, Sentry, Vercel.

## Repository Structure

```
vibe-context/
├── .claude-plugin/
│   └── marketplace.json      # Plugin registry
├── plugins/
│   ├── ai-engineer/
│   │   └── skills/
│   │       ├── openai-api/
│   │       ├── google-image/
│   │       └── llm-tracing/
│   ├── claude-skiller/
│   │   └── skills/
│   │       └── skill-creator/
│   └── integrations/
│       └── skills/
└── CLAUDE.md                 # Project instructions
```

## Adding a New Skill

1. Create directory: `plugins/<plugin-name>/skills/<skill-name>/`
2. Create `SKILL.md` with required frontmatter:
   ```yaml
   ---
   name: skill-name
   description: What it does. Use when [trigger keywords].
   ---
   ```
3. Add skill path to the appropriate plugin in `.claude-plugin/marketplace.json`

## Skill Naming Rules

- Max 64 characters
- Lowercase letters, numbers, hyphens only
- Cannot contain "anthropic" or "claude"
- Prefer gerund form: `processing-pdfs`, `generating-commits`

## License

MIT
