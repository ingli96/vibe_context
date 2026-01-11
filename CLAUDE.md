# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Claude Code Skills marketplace containing plugins with domain-specific Skills. Skills are loaded into Claude Code to provide specialized knowledge for tasks like AI development and third-party integrations.

## Architecture

```
.claude-plugin/
  marketplace.json     # Plugin registry - defines plugins and their skills
skills/
  skill-creator/       # Meta-skill for creating new Skills
  ai-engineer/         # AI/LLM development skills
    openai-api/        # OpenAI Responses API patterns
    google-ai/         # (placeholder)
  integrations/        # Third-party service skills (placeholders)
    supabase/
    stripe/
    sentry/
    vercel/
```

## Key Concepts

**Skills** are directories containing a `SKILL.md` file with YAML frontmatter (`name`, `description`) and markdown instructions. Claude auto-loads Skills when user requests match the description.

**Plugins** group related Skills and are defined in `marketplace.json`. Each plugin entry specifies a name, description, and array of skill paths.

## Adding a New Skill

1. Create directory: `skills/<category>/<skill-name>/`
2. Create `SKILL.md` with required frontmatter:
   ```yaml
   ---
   name: skill-name
   description: What it does. Use when [trigger keywords].
   ---
   ```
3. Add skill path to appropriate plugin in `.claude-plugin/marketplace.json`

## Skill Naming Rules

- Max 64 characters
- Lowercase letters, numbers, hyphens only
- Cannot contain "anthropic" or "claude"
- Prefer gerund form: `processing-pdfs`, `generating-commits`

## SKILL.md Guidelines

- Keep under 500 lines; use reference files for detailed docs
- Description must use third person (not "I can" or "You can")
- Include trigger keywords in description for proper discovery
- Reference files should be one level deep from SKILL.md
