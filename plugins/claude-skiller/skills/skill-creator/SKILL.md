---
name: skill-creator
description: Creates Claude Code Skills with proper structure and best practices. Use when user wants to create a new Skill, scaffold a Skill directory, or learn about Skill structure. Also use when updating, improving, or troubleshooting existing Skills.
---

# Creating Claude Code Skills

Skills are directories with `SKILL.md` files that teach Claude domain knowledge, workflows, and best practices. Claude automatically loads Skills when user requests match the Skill's description.

## This Repository's Structure

This marketplace uses **isolated plugin roots** to ensure each plugin only loads its own skills:

```
vibe-context/
├── .claude-plugin/
│   └── marketplace.json      # Plugin registry
└── plugins/
    ├── ai-engineer/          # LLM APIs, AI SDKs
    │   └── skills/
    │       └── openai-api/
    ├── claude-skiller/       # Skill creation tooling
    │   └── skills/
    │       └── skill-creator/
    └── integrations/         # Third-party services
        └── skills/
            └── (supabase, stripe, etc.)
```

**Why isolated roots?** Claude auto-loads ALL skills under `skills/` at the plugin root. The `skills` array in marketplace.json **supplements** defaults—it doesn't replace them. With `source: "./"`, installing one plugin loads every skill. Isolated roots prevent this.

### Adding a Skill to This Repo

1. **Choose target plugin**: `ai-engineer`, `claude-skiller`, or `integrations`
2. **Create directory**: `mkdir -p plugins/<plugin>/skills/<skill-name>`
3. **Create SKILL.md** with required frontmatter (see below)
4. **Update marketplace.json** - add `"./skills/<skill-name>"` to the plugin's `skills` array

```bash
# Example: Adding google-ai to ai-engineer
mkdir -p plugins/ai-engineer/skills/google-ai

# Example: Adding supabase to integrations
mkdir -p plugins/integrations/skills/supabase
```

## How Skills Work

Skills are **model-invoked**: Claude decides when to use them based on the request. You don't explicitly call a Skill - Claude automatically applies relevant Skills when the request matches their description.

### Progressive Loading (Three Levels)

| Level | When Loaded | Token Cost | Content |
|-------|------------|------------|---------|
| **Metadata** | Always (at startup) | ~100 tokens | `name` and `description` from YAML |
| **Instructions** | When Skill triggers | Under 5k tokens | SKILL.md body |
| **Resources** | As needed | Unlimited | Reference files, scripts |

The context window is shared with conversation history, other Skills, and your request. Keep SKILL.md under 500 lines; put detailed docs in separate files.

## Skill Directory Structure

```
my-skill/
└── SKILL.md              # Required - the only required file
```

**Optional additions** (only when SKILL.md would exceed 500 lines):
```
my-skill/
├── SKILL.md
├── references/
│   └── detailed-api.md   # Loaded when user asks about advanced topics
└── scripts/
    └── validate.py       # Executed without loading into context
```

## Required YAML Fields

### `name`

| Requirement | Details |
|-------------|---------|
| Max length | 64 characters |
| Allowed | Lowercase letters (a-z), numbers (0-9), hyphens (-) |
| Prohibited | Uppercase, underscores, spaces, XML tags |
| Reserved | Cannot contain "anthropic" or "claude" |

**Naming**: Use gerund form: `processing-pdfs`, `generating-commits`, `reviewing-code`

### `description`

| Requirement | Details |
|-------------|---------|
| Max length | 1024 characters |
| Point of view | Third person only (never "I can" or "You can") |
| Structure | `[Capabilities]. Use when [triggers].` |

The description is **critical for discovery**. Include specific capabilities AND trigger keywords:

```yaml
# Good - specific + triggers
description: Generates Supabase queries, auth flows, RLS policies, and edge functions. Use when working with Supabase, Supabase Auth, or PostgreSQL via Supabase.

# Bad - too vague
description: Helps with databases.
```

### Optional YAML Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `allowed-tools` | Restrict available tools | `Read, Grep, Glob` |
| `model` | Override model | `sonnet`, `haiku`, `opus` |
| `context: fork` | Run in isolated context | Use with `agent` field |
| `agent` | Agent type when forked | `Explore`, `Plan`, `general-purpose` |
| `user-invocable` | Show in slash menu | `true` (default) or `false` |

## SKILL.md Format

```yaml
---
name: skill-name
description: [Capabilities]. Use when [triggers].
---

# Skill Title

## DO NOT USE (Deprecated)
[Outdated patterns to avoid - prevents Claude from using old APIs/syntax]

## Quick Reference
[Tables for key data: endpoints, pricing tiers, config options, etc.]

## Quick Start
[Minimal working code - copy-paste ready, multiple languages if relevant]

## Core Patterns
[Main workflows, organized basic → advanced]

## Constraints / Limits
[Rate limits, size limits, required fields, etc.]

## References
[Link to extended docs for advanced topics]
```

**Effective patterns (from openai-api skill):**
- **DO NOT USE first** - Stops outdated patterns immediately
- **Tables for reference data** - Quick lookup (tiers, endpoints, options)
- **Decision helpers after tables** - "Use X for Y, use Z for W"
- **Python and TypeScript examples** - Both languages for each pattern
- **Comprehensive examples** - Full working code, not just snippets
- **Progressive complexity** - Basic setup → common operations → advanced
- **Constraints section** - Limits Claude must respect
- **Up-to-date documentation** - Current APIs, latest SDK versions, modern patterns
- **Single reference link** - Keep SKILL.md focused, overflow to `references/`

## Writing Effective Instructions

### Principle 1: Only Add What Claude Doesn't Know

Claude already knows general programming. Only include specific patterns, current APIs, or project conventions:

````markdown
<!-- GOOD - specific library pattern Claude might not know -->
Use pdfplumber for extraction:
```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```

<!-- BAD - generic knowledge Claude already has -->
PDF (Portable Document Format) files are a common file format...
````

### Principle 2: Match Detail to Risk

**High-risk** (data loss, security) → exact commands, no flexibility:
````markdown
## Database Migration
```bash
supabase db push --dry-run  # Always preview first
supabase db push            # Then apply
```
Never skip the dry-run.
````

**Low-risk** (exploration, formatting) → general guidance:
```markdown
## Exploring Schema
Query information_schema or use the dashboard to understand tables.
```

### Principle 3: Use Annotated Examples

Comments explain *why*, not just *what*:

````markdown
```typescript
// 1. Server-side client for auth persistence
const supabase = createServerClient(url, key, { cookies })

// 2. Always verify session before protected operations
const { data: { session } } = await supabase.auth.getSession()
if (!session) return unauthorized()

// 3. RLS automatically scopes queries to authenticated user
const { data } = await supabase.from('user_data').select('*')
```
````

### Principle 4: Checklists for Multi-Step Tasks

````markdown
## Deployment Workflow

```
- [ ] Run tests locally: `npm test && npm run lint`
- [ ] Create PR and get approval
- [ ] Deploy to staging and verify
- [ ] Deploy to production
```
````

### Principle 5: Feedback Loops for Quality-Critical Tasks

```markdown
## Editing Process
1. Make edits
2. Validate: `python scripts/validate.py`
3. If errors → fix and re-validate
4. Only proceed when validation passes
```

## Progressive Disclosure Patterns

When content exceeds 500 lines, split into reference files:

````markdown
# Main SKILL.md (~300 lines)

## Quick Start
[Copy-paste code]

## Core Patterns
[Most common operations]

## Advanced Topics
- **Auth patterns**: See [references/auth.md](references/auth.md)
- **RLS policies**: See [references/rls.md](references/rls.md)
````

Claude loads reference files only when user asks about those topics.

### Keep References One Level Deep

```markdown
# Bad - Claude may not fully read deeply nested files
SKILL.md → advanced.md → details.md → actual-info.md

# Good
SKILL.md → auth.md
SKILL.md → rls.md
```

## Complete Example

```yaml
---
name: code-review
description: Reviews code for quality, security, and maintainability. Use when reviewing PRs, checking code, or asking for code review.
---

# Code Review

## Quick Start
Run `git diff` to see changes, then analyze against the checklist.

## Review Checklist
- [ ] Readable code with clear names
- [ ] No hardcoded secrets
- [ ] Input validation at boundaries
- [ ] Appropriate error handling
- [ ] Tests cover changes

## Output Format
**Critical (must fix):** Security issues, data loss risks
**Warning (should fix):** Performance, missing error handling
**Suggestion:** Readability improvements

## Example
Input: Review user-auth.ts

Output:
### Critical
- Line 45: Password in plain text → use bcrypt

### Warning
- Line 23: Missing error handling for DB query

### Suggestion
- Lines 12-18: Extract to separate function
```

## Marketplace Configuration

### This Repo's marketplace.json

```json
{
  "name": "vibe-context",
  "plugins": [
    {
      "name": "ai-engineer",
      "source": "./plugins/ai-engineer",
      "skills": ["./skills/openai-api"]
    },
    {
      "name": "integrations",
      "source": "./plugins/integrations",
      "skills": []
    }
  ]
}
```

### Critical: Why Isolated Plugin Roots

**Wrong** - shared source loads ALL skills:
```json
{ "name": "plugin-a", "source": "./", "skills": ["./skills/a"] }
{ "name": "plugin-b", "source": "./", "skills": ["./skills/b"] }
// Installing plugin-a loads BOTH a and b!
```

**Correct** - isolated roots:
```json
{ "name": "plugin-a", "source": "./plugins/plugin-a", "skills": ["./skills/a"] }
{ "name": "plugin-b", "source": "./plugins/plugin-b", "skills": ["./skills/b"] }
// Each plugin only loads its own skills
```

### Testing

```bash
# Test locally
claude --plugin-dir ./plugins/ai-engineer

# Debug skill loading
claude --debug
```

## Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Vague descriptions | Won't trigger | Include specific trigger keywords |
| Too verbose | Wastes tokens | Only add what Claude doesn't know |
| Shared plugin source | All skills load | Use isolated plugin roots |
| Deep nesting | Partial reads | Keep references one level deep |
| "I can help..." | Discovery issues | Use third person only |

## Testing Checklist

- [ ] `name` is lowercase, hyphens only
- [ ] `description` uses third person + trigger keywords
- [ ] SKILL.md under 500 lines
- [ ] Quick Start has copy-paste code
- [ ] Common Mistakes section included
- [ ] References one level deep
- [ ] Added to correct plugin in marketplace.json
- [ ] Tested with `claude --plugin-dir`
