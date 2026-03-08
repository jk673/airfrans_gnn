---
model: claude-opus-4-6
description: Display Clarity command reference and disclaimer.
---

# /clarity:help

Display the Clarity command reference.

## Behaviour

Display the following content (render as markdown):

---

**Clarity by Ashby** — Plain-language review of AI development work

> Clarity is Claude reviewing Claude's work. It translates technical changes into plain language, identifies risks, and flags things worth questioning. It does not judge code quality or recommend fixes. Treat this as a helpful second perspective, not an authoritative audit.

### Commands

| Command | Purpose |
|---------|---------|
| `/clarity:explain` | Explain one thing — a file, error, output, or concept — in plain language. Quick and informal. |
| `/clarity:session` | Review what happened in the current session. Explains decisions, changes, dependencies, and risks in plain language. |
| `/clarity:project` | Review the full project. Explains what it is, how it works, what it depends on, and where the risks are. |
| `/clarity:config` | Set where Clarity saves review files. |
| `/clarity:help` | Show this reference. |

### Quick start

1. Build something with Claude Code
2. Run `/clarity:explain` to understand something specific — a file, an error, or a concept
3. Run `/clarity:session` to understand what just happened in this session
4. Or run `/clarity:project` to understand the whole project
5. Use the "What to Ask Claude Next" prompts in a new session

### What Clarity does

Clarity reads what Claude Code built, changed, or planned, and explains it in plain language. It identifies risks, explains decisions, translates jargon, and gives you specific prompts for follow-up. It is designed for people who build with AI but are not programmers.

### What Clarity does not do

- It does not fix code or suggest fixes
- It does not score or rate code quality
- It does not replace professional code review
- It does not guarantee it caught everything — it is Claude reviewing Claude's work

### Data handling

All processing runs locally. Review files are saved to your configured location (set via `/clarity:config`). No data is transmitted externally.

### About

Clarity is built by Ashby (ashby.systems). Apache 2.0 license.

---
