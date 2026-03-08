---
model: claude-opus-4-6
description: No argument = explain last message. Or give a file path, error, or concept for a plain-language explanation.
argument-hint: [file path, error message, or concept]
---

# /clarity:explain

Explain one thing in plain language.

## Purpose

This is the quick version of Clarity. No full review, no saved file, no structure. The user points at something and gets a short explanation they can understand.

## What it accepts

- **No argument (most common)** — explain Claude's most recent response
- **A file path** — explain what this file does and how it fits into the project
- **An error message** — explain what went wrong and what to do about it
- **A terminal output** — explain what just happened
- **A plan** — explain what Claude is proposing and what it commits you to
- **A concept or term** — explain it in plain language
- **A piece of code shown in the session** — explain what it does and why it matters

## Behaviour

### Step 1: Identify the target

Determine what the user wants explained:

1. If no argument is given, explain Claude's most recent response.
2. If a file path is provided, read the file.
3. If an error message or terminal output is quoted, explain that.
4. If a concept or term is provided, explain that.
5. If genuinely unclear, ask: "What would you like me to explain?"

### Step 2: Explain it

Use the language rules from the review-methodology skill (plain language, no jargon without inline definitions, specific consequences over abstract categories). Do NOT apply the full framework patterns from explanation-frameworks — those are for session and project reviews. Keep it light.

**Hard target: 3-8 sentences for most explanations.** Only go longer if the thing being explained genuinely requires it (e.g., a complex plan with multiple moving parts). When in doubt, shorter is better.

The explanation must:

- Start with what the thing IS or DOES (one sentence)
- Then cover only what's relevant from the list below — skip anything that doesn't apply
- If you name a technology, tool, or concept, define it in the same sentence. No bare name-drops.
- Be concise — this is a quick answer, not a report

**For each type, cover only what's relevant:**

- **Last response:** What Claude just did. What files changed and what each change means. Whether anything risky or worth questioning happened. Skip if nothing interesting to say.
- **Files:** What the file does. How it fits into the project. Anything noteworthy.
- **Errors:** What went wrong. Why. What to do next (a paste-ready prompt for Claude).
- **Terminal output:** What happened (success/failure/warning). Whether it requires action.
- **Plans:** What is being proposed. What it commits you to. What to question before approving.
- **Concepts:** What it means. Why it matters for this project.

### Step 3: End marker

End every explanation with:

---
*End of Clarity review. Ask a follow-up or continue your session.*

## Output rules

- Print the explanation directly in the terminal
- Do NOT save a file — this is quick and informal
- Do NOT produce a full review structure (no headers, no jargon buster, no "Worth Understanding")
- Aim for 3-8 sentences. A few paragraphs at most for complex topics.
- Use the language rules from the review-methodology skill throughout
- Do NOT end with "Want me to go deeper?" or similar — the end marker already tells the user they can ask follow-ups

## What this is not

- Not a full session review — use `/clarity:session` for that
- Not a full project review — use `/clarity:project` for that
- Not a fixer — if the user needs something fixed, suggest a prompt for a new session
