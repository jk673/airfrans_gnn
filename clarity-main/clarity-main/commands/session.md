---
model: claude-opus-4-6
description: Summarise this session in plain language. No args needed, or specify a focus area.
argument-hint: [optional: focus area]
---

# /clarity:session

Review the current session's work and produce a plain-language summary.

## Prerequisites

There must be recent Claude Code activity in the current session — files created, modified, deleted, or plans discussed. If there is nothing to review, say so.

## Full Workflow

Follow every step in order. Refer to the review-methodology skill as the governing document throughout.

### Step 1: First-run check

Check whether `${CLAUDE_PLUGIN_ROOT}/.config.json` exists and contains a save location.

If no configuration exists, prompt:

> **First-time setup:** Where should Clarity save review files?
> Enter a path to a folder. This can be changed later with `/clarity:config`.

If the user provides a path, save it. If they skip, use a `clarity/` directory in the current working directory. Only prompt on first use.

### Step 2: Disclaimer

Display before any review content:

> **Clarity by Ashby**
>
> Clarity is Claude reviewing Claude's work. It translates technical changes into plain language, identifies risks, and flags things worth questioning. It does not judge code quality or recommend fixes. Treat this as a helpful second perspective, not an authoritative audit.

### Step 3: Gather session data

Collect information from multiple sources. See the scope-management skill for detailed guidance.

**3a. Git changes (primary source)**

If the project uses git:
- Run `git diff` to see current uncommitted changes
- Run `git log` with today's date or session timeframe to identify committed changes
- List all files created, modified, or deleted

If the project does not use git, rely on conversation history and file system inspection.

**3b. Conversation history**

Read available conversation context for:
- Decisions made (framework choices, architecture, approach)
- Reasoning discussed (why choices were made)
- Plans or intentions stated but not yet implemented
- Problems encountered and how they were resolved

If earlier conversation context appears to be compressed or missing, note this limitation.

**3c. Focus area**

If the user provided an argument (e.g., `/clarity:session authentication`), focus the review on that area. Still mention other changes briefly but concentrate detail on the specified focus.

### Step 4: Produce the review

Using the templates at `${CLAUDE_PLUGIN_ROOT}/templates/session-review.md`, produce a review covering all sections. Apply the review-methodology skill for language and tone. Apply the explanation-frameworks skill for all technical explanations.

The review must cover these sections in order:

1. **What Was Decided** — Plans, approaches, choices made. No jargon. No code.

2. **What Changed** — Table of files created, modified, or deleted. For each: what the file does, what changed, and why. Use the explanation-frameworks skill for file change descriptions.

3. **Why Those Choices** — For every significant decision: what was chosen, why, what the alternatives were, and whether it locks the project in or is easy to change. Use the decision explanation format from the explanation-frameworks skill.

4. **New Dependencies** — Anything external the project now relies on. For each: what it is, why it was added, maintenance status, and what happens if it breaks. Use the dependency explanation format from the explanation-frameworks skill. If no new dependencies were added, state this.

5. **Risks and Gaps** — The most important section. Follow the risk assessment rules in the review-methodology skill. Be direct and specific. Cover: security, error handling, test coverage, production readiness, unvalidated assumptions. Never omit or soften this section.

6. **Test Coverage** — If tests were written or discussed: what they check (in plain language), what they do not check, and how to run them. If no tests exist or were discussed, state this and explain what it means.

7. **What to Ask Claude Next** — 2-3 specific, paste-ready prompts the user can take into a new session. Reference specific issues found in the review. Follow the rules in the review-methodology skill.

8. **Jargon Buster** — Every technical term used in the review, alphabetical, one-sentence definitions. Follow the jargon buster rules in the explanation-frameworks skill.

9. **Worth Understanding** — 1-2 concepts from this review that would help the user build better judgement. Follow the rules in the review-methodology skill.

### Step 5: Save and display

1. **Display the full review in the terminal** so the user can read it immediately
2. **Save a markdown file** using the filename format: `clarity-session-YYYY-MM-DD-HHMMSS.md`
3. Save to the configured location (from Step 1) or the `clarity/` directory in the current working directory
4. Create the directory if it does not exist. If directory creation fails (permissions, invalid path), save the review to the current working directory instead and inform the user: "Could not create [path]. Review saved to [fallback path] instead. Check your save location with `/clarity:config`."
5. Confirm the file location to the user

### Step 6: End marker

End every session review with:

---
*End of Clarity review. Ask a follow-up or continue your session.*

Do NOT end with "Anything you want me to dig deeper into?" or similar — the end marker already tells the user they can ask follow-ups.

## Language rules

Throughout this entire workflow, follow the language rules in the review-methodology skill. Every technical term must be defined. Every risk must be specific and consequence-led. No code blocks unless they help the user understand what to check or change.
