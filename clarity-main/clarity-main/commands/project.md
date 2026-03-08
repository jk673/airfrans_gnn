---
model: claude-opus-4-6
description: Summarise a project in plain language. Can get large — use on small/medium projects or specify a focus area.
argument-hint: [optional: project path or focus area]
---

# /clarity:project

Review the full project and produce a plain-language summary.

## Prerequisites

There must be a project in the current working directory, or the user must specify a path. If the directory is empty or contains no recognisable project files, say so.

## Full Workflow

Follow every step in order. Refer to the review-methodology skill as the governing document throughout. Refer to the scope-management skill for how to handle project size.

### Step 1: First-run check

Same as `/clarity:session` Step 1. Check for `${CLAUDE_PLUGIN_ROOT}/.config.json` and prompt for save location if not configured.

### Step 2: Disclaimer

Display before any review content:

> **Clarity by Ashby**
>
> Clarity is Claude reviewing Claude's work. It translates technical structure into plain language, identifies risks, and flags things worth questioning. This review covers the full project, not just a single session. Treat this as a helpful second perspective, not an authoritative audit.

### Step 3: Structure scan (Pass 1)

Follow the scope-management skill's two-pass approach.

Read:
1. Full file tree (excluding skip directories defined in scope-management skill)
2. Package files (package.json, requirements.txt, etc.)
3. Configuration files (.env.example, config files, CI/CD, docker files)
4. README and documentation
5. Git history (recent commits)

Produce a mental model of the project's architecture. Count total source files.

### Step 4: Size check

If the project has more than 100 source files (per the scope-management skill):

> This project has [n] source files. A full detailed review would be very long. Would you prefer:
> - **Full sweep** — I review everything, focusing most detail on high-priority files
> - **Focused review** — Tell me which area to focus on (e.g., "the authentication system", "the API endpoints", "recent changes")

Wait for the user's response before proceeding. If the response is ambiguous, offer examples: "Full sweep" or "Focused review on [area]".

If the user provided a focus area as an argument (e.g., `/clarity:project authentication`), treat this as a focused review without asking.

### Step 5: Detailed review (Pass 2)

Read source files in the priority order defined by the scope-management skill:

1. Entry points
2. Security-sensitive files
3. Configuration and environment
4. Core business logic
5. Data handling
6. Recently modified files
7. Tests
8. Other files (utilities, helpers, shared components) — summary level unless they appear in multiple categories above

Adjust depth based on the user's choice in Step 4:
- **Full sweep:** High-priority files get detailed review; lower-priority files get one-sentence summaries
- **Focused review:** Apply full detail to the specified area only; mention other areas at summary level

### Step 6: Produce the review

Using the template at `${CLAUDE_PLUGIN_ROOT}/templates/project-review.md`, produce a review covering all sections. Apply the review-methodology skill for language and tone. Apply the explanation-frameworks skill for all technical explanations.

The review must cover these sections in order:

1. **What This Project Is** — One paragraph, no jargon. What does it do? Who is it for?

2. **How It's Built** — Architecture in simple terms. Use the architecture description guidance from the explanation-frameworks skill. What are the main parts and how do they connect? Think floor plan, not wiring diagram.

3. **What It Depends On** — Every external library, service, API, or tool. For each: what it does, maintenance status, and what happens if it disappears. Use the dependency explanation format from the explanation-frameworks skill. Group by category (core framework, utilities, development tools, external services).

4. **Data and Privacy** — Where does data go? What is stored locally? What is sent externally? Are there API keys or credentials in the code? Is anything sensitive exposed? Be specific about every data flow you can identify.

5. **State of Completion** — What works, what is half-built, what is missing. Be honest about rough edges. If the project has a roadmap, backlog, or TODO comments, reference them.

6. **Risks and Concerns** — Same rules as session review but project-wide. Follow the risk assessment rules in the review-methodology skill. Cover: security, fragility, missing tests, single points of failure, dependency risks, things that would worry a careful reviewer. This is the most important section.

7. **If You Were Handing This to Someone Else** — What would they need to know? What is not obvious from the code? What would trip them up? What decisions were made that a new person would not understand without context? This section is for anyone inheriting or maintaining the project.

8. **What to Ask Claude Next** — 2-3 specific, paste-ready prompts. Follow the rules in the review-methodology skill.

9. **Jargon Buster** — Every technical term used in the review. Follow the rules in the explanation-frameworks skill.

10. **Worth Understanding** — 1-2 concepts. Follow the rules in the review-methodology skill.

### Step 7: Save and display

1. **Display the full review in the terminal** so the user can read it immediately
2. **Save a markdown file** using the filename format: `clarity-project-YYYY-MM-DD.md`
3. Save to the configured location or `clarity/` in the current working directory
4. Create the directory if it does not exist. If directory creation fails (permissions, invalid path), save the review to the current working directory instead and inform the user: "Could not create [path]. Review saved to [fallback path] instead. Check your save location with `/clarity:config`."
5. Confirm the file location to the user

### Step 8: End marker

End every project review with:

---
*End of Clarity review. Ask a follow-up or continue your session.*

Do NOT end with "Anything you want me to dig deeper into?" or similar — the end marker already tells the user they can ask follow-ups.

## Coverage statement

Every project review must include a coverage statement in the frontmatter (as shown in the template), immediately after the date:

> **Files reviewed:** [n] of [total] source files in detail
> **Files scanned:** [n] at summary level
> **Not reviewed:** [areas skipped and why. If all relevant areas were reviewed, write "None".]

## Language rules

Throughout this entire workflow, follow the language rules in the review-methodology skill. Every technical term must be defined. Every risk must be specific and consequence-led. No code blocks unless they help the user understand what to check or change.
