---
name: scope-management
description: Defines how Clarity manages review scope for sessions and projects of varying sizes. Covers git awareness for session reviews, the two-pass approach for project reviews, prioritisation rules, and transparency about coverage limits.
user-invocable: false
---

# Clarity Scope Management

## Purpose

Clarity reviews must be thorough. If you cannot review everything, say exactly what you reviewed and what you skipped. A review that claims to cover everything but silently skips files is worse than one that explicitly states what was and was not examined. This skill defines how to manage scope in both session and project reviews.

## Session Review Scope

### Primary sources (in order of reliability)

1. **Git changes** — Check if git is available by running `git rev-parse --git-dir 2>/dev/null`. If this succeeds, run `git diff` and `git log` to identify files created, modified, or deleted during the session. This is the most reliable source of what actually changed. If git is not available, proceed with conversation history and file system inspection.

2. **Conversation history** — Read the available conversation context for decisions, reasoning, and discussion that led to the changes.

3. **File system** — Read the actual files that were changed to understand their current state.

### Context compression awareness

Claude Code compresses earlier messages as conversations get long. If you detect that earlier context has been compressed or is missing:

- Note this in the review: "Earlier parts of this session may not be fully available for review. The changes below are confirmed from git history, but some discussion context may be missing."
- Rely more heavily on git history when conversation context is incomplete.
- Do not guess at decisions or reasoning that are not evidenced in the available context.

### What to cover in a session review

- Every file created, modified, or deleted (confirmed via git or conversation)
- Every significant decision discussed (framework choices, architecture, approach)
- Every new dependency added
- Configuration changes
- Security-relevant changes (authentication, API keys, data handling)

## Project Review Scope

### The two-pass approach

Project reviews use two passes to manage scope without missing critical elements.

#### Pass 1: Structure scan

Read the following to understand the project's shape:

1. **File tree** — Full directory listing to understand project organisation
2. **Package files** — package.json, requirements.txt, Cargo.toml, go.mod, Gemfile, or equivalent. These reveal all external dependencies.
3. **Configuration files** — .env.example (never .env), config files, CI/CD pipelines, docker files
4. **README and documentation** — What the project claims to be
5. **Git history** — Recent commits to understand what is actively being worked on

Produce a mental model of the project's architecture before reading any source code.

#### Pass 2: Prioritised deep read

Based on the structure scan, read source files in this priority order:

1. **Entry points** — main, index, app, or equivalent. How the application starts.
2. **Security-sensitive files** — Authentication, authorisation, API route handlers, middleware, data processing
3. **Configuration and environment** — How the application is configured, what it expects from its environment
4. **Core business logic** — The files that do the main thing the application exists to do
5. **Data handling** — Database schemas, models, data transformations, API calls to external services
6. **Recently modified files** — Files changed in recent git history
7. **Tests** — What exists, what is covered, what is missing

### Large project handling

Source files are files containing application logic (e.g., .js, .ts, .py, .rb, .go, .java, .md for plugin projects), excluding config files, stylesheets, lock files, tests (unless reviewing tests specifically), and assets.

If a project has more than 100 source files (excluding node_modules, vendor directories, build output, and similar):

1. Complete Pass 1 (structure scan) fully
2. Before starting Pass 2, inform the user:
   > This project has [n] source files. A full detailed review would be very long. Would you prefer:
   > - **Full sweep** — I review everything, focusing most detail on high-priority files
   > - **Focused review** — Tell me which area to focus on (e.g., "the authentication system", "the API endpoints", "recent changes")
3. If the user chooses full sweep, proceed with Pass 2 but adjust depth: high-priority files get detailed review, lower-priority files get one-sentence summaries.
4. If the user chooses focused review, apply Pass 2 priorities only within the specified area.

### Coverage transparency

Every project review must state:

- How many source files exist in the project
- How many were reviewed in detail
- How many were scanned at summary level
- Which areas were not reviewed and why

This appears in the review frontmatter as "Files reviewed / Files scanned / Not reviewed" (as shown in the project review template).

## What Not to Scan

Always skip:

- `node_modules/`, `vendor/`, `.venv/`, `__pycache__/`, `target/`, `build/`, `dist/` — dependency and build output directories
- `.git/` — git internals
- Binary files, images, fonts
- Lock files (package-lock.json, yarn.lock, etc.) — note their existence but do not review contents
- `.env` files — note their existence, warn if they contain secrets, but do not reproduce their contents in the review

## Handling Unfamiliar Technology

If the project uses a language, framework, or tool you are less confident about:

- State this explicitly: "This project uses [technology]. I am less familiar with its conventions, so some observations may not apply."
- Focus on universal concerns: security, error handling, dependency health, documentation
- Do not guess at framework-specific best practices you are unsure about
