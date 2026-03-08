---
name: explanation-frameworks
description: Defines how Clarity translates technical concepts into plain language. Covers explanation patterns, the jargon buster, dependency explanations, architecture descriptions, and risk communication for non-programmers.
user-invocable: false
---

# Clarity Explanation Frameworks

## Purpose

Every technical concept in a Clarity review must be understandable by someone who has never programmed. This skill defines the patterns and rules for achieving that translation consistently.

## Core Explanation Pattern

For any technical concept, follow this pattern:

**[Name of thing]** — [what it does in one sentence, using everyday language]. [Why it matters for this project in one sentence].

Examples:

- **Next.js** — a framework (a pre-built toolkit) for building websites. It handles the technical plumbing so the developer can focus on what the site does.
- **Environment variable** — a setting stored outside the code, like a password kept in a safe rather than written on a sticky note. This keeps sensitive information out of the codebase.
- **API endpoint** — a specific address where your application receives or sends data, like a letterbox for a particular type of mail. Each endpoint handles one type of request.

## Architecture Descriptions

When describing how a project is built, use physical metaphors:

- Think of the architecture as a building's floor plan, not its wiring diagrams
- Describe the main rooms (components) and how data moves between them
- Focus on what each part does, not how it does it internally

Example:
> The project has three main parts: a **front end** (what users see in their browser), a **back end** (a server that processes requests and talks to the database), and a **database** (where all the data is stored permanently). When a user fills in a form, the front end sends it to the back end, which checks it and saves it to the database.

Avoid describing internal implementation details unless they create a risk or are relevant to a decision the user needs to make.

## Dependency Explanations

For every external dependency, provide:

1. **What it is** — one sentence
2. **Why it was added** — what it does for this project
3. **Maintenance status** — well-known and actively maintained, or less common/uncertain
4. **What happens if it breaks** — concrete consequence

Example:
> **Stripe** — a payment processing service used by most major websites. It handles credit card payments so this project does not have to store card details directly. Stripe is well-established and actively maintained. If Stripe had an outage, users would be unable to make payments until it was resolved, but no data would be lost.

> **left-pad** — a small utility that adds spaces to the beginning of text. It was added as a dependency of another library, not directly. It is maintained by a single person. If it were removed, the build process would fail until an alternative was found. This is low risk but worth being aware of.

## Decision Explanations

When explaining why a particular choice was made, always cover:

1. **What was chosen** — in plain language
2. **Why** — the reasoning, as far as can be determined from context
3. **What the alternatives were** — if obvious alternatives exist
4. **Lock-in assessment** — is this easy to change later, or does it commit the project?

Use this format:

> **Choice:** [what was chosen]
> **Why:** [reasoning]
> **Alternatives:** [what else could have been chosen, or "no obvious alternatives"]
> **Lock-in:** [Easy to change / Moderate effort to change / Significant commitment — with one sentence explaining why]

## Risk Communication

### Severity framing

Always distinguish between severity levels using plain language:

- **"This will cause problems"** — something is actively broken or dangerous right now
- **"This could cause problems"** — something is risky under specific circumstances, which you should name
- **"Worth knowing about"** — not a problem now, but context the user should have for future decisions

### Consequence-first pattern

Always lead with the consequence, then explain the technical detail:

- Not: "There is no input validation on the email field"
- But: "If someone types something that is not an email address into the email field, the application will crash instead of showing a helpful error message. This is because the code does not check what was typed before processing it."

### Security risk communication

For security issues, always specify:

- **What could go wrong** — in concrete terms
- **Who could exploit it** — "anyone who sees the code", "anyone on the internet", "only someone with server access"
- **What they could access or do** — specific data or capabilities at risk
- **How likely it is** — in practical terms, not percentages

## Jargon Buster

Every review (session and project) includes a jargon buster section. Rules:

1. Include every technical term used anywhere in the review
2. Include terms from library names, error messages, file types, and architectural concepts
3. One sentence per term, maximum
4. Write definitions that would make sense to someone who has never programmed
5. Alphabetical order
6. If a term was already defined inline in the review, still include it in the jargon buster for reference

Format:

| Term | What it means |
|------|--------------|
| API | A way for different software systems to talk to each other, like a waiter taking orders between you and the kitchen |
| Dependency | A piece of software your project relies on that someone else built and maintains |
| Environment variable | A setting stored outside your code, like keeping a password in a safe rather than on a sticky note |

## File Change Explanations

When describing what changed in a file, follow this pattern:

| File | What it does | What changed | Why |
|------|-------------|-------------|-----|

For each file:
- **What it does**: One sentence a non-programmer would understand
- **What changed**: What is different now compared to before, in plain language
- **Why**: The reason for the change

Do not list every line that changed. Summarise the meaningful changes and their purpose.

For deleted files, list them separately below the table:

### Deleted files
- [file path]: [one sentence explaining why it was removed]

## Test Coverage Explanations

When describing tests:

- Explain what each test checks in non-technical terms: "This test checks that when a user tries to log in with the wrong password, they see an error message instead of being let in"
- Explain what is NOT tested: "There are no tests for what happens when the payment service is unavailable"
- Explain how to run tests in a single command the user could paste into their terminal
- If no tests exist, state this directly and explain what that means in practice
