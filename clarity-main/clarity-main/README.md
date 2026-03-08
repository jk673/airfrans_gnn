# Clarity by Ashby

**Plain-language explanations of AI development work.** Clarity works in any language. Claude explains in your language automatically. This README is in English; visit [ashby.systems/clarity](https://ashby.systems/clarity) for other languages.

Clarity is a free, open-source plugin for Claude (Cowork and Claude Code) that explains what AI built in language a non-programmer can understand. It walks through sessions and projects, explains decisions, flags risks, and gives you specific prompts for follow-up.

You build things with AI. Clarity helps you understand what was built and whether you should be worried about anything.

## What Clarity Does

1. Reads what Claude Code built, changed, or planned
2. Explains every change, decision, and dependency in plain language
3. Identifies risks and gaps with concrete consequences, not jargon
4. Tells you what is and is not tested
5. Gives you specific prompts to take into your next session
6. Teaches you one or two useful concepts along the way

Clarity does not judge code quality, suggest fixes, or tell you what to do. It translates and explains. You decide what happens next.

## Installation

1. Download this repository (Code > Download ZIP) and unzip it
2. In Claude Code, run:

```bash
claude plugin install ./path/to/Clarity
```

Replace `./path/to/Clarity` with the actual path to the unzipped folder.

## Quick Start

1. Build something with Claude Code
2. Run `/clarity:explain` to understand something specific — a file, an error, or a concept
3. Run `/clarity:session` to understand what just happened
4. Or run `/clarity:project` to understand the whole project
5. Use the "What to Ask Claude Next" prompts in a new session

That is the full workflow. A session explanation takes a few minutes. Full project walkthroughs take longer for large projects.

## Commands

| Command | What it does |
|---------|-------------|
| `/clarity:explain` | Explain one thing — a file, error, output, or concept |
| `/clarity:session` | Explain what happened in the current session |
| `/clarity:project` | Walk through the full project |
| `/clarity:config` | Set where Clarity saves explanation files |
| `/clarity:help` | Show command reference |

## What Clarity Is Not

- **Not a code reviewer.** It does not judge whether code is good or bad. It explains what happened.
- **Not a fixer.** It does not suggest or make code changes.
- **Not objective.** It is Claude explaining Claude's own work. It may miss things a human would catch.
- **Not a substitute for a developer.** If something in an explanation concerns you, share it with a developer for a second opinion.

## Saved Files

Explanations are saved as markdown files to your configured location (set via `/clarity:config`) or to a `clarity/` directory in your project. Session files are timestamped; project files are dated.

**An important note:** Clarity is Claude explaining Claude's own work. The explanations are best-effort translations, not authoritative audits. They are useful for understanding and asking better questions, not for making final safety or security determinations.

## Companion Plugin: Attest

Clarity explains what AI built. [Attest by Ashby](https://github.com/ashbysystems/attest) is the other half: a structured review workflow that surfaces questions about any AI-generated output, captures your decisions, and produces an auditable decision log. Different jobs. Same goal: keeping humans in control.

## Data Handling

All processing runs locally. No data is transmitted externally. Review files are stored in your configured save location. Apply your organisation's existing data handling and retention policies.

## Disclaimer

Clarity by Ashby explains AI development work in plain language. It does not assess code quality, security, or fitness for purpose. It does not provide technical, legal, or compliance advice. All explanations are best-effort. This is Claude explaining Claude's own work. Treat it with appropriate scepticism.

## License

Apache 2.0. See [LICENSE](LICENSE).

## How Clarity Was Built

Clarity was designed by the Ashby team and built using Claude Code. The explanation methodology, frameworks, and scope management rules are based on human experience explaining technical work to non-technical stakeholders. Claude was used as a development tool, not as a domain expert. All design decisions were made by humans.

## About Ashby

Ashby builds structured human oversight for AI operations. Clarity is our plain-language explanation plugin. Attest is our structured review workflow plugin. Requisite is our Article 14 (EU AI Act) compliance assessment tool.

Learn more at [ashby.systems](https://ashby.systems).
