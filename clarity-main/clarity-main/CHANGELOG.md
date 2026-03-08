# Changelog

All notable changes to Clarity are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- Environment variable: changed CLARITY_PLUGIN_ROOT to CLAUDE_PLUGIN_ROOT across all commands
- Default save location now consistent ("clarity/ in current working directory") everywhere
- Config command: added error handling for declined/failed directory creation
- Session/project commands: added fallback when save directory creation fails
- Session filename format: added seconds (HHMMSS) to prevent same-minute collisions
- Project command: clarified coverage statement placement, added "other files" to Pass 2
- Explain command: added rule requiring inline definitions for all named technologies

### Improved
- Review methodology: specified honesty clause location (disclaimer footer), language auto-detection rules, and history check depth (5 most recent reviews)
- Scope management: added git detection method, defined "source files", fixed scope notes terminology to match template
- Explanation frameworks: tightened architecture metaphor, added deleted files format
- Session template: added no-decisions guidance for "Why Those Choices", structured deleted files format
- Project template: fixed "Maintained?" to "Maintenance status", clarified "Not reviewed" when nothing skipped
- CHANGELOG: corrected file count (14 not 13), added Unreleased section
- Help command: clarified markdown rendering instruction

## [1.0.0] - 2026-02-09

### Added
- Initial build of Clarity plugin (14 files)
- Plugin manifest (.claude-plugin/plugin.json)
- 5 commands: explain, session, project, config, help
- 3 skills: review-methodology, scope-management, explanation-frameworks
- 2 templates: session-review, project-review
- Two-pass scope management for large project reviews
- Git-aware session reviews (git diff/log as primary data source)
- Context compression awareness for long sessions
- "What to Ask Claude Next" section with paste-ready prompts
- "Worth Understanding" section for concept education
- Jargon buster in both session and project reviews
- Configurable save location matching Attest pattern
- README with installation, usage, and disclaimer
- Apache 2.0 license
