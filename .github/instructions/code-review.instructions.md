---
applyTo: "**"
---

# Code Review Instructions

When performing code review, use `/CODE_REVIEW.md` as the source of truth for:

- Review checklist (reuse and boundaries, the package's contracts, public API and documentation, tests, modern Python and dependencies, security)
- Package-specific checks (validation at the boundary, no mutation of inputs, the fit / apply lifecycle, typed and chained errors, doctest output)
- Output format and priority levels
- Suggestion type emojis and review tone

Read "Reuse before you write" and "The contracts" in `/AGENTS.md` first, and check every function, class or module the diff adds against `docs/api/capabilities.md`.

Key principles:
- Sacrifice *cleverness* for *clarity*. Sacrifice *brevity* for *explicitness*.
- Don't worry about formatting — CI (ruff format, pre-commit) handles that automatically.
- Be **constructive** and **specific**. Acknowledge good patterns with 👍.
- Every suggestion must include a concrete alternative, not just criticism.
