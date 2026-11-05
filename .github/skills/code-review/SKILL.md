---
name: code-review
description: Review a change or pull request in this repository against CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the package already provides (docs/api/capabilities.md), and the tests that enforce the rules.
---

# Code review

Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](../../../AGENTS.md); for every function, class or module the diff adds, search [`docs/api/capabilities.md`](../../../docs/api/capabilities.md) for an existing equivalent; then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md) and report in its format.

The full procedure, with the reuse and correctness checklists, is the `mypackage-review` recipe in [`.claude/skills/mypackage-review/SKILL.md`](../../../.claude/skills/mypackage-review/SKILL.md) and the two reviewers in [`.claude/agents/`](../../../.claude/agents/).
