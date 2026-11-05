---
name: mypackage-review
description: Review a change or pull request in this repository against its own rules — CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the package already provides, and correctness at the boundaries (validation, mutation, the fit / apply lifecycle, numeric edge cases, doctest accuracy). Use when asked to review a diff, branch or PR in this repo.
---

<!-- template-example: the skill's name and step 4 are the example package's; rename the skill to <yourpackage>-review and rewrite step 4 from your contracts. -->

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note which modules, tests and docs it touches.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: a re-written validator, statistic, filter, padding rule or transform is the main way a package drifts from itself.
3. **Correctness** — run the `correctness-reviewer` subagent in parallel with step 2: validation at the boundary, mutation of inputs, the fit / apply lifecycle, numeric edge cases, error types and messages, doctest accuracy, test tolerances.
4. **Contracts** — for each module touched, the rules under "The contracts" in `AGENTS.md`: the `utils` validators and `'name' must …` messages, new lists out, `NotFittedError` before `fit`, `fit` returning `self`, frozen stateless transforms, CLI handlers writing only to the injected streams, exceptions deriving from `MypackageError` and their stdlib base.
5. **Checklist** — the rest of `CODE_REVIEW.md` (public API, API page and `docs/api/index.md` row, capability index, MyST prose, tests and tiers), skipping what ruff, ty or the tests already enforce.
6. **House style** — the `python-style` review checklist if the skill is installed, otherwise "House style" in `AGENTS.md`: docstring layout with a Unicode equation, comments above the line with the equation above each mathematical line, descriptive names, maths in the `eman-style-math` notation, `cyclopts` and `loguru` in new CLIs and scripts.
7. **Verify claims** — for anything you flag as a bug, run it (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with priority, file, lines and a concrete change, summary).
