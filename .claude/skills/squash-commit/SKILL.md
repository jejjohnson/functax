---
name: squash-commit
description: Write a Conventional-Commits squash commit message for a pull request in this repository from its commits and description. Use when asked for a squash / merge commit message.
---

Generate a squash commit message for a GitHub PR.

This is the repo's copy of the maintainer's `/squash-commit` command (jejagents), with this repo's types, scopes and release rules.

## Instructions

1. If an argument is provided (PR number or URL), fetch the PR details (`gh pr view`, or the GitHub MCP tools). Otherwise, detect the current branch and find its open PR.
2. Fetch all individual commit messages in the PR (`gh api` or `git log`).
3. Combine them into a single squash commit message following these rules.

### Format

```
<type>(<scope>): <concise summary of the overall change>

<1-3 sentence description combining the key changes from all commits.
Focus on the "why" and overall effect, not per-commit details.>

Co-authored-by: <every unique human Co-authored-by line from the individual commits>
```

### Rules

- Conventional Commits types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`; a lowercase subject (`conventional-commits.yml` enforces it on PR titles). The scope is optional and names the area.
  <!-- template-example: the scopes are the example package's modules; list yours. -->
  Scopes used here: `stats`, `smoothing`, `transforms`, `utils`, `cli`, `docs`, `agents`, `ci`, `deps`.
- release-please (`release-please-config.json`) builds the changelog from `feat`, `fix`, `perf` and `revert` commits and hides the other types, so the type decides whether a change is listed. Before 1.0 (`bump-minor-pre-major`, `bump-patch-for-minor-pre-major`), a breaking change bumps the minor version and a `feat` or `fix` the patch.
- The type/scope reflect the dominant change across all commits; mention the rest in the body.
- A breaking change keeps its `!` and `BREAKING CHANGE:` footer.
- Collapse redundant or incremental commits into one coherent description.
- Preserve every unique `Co-authored-by` line for a person. Drop AI attribution (`Co-Authored-By: Claude …`, `Claude-Session:`, "Generated with …") unless the maintainer asks to keep it.
- Keep the summary line under 72 characters.
- Do NOT list the individual commit messages as bullet points — this is a squash, not a merge.

### Output

Print ONLY the final squash commit message in a code block so the user can copy it directly. Do not add explanation before or after.
