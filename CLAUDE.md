# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public name in the package) before writing a helper; the "Reuse before you write" table in `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches their description (or run them as `/<name>`):
  - building: `add-scheme`, `add-guide-page`, `add-notebook`;
  - shipping: `pre-pr-check`, `functax-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Personal skills** from [jejagents](https://github.com/jejjohnson/jejagents), when installed in `~/.claude/skills/`, set the house style: `python-style` for any Python, `eman-style-math` for any maths, `code-sketch` for pseudocode, `eman-prose-tech` for prose in the maintainer's voice, `report` before `create-gh-issue`. The "House style" and "Personal skills" sections of `AGENTS.md` say which covers what; where a skill and `AGENTS.md` disagree, the departures documented in `AGENTS.md` win.
- **Subagents** (`.claude/agents/`), both read-only, both used by `functax-review`; run them on any diff that adds code, before committing:
  - `reuse-reviewer`: does the diff re-implement something in `docs/api/capabilities.md` or a shared private helper?
  - `numerics-reviewer`: dtype preservation, guarded `where` branches, explicit `inside` masks, static shapes, the NumPy-only base, adjoints and oracle tolerances.
- **Partial test runs** need `--no-cov` (`addopts` turns coverage on and its 90 % gate fails any subset).
- **GitHub.** GraphQL is blocked in Claude sessions, so issues, sub-issue and blocked-by links go through REST (`gh api repos/...`, as `create-gh-issue`, `link-gh-issues` and `.github/scripts/link-issues.sh` do). When the `gh` CLI is unavailable, use the GitHub MCP tools for the same operations (PRs, issues, review threads, check runs).
- **Commits.** No `Co-Authored-By: Claude` trailer and no "Generated with Claude Code" line, unless the maintainer asks.
