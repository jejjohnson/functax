---
name: create-gh-issue
description: Open GitHub issues on this repo from the templates in .github/ISSUE_TEMPLATE/, with labels, milestone and relationships, over the REST API. Use when asked to file / open an issue (feature, design, bug, research, epic) or to publish drafted backlog issues.
---

Open GitHub issues on this repo from its issue templates, with the right labels, milestone and relationships.

This is the repo's copy of the maintainer's `report` and `gh-issue` skills (jejagents): `report` decides what goes in the body, `gh-issue` how it is filed. Use them for any other repo. Here, the templates, labels and milestones below win, and links go through `link-gh-issues`.

Everything uses `gh api` over REST, because GraphQL (and so `gh issue create`, `gh issue list` and `gh project`) is blocked in Claude sessions. On your own machine both work.

## When to invoke

- "Open an issue for X" / "file an issue about Y"
- "Create a feature issue for a Savitzky–Golay filter"
- "Open an epic for Wave 2"
- "Publish the drafts in `.plans/wave-1-backlog.md` as real issues"
- "Turn this research note into a research issue"

## 1. Pick the template

The repo's own templates in `.github/ISSUE_TEMPLATE/` win over the generic report types; use their headings and fill them with the content rules in step 3.

| Template | Use when |
|---|---|
| `feature.md` | One deliverable: a function, transform, module, notebook, docs page |
| `bug.md` | A wrong result, a crash, a regression, with a reproduction |
| `design.md` | An open question about an API, a contract, a dependency or a package boundary |
| `research.md` | Map a library, paper or prior art onto this package; its output is the list of follow-up issues |
| `epic-wave.md` | A release-scoped wave (owns a milestone, groups theme epics) |
| `epic-theme.md` | Related feature / design / chore issues within a wave |

"Add X" defaults to a feature. A feature that needs a decision first becomes a design issue plus a feature blocked by it. A research issue never does the work itself.

## 2. Confirm scope

Before filing, confirm these with the user (or infer them and echo them back):

1. **Template** (step 1).
2. **Title**, following the template's `title:` frontmatter and the repo's recent titles (`gh api "repos/$R/issues?state=all&per_page=20" --jq '.[] | select(.pull_request|not) | .title'`):
   - Feature: `<scope>: <outcome>`, e.g. `feat(smoothing): savitzky–golay filter`
   - Bug: `bug: <what goes wrong>`
   - Design: `[Design] <the question>`
   - Research: `research: <topic>`
   - Epics: `[EPIC] Wave N: <title>` and `[EPIC] <theme title>`

   Subjects are lowercase and say the outcome, not the activity.
3. **Parent** theme or wave epic, if any.
4. **Wave and milestone**: the `wave:*` label and the `vX.Y-<slug>` milestone.
5. **Other labels**: `area:*`, `layer:*`, `priority:*` (step 4).

Then check for duplicates, searching titles and bodies of open and closed issues for the key name or phrase:

```bash
R=jejjohnson/<repo>
gh api --paginate "repos/$R/issues?state=all&per_page=100" \
  --jq '.[] | select(.pull_request|not) | "\(.number) \(.state) \(.title)"' \
  | grep -i "<term>"
```

A stale match is edited in place and re-parented, not closed and refiled; a partial match is linked as `Related`.

## 3. Write the body

Read `.github/ISSUE_TEMPLATE/<template>.md` and use it as the scaffold. Strip the YAML frontmatter and the pure-guidance HTML comments (`<!-- Delete if not needed. -->`); keep a comment only if it helps the filled-in body.

Never leave these sections empty:

| Template | Required |
|---|---|
| `feature.md` | Problem / Request, User Story, Motivation, Proposed API, Implementation Steps, Definition of Done, Testing, Relationships |
| `design.md` | Problem / Question, Proposed Options, Alternatives Considered, Relationships |
| `bug.md` | Problem, Reproduction, Expected Behavior, Actual Behavior, Environment, Relationships |
| `research.md` | Title, Context, §1 What the subject contains, §2 Comparison, §3 Summary Table, §5 Proposed Follow-up Issues, Relationships |
| `epic-wave.md` | Goal, Motivation, Theme Epics, Definition of Done (Wave), Relationships |
| `epic-theme.md` | Theme, Parent Wave, Issues, Definition of Done, Relationships |

On `feature.md` and `design.md`, `## Design Snapshot` and `## Mathematical Notes` may be renamed to fit (`## Demo To Implement`, `## Equations To Test`) or deleted.

Content rules (from `report`):

- **Self-contained.** Someone with only the issue, a person or an agent, can act on it without opening other repos, chats or docs. Paste the API, the equations and the reproduction; link only for depth.
- **Point first.** Each section opens with its point in one sentence.
- **Prose.** Not hard-wrapped; one sentence per line, sentences under 25 words, no "robust", "seamless" or "leverage". Numbers carry units.
- **Maths** follows `eman-style-math`: define every symbol, with its shape, before its first use; inline maths in Unicode (σ², Σ⁻¹, ∑) or `$…$`; display maths in a `math` fence, never `$$` mid-line; no custom macros on GitHub.
- **Code.** Lead the Design Snapshot and Proposed API with the exact snippet the implementer will reproduce. Sketches follow `code-sketch` (a shapes header, the equation above each line, a status marker on every call: `today (mypackage)`, `proposed (#N)`, `user`); code that will ship follows `python-style`.
- **Concrete steps.** Each implementation step names the file and the function: "Add `savitzky_golay` in `src/mypackage/smoothing.py`".
- **Recommend, then offer the alternative.** "Recommended: …" and "Alternative: …", one line each.
- **Evidence is marked, not blocked.** A claim checked in the code cites the name and file at a commit; a reproduction starts with `# verified: <date>` or `# not run`; a number says where it came from (a test, a notebook, a paper with DOI or arXiv id, or `(estimate)`); anything else unchecked ends with `(unverified)`. Never invent a reference, a name or a number to fill a section; leave `(unverified)` or `TODO`.
- **Relationships.** End with the `## Relationships` block (`Parent`, `Blocked by`, `Blocks`, `Related`), even when native links are applied too.

When the body has maths and the `eman-style-math` skill is installed, expand its macros and check the body before filing:

```bash
uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/expand_macros.py body.md
```

```bash
uv run --with cyclopts --with loguru --with pyyaml python ~/.claude/skills/eman-style-math/scripts/check_math.py body.md --target github
```

## 4. Pick labels

Every issue carries exactly one `type:*`, one or more `area:*`, at most one `layer:*`, one `wave:*` and one `priority:*`. Apply only labels that exist (`gh api "repos/$R/labels?per_page=100" --jq '.[].name'`).

<!-- template-example: the layer column maps the example package's modules onto the template's layer labels; remap it for your package. -->

| Work type | Labels |
|---|---|
| L0 primitive (a validator, a statistic, a filter: `utils`, `stats`, `smoothing`) | `type:feature`, `area:code`, `layer:0-primitives`, `wave:N`, `priority:p1` |
| L1 component (a transform, a pipeline feature: `transforms`) | `type:feature`, `area:code`, `layer:1-components`, `wave:N` |
| L2 public API / wrapper (a CLI subcommand, a top-level export: `cli`, `__init__`) | `type:feature`, `area:code`, `layer:2-models`, `wave:N` |
| Test coverage | `type:feature`, `area:testing`, `wave:N` |
| Notebook / docs page | `type:docs`, `area:docs`, `wave:N` |
| Design / ADR | `type:design`, `area:code` (or `area:engineering`), `wave:N` |
| Research / survey | `type:research`, `area:code` (or the relevant area) |
| Engineering / CI / packaging | `type:chore`, `area:engineering`, `wave:0` |

The labels come from `.github/scripts/create-labels.sh` (`uv run poe gh-labels`): `type:{epic-wave,epic-theme,feature,bug,design,chore,docs,research}`, `area:{engineering,testing,docs,code}`, `dependencies`, `layer:{0-primitives,1-components,2-models}`, `wave:0`…`wave:4`, `priority:{p0,p1,p2}`. A project that needs more areas (`area:algorithmic`, `area:performance`, …) adds them to that script first. Check `docs/contributing.md` for the current waves and milestones.

## 5. Create

Write the body to a file; never pass code fences through a command-line string. The REST API takes the milestone's **number**, so look it up by title first:

```bash
gh api "repos/$R/milestones?state=open" --jq '.[] | "\(.number) \(.title)"'
```

```bash
gh api repos/$R/issues \
  -f title="feat(smoothing): savitzky–golay filter" \
  -F body=@body.md \
  -f "labels[]=type:feature" -f "labels[]=area:code" -f "labels[]=layer:0-primitives" \
  -f "labels[]=wave:1" -f "labels[]=priority:p1" \
  -F milestone=<milestone number> \
  --jq '"\(.number) \(.id) \(.html_url)"'
```

Keep both numbers from the result: `number` for people and links in text, `id` for the link endpoints.

To edit an existing issue: `gh api -X PATCH repos/$R/issues/<n> -F body=@body.md` (and `-f title=`, `-f state=closed`, `-f "labels[]=…"`).

When the user wants to review in the browser instead, `gh issue create --web` opens the template chooser (on their own machine; it needs GraphQL).

## 6. Link

Apply the native links with `link-gh-issues`:

```bash
uv run poe gh-sub <theme-epic#> <new-issue#>     # sub-issue of its theme epic
uv run poe gh-block <new-issue#> <blocker#>      # blocked by another issue
```

Keep the prose `## Relationships` lines in the body; they coexist with the native links.

## 7. Report back

One issue: its URL and a one-line summary. Several: a draft → issue table, plus anything skipped (an existing duplicate, a label that does not exist, a link that failed):

```
Draft ID  → issue
MP-05     → #42   feat(smoothing): ...
MP-06     → #43   feat(cli): ...
```

## Publishing a wave backlog

When the user has a `.plans/<wave>-backlog.md` drafted from `docs/templates/wave-backlog.md`:

1. Split the backlog into drafts: each starts with `# <title>` and `Draft ID: \`<PREFIX>-NN\``, and drafts are separated by `---`.
2. Match each draft to its template by its first section (`## Goal` → epic-wave, `## Theme` → epic-theme, `## Problem / Request` → feature, `## Problem / Question` → design, `## Context` with numbered `## 1.` sections → research).
3. Run the maths check on every body.
4. Create the wave epic first, then the theme epics, then the leaf issues, recording draft ID → number → id as you go. Pace the calls, about a second apart, to avoid secondary rate limits.
5. Replace draft IDs in the bodies with `#numbers` and PATCH each body.
6. Apply the sub-issue and blocked-by links with `link-gh-issues`, then read them back.
7. Replace the draft IDs in the backlog file too, or move it to `.plans/archive/`.

## Common pitfalls

- **Missing label:** the create call fails with a 422. Run `uv run poe gh-labels` on a fresh repo.
- **Missing milestone:** create it with `gh api repos/$R/milestones -f title="vX.Y-<slug>"`, then use its number.
- **Frontmatter in the body:** strip the template's `---` block before writing the body file.
- **Backticks:** pass bodies with `-F body=@file`, never inline, so code fences survive.

## Related skills

- [`link-gh-issues`](../link-gh-issues/SKILL.md): native sub-issue and blocked-by links, after the issues exist.
- [`squash-commit`](../squash-commit/SKILL.md): the squash commit message when the issue's PR merges.
