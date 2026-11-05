---
name: link-gh-issues
description: Apply native GitHub issue relationships (sub-issues, blocked-by) over the REST API. Use when asked to link issues as parent/child, mark one blocked by another, or inspect an issue's relationships.
---

Apply native GitHub issue relationships (sub-issues and blocked-by) over the REST API.

This is the repo's copy of the linking half of the maintainer's `gh-issue` skill (jejagents), with this repo's helper script and poe tasks. It works on top of the prose `## Relationships` block in `docs/contributing.md`: the prose stays as the human-readable record, and this skill applies the matching native link so GitHub's UI and automation pick up the hierarchy.

Everything here uses `gh api` over REST, because GraphQL (and so `gh issue create`, `gh issue list`, `gh repo view` and `gh project`) is blocked in Claude sessions. On your own machine both work.

## When to invoke

- "Link issue #42 as a sub-issue of #7"
- "Mark #44 as blocked by #43"
- "Wire up the sub-issues for the Wave 1 theme epic #29"
- "Apply the relationships from this epic's body": parse its checklist and its `Parent` / `Blocked by` lines, and apply each link
- "What's blocking #44?": read the relationships back

## The quick path

`.github/scripts/link-issues.sh` wraps the endpoints below, resolves numbers to ids, treats an existing link as a no-op and paces bulk calls:

```bash
uv run poe gh-sub 7 42 43 44    # 42, 43 and 44 become sub-issues of 7
uv run poe gh-block 44 43       # 44 is blocked by 43
uv run poe gh-show 44           # parent, sub-issues, blocked by, blocking
```

`unsub` and `unblock` are only on the script (`bash .github/scripts/link-issues.sh unsub <parent> <child>`). Set `GH_REPO=owner/repo` to act on another repo.

Prefer the script for more than two or three links.

## The endpoints

Both link endpoints take the issue's numeric **`id`**, not its `number`. Sub-issues may live in another repo of the same owner.

```bash
R=jejjohnson/<repo>
id() { gh api "repos/$1/issues/$2" --jq .id; }

# child becomes a sub-issue of the parent (add -F replace_parent=true to move it)
gh api repos/$R/issues/<parent n>/sub_issues -F sub_issue_id=$(id $R <child n>)

# <n> is blocked by <blocker n>
gh api repos/$R/issues/<n>/dependencies/blocked_by -F issue_id=$(id $R <blocker n>)

# remove them again
gh api -X DELETE repos/$R/issues/<parent n>/sub_issue -F sub_issue_id=$(id $R <child n>)
gh api -X DELETE repos/$R/issues/<n>/dependencies/blocked_by/$(id $R <blocker n>)

# read back
gh api repos/$R/issues/<n>/parent --jq '.number'
gh api repos/$R/issues/<n>/sub_issues --jq '.[] | "\(.number) \(.title)"'
gh api repos/$R/issues/<n>/dependencies/blocked_by --jq '.[].number'
gh api repos/$R/issues/<n>/dependencies/blocking --jq '.[].number'
```

"A blocks B" means B is blocked by A: call the `blocked_by` endpoint on B with A's id.

## Bulk-apply from an epic body

1. Fetch the epic's body: `gh api repos/$R/issues/<N> --jq .body`.
2. Every `#NNN` in its `## Issues` (or `## Canonical Child Issues`) checklist becomes a sub-issue of the epic.
3. Apply its `Blocked by:` and `Blocks:` lines from the `## Relationships` block.
4. Skip `Related:` lines; they have no native link and stay as prose.
5. Read the links back with `uv run poe gh-show <N>` and report them in one table.

## Common errors

- **422 saying the link already exists:** a no-op; report it and move on.
- **404 on an issue:** the number is wrong or the repo is not the one you meant; check `GH_REPO` or the git remote.
- **422 when linking across owners:** sub-issues must belong to the same owner.
- **403 or 404 on a link endpoint with a valid issue:** the `gh` token needs `repo` scope (`gh auth refresh -s repo` on your machine).

## What to keep as prose vs native

The prose `## Relationships` block in the issue body is the human-readable record:

```markdown
## Relationships
- Parent: #7
- Blocked by: #42
- Blocks: #55
- Related: #38
```

Leave those lines in place: they show up in diffs, search and tools that never call the link endpoints. The native link is for GitHub's UI and automation. Both coexist.

## Without the `gh` CLI

The GitHub MCP tools cover the same ground: `sub_issue_write` adds, removes or reprioritises a sub-issue; read relationships with `issue_read`.
