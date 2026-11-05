#!/usr/bin/env bash
#
# Apply native GitHub issue relationships via the REST API.
#
# Supports two relationship types:
#   - sub-issues (parent <-> child hierarchy)
#   - blocked-by / blocks (typed dependency)
#
# Usage:
#   bash .github/scripts/link-issues.sh sub <parent> <child> [<child> ...]
#   bash .github/scripts/link-issues.sh block <issue> <blocking-issue>
#   bash .github/scripts/link-issues.sh unsub <parent> <child>
#   bash .github/scripts/link-issues.sh unblock <issue> <blocking-issue>
#   bash .github/scripts/link-issues.sh show <issue>
#
# All issue arguments are issue numbers (e.g. 42) in the current repo; the
# link endpoints take the issue's numeric `id`, which is resolved for you.
# Set GH_REPO=owner/repo to act on another repo.
#
# Or via the poe tasks in pyproject.toml:
#   uv run poe gh-sub 7 42 43 44
#   uv run poe gh-block 44 43
#   uv run poe gh-show 44
#
# REST rather than GraphQL, because GraphQL is blocked in Claude sessions;
# on your own machine both work.
#
# Requires: `gh` CLI authenticated with repo scope.

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH. Install from https://cli.github.com/" >&2
  exit 1
fi

# `{owner}/{repo}` is filled in by gh from GH_REPO or the git remote.
REPO="repos/{owner}/{repo}"

# Resolve issue number -> numeric issue id (what the link endpoints take).
id_for() {
  local number="$1"
  gh api "${REPO}/issues/${number}" --jq '.id' 2>/dev/null \
    || { echo "error: could not resolve issue #${number} — wrong repo or number?" >&2; return 1; }
}

# Link one child under a parent. Idempotent: reports no-op if already linked.
add_sub_issue() {
  local parent="$1" child="$2"
  local child_id out
  child_id=$(id_for "$child")
  if out=$(gh api "${REPO}/issues/${parent}/sub_issues" -F sub_issue_id="$child_id" 2>&1); then
    printf '  ✓ sub-issue: #%s -> parent #%s\n' "$child" "$parent"
  elif grep -qiE "already|duplicate|exists" <<<"$out"; then
    printf '  • no-op: #%s is already a sub-issue of #%s\n' "$child" "$parent"
  else
    printf '  ✗ failed #%s -> #%s: %s\n' "$child" "$parent" "$out" >&2
    return 1
  fi
}

remove_sub_issue() {
  local parent="$1" child="$2"
  local child_id
  child_id=$(id_for "$child")
  gh api -X DELETE "${REPO}/issues/${parent}/sub_issue" -F sub_issue_id="$child_id" >/dev/null
  printf '  ✓ removed sub-issue link: #%s -/> #%s\n' "$child" "$parent"
}

# Mark `issue` as blocked by `blocker`.
add_blocked_by() {
  local issue="$1" blocker="$2"
  local blocker_id out
  blocker_id=$(id_for "$blocker")
  if out=$(gh api "${REPO}/issues/${issue}/dependencies/blocked_by" -F issue_id="$blocker_id" 2>&1); then
    printf '  ✓ #%s is now blocked by #%s\n' "$issue" "$blocker"
  elif grep -qiE "already|duplicate|exists" <<<"$out"; then
    printf '  • no-op: #%s is already blocked by #%s\n' "$issue" "$blocker"
  else
    printf '  ✗ failed #%s blocked-by #%s: %s\n' "$issue" "$blocker" "$out" >&2
    return 1
  fi
}

remove_blocked_by() {
  local issue="$1" blocker="$2"
  local blocker_id
  blocker_id=$(id_for "$blocker")
  gh api -X DELETE "${REPO}/issues/${issue}/dependencies/blocked_by/${blocker_id}" >/dev/null
  printf '  ✓ removed blocked-by: #%s no longer blocked by #%s\n' "$issue" "$blocker"
}

# Print parent / sub-issues / blocking / blocked-by for an issue.
list_numbers() {
  gh api "$1" --jq '.[] | "    #\(.number) [\(.state)] \(.title)"' 2>/dev/null || true
}

show_issue() {
  local number="$1"
  gh api "${REPO}/issues/${number}" --jq '"#\(.number) [\(.state)] \(.title)"'
  echo "  parent:"
  local parent
  if parent=$(gh api "${REPO}/issues/${number}/parent" \
      --jq '"    #\(.number) [\(.state)] \(.title)"' 2>/dev/null); then
    echo "$parent"
  else
    echo "    (none)"
  fi
  echo "  sub-issues:"
  list_numbers "${REPO}/issues/${number}/sub_issues"
  echo "  blocked by:"
  list_numbers "${REPO}/issues/${number}/dependencies/blocked_by"
  echo "  blocking:"
  list_numbers "${REPO}/issues/${number}/dependencies/blocking"
}

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

cmd="${1:-}"; shift || true
case "$cmd" in
  sub)
    [[ $# -ge 2 ]] || usage
    parent="$1"; shift
    for child in "$@"; do
      add_sub_issue "$parent" "$child"
      sleep 1  # pace bulk links to avoid secondary rate limits
    done
    ;;
  unsub)
    [[ $# -eq 2 ]] || usage
    remove_sub_issue "$1" "$2"
    ;;
  block)
    [[ $# -eq 2 ]] || usage
    add_blocked_by "$1" "$2"
    ;;
  unblock)
    [[ $# -eq 2 ]] || usage
    remove_blocked_by "$1" "$2"
    ;;
  show)
    [[ $# -eq 1 ]] || usage
    show_issue "$1"
    ;;
  *)
    usage
    ;;
esac
