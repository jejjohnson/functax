#!/usr/bin/env bash
#
# Bootstrap functax's label taxonomy.
#
# Creates labels for: type / area / wave / priority.
# Idempotent — uses `gh label create --force` so re-runs are safe.
#
# Usage:
#   bash .github/scripts/create-labels.sh
#   or:
#   uv run poe gh-labels
#
# Requires: `gh` CLI authenticated with repo scope.
#
# Customise by editing the `create` calls below; run again to apply.

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH. Install from https://cli.github.com/" >&2
  exit 1
fi

# Fail early with a clear message if gh is not authenticated.
if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

create() {
  local name="$1"
  local color="$2"
  local desc="$3"
  gh label create "$name" --color "$color" --description "$desc" --force >/dev/null
  printf '  ✓ %s\n' "$name"
}

echo "Creating type:* labels..."
create "type:epic-wave"   "5319e7" "Release-scoped mega-epic (L1)"
create "type:epic-theme"  "8b5cf6" "Parallel-safe theme under a wave (L2)"
create "type:feature"     "a2eeef" "New feature or enhancement"
create "type:bug"         "d73a4a" "Something isn't working"
create "type:design"      "fbca04" "Design / ADR issue for a new API"
create "type:chore"       "c5def5" "Engineering maintenance"
create "type:docs"        "0075ca" "Documentation work"
create "type:research"    "bfe5bf" "Comparative analysis / prior-art survey / investigation"

echo "Creating area:* labels..."
# One area per public module, plus the cross-cutting ones.
create "area:geometry"    "c2e0c6" "functax.geometry"
create "area:basis"       "c2e0c6" "functax.basis"
create "area:diffops"     "c2e0c6" "functax.diffops"
create "area:interp"      "c2e0c6" "functax.interp"
create "area:quad"        "c2e0c6" "functax.quad"
create "area:transform"   "c2e0c6" "functax.transform"
create "area:project"     "c2e0c6" "functax.project"
create "area:engineering" "ededed" "Build, packaging, CI, tooling"
create "area:testing"     "bfdadc" "Test suite"
create "area:docs"        "1d76db" "Docs + notebooks"

echo "Creating Dependabot / misc labels..."
create "dependencies" "ededed" "Pull requests that update a dependency file"

echo "Creating wave:* labels..."
# One per wave epic (functax#8 to functax#13).
create "wave:0" "bfd4f2" "Wave 0"
create "wave:1" "bfd4f2" "Wave 1"
create "wave:2" "bfd4f2" "Wave 2"
create "wave:3" "bfd4f2" "Wave 3"
create "wave:4" "bfd4f2" "Wave 4"
create "wave:5" "bfd4f2" "Wave 5"

echo "Creating priority:* labels..."
create "priority:p0" "b60205" "Blocker for current wave"
create "priority:p1" "d93f0b" "High priority"
create "priority:p2" "fbca04" "Normal priority"

echo
echo "Done. View labels: gh label list"
