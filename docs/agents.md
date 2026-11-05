# Building with agents

<!-- template-example: this page describes the agent support of the bundled example package; rewrite it for your package on adoption. -->

`mypackage` exists so that summarising, smoothing and rescaling a numeric
series is a call to a small, validated, pure function rather than a loop
written again in every project. Coding agents tend to re-implement what they
cannot see — a variance loop, a rolling median over a `deque`, a z-score
scaler fitted on the wrong split — and each copy loses the validation, the
boundary handling or the fit / apply discipline. The project ships three
things that let them find the package instead.

## The capability index

The [capability index](https://jejjohnson.github.io/pypackage_template/reference/capabilities/)
lists every public name in `mypackage` and `mypackage.cli`, grouped by the
module that defines it, with a one-line summary. It is regenerated from the
code (`uv run poe capabilities`) and checked in the test suite, so it never
drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/pypackage_template
/plugin install mypackage@mypackage
```

The plugin adds:

- **`series-analysis-with-mypackage`** (skill) — loads whenever a task
  summarises, smooths, rescales or cleans a numeric series: what lives
  where, the rules (inputs are never mutated, fit then apply, structural
  transforms, typed errors), a worked example, and a "don't write it — use
  mypackage" table.
- **`mypackage-reuse-reviewer`** (subagent) — a read-only check of a diff
  for series code the package already provides, and for misuse.

## llms.txt

For other agents and tools, the site serves
[`llms.txt`](https://jejjohnson.github.io/pypackage_template/llms.txt): a
curated map of the package and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on
`mypackage`:

```markdown
## Numeric series: build on mypackage

This project uses mypackage for statistics, smoothing and transforms of
one-dimensional numeric series. Before writing a statistic, a rolling
window, a scaler or an input validator, search the capability index
(https://jejjohnson.github.io/pypackage_template/reference/capabilities/) or
`mypackage.__all__`, and compose what exists:

- Statistics: `mp.summarize` (mean, variance, std, median, extrema),
  `mp.quantile`, `mp.zscores`; streaming, `mp.RunningStats`.
- Smoothing: `mp.moving_average`, `mp.median_filter` (for spikes),
  `mp.exponential_moving_average`, with `padding=` from `mp.Padding`.
- Rescaling: fit `mp.Standardize` / `mp.MinMaxScale` on training data only
  and reuse the fitted object; chain steps with `mp.chain` / `mp.Pipeline`.
- Never mutate the caller's list; catch `mp.MypackageError`, not a bare
  `except`.
```

## Working on the package itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/pypackage_template/blob/main/AGENTS.md)
in the repository: the module map, "reuse before you write", the contracts
and the tests that enforce them, and recipe skills for adding functions,
transforms, CLI commands, guide pages and notebooks. Adopting the template
for a project of your own is a recipe too: `adopt-template` in
`.claude/skills/`.
