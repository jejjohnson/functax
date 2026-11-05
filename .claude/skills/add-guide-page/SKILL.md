---
name: add-guide-page
description: Add or update a prose page in the MyST half of the docs — a guide page under docs/guide (or a top-level page under docs/) in MyST Markdown with directives, xref links to the API by top-level name, a toc entry in docs/myst.yml, a unique basename, a link back from the API page, and the assembled docs build. Use when asked to write or fix a guide, tutorial text, explanation or any non-notebook prose page.
---

# Add or update a guide page

Read "Documentation" in `AGENTS.md` first; `docs/README.md` explains the two-tool build.

## 1. Where it goes

- A guide to one part of the package: `docs/guide/<topic>.md`, listed under `Guide` in the `toc` of `docs/myst.yml`.
- A project-level page (contributing, changelog, agents): `docs/<name>.md`, in the matching toc section.
- An executable walkthrough is a notebook instead (`add-notebook`); an API page under `docs/api/` is MkDocs, not MyST (`add-function`).
- The basename must be unique across `docs/`, `docs/guide/` and `docs/notebooks/`: the page is served at `/<basename>/` (`_` → `-`), and frontmatter `slug:` is ignored.

## 2. Write it in MyST

- Directives, not MkDocs-Material syntax: `:::{note}`, `:::{tip}`, `:::{dropdown} Title`, `::::{tab-set}` / `:::{tab-item}`, `::::{grid}` / `:::{grid-item-card}` (see `docs/index.md` and `docs/guide/*.md`).
- Maths with `$…$` and `$$…$$` in the `eman-style-math` notation ("Maths notation" in `AGENTS.md`): the kind-named macros (`\vect{x}`, `\gmat{\Sigma}`) that `docs/myst.yml` loads, every symbol defined with its shape before its first use, each expression on one source line, no spacing macros (`\,`, `\quad`).
- Prose in the maintainer's voice (`eman-prose-tech`, docs register), not hard-wrapped: one line per paragraph. Diagrams are Mermaid (a `{mermaid}` directive), never ASCII art.
- Link the API with the `xref:` protocol and the **top-level** name: ``[`summarize`](xref:api#mypackage.summarize)``. Links to other prose pages are relative paths to their source files (`[CLI](guide/cli.md)`).
- Code blocks are illustrative (nothing executes them): run every snippet yourself and paste real output.

## 3. Wire it in

- The `toc` entry in `docs/myst.yml`.
- If an API page covers the same module, link the guide from it the way `docs/api/stats.md` does (`[statistics guide](../statistics/)`: the flat URL, written relative to the source file in `docs/api/`, because Zensical resolves links from the source file and adds one level for the page's directory URL, so it emits `../../statistics/` from `/reference/stats/`).

## 4. Verify

`uv run poe docs` builds both halves, assembles them and fails on any broken internal link, including links between the halves; `uv run poe docs-check` validates both halves without rendering the theme. Both need the mystmd CLI (`npm install -g mystmd`; if the theme download is blocked, see `docs/README.md`).
