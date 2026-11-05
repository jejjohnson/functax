---
name: add-notebook
description: Add or update an executed example notebook in the docs — jupytext percent .py while drafting, an executed .ipynb as the committed source, a toc entry in docs/myst.yml, a unique basename, xref links into the API reference, ruff on the notebook cells, and the assembled docs build. Use when asked to write, extend, re-run or fix a tutorial / example notebook under docs/notebooks.
---

# Add or update a notebook

The full standard is `.github/instructions/docs-examples.instructions.md`; read it first. This is the checklist.

## 1. Plan it

- One question per notebook; check the existing ones in `docs/notebooks/` and the `toc` in `docs/myst.yml` so you extend rather than duplicate.
- The basename must be unique across `docs/`, `docs/guide/` and `docs/notebooks/`: mystmd builds flat URLs from basenames, underscores becoming hyphens (`smoothing_deep_dive.ipynb` → `/smoothing-deep-dive/`), and a clash gets an unstable `-1` suffix.
- Use the public API the way a user would (`import mypackage as mp`); no private names.

## 2. Draft in `.py`, commit the `.ipynb`

1. Write `docs/notebooks/<name>.py` in jupytext percent format (header in the instructions file); first markdown cell: title + Colab badge; first code cell: the Colab install guard; then the `watermark` readout.
2. Smoke-run: `uv run --group docs python docs/notebooks/<name>.py`.
3. Convert and execute:

   ```bash
   uv run --group docs jupytext --to notebook docs/notebooks/<name>.py
   uv run --group docs jupyter nbconvert --to notebook --execute \
     docs/notebooks/<name>.ipynb --inplace --ExecutePreprocessor.timeout=180
   ```

4. Delete the `.py`; the executed `.ipynb` is the source of truth. To fix something later, regenerate the `.py` (`jupytext --to py:percent`), edit, re-convert and **re-execute**; never hand-edit committed source cells.

## 3. Content rules

- Figures inline with `plt.show()`: no `savefig`, no committed PNGs; keep the notebook under the pre-commit `check-added-large-files` limit (500 kB by default) with few, small figures.
- Link API names with ``[`summarize`](xref:api#mypackage.summarize)`` — the **top-level** name; a missing target fails the strict build.
- Every number quoted in a markdown cell matches what the cells print.
- Maths in markdown cells follows `eman-style-math` (the macros in `docs/myst.yml`, every symbol defined with its shape, one expression per source line); prose follows `eman-prose-tech` (docs register); code cells follow `python-style` (comments above the line, the Unicode equation above each mathematical line).
- Random draws, if any, are seeded so a re-run reproduces the outputs.
- Code cells ≤ 88 characters (ruff lints the `.ipynb` cells, not the `.py`).

## 4. Wire it in and verify

- Add `- file: notebooks/<name>.ipynb` under the right section of the `toc` in `docs/myst.yml` (the examples sit under `Examples`).
- `uv run --group lint ruff format docs/notebooks/` and `uv run --group lint ruff check .`
- `uv run poe docs-check` (strict validation of both halves) and `uv run poe docs` (builds, assembles and checks every internal link) both need the mystmd CLI (`npm install -g mystmd`); if the theme download is blocked, see `docs/README.md`. Without mystmd, run `uv run poe docs-api` and say in the PR that the prose half was not checked.
