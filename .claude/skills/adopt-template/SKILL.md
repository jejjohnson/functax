---
name: adopt-template
description: Turn a fresh copy of this template into your own project — rename the package, the repository and every reference to them, delete or replace the example package and its tests, docs and notebooks, rewrite every passage marked template-example in the agent files from the new code, rewrite the example building skills and reviewers for the new extension points, regenerate the capability index, update labels, release-please and docs URLs, and verify. Use when asked to adopt, bootstrap, rename or start a project from this template, or to remove the example package.
---

# Adopt the template

Work on a branch, in this order: rename first and prove the copy still passes (step 3), then replace the example (step 4), then rewrite the agent files from the code that now exists (steps 5–6). Every step ends green.

## 1. Choose the names

| Placeholder | Meaning | Template value |
|---|---|---|
| `NEWPKG` | import package and distribution name (snake_case) | `mypackage` |
| `NewPkg` | its CamelCase form, for the base exception | `Mypackage` (in `MypackageError`) |
| `NEWREPO` | GitHub repository name | `pypackage_template` |
| `OWNER` | GitHub owner | `jejjohnson` |

Also the author's name and email (`pyproject.toml`, `mkdocs.yml` `site_author`, `docs/myst.yml` `authors`, `LICENSE`).

## 2. Find every reference

```bash
git grep -n -e mypackage -e Mypackage -e pypackage_template   # contents
git ls-files | grep -i mypackage                                # paths
git grep -n jejjohnson                                          # owner, URLs
git grep -n "template-example"                                  # passages to rewrite (step 5)
```

The paths are `src/mypackage/`, `.claude/skills/mypackage-review/` and the plugin under `plugins/mypackage/` (its skill and agent file names carry the package name too).

## 3. Rename, and prove the copy still passes

```bash
NEWPKG=yourpkg NEWPKG_CAMEL=Yourpkg NEWREPO=your-repo
git mv src/mypackage "src/$NEWPKG"
git mv .claude/skills/mypackage-review ".claude/skills/$NEWPKG-review"
git mv plugins/mypackage "plugins/$NEWPKG"
git mv "plugins/$NEWPKG/skills/series-analysis-with-mypackage" \
       "plugins/$NEWPKG/skills/series-analysis-with-$NEWPKG"
git mv "plugins/$NEWPKG/agents/mypackage-reuse-reviewer.md" \
       "plugins/$NEWPKG/agents/$NEWPKG-reuse-reviewer.md"
git grep -l -e mypackage -e Mypackage -e pypackage_template -- \
    ':!uv.lock' ':!CHANGELOG.md' ':!.claude/skills/adopt-template' \
  | xargs sed -i -e "s/mypackage/$NEWPKG/g" -e "s/Mypackage/$NEWPKG_CAMEL/g" \
                 -e "s/pypackage_template/$NEWREPO/g"
uv lock && uv sync --all-groups
uv run poe format         # the new names re-sort imports and __all__
uv run poe lint           # a longer name can push a line past 88 characters
uv run poe capabilities   # the index now names the new package
uv run poe test-cov && uv run poe typecheck
```

(`sed -i ''` on macOS.) `uv run poe lint` (and `uv run poe format`, which exits non-zero for the same reason) flags any line the new name made too long: wrap it by hand (a longer name usually pushes the two inventory strings in `test_reads_entries` in `tests/test_build_docs.py` past the limit; split each string in two). Then by hand:

- `jejjohnson` → your owner where it is a URL or a repository (`docs/myst.yml`, `mkdocs.yml`, `pyproject.toml` `[project.urls]`, `README.md` badges, the notebooks' Colab badges); the author's name and email.
- `CHANGELOG.md`: start it fresh; reset `.release-please-manifest.json` to your first version (and `version` in `pyproject.toml`, `__version__` in `src/<pkg>/__init__.py` to match).
- The executed notebooks were renamed as text, but their stored outputs came from the old package: re-execute the ones you keep (`add-notebook`), or delete them in step 4.

## 4. Replace the example package

Keep one module as a reference while you write your own, or delete it all. The example touches:

| What | Where |
|---|---|
| Code | `src/<pkg>/*.py` (keep `__init__.py`, `py.typed`; `exceptions.py` is a good base) |
| Tests | `tests/test_{stats,smoothing,transforms,utils,cli,exceptions}.py`, the fixtures in `tests/conftest.py`, `SUBMODULES` in `tests/test_public_api.py` |
| CLI | `[project.scripts]` in `pyproject.toml`, `tests/test_cli.py` (its `test_entry_point_is_declared`), `docs/guide/cli.md`, `docs/api/cli.md` |
| API reference | `docs/api/*.md`, the `nav` of `mkdocs.yml`, the module table in `docs/api/index.md` |
| Prose | `docs/index.md`, `docs/guide/*.md`, `docs/notebooks/*.ipynb`, the `toc` of `docs/myst.yml` |
| Generic tests to keep | `tests/test_public_api.py`, `tests/test_capabilities.py`, `tests/test_build_docs.py` |

The coverage gate (`fail_under = 90`) fails an empty package: write tests with the code. `uv run poe capabilities` after every public API change.

## 5. Rewrite the agent files from the new code

`git grep -n "template-example"` lists every passage written for the example. Rewrite each from your code, check every name, path and command it cites, then delete the marker:

- `AGENTS.md`: what the repo is (module map), boundaries, what it is built on, the reuse table, the contracts, what enforces them, the random-draws note;
- `.github/copilot-instructions.md`, `CODE_REVIEW.md` (section 2 and "Package-Specific Checks" — run every code example you put there), `.github/instructions/python-coding.instructions.md`;
- `scripts/capabilities.py`: list the libraries you build on in `UPSTREAM` (with a blurb) and any deliberate name clash in `ALLOWED_SHARED_NAMES`.

The unmarked rules (git and PR rules, the review-thread commands, test tiers, the pre-commit checklist, the docs toolchain, the house style) stay as they are. If the new package uses JAX, the JAX rules of `python-style` now apply in full ("House style" in `AGENTS.md`); rewrite the paragraph there that records the stdlib example's departures from it.

## 6. Rewrite the skills and reviewers

- The building skills `add-function`, `add-transform`, `add-cli-command` are the example's extension points: rewrite one skill per extension point of your package (where it goes, the base class or protocol, the registry or test list it joins, the docs page, the exact `pytest` command that selects its tests), and delete the ones that do not apply.
- `.claude/skills/<pkg>-review`: rewrite step 4 from your contracts.
- `.claude/agents/correctness-reviewer.md`: replace it with a reviewer for your domain (numerics, I/O, concurrency, security, …); `.claude/agents/reuse-reviewer.md`: replace its list of usual re-implementations.
- `.claude/skills/squash-commit`: your scopes.
- Update the skill lists in `CLAUDE.md`, the "Recipes" table in `AGENTS.md` and `.github/copilot-instructions.md`.
- The downstream plugin (`plugins/<pkg>/`, `.claude-plugin/marketplace.json`): rename its skill to `<verb>-with-<pkg>` and rewrite it for your package — where to find the capability index, what lives where, the rules, a worked example that runs, a "don't write it" table; rewrite its reuse reviewer; point `tests/test_plugin_skill.py` at them and rewrite its claims from the new prose. Rewrite `docs/llms.txt` and `docs/agents.md` and check every URL in them against the built site (`uv run poe docs`, then look in `public/`). A package nobody else builds on can delete the plugin, `llms.txt` (and its entry in `ROOT_FILES`), `docs/agents.md` and the test instead.

## 7. Labels, releases, URLs

- `.github/scripts/create-labels.sh`: your `area:*` / `layer:*` / `wave:*` labels, then `uv run poe gh-labels`; the label table in `.claude/skills/create-gh-issue/SKILL.md` and `docs/contributing.md`.
- `release-please-config.json`: `package-name` (renamed in step 3) and `extra-files`.
- Docs URLs: `site.nav` in `docs/myst.yml` (absolute, your Pages URL), `repo_url` / `repo_name` in `mkdocs.yml`, the `homepage` and `repository` in `plugins/<pkg>/.claude-plugin/plugin.json` and the owner in `.claude-plugin/marketplace.json`; enable GitHub Pages (Actions) and, for coverage, `CODECOV_TOKEN`.
- `.github/ISSUE_TEMPLATE/config.yml`: your Discussions URL, if any.
- `README.md`: rewrite for your project.

## 8. Verify

- The `pre-pr-check` skill, including `uv run poe docs`.
- `git grep -n -e mypackage -e Mypackage -e pypackage_template` finds nothing you did not mean to keep.
- Remove the description of the marker convention, which no longer applies: the note at the top of `AGENTS.md`, the "Template markers" bullet in `CLAUDE.md`, the sentence in `.github/copilot-instructions.md` and the paragraph in `README.md`. `git grep -n "template-example"` then finds only this skill.
- Delete this skill (`.claude/skills/adopt-template/`), its row in the "Recipes" table of `AGENTS.md` and its mention in `CLAUDE.md`: the project is adopted.
