"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/mypackage/skills/series-analysis-with-mypackage/SKILL.md`` is what
agents in downstream projects read before writing code on the package, so a
stale name or a broken example there teaches them the wrong API. These tests
run its worked example, check the numbers its prose quotes, and check that
every ``mp.X`` / ``mypackage.X`` / ``mypackage.cli.X`` it and the plugin's
reviewer name is a current public name.

On adoption, point ``SKILL`` / ``AGENT`` at your plugin, ``_ALIASES`` /
``_NAME`` at your package, and rewrite the claims below from your prose.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "mypackage"
SKILL = PLUGIN / "skills" / "series-analysis-with-mypackage" / "SKILL.md"
AGENT = PLUGIN / "agents" / "mypackage-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the repository root.
    pytest.skip("the plugin is not present", allow_module_level=True)

_ALIASES = {"mp": "mypackage"}
# Not preceded by a word character, dot or slash, so URLs do not match.
_NAME = re.compile(r"(?<![\w./])(mp|mypackage\.cli|mypackage)\.([A-Za-z_]\w*)")
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


# The example is pure Python on 20 numbers and runs in milliseconds, so it
# sits in the fast tier; an example that fits or compiles for over a second
# would be marked ``@pytest.mark.slow``.
def test_worked_example_runs_and_its_claims_hold() -> None:
    (block,) = [b for b in _BLOCKS.findall(SKILL.read_text()) if "readings" in b]
    ns: dict = {}
    exec(block, ns)
    raw, clean = ns["raw"], ns["clean"]
    # The prose quotes rounded values of exact computations on fixed data, so
    # each check rounds to the digits the prose gives.
    # "drags the raw mean to 25.27 °C and the standard deviation to 16.4,
    # while the median stays at 21.7"
    assert round(raw.mean, 2) == 25.27
    assert round(raw.std, 1) == 16.4
    assert round(raw.median, 1) == 21.7
    # "the cleaned maximum is 22.9, the standard deviation 0.90"
    assert clean.maximum == 22.9
    assert round(clean.std, 2) == 0.90
    # "the moving average smears it over three samples near 46"
    assert [round(v) for v in ns["smeared"] if v > 30] == [46, 46, 46]
    # "1.8 to 2.5 training standard deviations above the training mean, so
    # the ±3 clip leaves them alone"
    z_test = ns["z_test"]
    assert (round(min(z_test), 1), round(max(z_test), 1)) == (1.8, 2.5)
    assert max(abs(z) for z in [*ns["z_train"], *z_test]) < 3.0
    # "RunningStats agrees with summarize": Welford's update and the two-pass
    # formula differ only by float rounding (~1e-16 relative here).
    running = ns["running"]
    assert running.mean == pytest.approx(clean.mean, rel=1e-12)
    assert running.std == pytest.approx(clean.std, rel=1e-12)
    # The user-defined transform is a Transform without a base class, and an
    # unfitted Standardize raises rather than passing data through.
    assert isinstance(ns["Difference"](), ns["mp"].Transform)
    assert len(ns["steps"]) == len(ns["test"]) - 1
    assert ns["message"].startswith("Standardize is not fitted")


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path) -> None:
    stale = []
    for prefix, attr in set(_NAME.findall(path.read_text())):
        module_name = _ALIASES.get(prefix, prefix)
        module = importlib.import_module(module_name)
        if attr in getattr(module, "__all__", dir(module)):
            continue
        try:  # a submodule path such as ``mypackage.cli``
            importlib.import_module(f"{module_name}.{attr}")
        except ImportError:
            stale.append(f"{module_name}.{attr}")
    assert not stale, f"{path.name} names objects that do not exist: {sorted(stale)}"
