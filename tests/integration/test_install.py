"""How functax installs and imports: the NumPy-only base never loads JAX, a module that
needs the extra names it, and a plan key is the same in a fresh process. Each test
runs a subprocess.
"""

from __future__ import annotations

import subprocess
import sys

import jax.numpy as jnp
import pytest

from functax.geometry import (
    AffineGrid,
    Points,
)
from functax.interp import (
    Linear,
    Policies,
    plan_key,
)


AFFINE = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))


BASE_MODULES = [
    "functax",
    "functax.geometry",
    "functax.geometry.affine",
    "functax._src.geometry._affine",
    "functax._src.geometry._transform",
]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter, so no module is imported yet."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


class TestBaseInstall:
    @pytest.mark.parametrize("module", BASE_MODULES)
    def test_base_modules_never_import_jax(self, module: str) -> None:
        result = _run(
            f"import sys, {module}; "
            "loaded = sorted(n for n in ('jax', 'equinox', 'lineax', 'einx', "
            "'jaxtyping') if n in sys.modules); "
            "assert not loaded, loaded"
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("module", BASE_MODULES)
    def test_base_modules_import_with_jax_blocked(self, module: str) -> None:
        result = _run(f"import sys; sys.modules['jax'] = None; import {module}")
        assert result.returncode == 0, result.stderr

    def test_the_kernels_never_import_jax(self) -> None:
        code = (
            "import sys, functax.geometry as g, numpy as np\n"
            "g.to_pixel(np.zeros(6) + [1, 0, 0, 0, -1, 0], np.zeros((1, 2)))\n"
            "g.from_pixel(np.zeros(6) + [1, 0, 0, 0, -1, 0], np.zeros((1, 2)))\n"
            "import types\n"
            "grid = types.SimpleNamespace(\n"
            "    transform=(1.0, 0, 0, 0, -1.0, 0), shape=(2, 2)\n"
            ")\n"
            "g.pixel_centres(grid)\n"
            "g.window(grid, (0.0, -1.0, 1.0, 0.0))\n"
            "assert 'jax' not in sys.modules, 'jax was imported'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestMissingExtra:
    def test_geometry_types_name_the_extra_when_jax_is_blocked(self) -> None:
        result = _run(
            "import sys; sys.modules['jax'] = None\n"
            "import functax.geometry\n"
            "try:\n"
            "    functax.geometry.Axis\n"
            "except ImportError as err:\n"
            "    assert 'functax[jax]' in str(err), err\n"
            "else:\n"
            "    raise AssertionError('Axis imported without JAX')\n"
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("module", ["functax.interp", "functax.basis"])
    def test_a_jax_module_names_the_extra_when_jax_is_blocked(
        self, module: str
    ) -> None:
        # gate G-message of #48: a module that needs JAX says which extra to install
        result = _run(
            "import sys; sys.modules['jax'] = None\n"
            "try:\n"
            f"    import {module}\n"
            "except ImportError as err:\n"
            "    assert 'functax[jax]' in str(err), err\n"
            "else:\n"
            f"    raise AssertionError('{module} imported without JAX')\n"
        )
        assert result.returncode == 0, result.stderr


class TestAcrossProcesses:
    def test_is_identical_across_processes(self) -> None:
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', True)\n"
            "import jax.numpy as jnp\n"
            "from functax.geometry import AffineGrid, Points\n"
            "from functax.interp import Linear, Policies, plan_key\n"
            "grid = AffineGrid([10.0, 0.0, 500.0, 0.0, -10.0, 900.0], (4, 5))\n"
            "points = Points(jnp.array([[515.0, 885.0], [540.0, 870.0]]))\n"
            "policies = Policies(extent='cells', boundary='edge')\n"
            "print(plan_key(Linear(), grid, points, policies=policies))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        points = Points(jnp.array([[515.0, 885.0], [540.0, 870.0]]))
        policies = Policies(extent="cells", boundary="edge")
        expected = plan_key(Linear(), AFFINE, points, policies=policies)
        assert result.stdout.strip() == expected
