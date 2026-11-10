# Quadrature

A quadrature rule is data: nodes $x_n$ and weights $w_n$ with $\int g \mathrm{d}x \approx \sum_n w_n g(x_n)$. `Rule` holds the two arrays and nothing else; integrix sums, scans and adapts with it (integrix design decision 3), and unpacking a rule gives `(nodes, weights)`.

Four constructors cover the 1-D rules. `gauss_legendre(n, interval)` is the $n$-point Gauss–Legendre rule, exact for polynomials of degree up to $2n - 1$, and `clenshaw_curtis(n, interval)` the $n$-point Clenshaw–Curtis rule on the Chebyshev extreme points, exact to degree $n - 1$ ($n$ when $n - 1$ is even). Both are built on the host in float64, so they can size static arrays, and map affinely from $[-1, 1]$ onto any interval. `trapezoid(nodes)` and `simpson(nodes)` take the nodes a field already lives on, uniform or not, ascending or descending; they equal `scipy.integrate.trapezoid` and `scipy.integrate.simpson` (with its correction for an even node count) and are traceable, so their weights are differentiable in the nodes. The sphere grids in `functax.geometry` carry their own weights.

::: functax.quad
