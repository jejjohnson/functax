# Quadrature

A quadrature rule is data: nodes $t_k$ and weights $w_k$ with $\int f \approx \sum_k w_k f(t_k)$. Procedures that adapt or march (adaptive quadrature, scans, time stepping) belong to integrix. `gauss_legendre` is the $n$-point Gauss–Legendre rule on $[-1, 1]$, exact for polynomials of degree up to $2n - 1$, built on the host so it can size static arrays. The sphere grids in `functax.geometry` carry their own weights.

::: functax.quad
