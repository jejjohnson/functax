# Changelog

## [0.0.4](https://github.com/jejjohnson/functax/compare/v0.0.3...v0.0.4) (2026-11-10)


### Features

* **basis:** move geonnax's EOF and graph-Laplacian bases into functax.basis.eigen ([#78](https://github.com/jejjohnson/functax/issues/78)) ([8d42db4](https://github.com/jejjohnson/functax/commit/8d42db4e1b00d1784cc5ceca9acb22a329a305f3)), closes [#19](https://github.com/jejjohnson/functax/issues/19)
* **basis:** move geonnax's radial, Gabor and wavelet bases into functax.basis.local ([#77](https://github.com/jejjohnson/functax/issues/77)) ([f6c05aa](https://github.com/jejjohnson/functax/commit/f6c05aabf63659f312050020c89943c61972b577))
* **diffops:** move geonnax's field derivatives into functax.diffops ([#80](https://github.com/jejjohnson/functax/issues/80)) ([24a8248](https://github.com/jejjohnson/functax/commit/24a82484ae8c14f481c2c44e899f0d794284943d)), closes [#22](https://github.com/jejjohnson/functax/issues/22)
* **transform:** move geonnax's spherical-harmonic transform and NUFFT into functax.transform ([#82](https://github.com/jejjohnson/functax/issues/82)) ([a83b571](https://github.com/jejjohnson/functax/commit/a83b5711406a5a809ead6a5cbf53880266c07d31)), closes [#23](https://github.com/jejjohnson/functax/issues/23)

## [0.0.3](https://github.com/jejjohnson/functax/compare/v0.0.2...v0.0.3) (2026-11-10)


### Features

* **basis:** move geonnax's box Fourier and vector bases into functax.basis ([#75](https://github.com/jejjohnson/functax/issues/75)) ([a0dc7f2](https://github.com/jejjohnson/functax/commit/a0dc7f2e6fb0b5375d7a395b383936f526ba8428)), closes [#19](https://github.com/jejjohnson/functax/issues/19)
* **basis:** move geonnax's Fourier features into functax.basis.features ([#70](https://github.com/jejjohnson/functax/issues/70)) ([29e3ac9](https://github.com/jejjohnson/functax/commit/29e3ac96630f6821de603705a0e6fc729c12294e))
* **basis:** move geonnax's Legendre and sphere bases into functax.basis ([#73](https://github.com/jejjohnson/functax/issues/73)) ([de9177b](https://github.com/jejjohnson/functax/commit/de9177bd176726330b9ace0edc63d2aa07562113)), closes [#19](https://github.com/jejjohnson/functax/issues/19)
* **basis:** move geonnax's Slepian bases into functax.basis.slepian ([#74](https://github.com/jejjohnson/functax/issues/74)) ([9c3c542](https://github.com/jejjohnson/functax/commit/9c3c542bee234aa8576db894a5698ed9a0fdb983)), closes [#19](https://github.com/jejjohnson/functax/issues/19)
* **geometry:** move geonnax's sphere grids, icosphere and frames into functax.geometry ([#71](https://github.com/jejjohnson/functax/issues/71)) ([f3b9752](https://github.com/jejjohnson/functax/commit/f3b9752369fa8f6fd7a28f76e438e731a80e7b2b)), closes [#20](https://github.com/jejjohnson/functax/issues/20)

## [0.0.2](https://github.com/jejjohnson/functax/compare/v0.0.1...v0.0.2) (2026-11-09)


### Features

* **geometry:** add affine raster maths on any array namespace ([#53](https://github.com/jejjohnson/functax/issues/53)) ([b34ac0a](https://github.com/jejjohnson/functax/commit/b34ac0ab5cd2f47af165bc40dfb3bca722bc5d6e))
* **geometry:** add CRS-free axes, grids and point sets ([#51](https://github.com/jejjohnson/functax/issues/51)) ([73de6f6](https://github.com/jejjohnson/functax/commit/73de6f634cc7f826d550ed789955d6e522ab76fc))
* **geometry:** add the sphere and chord, geodesic and pairwise distances ([#54](https://github.com/jejjohnson/functax/issues/54)) ([94fd28b](https://github.com/jejjohnson/functax/commit/94fd28ba4d9e6df3607116c56f5bd37350d4d28d))
* **interp:** add 1-D cubic splines, PCHIP, Akima and Steffen, with NonlinearPlan.linearize ([#63](https://github.com/jejjohnson/functax/issues/63)) ([d6320bb](https://github.com/jejjohnson/functax/commit/d6320bbdc4aec37c1d4748c89719bb6a2e7105ef))
* **interp:** add DerivativePlan and exact derivatives in the target coordinates ([#60](https://github.com/jejjohnson/functax/issues/60)) ([00fe9bb](https://github.com/jejjohnson/functax/commit/00fe9bbc8ae2c5d70e24e271792fd2d72d3deaf0))
* **interp:** add GaussianPlan and the Brownian-bridge variance of linear interpolation ([#66](https://github.com/jejjohnson/functax/issues/66)) ([80153f2](https://github.com/jejjohnson/functax/commit/80153f2b8918df89661a7720c32457cca98a927d)), closes [#31](https://github.com/jejjohnson/functax/issues/31)
* **interp:** add Interpolant for moving targets, read inside the trace ([#62](https://github.com/jejjohnson/functax/issues/62)) ([be767b9](https://github.com/jejjohnson/functax/commit/be767b98a02cefa7d60754708da4cbaad333af0f))
* **interp:** add Keys cubic convolution and B-splines of order 0 to 5 ([#58](https://github.com/jejjohnson/functax/issues/58)) ([e973300](https://github.com/jejjohnson/functax/commit/e9733006c94776d18556d657c9f84bbd81297c45))
* **interp:** add locate and the outside, missing and boundary policies ([#56](https://github.com/jejjohnson/functax/issues/56)) ([a04c1bf](https://github.com/jejjohnson/functax/commit/a04c1bf04fc8a793d4aabdb6423a684b1f8f3538))
* **interp:** add locate_curvilinear and curvilinear sources for Nearest and Linear ([#67](https://github.com/jejjohnson/functax/issues/67)) ([f37cc1b](https://github.com/jejjohnson/functax/commit/f37cc1b01c75dbf93ed100d6fb28e110d0d52ba1)), closes [#32](https://github.com/jejjohnson/functax/issues/32)
* **interp:** add plan composition, windows and the gaussx export ([#61](https://github.com/jejjohnson/functax/issues/61)) ([03e0954](https://github.com/jejjohnson/functax/commit/03e0954380a1884e44d30a360ba87b89aa05bb14)), closes [#30](https://github.com/jejjohnson/functax/issues/30)
* **interp:** add Scheme and Plan, with Nearest, Linear and Tensor on D-dimensional grids ([#57](https://github.com/jejjohnson/functax/issues/57)) ([1b31c8f](https://github.com/jejjohnson/functax/commit/1b31c8f6a321a8771f1c6d75e31ecbf60799cb86))
* **interp:** add Stencil, the sparse read-out H with its hand-written adjoint ([#55](https://github.com/jejjohnson/functax/issues/55)) ([e7eb8f9](https://github.com/jejjohnson/functax/commit/e7eb8f9423666a0f32fcda1aa5ac88611d823fcc))
* **interp:** add strict-JSON scheme configs, plan bytes and plan_key ([#68](https://github.com/jejjohnson/functax/issues/68)) ([23d0f61](https://github.com/jejjohnson/functax/commit/23d0f618f565fcbf32416e96b13dc7fd8cb13d3b))

## Changelog
