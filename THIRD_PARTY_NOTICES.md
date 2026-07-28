# Third-party notices

MARISA-B is distributed under the GNU General Public License, version 3 only
(`GPL-3.0-only`), as stated in the repository-root `LICENSE` file.  That file
is a byte-for-byte copy of
`external/ACTio-ReACTio/reactions/COPYING` at the pinned submodule commit
listed below.

This notice records third-party material that is compiled, linked, or used to
generate source distributed by this repository.  Scientific citations are
documented separately from copyright and software-license provenance.

## ACTio-ReACTio

- Project: ACTio-ReACTio.
- Upstream: <https://github.com/nebblu/ACTio-ReACTio>.
- Pinned commit: `a36dda02db6b537a4ddcc7bf044c446c10a58373`.
- Local form: Git submodule at `external/ACTio-ReACTio`.
- Upstream tree inventory SHA256:
  `d2f3b86ac85016aa1c386dd91299284ea72a83ae0480652be02f60c15655d48d`.
- Repository-level license: MIT; `external/ACTio-ReACTio/LICENSE` has SHA256
  `b8b9c247a9d453f0669a8c6104b08293b31851a6cdf0f0dcbe08b5bf6dcaf5a1`.
- Compiled `reactions` subtree license: GPL-3.0-only;
  `external/ACTio-ReACTio/reactions/COPYING` has SHA256
  `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903`.
- Repository-level MIT copyright notice: Copyright (c) 2022 Benjamin Bose,
  Maria Tsedrik, Tilman Troester, and Qianli Xia.

The default native/EFT-v2 targets compile `reactions/src/Common.cpp`,
`PowerSpectrum.cpp`, `array.cpp`, and `Quadrature.cpp`.  The optional triangle
CLI and ReACT diagnostic adapter additionally compile `BSPT.cpp`,
`Cosmology.cpp`, `InterpolatedPS.cpp`, `SPT.cpp`, `SpecialFunctions.cpp`, and
`Spline.cpp`, together with transitively included headers.  MARISA-B also uses
interfaces from this subtree in its native and EFT-v2 sources.  The submodule
is not modified by this repository.

## OneLoopBispectrum kernel tables

- Project: OneLoopBispectrum.
- Authors identified by upstream: Oliver Philcox, Mikhail Ivanov, and Marko
  Simonovic.
- Upstream: <https://github.com/oliverphilcox/OneLoopBispectrum>.
- Pinned commit: `5d951372e6a49d006e72d73247b5e085bfe50b73`.
- License: GPL-3.0-only.
- Upstream `LICENSE` SHA256:
  `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`.

The following upstream WDX inputs were verified directly against that pinned
commit:

| Upstream file | SHA256 |
|---|---|
| `kernels/tab222freq-flat.wdx` | `803531c5786dd5b4416d277a3abe74f57204444902949b42bf82b54fe8125fd1` |
| `kernels/tab222derivs-flat.wdx` | `5eb75bd7033d2786fb885ff448215ca6f1b8b00e4084aeb43552eec89b3cba7b` |
| `kernels/tab321freq-flat.wdx` | `9d348de88bb263e63d695d1d29c3c9141373e6a27b91973b00ad6e5768e86171` |
| `kernels/tab321derivs-flat.wdx` | `99ad3047700fea373762c95b008ea2ece1e670869da37e3c3fb8e377f54bb280` |
| `kernels/tab411freq-flat.wdx` | `8a3cf8f0853504c06c705bc8598e0ab71278458693b02d777f9e4031bdd09f33` |
| `kernels/tab411derivs-flat.wdx` | `df91008f35ef12ffb7163716e7897b449264e2f9ddb52c57a70413964639a059` |
| `kernels/b411uv-flat.wdx` | `5ed29ddf0b6483116c44876771297e470f64413c1ad9225f545706fae15db291` |

On 2026-07-23, MARISA-B decoded these inputs, selected the real-space operator
rows, and converted the resulting exact rational data into C++ constant
tables.  The modified/generated source is:

- `src/eft_v2/generated_b222_b321i_tables.h`, SHA256
  `6cf00d544606acbc7f8317bb84180b65a841a7175e5e513d673265e7508bbcc5`;
- `src/eft_v2/generated_b411_table.h`, SHA256
  `8e08e579fec06ab19b6bb9d8a48a4aef5db3173c68d42afd1f2bbae8638c46c6`;
- `scripts/generate_eft_v2_b411_table.py`, SHA256
  `42618405f57d5816b320acda48ec4d1d428549ec82eac040c6b46fc78b7a6ef2`.

The generated headers are compiled by the EFT-v2 FFTLog/DR implementation.
The upstream WDX files themselves are not vendored in this repository; their
exact identity and their transformation into the distributed C++ source are
recorded above.

## GNU Scientific Library

MARISA-B links against the GNU Scientific Library (GSL), which is licensed
under GPL-3.0-or-later.  GSL source and binaries are not vendored here.  The
exact GSL package and license must be recorded by each formal build manifest.
