# RSD P0+B000 controlled test

Status: COMPLETE; converged 18/18.

P0 uses the real-space 12-bin edges/density, kmax_P=0.08, with its actual shared CPU binner window. B-only target/model/priors are unchanged.
Targets are same 100-box means. Full matched Fid500 single-box covariance is unscaled: V=1, never /100 or /500.
One independent Pshot residual parameter is added; no IR/FoG/loops or bphidelta/stochastic repair.
Mean100 + Csingle is a forecast/model test, not a calibrated likelihood of the mean.
Posterior truth coverage alone is not a physical validity certificate.

| sample | kmaxB | fNL median ± std | B-only median ± std | std ratio | b1 median | MAP fNL | Δχ² truth |
|---|---:|---:|---:|---:|---:|---:|---:|
| fiducial | 0.08 | -3.26 ± 95.94 | 20.06 ± 191.15 | 0.502 | 1.931 | -1.59 | 0.000 |
| fiducial | 0.10 | 0.76 ± 94.30 | 32.42 ± 163.74 | 0.576 | 1.917 | 1.23 | 0.000 |
| fiducial | 0.12 | 5.83 ± 91.01 | 65.87 ± 157.85 | 0.577 | 1.915 | 13.22 | 0.024 |
| fiducial | 0.14 | 1.85 ± 87.44 | 71.12 ± 148.28 | 0.590 | 1.916 | 10.12 | 0.015 |
| fiducial | 0.18 | 27.80 ± 88.07 | 91.76 ± 123.41 | 0.714 | 1.872 | 37.92 | 0.200 |
| fiducial | 0.20 | 19.69 ± 84.61 | 85.14 ± 113.78 | 0.744 | 1.869 | 29.19 | 0.124 |
| LC_p | 0.08 | 101.42 ± 97.50 | 127.12 ± 170.33 | 0.572 | 1.918 | 99.84 | -0.001 |
| LC_p | 0.10 | 107.35 ± 93.31 | 155.04 ± 149.37 | 0.625 | 1.906 | 101.45 | 0.000 |
| LC_p | 0.12 | 110.20 ± 92.10 | 189.61 ± 144.05 | 0.639 | 1.905 | 115.98 | 0.039 |
| LC_p | 0.14 | 104.68 ± 86.79 | 180.30 ± 136.28 | 0.637 | 1.907 | 112.07 | 0.021 |
| LC_p | 0.18 | 139.96 ± 86.54 | 209.26 ± 119.98 | 0.721 | 1.857 | 147.10 | 0.316 |
| LC_p | 0.20 | 127.67 ± 82.89 | 194.26 ± 113.80 | 0.728 | 1.861 | 135.80 | 0.197 |
| LC_m | 0.08 | -107.89 ± 97.46 | -87.61 ± 205.02 | 0.475 | 1.946 | -100.81 | 0.000 |
| LC_m | 0.10 | -103.43 ± 94.30 | -89.31 ± 170.43 | 0.553 | 1.928 | -97.76 | 0.001 |
| LC_m | 0.12 | -98.13 ± 91.85 | -60.59 ± 163.80 | 0.561 | 1.919 | -87.07 | 0.025 |
| LC_m | 0.14 | -101.09 ± 89.86 | -47.97 ± 152.83 | 0.588 | 1.917 | -89.06 | 0.016 |
| LC_m | 0.18 | -79.29 ± 90.95 | -22.23 ± 121.66 | 0.748 | 1.880 | -67.82 | 0.136 |
| LC_m | 0.20 | -89.51 ± 87.07 | -27.89 ± 113.06 | 0.770 | 1.875 | -77.11 | 0.075 |
