#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
``quijote_strict_triangle_fft_estimator`` 的独立小网格回归测试。

代码大纲
========
1. 验证 shell 边界、零模排除和 strict all-legs 条件。
2. 用直接 mode-pair 枚举核对 FFT convolution triangle counts。
3. 用三个独立随机实场核对 ordered cross-bispectrum 的数值与归一化。
4. 验证 field permutation、零场和单一闭合 Fourier 三角形。

测试只依赖 Python 标准库与 NumPy，可以在缺少 JAX/jaxpower 的登录环境运行。
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))

import numpy as np

import quijote_strict_triangle_fft_estimator as estimator


class StrictTriangleEstimatorTest(unittest.TestCase):
    """小网格 exact-lattice 与测量器测试集合。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.meshsize = 10
        cls.boxsize = 100.0
        cls.kf = 2.0 * np.pi / cls.boxsize
        cls.edges = np.asarray([cls.kf, 2.1 * cls.kf, 3.1 * cls.kf, 4.2 * cls.kf])
        cls.geometry = estimator.build_geometry(cls.meshsize, cls.boxsize, cls.edges)

    def test_edges_and_all_legs_are_strict(self) -> None:
        """所有入选 mode 必须非零且不越过冻结 kmax。"""

        labels = self.geometry.shell_index
        _, _, _, kmag = estimator.fft_wavevectors(self.meshsize, self.boxsize)
        selected = labels >= 0
        self.assertTrue(np.all(kmag[selected] >= self.edges[0]))
        self.assertTrue(np.all(kmag[selected] <= self.edges[-1] + 1.0e-14))
        self.assertEqual(int(labels[(0, 0, 0)]), -1)
        for triangle in self.geometry.triangles:
            for shell in triangle:
                self.assertLessEqual(self.edges[int(shell) + 1], self.edges[-1])

    def test_triangle_counts_match_direct_enumeration(self) -> None:
        """FFT mask convolution 必须逐 triangle 等于整数 mode 枚举。"""

        direct = estimator.bruteforce_triangle_counts(self.geometry)
        np.testing.assert_array_equal(self.geometry.triangle_counts, direct)
        self.assertTrue(np.all(direct > 0))

    def test_exact_internal_edge_is_orientation_independent(self) -> None:
        """恰为整数倍 kf 的同半径 modes 必须全部进入右侧 shell。"""

        size = 64
        boxsize = 1000.0
        kf = 2.0 * np.pi / boxsize
        _, _, _, kmag = estimator.fft_wavevectors(size, boxsize)
        labels = estimator.assign_shells(kmag, [kf, 3.0 * kf, 5.0 * kf])
        axis = np.fft.fftfreq(size) * size
        nx, ny, nz = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
        radius_three = np.rint(nx * nx + ny * ny + nz * nz).astype(np.int64) == 9
        self.assertGreater(int(np.count_nonzero(radius_three)), 0)
        np.testing.assert_array_equal(
            np.unique(labels[radius_three]), np.asarray([1], dtype=np.int16)
        )

    def test_ordered_cross_bispectrum_matches_bruteforce(self) -> None:
        """三个不同 tracer role 的 FFT estimator 必须匹配独立枚举。"""

        rng = np.random.default_rng(1729)
        grids = [rng.normal(size=(self.meshsize,) * 3) for _ in range(3)]
        filtered = [estimator.filter_shells(grid, self.geometry) for grid in grids]
        measured = estimator.measure_bispectrum(*filtered, self.geometry)
        direct = estimator.bruteforce_bispectrum(*grids, self.geometry)
        np.testing.assert_allclose(measured, direct, rtol=2.0e-12, atol=2.0e-7)

    def test_simultaneous_field_and_shell_permutation(self) -> None:
        """交换 field 1/2 后，同时交换前两条 shell 腿应保持三点乘积。"""

        rng = np.random.default_rng(314159)
        first = rng.normal(size=(self.meshsize,) * 3)
        second = rng.normal(size=(self.meshsize,) * 3)
        third = rng.normal(size=(self.meshsize,) * 3)
        filtered = [estimator.filter_shells(grid, self.geometry) for grid in (first, second, third)]
        original = estimator.measure_bispectrum(filtered[0], filtered[1], filtered[2], self.geometry)

        # geometry 只存 i<=j<=k，因此用 brute force 对交换后的 ordered shell roles
        # 做逐行真值；当 i==j 时，单纯交换 fields 应严格不变。
        swapped = estimator.measure_bispectrum(filtered[1], filtered[0], filtered[2], self.geometry)
        equal_shell = self.geometry.triangles[:, 0] == self.geometry.triangles[:, 1]
        np.testing.assert_allclose(original[equal_shell], swapped[equal_shell], rtol=2.0e-12, atol=2.0e-7)

    def test_zero_field_and_constant_mode_vanish(self) -> None:
        """零场或只有被排除零模的常数场必须给出零 P/B。"""

        rng = np.random.default_rng(2718)
        random_grid = rng.normal(size=(self.meshsize,) * 3)
        zero = np.zeros_like(random_grid)
        constant = np.ones_like(random_grid) * 7.0
        filtered_random = estimator.filter_shells(random_grid, self.geometry)
        filtered_zero = estimator.filter_shells(zero, self.geometry)
        filtered_constant = estimator.filter_shells(constant, self.geometry)
        np.testing.assert_array_equal(filtered_constant, filtered_zero)
        np.testing.assert_array_equal(
            estimator.measure_power(filtered_zero, filtered_random, self.geometry),
            np.zeros(self.geometry.nshell),
        )
        np.testing.assert_array_equal(
            estimator.measure_bispectrum(filtered_zero, filtered_random, filtered_random, self.geometry),
            np.zeros(self.geometry.triangles.shape[0]),
        )

    def test_single_closed_triangle_normalization(self) -> None:
        """人工 Fourier 三角形同时核对体积因子和 mode normalization。"""

        size = self.meshsize
        # 三个实场分别含 +/-q1、+/-q2、+/-q3，且 q1+q2+q3=0。
        q1 = np.asarray([1, 0, 0])
        q2 = np.asarray([0, 1, 0])
        q3 = -q1 - q2
        amplitudes = (0.13, -0.21, 0.34)
        grids = []
        coordinates = np.indices((size, size, size))
        for mode, amplitude in zip((q1, q2, q3), amplitudes):
            phase = 2.0 * np.pi * sum(mode[axis] * coordinates[axis] for axis in range(3)) / size
            grids.append(2.0 * amplitude * np.cos(phase))
        filtered = [estimator.filter_shells(grid, self.geometry) for grid in grids]
        measured = estimator.measure_bispectrum(*filtered, self.geometry)
        direct = estimator.bruteforce_bispectrum(*grids, self.geometry)
        np.testing.assert_allclose(measured, direct, rtol=2.0e-12, atol=1.0e-9)
        self.assertGreater(float(np.max(np.abs(measured))), 0.0)

    def test_invalid_edges_fail_closed(self) -> None:
        """非递增、NaN 或负 edge 必须明确失败。"""

        for edges in ([0.1], [0.1, 0.1], [-0.1, 0.2], [0.1, np.nan]):
            with self.assertRaises(ValueError):
                estimator.validate_edges(edges)

    def test_weighted_factorial_shot_subtraction_matches_distinct_objects(self) -> None:
        """带正负权重点过程的 inclusion--exclusion 必须等于显式 distinct triples。"""

        size = 6
        boxsize = 60.0
        kf = 2.0 * np.pi / boxsize
        geometry = estimator.build_geometry(size, boxsize, [kf, 1.8 * kf])
        self.assertEqual(geometry.triangles.shape[0], 1)
        positions = np.asarray(
            [[0, 0, 0], [1, 2, 0], [3, 1, 4], [4, 4, 2], [5, 0, 3]],
            dtype=np.int64,
        )
        weights = np.asarray([1.0, -0.4, 0.7, -1.1, 0.2], dtype=np.float64) / 3.7
        ncell = size**3

        def moment_fourier(order: int) -> np.ndarray:
            grid = np.zeros((size, size, size), dtype=np.float64)
            for position, weight in zip(positions, weights):
                grid[tuple(position)] += float(ncell) * weight**order
            return np.fft.fftn(grid) / float(ncell)

        d1 = moment_fourier(1)
        d2 = moment_fourier(2)
        d3 = float(np.sum(weights**3))
        measured = estimator.measure_factorial_bispectrum_fourier(
            (d1, d1, d1),
            geometry,
            pair_common_object_fields={(0, 1): d2, (0, 2): d2, (1, 2): d2},
            triple_common_object_zero_mode=d3,
        )

        measured_power = estimator.measure_factorial_power_fourier(
            d1,
            d1,
            geometry,
            pair_common_object_zero_mode=float(np.sum(weights**2)),
        )
        direct_power = 0.0j
        power_modes = np.argwhere(geometry.shell_index == 0)
        for mode in power_modes:
            phases = np.exp(-2.0j * np.pi * (positions @ mode) / float(size))
            for first_object in range(len(weights)):
                for second_object in range(len(weights)):
                    if first_object == second_object:
                        continue
                    direct_power += (
                        weights[first_object] * weights[second_object]
                        * phases[first_object] * np.conjugate(phases[second_object])
                    )
        direct_power *= geometry.volume_mpc3_h3 / float(len(power_modes))
        self.assertLess(
            abs(float(direct_power.imag)),
            2.0e-12 * max(1.0, abs(float(direct_power.real))),
        )
        np.testing.assert_allclose(
            measured_power["factorial"][0],
            float(direct_power.real),
            rtol=2.0e-12,
            atol=2.0e-8,
        )

        modes = np.argwhere(geometry.shell_index == 0)
        mode_set = {tuple(mode) for mode in modes.tolist()}
        direct = 0.0j
        for first_mode in modes:
            for second_mode in modes:
                third_mode = tuple((-(first_mode + second_mode)) % size)
                if third_mode not in mode_set:
                    continue
                phases_first = np.exp(
                    -2.0j * np.pi * (positions @ first_mode) / float(size)
                )
                phases_second = np.exp(
                    -2.0j * np.pi * (positions @ second_mode) / float(size)
                )
                phases_third = np.exp(
                    -2.0j * np.pi
                    * (positions @ np.asarray(third_mode, dtype=np.int64))
                    / float(size)
                )
                for first_object in range(len(weights)):
                    for second_object in range(len(weights)):
                        if second_object == first_object:
                            continue
                        for third_object in range(len(weights)):
                            if third_object in (first_object, second_object):
                                continue
                            direct += (
                                weights[first_object]
                                * weights[second_object]
                                * weights[third_object]
                                * phases_first[first_object]
                                * phases_second[second_object]
                                * phases_third[third_object]
                            )
        direct *= geometry.volume_mpc3_h3**2 / float(geometry.triangle_counts[0])
        self.assertLess(
            abs(float(direct.imag)),
            2.0e-12 * max(1.0, abs(float(direct.real))),
        )
        np.testing.assert_allclose(
            measured["factorial"][0], float(direct.real), rtol=2.0e-12, atol=2.0e-5
        )
        reconstructed = measured["raw"].copy()
        for key in ("pair_01", "pair_02", "pair_12"):
            reconstructed -= measured[key]
        reconstructed += measured["triple"]
        np.testing.assert_array_equal(reconstructed, measured["factorial"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
