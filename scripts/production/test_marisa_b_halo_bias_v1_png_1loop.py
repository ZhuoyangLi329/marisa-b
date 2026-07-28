#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MARISA-B halo bias-v1 local-PNG truncated one-loop 的可重复回归测试。

执行大纲
--------
1. 在有限周期动量格点上做 quadratic substitution 与 exact Wick enumeration；
2. 通过公开 C++ CLI 核对新 one-loop mode 的 tree 与 Barreira tree mode；
3. 独立核对 finite-R post K3/K4，并验收 pre/post 全六置换稳定性；
4. 核对旧 DM mode 的 JSON 边界没有被新字段污染；
5. 把 matter 极限与旧显式 B0 routing 的大 cutoff 差异记录为诊断。

最后一项不是系数 oracle：有限球形硬 cutoff 不保持 loop-momentum 平移，故两种
routing 只在 regulator 边界影响足够小时接近。测试使用大 qmax 并设置宽松的
2% 诊断阈值；KIC 组合系数的严格验收来自第一项的 exact Wick identity。
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MARISA_B_DATA_ROOT", PROJECT_ROOT)).resolve()
DEFAULT_BINARY = PROJECT_ROOT / "build" / "marisa_b" / "marisa_b_triangle"
DEFAULT_TABLE = (
    DATA_ROOT
    / "analysis"
    / "theory_vectors"
    / "marisa_b_v0"
    / "marisa_b_pre_local_png_1loop_z1_diag15_kmax0p3_nmu48_eps1e3_partial446_20260616_plin_pphi_m_z1.dat"
)
CURRENT_Z05_TABLE = (
    PROJECT_ROOT
    / "analysis"
    / "theory_vectors"
    / "marisa_b_v0"
    / "marisa_b_pre_gaussian_point_diag15_kmax0p3_nmu12_20260615_plin_z0.5.dat"
)
HISTORICAL_PRE_PAYLOAD = (
    PROJECT_ROOT
    / "analysis"
    / "theory_vectors"
    / "marisa_b_v0"
    / "marisa_b_native_triangle_eq005_eps1e3_smoke.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--png-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--skip-matter-routing-diagnostic",
        action="store_true",
        help="跳过 qmax=70 的新/旧 matter routing 诊断，只运行严格恒等式与快速 CLI 测试。",
    )
    return parser.parse_args()


def relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1.0e-300)
    return abs(left - right) / scale


def run_binary(
    binary: Path,
    table: Path,
    mode: str,
    triangles: list[tuple[float, float, float]],
    *,
    qmin: float = 0.01,
    qmax: float = 0.02,
    epsrel: float = 0.1,
    p13_epsrel: float = 0.1,
    smoothing_radius: float = 15.0,
    bias_recon: float = 1.0,
    bias: tuple[float, float, float, float, float] = (2.7, 0.3, -0.2, 5.8, 3.5),
    bphi2: float = 0.0,
) -> dict[str, Any]:
    """调用公开 CLI；参数与返回值都保留双精度 JSON，不生成仓库内文件。"""

    b1, b2, bk2, bphi, bphidelta = bias
    command = [
        str(binary),
        "--pk-table",
        str(table),
        "--png-table",
        str(table),
        "--mode",
        mode,
        "--backend",
        "native_cpp",
        "--qmin",
        f"{qmin:.17g}",
        "--qmax",
        f"{qmax:.17g}",
        "--epsrel",
        f"{epsrel:.17g}",
        "--p13-epsrel",
        f"{p13_epsrel:.17g}",
        "--smoothing-radius",
        f"{smoothing_radius:.17g}",
        "--bias-recon",
        f"{bias_recon:.17g}",
        "--b1",
        f"{b1:.17g}",
        "--b2",
        f"{b2:.17g}",
        "--bk2",
        f"{bk2:.17g}",
        "--bphi",
        f"{bphi:.17g}",
        "--bphidelta",
        f"{bphidelta:.17g}",
        "--bphi2",
        f"{bphi2:.17g}",
    ]
    for k1, k2, mu12 in triangles:
        command.extend(["--triangle", f"{k1:.17g},{k2:.17g},{mu12:.17g}"])
    single_core_env = os.environ.copy()
    single_core_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=single_core_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"CLI failed ({mode}): {completed.stderr}")
    return json.loads(completed.stdout)


def exact_lattice_wick_test() -> dict[str, Any]:
    """用两条独立路径在 Z_11 上验收 quadratic-IC 与 Wick 组合系数。

    expected 路径从 ``delta_L=g+f Q*g*g`` 出发：逐个替换 Gaussian 场中的
    一个 seed，再对所有带标签外腿做 n! 对称化。它不知道 KIC 的 pair-sum
    闭式，也不调用下方被测的闭式 kernel。diagram 路径才使用
    ``(2/n) sum_ij Q_ij K^G_{n-1}``。因此两者相等会真正验收 2/n、零模删除，
    以及 K2/K3/K4，而不是让同一个 helper 与自己比较。
    """

    ngrid = 11
    external = (1, 2, 8)  # 1+2+8=0 mod 11
    coefficients = {1: 1.1, 2: 0.37, 3: -0.12, 4: 0.045}

    def signed_size(momentum: int) -> int:
        wrapped = momentum % ngrid
        return min(wrapped, ngrid - wrapped)

    def power(momentum: int) -> float:
        if momentum % ngrid == 0:
            return 0.0
        return 0.7 + 0.09 * signed_size(momentum)

    def transfer_nonzero(momentum: int) -> float:
        """只允许非零模；若实现错误地外推 M(0)，测试应立即失败。"""

        if momentum % ngrid == 0:
            raise AssertionError("quadratic-IC path queried forbidden merged M(0)")
        size = signed_size(momentum)
        return 0.8 + 0.11 * size + 0.007 * size * size

    @functools.cache
    def gaussian_kernel(arguments: tuple[int, ...]) -> float:
        # 对称、parity-even，但并非常数；这样各 merged-pair 路径彼此可区分。
        shape = sum(signed_size(item) ** 2 for item in arguments)
        return coefficients[len(arguments)] * (1.0 + 0.013 * shape)

    def pair_q(left: int, right: int) -> float:
        merged = (left + right) % ngrid
        if merged == 0 or left % ngrid == 0 or right % ngrid == 0:
            return 0.0
        return transfer_nonzero(merged) / (
            transfer_nonzero(left) * transfer_nonzero(right)
        )

    @functools.cache
    def quadratic_substitution_expected(arguments: tuple[int, ...]) -> float:
        """直接替换一个 Gaussian seed，并做 n! 外腿对称化的 expected。"""

        order = len(arguments)
        if order < 2:
            return 0.0
        gaussian_order = order - 1
        accumulated = 0.0
        # permutation[0:2] 是 quadratic seed 的两个有序输入；其余输入依次
        # 填入剩余 Gaussian slots。再遍历 quadratic seed 所在的 slot。
        for permutation in itertools.permutations(arguments):
            # 这里故意不调用 candidate 路径的 pair_q helper；直接从
            # delta_L 的 quadratic substitution 定义重建 Q(left,right)。
            merged = (permutation[0] + permutation[1]) % ngrid
            if (
                merged == 0
                or permutation[0] % ngrid == 0
                or permutation[1] % ngrid == 0
            ):
                continue
            q_value = transfer_nonzero(merged) / (
                transfer_nonzero(permutation[0])
                * transfer_nonzero(permutation[1])
            )
            for quadratic_slot in range(gaussian_order):
                gaussian_arguments = list(permutation[2:])
                gaussian_arguments.insert(quadratic_slot, merged)
                accumulated += q_value * gaussian_kernel(tuple(gaussian_arguments))
        return accumulated / math.factorial(order)

    @functools.cache
    def kic_closed_form_under_test(arguments: tuple[int, ...]) -> float:
        """与 native KIC 约定同形的闭式；只供 diagram/candidate 路径使用。"""

        order = len(arguments)
        if order < 2:
            return 0.0
        pair_sum = 0.0
        for i in range(order):
            for j in range(i + 1, order):
                q_value = pair_q(arguments[i], arguments[j])
                if q_value == 0.0:
                    continue
                gaussian_arguments = [
                    (arguments[i] + arguments[j]) % ngrid,
                    *(
                        arguments[index]
                        for index in range(order)
                        if index != i and index != j
                    ),
                ]
                pair_sum += q_value * gaussian_kernel(tuple(gaussian_arguments))
        return (2.0 / order) * pair_sum

    order_inputs = {
        2: (2, 3),
        3: (2, 9, 4),   # 2+9=0：必须删除这一 merged zero-mode pair。
        4: (1, 10, 3, 5),  # 1+10=0：K4 再独立覆盖一次零模删除。
    }
    order_checks: dict[str, dict[str, float]] = {}
    for order, arguments in order_inputs.items():
        expected = quadratic_substitution_expected(arguments)
        candidate = kic_closed_form_under_test(arguments)
        raw_pair_sum = candidate / (2.0 / order)
        inferred_factor = expected / raw_pair_sum
        error = relative_difference(candidate, expected)
        factor_error = relative_difference(inferred_factor, 2.0 / order)
        if error > 2.0e-14 or factor_error > 2.0e-14:
            raise AssertionError(
                f"independent quadratic-IC K{order} substitution failed: "
                f"value={error}, factor={factor_error}"
            )
        order_checks[f"K{order}"] = {
            "substitution_expected": expected,
            "kic_closed_form": candidate,
            "relative_error": error,
            "inferred_pair_sum_factor": inferred_factor,
            "required_pair_sum_factor": 2.0 / order,
        }

    zero_mode_pairs = ((2, 9), (1, 10))
    zero_mode_q_values = {
        f"{left}+{right}": pair_q(left, right) for left, right in zero_mode_pairs
    }
    if any(value != 0.0 for value in zero_mode_q_values.values()):
        raise AssertionError(f"merged zero-mode was not removed: {zero_mode_q_values}")

    def momentum_tuples(order: int, total: int):
        for head in itertools.product(range(ngrid), repeat=order - 1):
            yield head + ((total - sum(head)) % ngrid,)

    @functools.cache
    def wick(momenta: tuple[int, ...]) -> float:
        if not momenta:
            return 1.0
        first = momenta[0]
        total = 0.0
        for partner in range(1, len(momenta)):
            covariance = power(first) if (first + momenta[partner]) % ngrid == 0 else 0.0
            if covariance != 0.0:
                total += covariance * wick(momenta[1:partner] + momenta[partner + 1 :])
        return total

    def direct_three_field_response(total_degree: int) -> float:
        total = 0.0
        for response_leg in range(3):
            other_legs = [index for index in range(3) if index != response_leg]
            for response_order in range(2, 5):
                for first_order in range(1, 5):
                    second_order = total_degree - response_order - first_order
                    if not 1 <= second_order <= 4:
                        continue
                    orders = [0, 0, 0]
                    orders[response_leg] = response_order
                    orders[other_legs[0]] = first_order
                    orders[other_legs[1]] = second_order
                    for args0 in momentum_tuples(orders[0], external[0]):
                        for args1 in momentum_tuples(orders[1], external[1]):
                            for args2 in momentum_tuples(orders[2], external[2]):
                                arguments = (args0, args1, args2)
                                expectation = wick(args0 + args1 + args2)
                                if expectation == 0.0:
                                    continue
                                kernels = [gaussian_kernel(item) for item in arguments]
                                # expected 只来自逐 seed substitution，绝不调用 KIC 闭式。
                                kernels[response_leg] = quadratic_substitution_expected(
                                    arguments[response_leg]
                                )
                                total += math.prod(kernels) * expectation
        return total

    def product_response(factors: list[tuple[float, float]]) -> float:
        total = 0.0
        for response_index, (_, response) in enumerate(factors):
            term = response
            for index, (gaussian, _) in enumerate(factors):
                if index != response_index:
                    term *= gaussian
            total += term
        return total

    p_external = [power(value) for value in external]
    cyclic_pairs = ((0, 1), (1, 2), (2, 0))
    tree = 0.0
    for i, j in cyclic_pairs:
        factors = [
            (gaussian_kernel((external[i],)), kic_closed_form_under_test((external[i],))),
            (gaussian_kernel((external[j],)), kic_closed_form_under_test((external[j],))),
            (
                gaussian_kernel((external[i], external[j])),
                kic_closed_form_under_test((external[i], external[j])),
            ),
        ]
        tree += 2.0 * product_response(factors) * p_external[i] * p_external[j]

    b222 = 0.0
    b321i = 0.0
    b411 = 0.0
    for loop in range(ngrid):
        args_a = (loop, (external[0] - loop) % ngrid)
        args_b = ((-loop) % ngrid, (external[1] + loop) % ngrid)
        args_c = ((external[0] - loop) % ngrid, (external[1] + loop) % ngrid)
        b222 += (
            8.0
            * product_response(
                [
                    (gaussian_kernel(args_a), kic_closed_form_under_test(args_a)),
                    (gaussian_kernel(args_b), kic_closed_form_under_test(args_b)),
                    (gaussian_kernel(args_c), kic_closed_form_under_test(args_c)),
                ]
            )
            * power(loop)
            * power(args_a[1])
            * power(args_b[1])
        )
        for split_index in range(3):
            split_minus_loop = (external[split_index] - loop) % ngrid
            for linear_index in range(3):
                if linear_index == split_index:
                    continue
                args1 = (external[linear_index],)
                args2 = (loop, split_minus_loop)
                args3 = (external[linear_index], loop, split_minus_loop)
                b321i += (
                    6.0
                    * product_response(
                        [
                            (gaussian_kernel(args1), kic_closed_form_under_test(args1)),
                            (gaussian_kernel(args2), kic_closed_form_under_test(args2)),
                            (gaussian_kernel(args3), kic_closed_form_under_test(args3)),
                        ]
                    )
                    * power(external[linear_index])
                    * power(loop)
                    * power(split_minus_loop)
                )
        for i, j in cyclic_pairs:
            args_i = (external[i],)
            args_j = (external[j],)
            args4 = (external[i], external[j], loop, (-loop) % ngrid)
            b411 += (
                12.0
                * product_response(
                    [
                        (gaussian_kernel(args_i), kic_closed_form_under_test(args_i)),
                        (gaussian_kernel(args_j), kic_closed_form_under_test(args_j)),
                        (gaussian_kernel(args4), kic_closed_form_under_test(args4)),
                    ]
                )
                * power(external[i])
                * power(external[j])
                * power(loop)
            )

    i3_gaussian = [
        sum(gaussian_kernel((k, loop, (-loop) % ngrid)) * power(loop) for loop in range(ngrid))
        for k in external
    ]
    i3_response = [
        sum(
            kic_closed_form_under_test((k, loop, (-loop) % ngrid)) * power(loop)
            for loop in range(ngrid)
        )
        for k in external
    ]
    b321ii = 0.0
    for i, j in cyclic_pairs:
        g2 = gaussian_kernel((external[i], external[j]))
        r2 = kic_closed_form_under_test((external[i], external[j]))
        g1i = gaussian_kernel((external[i],))
        g1j = gaussian_kernel((external[j],))
        r1i = kic_closed_form_under_test((external[i],))
        r1j = kic_closed_form_under_test((external[j],))
        b321ii += 6.0 * p_external[i] * p_external[j] * (
            r2 * (g1i * i3_gaussian[j] + g1j * i3_gaussian[i])
            + g2
            * (
                r1i * i3_gaussian[j]
                + g1i * i3_response[j]
                + r1j * i3_gaussian[i]
                + g1j * i3_response[i]
            )
        )

    direct_tree = direct_three_field_response(4)
    direct_loop = direct_three_field_response(6)
    diagram_loop = b222 + b321i + b321ii + b411
    if relative_difference(tree, direct_tree) > 5.0e-13:
        raise AssertionError(
            f"exact lattice tree Wick identity failed: direct={direct_tree}, diagrams={tree}"
        )
    if relative_difference(diagram_loop, direct_loop) > 5.0e-13:
        raise AssertionError(
            f"exact lattice one-loop Wick identity failed: direct={direct_loop}, "
            f"diagrams={diagram_loop}"
        )
    return {
        "quadratic_ic_order_checks": order_checks,
        "merged_zero_mode_q_values": zero_mode_q_values,
        "direct_tree": direct_tree,
        "diagram_tree": tree,
        "direct_one_loop": direct_loop,
        "diagram_B222": b222,
        "diagram_B321I": b321i,
        "diagram_B321II": b321ii,
        "diagram_B411": b411,
        "diagram_one_loop": diagram_loop,
        "relative_error_one_loop": relative_difference(diagram_loop, direct_loop),
    }


def finite_r_post_product_rule_test() -> dict[str, Any]:
    """独立验收 finite-R reconstruction 对 K3/K4 的一次 response product rule。

    candidate 按 native set-partition 公式显式传播 dual number；expected 则从
    ``delta_h exp(-i k.s)`` 的场级有序 density/shift 展开出发，对 n! 外腿和
    l! 个不可区分 shift factors 直接求和。expected 不调用 set-partition 或
    dual/product-response helper，并用 complex-step 取导数；R=1.7 明确不是
    R->infinity。
    """

    smoothing_radius = 1.7
    bias_recon = 2.3
    complex_step = 1.0e-30

    @functools.cache
    def set_partitions(items: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
        if not items:
            return ((),)
        first = items[0]
        out: list[tuple[tuple[int, ...], ...]] = []
        for partition in set_partitions(items[1:]):
            out.append(((first,), *partition))
            for block_index in range(len(partition)):
                blocks = list(partition)
                blocks[block_index] = (first, *blocks[block_index])
                out.append(tuple(blocks))
        return tuple(out)

    def pre_gaussian(arguments: tuple[float, ...]) -> float:
        order = len(arguments)
        return (0.43 + 0.19 * order) * (
            1.0 + 0.07 * sum(item * item for item in arguments)
        )

    def pre_response(arguments: tuple[float, ...]) -> float:
        order = len(arguments)
        return (-0.21 + 0.08 * order) * (
            1.0 + 0.05 * sum(abs(item) for item in arguments)
        )

    def shift_geometry(total: float, block_arguments: tuple[float, ...]) -> float:
        block_sum = sum(block_arguments)
        if abs(block_sum) <= 1.0e-14:
            return 0.0
        window = math.exp(-0.5 * (smoothing_radius * abs(block_sum)) ** 2)
        return -(total * block_sum / (block_sum * block_sum)) * window / bias_recon

    def reconstruction_terms(arguments: tuple[float, ...]):
        order = len(arguments)
        total = sum(arguments)
        for mask in range(1, 1 << order):
            density_labels = tuple(index for index in range(order) if mask & (1 << index))
            remaining_labels = tuple(index for index in range(order) if not mask & (1 << index))
            density_arguments = tuple(arguments[index] for index in density_labels)
            for partition in set_partitions(remaining_labels):
                coefficient = math.factorial(len(density_labels)) / math.factorial(order)
                factors = [density_arguments]
                shift_factors: list[float] = [1.0]
                for block in partition:
                    block_arguments = tuple(arguments[index] for index in block)
                    coefficient *= math.factorial(len(block))
                    factors.append(block_arguments)
                    shift_factors.append(shift_geometry(total, block_arguments))
                yield coefficient, factors, shift_factors

    @functools.cache
    def positive_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
        """把 total 写成 parts 个正整数；场级展开用，不依赖 set partitions。"""

        if parts == 1:
            return ((total,),) if total >= 1 else ()
        out: list[tuple[int, ...]] = []
        for first in range(1, total - parts + 2):
            for remainder in positive_compositions(total - first, parts - 1):
                out.append((first, *remainder))
        return tuple(out)

    def direct_field_level_post(
        arguments: tuple[float, ...], deformation: complex
    ) -> complex:
        """直接展开 density 乘 exp(shift)，不调用 candidate 的 partition 枚举。"""

        order = len(arguments)
        total_momentum = sum(arguments)
        value = 0.0j
        labels = tuple(range(order))
        for shift_count in range(order):
            for block_sizes in positive_compositions(order, shift_count + 1):
                for permutation in itertools.permutations(labels):
                    offset = block_sizes[0]
                    density_arguments = tuple(
                        arguments[index] for index in permutation[:offset]
                    )
                    term = (
                        pre_gaussian(density_arguments)
                        + deformation * pre_response(density_arguments)
                    )
                    for block_size in block_sizes[1:]:
                        block_arguments = tuple(
                            arguments[index]
                            for index in permutation[offset : offset + block_size]
                        )
                        offset += block_size
                        shift = shift_geometry(total_momentum, block_arguments)
                        term *= shift * (
                            pre_gaussian(block_arguments)
                            + deformation * pre_response(block_arguments)
                        )
                    value += term / (
                        math.factorial(order) * math.factorial(shift_count)
                    )
        return value

    def dual_post(arguments: tuple[float, ...]) -> tuple[float, float]:
        gaussian_total = 0.0
        response_total = 0.0
        for coefficient, factors, shift_factors in reconstruction_terms(arguments):
            gaussian_values = [
                shift * pre_gaussian(factor_arguments)
                for factor_arguments, shift in zip(factors, shift_factors)
            ]
            response_values = [
                shift * pre_response(factor_arguments)
                for factor_arguments, shift in zip(factors, shift_factors)
            ]
            gaussian_total += coefficient * math.prod(gaussian_values)
            for response_index, response_value in enumerate(response_values):
                term = coefficient * response_value
                for index, gaussian_value in enumerate(gaussian_values):
                    if index != response_index:
                        term *= gaussian_value
                response_total += term
        return gaussian_total, response_total

    arguments_by_order = {
        3: (0.17, -0.29, 0.41),
        4: (0.17, -0.29, 0.41, -0.11),
    }
    checks: dict[str, dict[str, float]] = {}
    for order, arguments in arguments_by_order.items():
        gaussian_candidate, response_candidate = dual_post(arguments)
        undeformed = direct_field_level_post(arguments, 0.0j).real
        response_expected = direct_field_level_post(
            arguments, 1.0j * complex_step
        ).imag / complex_step
        gaussian_error = relative_difference(gaussian_candidate, undeformed)
        response_error = relative_difference(response_candidate, response_expected)
        if gaussian_error > 2.0e-14 or response_error > 2.0e-14:
            raise AssertionError(
                f"finite-R post K{order} product rule failed: "
                f"G={gaussian_error}, R={response_error}"
            )
        checks[f"K{order}"] = {
            "gaussian_direct": undeformed,
            "gaussian_dual": gaussian_candidate,
            "gaussian_relative_error": gaussian_error,
            "response_complex_step": response_expected,
            "response_dual_product_rule": response_candidate,
            "response_relative_error": response_error,
        }
    return {
        "smoothing_radius": smoothing_radius,
        "bias_recon": bias_recon,
        "complex_step": complex_step,
        "kernel_order_checks": checks,
    }


def finite_tree_fnl2_reference_test(binary: Path, table: Path) -> dict[str, Any]:
    """Independently evaluate the complete pre/post finite-PNG tree.

    An equilateral triangle makes all required P_L and M values exact table
    knots, so this reference does not duplicate the native spline.  The
    calculation below is written directly from the published cyclic formula
    and from the field-level reconstruction product rule.  It never reads the
    native component values when constructing the expectation.
    """

    table_rows: list[tuple[float, float, float]] = []
    with table.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split()
            if len(columns) < 4:
                continue
            table_rows.append(
                (float(columns[0]), float(columns[1]), float(columns[3]))
            )
    if len(table_rows) < 8:
        raise AssertionError("PNG transfer table lacks four-column rows")
    k, power, transfer = min(table_rows, key=lambda row: abs(row[0] - 0.055))
    if power <= 0.0 or transfer <= 0.0:
        raise AssertionError("invalid P_L or M table knot")

    b1, b2, bk2, bphi, bphidelta = (2.7, 0.4, -0.2, 5.8, 3.5)
    bphi2 = -4.2
    mu = -0.5
    triangle = (k, k, mu)
    row = run_binary(
        binary,
        table,
        "pre_recon_halo_bias_v1_local_png_tree",
        [triangle],
        bias=(b1, b2, bk2, bphi, bphidelta),
        bphi2=bphi2,
    )["results"][0]

    f2 = 5.0 / 7.0 + mu + 2.0 * mu * mu / 7.0
    s2 = mu * mu - 1.0 / 3.0
    inv_m = 1.0 / transfer
    inv_m_sum = 2.0 * inv_m
    inv_m_product = inv_m * inv_m
    pp = power * power
    b0hat_pair = 2.0 * pp / transfer
    b0hat = 3.0 * b0hat_pair
    inv_m_all = 3.0 * inv_m

    expected = {
        "Bhalo_tree_fNL2_bphi_B0": (
            b1 * b1 * bphi * inv_m_all * b0hat
        ),
        "Bhalo_tree_fNL2_bphi_sq_advection": 3.0
        * b1
        * bphi
        * bphi
        * pp
        * inv_m_sum
        * mu
        * (2.0 * inv_m),
        "Bhalo_tree_fNL2_bphi_sq_F2": 3.0
        * 2.0
        * bphi
        * bphi
        * inv_m_product
        * b1
        * f2
        * pp,
        "Bhalo_tree_fNL2_bphi_sq_b2": 3.0
        * bphi
        * bphi
        * b2
        * inv_m_product
        * pp,
        "Bhalo_tree_fNL2_bphi_sq_bK2": 3.0
        * 2.0
        * bphi
        * bphi
        * bk2
        * s2
        * inv_m_product
        * pp,
        "Bhalo_tree_fNL2_bphi_bphidelta": 3.0
        * b1
        * bphi
        * bphidelta
        * inv_m_sum
        * inv_m_sum
        * pp,
        "Bhalo_tree_fNL2_bphi2_operator": 3.0
        * b1
        * b1
        * bphi2
        * inv_m_product
        * pp,
        "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis": 3.0
        * bphi
        * bphi
        * power
        * inv_m_product,
    }
    expected["Bhalo_tree_fNL2_deterministic"] = sum(
        value
        for key, value in expected.items()
        if key != "Bhalo_tree_fNL2_stochastic_alpha3PNG_basis"
    )
    relative_errors = {
        key: relative_difference(float(row[key]), value)
        for key, value in expected.items()
    }
    if max(relative_errors.values()) > 2.0e-11:
        raise AssertionError(
            f"finite-tree fNL2 independent reference failed: {relative_errors}"
        )

    reconstruction_radius = 15.0
    reconstruction_bias = 2.7340475186190334
    post_row = run_binary(
        binary,
        table,
        "post_recon_halo_bias_v1_local_png_tree",
        [triangle],
        smoothing_radius=reconstruction_radius,
        bias_recon=reconstruction_bias,
        bias=(b1, b2, bk2, bphi, bphidelta),
        bphi2=bphi2,
    )["results"][0]
    # For an equilateral cyclic pair, k_out=-(k_i+k_j) and
    # k_out.k_i/k_i^2=-1/2.  Thus the two shift factors sum to -W/b_rec.
    shift_sum = -math.exp(
        -0.5 * (reconstruction_radius * k) ** 2
    ) / reconstruction_bias
    expected_reconstruction = (
        3.0
        * pp
        * shift_sum
        * b1
        * b1
        * bphi
        * bphi
        * 6.0
        * inv_m_product
    )
    post_expected = dict(expected)
    post_expected[
        "Bhalo_tree_fNL2_bphi_sq_reconstruction"
    ] = expected_reconstruction
    post_expected["Bhalo_tree_fNL2_deterministic"] = (
        expected["Bhalo_tree_fNL2_deterministic"]
        + expected_reconstruction
    )
    post_relative_errors = {
        key: relative_difference(float(post_row[key]), value)
        for key, value in post_expected.items()
    }
    if max(post_relative_errors.values()) > 2.0e-11:
        raise AssertionError(
            "finite-R tree fNL2 independent reference failed: "
            f"{post_relative_errors}"
        )

    rinf_row = run_binary(
        binary,
        table,
        "post_recon_halo_bias_v1_local_png_tree",
        [triangle],
        smoothing_radius=1.0e6,
        bias_recon=reconstruction_bias,
        bias=(b1, b2, bk2, bphi, bphidelta),
        bphi2=bphi2,
    )["results"][0]
    rinf_keys = tuple(post_expected)
    rinf_errors = {
        key: relative_difference(
            float(rinf_row[key]),
            0.0
            if key
            == "Bhalo_tree_fNL2_bphi_sq_reconstruction"
            else float(row[key]),
        )
        for key in rinf_keys
    }
    if max(rinf_errors.values()) > 2.0e-12:
        raise AssertionError(
            f"finite-tree fNL2 R-infinity test failed: {rinf_errors}"
        )

    component_keys = tuple(post_expected)
    k1, k2, mu12 = (0.05, 0.06, -0.3)
    sin12 = math.sqrt(1.0 - mu12 * mu12)
    vectors = (
        (0.0, 0.0, k1),
        (k2 * sin12, 0.0, k2 * mu12),
        (-k2 * sin12, 0.0, -(k1 + k2 * mu12)),
    )

    def norm(vector: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    def cosine(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> float:
        return sum(a * b for a, b in zip(left, right)) / (
            norm(left) * norm(right)
        )

    triangle_representations = [
        (norm(vectors[i]), norm(vectors[j]), cosine(vectors[i], vectors[j]))
        for i, j in itertools.permutations(range(3), 2)
    ]
    permutation_spreads: dict[str, dict[str, float]] = {}
    for stage in ("pre", "post"):
        permutation_rows = run_binary(
            binary,
            table,
            f"{stage}_recon_halo_bias_v1_local_png_tree",
            triangle_representations,
            smoothing_radius=reconstruction_radius,
            bias_recon=reconstruction_bias,
            bias=(b1, b2, bk2, bphi, bphidelta),
            bphi2=bphi2,
        )["results"]
        stage_spreads = {}
        for key in component_keys:
            values = [float(item[key]) for item in permutation_rows]
            stage_spreads[key] = (
                max(values) - min(values)
            ) / max(
                max(abs(value) for value in values),
                1.0e-300,
            )
        if max(stage_spreads.values()) > 2.0e-12:
            raise AssertionError(
                "finite-tree fNL2 permutation test failed: "
                f"{stage} {stage_spreads}"
            )
        permutation_spreads[stage] = stage_spreads

    matter_row = run_binary(
        binary,
        table,
        "post_recon_halo_bias_v1_local_png_tree",
        [triangle],
        smoothing_radius=reconstruction_radius,
        bias_recon=reconstruction_bias,
        bias=(1.0, 0.0, 0.0, 0.0, 0.0),
        bphi2=0.0,
    )["results"][0]
    matter_limit_max_abs = max(abs(float(matter_row[key])) for key in component_keys)
    if matter_limit_max_abs != 0.0:
        raise AssertionError(
            f"finite-tree halo fNL2 matter limit failed: {matter_limit_max_abs}"
        )

    f_test = 100.0
    gaussian = float(row["Btree"])
    linear = float(row["dBdfNL_local_tree"])
    quadratic = float(row["Bhalo_tree_fNL2_deterministic"])
    finite_plus = gaussian + f_test * linear + f_test * f_test * quadratic
    finite_minus = gaussian - f_test * linear + f_test * f_test * quadratic
    recovered_odd = (finite_plus - finite_minus) / (2.0 * f_test)
    recovered_even = (
        finite_plus + finite_minus - 2.0 * gaussian
    ) / (2.0 * f_test * f_test)
    parity_errors = {
        "odd": relative_difference(recovered_odd, linear),
        "even": relative_difference(recovered_even, quadratic),
    }
    if max(parity_errors.values()) > 2.0e-12:
        raise AssertionError(f"finite-tree fNL parity failed: {parity_errors}")

    return {
        "reference": "Dizgah et al. 2020 Eq. (2.58) and Eq. (2.65)",
        "equilateral_table_knot": k,
        "pre_relative_errors": relative_errors,
        "post_relative_errors": post_relative_errors,
        "r_infinity_relative_errors": rinf_errors,
        "permutation_relative_spreads": permutation_spreads,
        "matter_limit_max_abs": matter_limit_max_abs,
        "finite_fNL_parity_relative_errors": parity_errors,
    }


def cli_tests(binary: Path, table: Path, run_matter_diagnostic: bool) -> dict[str, Any]:
    triangle = (0.05, 0.06, -0.3)
    tree_payload = run_binary(
        binary, table, "pre_recon_halo_bias_v1_local_png_tree", [triangle]
    )
    loop_payload = run_binary(
        binary, table, "pre_recon_halo_bias_v1_local_png_1loop_truncated", [triangle]
    )
    tree_row = tree_payload["results"][0]
    loop_row = loop_payload["results"][0]
    tree_error = relative_difference(
        float(tree_row["dBdfNL_local_tree"]), float(loop_row["dBdfNL_local_tree"])
    )
    if tree_error > 1.0e-12:
        raise AssertionError(f"Barreira tree golden test failed: {tree_error}")

    post_tree_row = run_binary(
        binary, table, "post_recon_halo_bias_v1_local_png_tree", [triangle]
    )["results"][0]
    post_loop_row = run_binary(
        binary, table, "post_recon_halo_bias_v1_local_png_1loop_truncated", [triangle]
    )["results"][0]
    post_tree_error = relative_difference(
        float(post_tree_row["dBdfNL_local_tree"]),
        float(post_loop_row["dBdfNL_local_tree"]),
    )
    if post_tree_error > 1.0e-12:
        raise AssertionError(f"finite-R post tree golden test failed: {post_tree_error}")

    post_rinf = run_binary(
        binary,
        table,
        "post_recon_halo_bias_v1_local_png_1loop_truncated",
        [triangle],
        smoothing_radius=1.0e6,
    )["results"][0]
    rinf_keys = (
        "Btree",
        "B222",
        "B321I",
        "B321II",
        "B411",
        "Btotal",
        "dBdfNL_local_tree",
        "dBdfNL_local_B222",
        "dBdfNL_local_B321I",
        "dBdfNL_local_B321II",
        "dBdfNL_local_B411",
        "dBdfNL_local_total",
    )
    rinf_error = max(
        relative_difference(float(loop_row[key]), float(post_rinf[key])) for key in rinf_keys
    )
    if rinf_error > 1.0e-12:
        raise AssertionError(f"R->infinity post/pre test failed: {rinf_error}")

    k1, k2, mu12 = triangle
    sin12 = math.sqrt(max(0.0, 1.0 - mu12 * mu12))
    vectors = (
        (0.0, 0.0, k1),
        (k2 * sin12, 0.0, k2 * mu12),
        (-k2 * sin12, 0.0, -(k1 + k2 * mu12)),
    )

    def vector_norm(vector: tuple[float, float, float]) -> float:
        return math.sqrt(sum(component * component for component in vector))

    def vector_mu(
        left: tuple[float, float, float], right: tuple[float, float, float]
    ) -> float:
        denominator = vector_norm(left) * vector_norm(right)
        cosine = sum(a * b for a, b in zip(left, right)) / denominator
        return max(-1.0, min(1.0, cosine))

    # 三条带标签外腿的全部有序 pair，共 3!=6 个 CLI 表示；不再只测 cyclic 3 个。
    six_permutations = [
        (vector_norm(vectors[i]), vector_norm(vectors[j]), vector_mu(vectors[i], vectors[j]))
        for i, j in itertools.permutations(range(3), 2)
    ]
    permutation_diagnostics: dict[str, Any] = {
        "triangle_representation_count": len(six_permutations),
        "qmin": 1.0e-6,
        "qmax": 50.0,
        "epsrel": 0.002,
        "p13_epsrel": 0.002,
        "smoothing_radius": 15.0,
        "strict_tree_full_s3_relative_tolerance": 1.0e-12,
        "strict_one_loop_full_s3_relative_tolerance": 1.0e-3,
        "fixed_cutoff_scope": (
            "All six input representations are runtime-sorted to the same external-side "
            "canonical route before every new halo-v1 loop evaluation. Tree and every "
            "one-loop topology are therefore full-S3 hard checks in this frozen scheme. "
            "Only comparisons to the legacy explicit-PNG route or to an unsorted native "
            "route require affine q shifts and remain fixed-cutoff diagnostics."
        ),
    }
    reported_keys = (
        "Btree",
        "B222",
        "B321I",
        "B321II",
        "B411",
        "Btotal",
        "dBdfNL_local_tree",
        "dBdfNL_local_B222",
        "dBdfNL_local_B321I",
        "dBdfNL_local_B321II",
        "dBdfNL_local_B411",
        "dBdfNL_local_total",
    )
    def relative_spread(values: list[float]) -> float:
        return (max(values) - min(values)) / max(
            max(abs(value) for value in values), 1.0e-300
        )

    for stage in ("pre", "post"):
        permutation_rows = run_binary(
            binary,
            table,
            f"{stage}_recon_halo_bias_v1_local_png_1loop_truncated",
            six_permutations,
            qmin=1.0e-6,
            qmax=50.0,
            epsrel=0.002,
            p13_epsrel=0.002,
            smoothing_radius=15.0,
        )["results"]
        values_by_key = {
            key: [float(row[key]) for row in permutation_rows]
            for key in reported_keys
        }
        full_six_spreads = {
            key: relative_spread(values) for key, values in values_by_key.items()
        }
        if (
            full_six_spreads["Btree"] > 1.0e-12
            or full_six_spreads["dBdfNL_local_tree"] > 1.0e-12
        ):
            raise AssertionError(
                f"{stage} finite-R tree full-S3 symmetry failed: {full_six_spreads}"
            )
        failed_loop_checks = {
            key: spread
            for key, spread in full_six_spreads.items()
            if key not in {"Btree", "dBdfNL_local_tree"} and spread > 1.0e-3
        }
        if failed_loop_checks:
            raise AssertionError(
                f"{stage} canonical-route one-loop full-S3 stability failed: "
                f"{failed_loop_checks}"
            )
        permutation_diagnostics[stage] = {
            "values": values_by_key,
            "full_six_relative_spreads": full_six_spreads,
            "integration_metadata": [row["integration_metadata"] for row in permutation_rows],
        }

    legacy = run_binary(binary, table, "pre_recon_gaussian", [triangle])["results"][0]
    if "dBdfNL_local_B222" in legacy:
        raise AssertionError("legacy DM JSON schema was polluted by halo bias-v1 response fields")
    metadata = loop_payload["metadata"]
    required_metadata = {
        "model_name": "MARISA-B halo bias-v1 truncated fixed-cutoff one-loop local-PNG response",
        "renormalization": "fixed_cutoff_unrenormalized",
        "tree_reference": "Barreira_2022_exact",
        "loop_momentum_routing": "canonical_sorted_external_legs_native_gaussian_diagram_logq_with_spherical_qmin_qmax",
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise AssertionError(f"missing/incorrect metadata {key}: {metadata.get(key)!r}")

    # 历史 pre-DM payload 的原表已经归档；当前表是同一份 Quijote z=.5 CLASS P_L。
    # 用历史 command 的全部数值设置重跑，要求八个旧 diagram/总量逐 bit 不变。
    historical_payload = json.loads(HISTORICAL_PRE_PAYLOAD.read_text(encoding="utf-8"))
    historical_row = historical_payload["marisa_b"]["payload"]["results"][0]
    rerun_row = run_binary(
        binary,
        CURRENT_Z05_TABLE,
        "pre_recon_gaussian",
        [(0.05, 0.05, -0.5)],
        qmin=1.0e-4,
        qmax=30.0,
        epsrel=0.001,
        p13_epsrel=0.01,
        smoothing_radius=10.0,
        bias=(1.0, 0.0, 0.0, 0.0, 0.0),
    )["results"][0]
    legacy_components = (
        "Btree",
        "B222",
        "B321I",
        "B321II",
        "B411",
        "Bloopterms",
        "B1loop",
        "Btotal",
    )
    legacy_numeric_errors = {
        key: relative_difference(float(rerun_row[key]), float(historical_row[key]))
        for key in legacy_components
    }
    if max(legacy_numeric_errors.values()) != 0.0:
        raise AssertionError(f"legacy pre-DM numeric baseline changed: {legacy_numeric_errors}")

    # b1=1,b2=bK2=0 时新 Gaussian halo kernel 必须退化到旧 DM kernel。
    # 新 halo-v1 用 log(q/k1) 而旧 DM 保留历史 linear-q cubature，因此 loop
    # 数值不要求逐 bit 相等；规格要求是差值不超过各自 cubature error budget。
    gaussian_matter_identity: dict[str, Any] = {}
    for stage in ("pre", "post"):
        old_gaussian = run_binary(
            binary,
            table,
            f"{stage}_recon_gaussian",
            [triangle],
            qmin=1.0e-4,
            qmax=3.0,
            epsrel=0.002,
            p13_epsrel=0.002,
            smoothing_radius=15.0,
            bias=(1.0, 0.0, 0.0, 0.0, 0.0),
        )["results"][0]
        new_gaussian = run_binary(
            binary,
            table,
            f"{stage}_recon_halo_bias_v1_gaussian",
            [triangle],
            qmin=1.0e-4,
            qmax=3.0,
            epsrel=0.002,
            p13_epsrel=0.002,
            smoothing_radius=15.0,
            bias=(1.0, 0.0, 0.0, 0.0, 0.0),
        )["results"][0]
        topology_errors = {
            key: relative_difference(float(new_gaussian[key]), float(old_gaussian[key]))
            for key in ("Btree", "B222", "B321I", "B321II", "B411", "Btotal")
        }
        error_budget_checks: dict[str, dict[str, float]] = {}
        for key in ("B222", "B321I", "B411"):
            old_error = float(old_gaussian["integration_metadata"][key]["abserr"])
            new_error = float(new_gaussian["integration_metadata"][key]["abserr"])
            absolute_difference = abs(float(new_gaussian[key]) - float(old_gaussian[key]))
            one_sigma_budget = old_error + new_error
            ratio = absolute_difference / max(one_sigma_budget, 1.0e-300)
            error_budget_checks[key] = {
                "absolute_difference": absolute_difference,
                "old_abserr": old_error,
                "new_abserr": new_error,
                "difference_over_summed_abserr": ratio,
            }
            if ratio > 5.0:
                raise AssertionError(
                    f"Gaussian matter-limit {key} exceeds cubature budget: "
                    f"{stage} {error_budget_checks[key]}"
                )
        if topology_errors["Btree"] > 1.0e-12:
            raise AssertionError(f"Gaussian matter-limit tree failed: {stage} {topology_errors}")
        if topology_errors["B321II"] > 1.0e-3 or topology_errors["Btotal"] > 1.0e-3:
            raise AssertionError(f"Gaussian matter-limit total/P13 topology failed: {stage} {topology_errors}")
        gaussian_matter_identity[stage] = {
            "relative_errors": topology_errors,
            "cubature_error_budget_checks": error_budget_checks,
            "maximum_allowed_difference_over_summed_abserr": 5.0,
        }

    matter_diagnostic: dict[str, Any] | None = None
    if run_matter_diagnostic:
        matter_bias = (1.0, 0.0, 0.0, 0.0, 0.0)
        diagnostic: dict[str, Any] = {}
        for stage in ("pre", "post"):
            new_mode = f"{stage}_recon_halo_bias_v1_local_png_1loop_truncated"
            old_mode = f"{stage}_recon_local_png_1loop"
            new_row = run_binary(
                binary,
                table,
                new_mode,
                [triangle],
                qmin=1.0e-6,
                qmax=70.0,
                epsrel=0.01,
                p13_epsrel=0.01,
                bias=matter_bias,
            )["results"][0]
            old_row = run_binary(
                binary,
                table,
                old_mode,
                [triangle],
                qmin=1.0e-6,
                qmax=70.0,
                epsrel=0.01,
                p13_epsrel=0.01,
                bias=matter_bias,
            )["results"][0]
            loop_error = relative_difference(
                float(new_row["dBdfNL_local_1loop"]),
                float(old_row["dBdfNL_local_1loop"]),
            )
            if loop_error > 0.02:
                raise AssertionError(f"large-cutoff matter routing diagnostic failed: {stage} {loop_error}")
            diagnostic[stage] = {
                "new_directional_derivative": float(new_row["dBdfNL_local_1loop"]),
                "legacy_explicit_B0": float(old_row["dBdfNL_local_1loop"]),
                "relative_routing_difference": loop_error,
            }
        matter_diagnostic = diagnostic

    return {
        "barreira_tree_relative_error": tree_error,
        "finite_r_post_tree_relative_error": post_tree_error,
        "r_infinity_max_relative_error": rinf_error,
        "permutation_diagnostics": permutation_diagnostics,
        "legacy_schema_clean": True,
        "metadata": required_metadata,
        "legacy_pre_numeric_baseline_relative_errors": legacy_numeric_errors,
        "gaussian_b1_unity_new_old_relative_errors": gaussian_matter_identity,
        "matter_large_cutoff_routing_diagnostic": matter_diagnostic,
    }


def main() -> None:
    args = parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(f"compile the MARISA-B binary first: {args.binary}")
    if not args.png_table.is_file():
        raise FileNotFoundError(args.png_table)
    result = {
        "status": "passed",
        "exact_lattice_wick": exact_lattice_wick_test(),
        "finite_r_post_product_rule": finite_r_post_product_rule_test(),
        "finite_tree_fnl2_reference": finite_tree_fnl2_reference_test(
            args.binary,
            args.png_table,
        ),
        "cli": cli_tests(
            args.binary,
            args.png_table,
            run_matter_diagnostic=not args.skip_matter_routing_diagnostic,
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
