#!/usr/bin/env python3
"""四方法理论映射与 current-p 参数示例；仅依赖 Python 标准库。

模块大纲：读取冻结校准；区分 tracer 与位移参数；计算 PNG bias 和
adaptive 原子基底的归一化；打印 JSON，不生成 catalogue、B 模板或拟合。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/reconstruction_current_p_example.json"


def finite(value: float, name: str) -> float:
    """拒绝非有限数，避免把无效设置打印成看似可用的示例。"""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def tracer_biases(b1, p_halo, b2, gamma2, delta_c=1.686, anchor_p=1.0):
    """复现当前 pre IR 扫描的 bias 约定，不更改生产适配器的默认值。"""
    b1, p_halo, b2, gamma2, delta_c, anchor_p = (
        finite(value, name) for value, name in zip(
            (b1, p_halo, b2, gamma2, delta_c, anchor_p),
            ("b1", "p_halo", "b2", "gamma2", "delta_c", "anchor_p"),
        )
    )
    if b1 <= 0 or delta_c <= 0:
        raise ValueError("b1 and delta_c must be positive")
    b2_native = b2 - 4.0 * gamma2 / 3.0
    b2_lagrangian = b2_native - 8.0 * (b1 - 1.0) / 21.0
    return {
        "b1": b1, "p_halo": p_halo, "b2_native": b2_native,
        "bphi": 2.0 * delta_c * (b1 - p_halo),
        # current-p 只替换 bphi；bphidelta 仍采用独立的 p=1 anchor。
        "bphidelta": 2.0 * delta_c * (b1 - anchor_p)
        + 2.0 * (delta_c * b2_lagrangian - b1 + 1.0),
        "bphi2": 4.0 * delta_c * (delta_c * b2_lagrangian - 2.0 * (b1 - 1.0)),
    }


def reconstruction_denominator(b_rec, p_rec, fnl_rec, transfer_m, delta_c=1.686):
    """仅算给定一个 M(k) 的分母，不假造功率谱或完整位移核。"""
    b_rec, p_rec, fnl_rec, transfer_m, delta_c = (
        finite(value, name) for value, name in zip(
            (b_rec, p_rec, fnl_rec, transfer_m, delta_c),
            ("b_rec", "p_rec", "fnl_rec", "transfer_m", "delta_c"),
        )
    )
    if b_rec <= 0 or transfer_m <= 0 or delta_c <= 0:
        raise ValueError("b_rec, transfer_m and delta_c must be positive")
    bphi_rec = 2.0 * delta_c * (b_rec - p_rec)
    effective = b_rec + fnl_rec * bphi_rec / transfer_m
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("non-positive adaptive denominator; no clipping is applied")
    return bphi_rec, effective


def summarize(config, *, fnl=100.0, fnl_rec=100.0, transfer_m=1000.0,
              b2=0.0, gamma2=0.0, p_rec_override=None):
    """输出五行（pre 对照加四方法）；post 的 parent 校准明确标为示例。"""
    fnl = finite(fnl, "fnl")
    fnl_rec = finite(fnl_rec, "fnl_rec")
    transfer_m = finite(transfer_m, "transfer_m")
    if transfer_m <= 0:
        raise ValueError("transfer_m must be positive")
    delta_c = config["delta_c"]
    rows = {}
    for method, entry in config["methods"].items():
        tracer = config["tracers"][entry["tracer"]]
        bias = tracer_biases(tracer["b1_theory"], tracer["p_halo"], b2, gamma2,
                             delta_c, config["bphidelta_universal_anchor_p"])
        row = {
            "tracer": entry["tracer"], "kernel": entry["kernel"],
            "calibration_use": entry["calibration_use"], "bias": bias,
            "Z1_at_supplied_M": bias["b1"] + fnl * bias["bphi"] / transfer_m,
            "reconstruction": None,
        }
        if entry["kernel"] not in ("pre", "fixed_brec", "adaptive_brec"):
            raise ValueError(f"unsupported kernel: {entry['kernel']}")
        if entry["kernel"] != "pre":
            rec = dict(tracer["reconstruction_example"])
            if entry["kernel"] == "adaptive_brec":
                p_rec = rec["p_rec"] if p_rec_override is None else p_rec_override
                bphi_rec, effective = reconstruction_denominator(
                    rec["b_rec"], p_rec, fnl_rec, transfer_m, delta_c)
                ratio = bphi_rec / rec["b_rec"]
                rec.update(p_rec=p_rec, fNL_rec=fnl_rec, bphi_rec=bphi_rec,
                           denominator_at_supplied_M=effective)
                # 这些系数还必须乘 native 导出的 raw basis；本例不计算这些基底。
                rec["adaptive_raw_basis_prefactors"] = {
                    "gaussian_linear": bias["b1"]**4 * ratio,
                    "gaussian_quadratic": bias["b1"]**4 * ratio**2,
                    "png_linear_cross": bias["b1"]**3 * bias["bphi"] * ratio,
                }
            else:
                # std 不使用 p_rec 或 fNL_rec；不把 tracer p 误传给位移算法。
                rec.update(p_rec=None, fNL_rec=None, bphi_rec=None,
                           denominator_at_supplied_M=rec["b_rec"])
            row["reconstruction"] = rec
        rows[method] = row
    return {
        "status": "illustration_only_not_a_fit_or_IR_prediction",
        "snapshot_date": config["snapshot_date"],
        "fNL_cosmology": fnl, "fNL_rec_example": fnl_rec,
        "transfer_M_example": transfer_m,
        "warning": "M is an illustrative positive input, not a computed cosmology table; b2/gamma2 are illustrative nuisance inputs. Parent-tracer calibration reuse does not establish post-reconstruction validity.",
        "methods": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fnl", type=float, default=100.0)
    parser.add_argument("--fnl-rec", type=float, default=100.0)
    parser.add_argument("--transfer-m", type=float, default=1000.0)
    parser.add_argument("--b2", type=float, default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--p-rec", type=float, default=None,
                        help="显式改变示例的重构 p，不改变 p_halo，也不修改真实 catalogue")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = summarize(config, fnl=args.fnl, fnl_rec=args.fnl_rec,
                           transfer_m=args.transfer_m, b2=args.b2, gamma2=args.gamma2,
                           p_rec_override=args.p_rec)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
