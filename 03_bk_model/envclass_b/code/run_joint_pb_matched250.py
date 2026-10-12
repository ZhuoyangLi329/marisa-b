#!/usr/bin/env python3
"""最终 matched250 P+B 拟合：pre250 对比 sto_marisa p-aware P250+B250。

sto_marisa real000--099 从只读旧产物读取，real100--249 从新仓库扩产目录读取。
P 与 B 必须来自同一个 p-aware realization；协方差继续冻结为现行 fid500 联合
样本协方差，以便只检验目标均值由 100 扩到 250 后的变化。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import h5py
import numpy as np
from iminuit import Minuit
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as chi2_distribution


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
HERE = ROOT / "03_bk_model/envclass_b"
VENDOR = HERE / "vendor"
ANALYSIS = HERE / "analysis"
WORK = HERE / "work/sto_marisa_v4_lcp250"
LATTICE = ROOT / "02_reconstruction_on_pk/inputs/p0_finite_local_png_lattice.npz"
OLD_HEAD = Path(
    "/pscratch/sd/l/lzy/reconstruction-png/"
    "result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_"
    "sto_marisa_b000_pk_v2_p1282/full/LC_p"
)
NEW_TAIL = ROOT / "01_reconstruction_methods/outputs/sto_marisa_v4_lcp250/full/LC_p"
OUT = ANALYSIS / "JOINT_PB_PRE_VS_STOMARISA_V4_MATCHED250.json"
VOLUME = 42.875
FNL_BOUNDS = (-300.0, 300.0)
PRIOR_WEIGHT = 1.0 / 25.0
NAMES = ("fNL", "b1", "Pshot", "b2", "gamma2", "Ashot_residual", "Bshot_residual")
CASES = {
    "pre": {"b1": 1.93, "p": 1.179, "method": "pre"},
    "sto_marisa": {
        "b1": 2.2280546353635335,
        "p": 1.2822,
        "method": "marisa",
        "b_rec": 2.2280546353635335,
        "p_rec": 1.2822,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vendor_load(name: str, filename: str):
    import types

    path = VENDOR / "py" / filename
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def realization_root(real: int) -> Path:
    base = OLD_HEAD if real < 100 else NEW_TAIL
    return base / f"real{real:03d}"


def validate_metadata(path: Path, real: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest") or {}
    expected = {
        "cosmology": "LC_p",
        "realization": real,
        "input_mode": "sto-weighted",
        "recon_class": "marisa-plane",
        "png_p": 1.2822,
        "fnl_loc": 100.0,
        "bias": 2.2280546353635335,
        "measure_pk": True,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if actual != value:
            raise RuntimeError(f"real{real:03d} metadata 失配: {key}={actual!r}, 预期 {value!r}")


def load_paware_target() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    k_rows: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    b_k_rows: list[np.ndarray] = []
    b_edge_rows: list[np.ndarray] = []
    b_nmodes_rows: list[np.ndarray] = []
    for real in range(250):
        root = realization_root(real)
        required = [root / name for name in ("DONE", "metadata.json", "post_pk.npz", "post_b000.h5")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"matched250 不完整，real{real:03d} 缺失: {missing[0]}")
        validate_metadata(root / "metadata.json", real)
        with np.load(root / "post_pk.npz", allow_pickle=False) as data:
            pk = np.asarray(data["pk"], dtype=np.float64)
            kval = np.asarray(data["k"], dtype=np.float64)
            edges = np.asarray(data["edges"], dtype=np.float64)
        with h5py.File(root / "post_b000.h5", "r") as data:
            bval = np.asarray(data["0_0_0/value"][()], dtype=np.float64)
            b_k_rows.append(np.asarray(data["0_0_0/k"][()]))
            b_edge_rows.append(np.asarray(data["0_0_0/k_edges"][()]))
            b_nmodes_rows.append(np.asarray(data["0_0_0/nmodes"][()]))
        if pk.shape[0] < 12 or kval.shape[0] < 12 or edges.shape[0] < 13 or bval.shape != (120,):
            raise RuntimeError(
                f"real{real:03d} shape 异常: P{pk.shape} k{kval.shape} edges{edges.shape} B{bval.shape}"
            )
        p_rows.append(pk[:12])
        k_rows.append(kval[:12])
        edge_rows.append(edges[:13])
        b_rows.append(bval)

    p = np.asarray(p_rows)
    b = np.asarray(b_rows)
    k = np.asarray(k_rows)
    edges = np.asarray(edge_rows)
    b_k = np.asarray(b_k_rows)
    b_edges = np.asarray(b_edge_rows)
    b_nmodes = np.asarray(b_nmodes_rows)
    if p.shape != (250, 12) or b.shape != (250, 120):
        raise RuntimeError(f"matched250 shape 异常: P{p.shape}, B{b.shape}")
    if not np.isfinite(p).all() or not np.isfinite(b).all():
        raise RuntimeError("matched250 含非有限 P/B 数值")
    if not np.allclose(k, k[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("matched250 P k-center 在 realization 间漂移")
    if not np.allclose(edges, edges[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("matched250 P edges 在 realization 间漂移")
    if not np.allclose(b_k, b_k[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("matched250 B k 在 realization 间漂移")
    if not np.allclose(b_edges, b_edges[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("matched250 B k_edges 在 realization 间漂移")
    if not np.array_equal(b_nmodes, np.broadcast_to(b_nmodes[0], b_nmodes.shape)):
        raise RuntimeError("matched250 B nmodes 在 realization 间漂移")
    return p, b, k[0], edges[0]


def p_theory(k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(LATTICE, allow_pickle=False) as data:
        k_old = np.asarray(data["pk_k"], dtype=float)
        arrays = [
            np.asarray(data["PL_lattice"], dtype=float),
            np.asarray(data["PL_over_M_lattice"], dtype=float),
            np.asarray(data["Pphi_lattice"], dtype=float),
        ]

    def interp(values: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(k), np.log(k_old), np.log(np.maximum(values, 1.0e-300))))

    return tuple(interp(values) for values in arrays)  # type: ignore[return-value]


def configure_models():
    low = vendor_load("quijote_halo_z0p5_direct_finite_fnl_models", "low.py")
    scan = vendor_load("quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models", "scan.py")
    low.RECOVERY_RUNNER = VENDOR / "py/recovery.py"
    low.FULL_RUNNER = VENDOR / "py/full.py"
    low.P_MATRIX = VENDOR / "data/PRE_P0_FNL250_MATRIX.npz"
    low.B_MATRIX_ROOT = VENDOR / "data"
    low.B_MATRICES = {
        "standard": VENDOR / "data/STANDARD_FNL250_MATRIX.npz",
        "marisa": VENDOR / "data/MARISA_FNL250_MATRIX.npz",
    }
    low.MARISA_FINITE = VENDOR / "data/marisa_finite_kmax0p08.npz"
    low.MARISA_FINITE_MANIFEST = low.MARISA_FINITE.with_suffix(".json")
    scan.HIGH_BACKEND = VENDOR / "py/high_backend.py"
    scan.PRE_TREE = VENDOR / "data/pre_tree_kmax0p20.npz"
    scan.PRE_TREE_MANIFEST = scan.PRE_TREE.with_suffix(".json")
    scan.MARISA_FINITE = VENDOR / "data/marisa_finite_kmax0p20.npz"
    scan.MARISA_FINITE_MANIFEST = scan.MARISA_FINITE.with_suffix(".json")
    sys.path.insert(0, str(VENDOR / "py"))
    return low, scan, scan.load_runtime()


def fit_case(
    case_name: str,
    cfg: dict,
    *,
    low,
    scan,
    runtime,
    p_fid: np.ndarray,
    b_fid: np.ndarray,
    p_target: np.ndarray,
    b_target: np.ndarray,
    p_templates: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[dict]:
    p_halo = float(cfg["p"])
    low.bphi_from_b1 = lambda b1: 2.0 * low.DELTA_C * (float(b1) - p_halo)
    if case_name == "sto_marisa":
        low.B_REC = float(cfg["b_rec"])
        low.P_REC = float(cfg["p_rec"])
        low.BPHI_REC = 2.0 * low.DELTA_C * (low.B_REC - low.P_REC)
    model = scan.DirectFiniteScanB000Model(runtime, cfg["method"])
    context = model.base_context
    prior_order = list(context.names)
    prior_mean = {name: float(context.prior.mean[prior_order.index(name)]) for name in prior_order}
    whitener = np.asarray(context.prior.whitener, dtype=np.float64)
    PL, PM, Pphi = p_templates
    nominal_b1 = float(cfg["b1"])
    rows: list[dict] = []

    for cut in scan.CUTS:
        idx = np.asarray(scan.selected_indices(runtime.geometry, cut), dtype=np.int64)
        target = np.concatenate([p_target, b_target[idx]])
        fid_cols = np.concatenate([p_fid, b_fid[:, idx]], axis=1)
        nfid, ndata = fid_cols.shape
        hartlap = (float(nfid) - ndata - 2.0) / (float(nfid) - 1.0)
        if hartlap <= 0.0:
            raise RuntimeError(f"Hartlap 非正: nfid={nfid}, ndata={ndata}")
        covariance = np.cov(fid_cols, rowvar=False, ddof=1) / VOLUME
        chol = np.linalg.cholesky(covariance) / math.sqrt(hartlap)

        def objective_impl(values) -> float:
            params = dict(zip(NAMES, [float(value) for value in values]))
            b1 = params["b1"]
            bphi = 2.0 * low.DELTA_C * (b1 - p_halo)
            p_pred = (
                b1**2 * PL
                + params["fNL"] * 2.0 * b1 * bphi * PM
                + params["fNL"] ** 2 * bphi**2 * Pphi
                + params["Pshot"]
            )
            bparams = {
                "b1": b1,
                "b2": params["b2"],
                "gamma2": params["gamma2"],
                "Ashot_residual": params["Ashot_residual"],
                "Bshot_residual": params["Bshot_residual"],
            }
            prediction = np.concatenate([p_pred, model(params["fNL"], bparams, idx)])
            white = solve_triangular(chol, prediction - target, lower=True)
            prior_values = np.asarray([bparams.get(name, prior_mean[name]) for name in prior_order])
            prior_base = np.asarray([prior_mean[name] for name in prior_order])
            prior_white = whitener @ (prior_values - prior_base)
            return float(white @ white) + PRIOR_WEIGHT * float(prior_white @ prior_white)

        signature = ", ".join(NAMES)
        namespace = {"_impl": objective_impl}
        exec(
            "def _objective(" + signature + "):\n"
            "    return _impl((" + ", ".join(NAMES) + ",))\n",
            namespace,
        )
        objective = namespace["_objective"]
        fits = []
        for f0 in (100.0, 0.0, 40.0, 160.0):
            start = {
                "fNL": f0,
                "b1": nominal_b1,
                "Pshot": 0.0,
                "b2": prior_mean["b2"],
                "gamma2": prior_mean["gamma2"],
                "Ashot_residual": prior_mean["Ashot_residual"],
                "Bshot_residual": prior_mean["Bshot_residual"],
            }
            fit = Minuit(objective, **start)
            fit.errordef = Minuit.LEAST_SQUARES
            fit.limits["fNL"] = FNL_BOUNDS
            fit.limits["b1"] = (0.1, 8.0)
            fit.limits["Pshot"] = (-1.0e5, 1.0e5)
            fit.limits["b2"] = (-4.0, 6.0)
            fit.limits["gamma2"] = (-2.0, 4.0)
            fit.limits["Ashot_residual"] = (-10.0 * (abs(start["Ashot_residual"]) + 1.0e6), 10.0 * (abs(start["Ashot_residual"]) + 1.0e6))
            fit.limits["Bshot_residual"] = (-10.0 * (abs(start["Bshot_residual"]) + 1.0e6), 10.0 * (abs(start["Bshot_residual"]) + 1.0e6))
            fit.errors["fNL"] = 10.0
            fit.errors["b1"] = 0.15
            fit.errors["Pshot"] = 1000.0
            fit.errors["b2"] = 0.08
            fit.errors["gamma2"] = 0.05
            fit.strategy, fit.tol = 1, 1.0e-6
            fit.migrad(ncall=200000, iterate=8, use_simplex=True)
            fits.append(fit)
        valid = [fit for fit in fits if fit.valid and math.isfinite(float(fit.fval))]
        best = min(valid or fits, key=lambda fit: float(fit.fval))
        best.hesse()
        best.minos("fNL", ncall=100000)
        error = best.merrors["fNL"]
        minos_valid = bool(error.is_valid and error.lower < 0.0 < error.upper)
        params = {name: float(best.values[name]) for name in NAMES}
        bphi = 2.0 * low.DELTA_C * (params["b1"] - p_halo)
        p_prediction = (
            params["b1"] ** 2 * PL
            + params["fNL"] * 2.0 * params["b1"] * bphi * PM
            + params["fNL"] ** 2 * bphi**2 * Pphi
            + params["Pshot"]
        )
        bparams = {name: params[name] for name in ("b1", "b2", "gamma2", "Ashot_residual", "Bshot_residual")}
        prediction = np.concatenate([p_prediction, model(params["fNL"], bparams, idx)])
        white = solve_triangular(chol, prediction - target, lower=True)
        data_chi2 = float(white @ white)
        row = {
            "case": case_name,
            "kmax_h_mpc": float(cut),
            "ndata": int(ndata),
            "n_fid": int(nfid),
            "n_target": 250,
            "map_fNL": params["fNL"],
            "minos_fNL": {"lower": float(error.lower), "upper": float(error.upper), "valid": minos_valid},
            "map": params,
            "data_chi2": data_chi2,
            "dof": int(ndata - len(NAMES)),
            "gof_p_value_data_only": float(chi2_distribution.sf(data_chi2, max(ndata - len(NAMES), 1))),
            "hartlap": float(hartlap),
        }
        rows.append(row)
        if minos_valid:
            interval = f"[{params['fNL'] + error.lower:+.1f},{params['fNL'] + error.upper:+.1f}]"
        else:
            interval = f"[MINOS invalid: lower={error.lower:+.2f}, upper={error.upper:+.2f}]"
        print(
            f"[{case_name:<11} kmax={cut:.2f}] fNL={params['fNL']:+8.2f} {interval} "
            f"b1={params['b1']:.3f} gof={row['gof_p_value_data_only']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    sm_p_target_rows, sm_b_target_rows, p_k, p_edges = load_paware_target()
    low, scan, runtime = configure_models()
    with np.load(VENDOR / "data/PRE_P0_FNL250_MATRIX.npz", allow_pickle=False) as data:
        pre_p_k = np.asarray(data["pk_k"], dtype=np.float64)
        pre_p_fid = np.asarray(data["fiducial_pre_P0"], dtype=np.float64)
        pre_p_target_rows = np.asarray(data["LC_p_pre_P0"], dtype=np.float64)
    with np.load(VENDOR / "data/STANDARD_FNL250_MATRIX.npz", allow_pickle=False) as data:
        pre_b_fid = np.asarray(data["fiducial_pre_B000"], dtype=np.float64)
        pre_b_target_rows = np.asarray(data["LC_p_pre_B000"], dtype=np.float64)
    with np.load(VENDOR / "data/sto_marisa_matrix.npz", allow_pickle=False) as data:
        sm_p_fid = np.asarray(data["fiducial_post_P0"], dtype=np.float64)[:, :12]
        sm_b_fid = np.asarray(data["fiducial_post_B000"], dtype=np.float64)

    expected_shapes = {
        "pre_p_fid": (500, 12),
        "pre_b_fid": (500, 120),
        "pre_p_target": (250, 12),
        "pre_b_target": (250, 120),
        "sm_p_fid": (500, 12),
        "sm_b_fid": (500, 120),
        "sm_p_target": (250, 12),
        "sm_b_target": (250, 120),
    }
    arrays = {
        "pre_p_fid": pre_p_fid,
        "pre_b_fid": pre_b_fid,
        "pre_p_target": pre_p_target_rows,
        "pre_b_target": pre_b_target_rows,
        "sm_p_fid": sm_p_fid,
        "sm_b_fid": sm_b_fid,
        "sm_p_target": sm_p_target_rows,
        "sm_b_target": sm_b_target_rows,
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"{name} shape={arrays[name].shape}, 预期 {shape}")
    print("data: pre P(250,12)+B(250,120) | sto_marisa p-aware P(250,12)+B(250,120)")

    rows = {
        "pre": fit_case(
            "pre",
            CASES["pre"],
            low=low,
            scan=scan,
            runtime=runtime,
            p_fid=pre_p_fid,
            b_fid=pre_b_fid,
            p_target=pre_p_target_rows.mean(axis=0),
            b_target=pre_b_target_rows.mean(axis=0),
            p_templates=p_theory(pre_p_k),
        ),
        "sto_marisa": fit_case(
            "sto_marisa",
            CASES["sto_marisa"],
            low=low,
            scan=scan,
            runtime=runtime,
            p_fid=sm_p_fid,
            b_fid=sm_b_fid,
            p_target=sm_p_target_rows.mean(axis=0),
            b_target=sm_b_target_rows.mean(axis=0),
            p_templates=p_theory(p_k),
        ),
    }
    summary_path = WORK / "manifest_summary.json"
    payload = {
        "schema": "joint-pb-pre-vs-stomarisa-v4-matched250",
        "created_utc": utc_now(),
        "caliber": (
            "Single final product. pre uses aligned FNL250 P250+B250. sto_marisa uses aligned p-aware "
            "P250+B250: read-only real000-099 plus new-repo real100-249. p=1.2822 and b_rec=2.2281; "
            "b1 is free. Fid500 P-B joint covariance is frozen to the current sto_marisa matrix."
        ),
        "target_contract": {
            "realizations": list(range(250)),
            "n_target": 250,
            "head_root": str(OLD_HEAD),
            "tail_root": str(NEW_TAIL),
            "p_k_h_mpc": p_k.tolist(),
            "pre_p_k_h_mpc": pre_p_k.tolist(),
            "p_edges_h_mpc": p_edges.tolist(),
            "p_bins_used": 12,
            "p_and_b_same_realization": True,
            "png_p": 1.2822,
            "manifest_summary": str(summary_path),
            "manifest_summary_sha256": sha256(summary_path),
        },
        "covariance_contract": {
            "n_fid": 500,
            "volume_factor": VOLUME,
            "hartlap_per_cut": True,
            "p_b_cross_covariance": True,
            "note": "fid500 frozen to the current matrix; only the LC_p target ensemble is expanded",
        },
        "cases": {name: {"b1_start": cfg["b1"], "p": cfg["p"]} for name, cfg in CASES.items()},
        "rows": rows,
        "runner_sha256": sha256(Path(__file__)),
    }
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
