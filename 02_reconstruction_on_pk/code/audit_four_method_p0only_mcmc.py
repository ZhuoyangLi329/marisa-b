#!/usr/bin/env python3
"""独立审计四方法 P0-only matched100 MCMC。

代码大纲
========
1. 从原 baseline/STO matrices 和 600 个 Marisa product 重建统一样本矩阵。
2. 独立重算 matched100 targets、Fid500 covariance、finite-PNG prediction 与 logp。
3. 重算八组 chain convergence、posterior quantiles、truth coverage 和相对 pre 信息比。
4. 检查 PDF 页数、PNG 禁令、全部输入/输出哈希并写最终 audit。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

import run_quijote_halo_z0p5_four_method_p0only_mcmc as run


ROOT = run.ROOT
ANALYSIS = run.ANALYSIS
FIGURES = run.FIGURES
OUTPUT = ANALYSIS / "P0ONLY_MCMC_INDEPENDENT_AUDIT.json"
REPORT = ANALYSIS / "P0ONLY_MCMC_INDEPENDENT_AUDIT.md"
BASELINE = (
    ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/baseline_pre_standard_matrix.npz"
)
STO = BASELINE.with_name("sto_marisa_matrix.npz")
MARISA_ROOT = ROOT / "result_quijote_halo_z0p5_marisa_p0only_fid500_lcp100_v1/full"


def sha256(path: Path) -> str:
    """返回文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """转换 numpy/Path 为 JSON 类型。"""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写 JSON。"""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """读取 NPZ。"""

    with np.load(path, allow_pickle=False) as stream:
        return {name: np.asarray(stream[name]) for name in stream.files}


def compare(name: str, left: np.ndarray, right: np.ndarray, *, atol: float = 0.0) -> dict[str, Any]:
    """比较两个数组并返回最大差。"""

    left = np.asarray(left)
    right = np.asarray(right)
    shape = left.shape == right.shape
    maximum = math.inf if not shape else float(np.max(np.abs(left - right), initial=0.0))
    exact = bool(shape and np.array_equal(left, right))
    passed = bool(exact if atol == 0.0 else shape and maximum <= atol)
    return {"name": name, "shape_equal": shape, "exact": exact, "max_abs": maximum, "atol": atol, "pass": passed}


def original_samples() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """从原始活动产物独立重建四方法 Fid500/LCp100。"""

    baseline = load_npz(BASELINE)
    sto = load_npz(STO)
    samples = {
        "pre": (
            np.asarray(baseline["fiducial_pre_P0"][:, :12], dtype=float),
            np.asarray(baseline["LC_p_pre_P0"][:100, :12], dtype=float),
        ),
        "standard": (
            np.asarray(baseline["fiducial_post_P0"][:, :12], dtype=float),
            np.asarray(baseline["LC_p_post_P0"][:100, :12], dtype=float),
        ),
        "sto_marisa": (
            np.asarray(sto["fiducial_post_P0"][:, :12], dtype=float),
            np.asarray(sto["LC_p_post_P0"][:100, :12], dtype=float),
        ),
    }
    rows: dict[str, list[np.ndarray]] = {"fiducial": [], "LC_p": []}
    for cosmology, count in (("fiducial", 500), ("LC_p", 100)):
        for realization in range(count):
            path = MARISA_ROOT / cosmology / f"real{realization:03d}" / "post_pk.npz"
            with np.load(path, allow_pickle=False) as stream:
                rows[cosmology].append(np.asarray(stream["pk"], dtype=float))
    samples["marisa"] = (np.vstack(rows["fiducial"]), np.vstack(rows["LC_p"]))
    return samples


def independent_prediction(theory: dict[str, np.ndarray], physical: np.ndarray) -> np.ndarray:
    """不调用 production Case，独立实现 finite local-PNG P0。"""

    values = np.asarray(physical, dtype=float)
    f, b1, shot = values[:, 0], values[:, 1], values[:, 2]
    bp = 2.0 * 1.686 * (b1 - 1.12)
    return (
        b1[:, None] ** 2 * theory["PL_lattice"][None, :]
        + 2.0 * f[:, None] * (b1 * bp)[:, None] * theory["PL_over_M_lattice"][None, :]
        + f[:, None] ** 2 * bp[:, None] ** 2 * theory["Pphi_lattice"][None, :]
        + shot[:, None]
    )


def pdf_pages(path: Path) -> int:
    """用现有 pdfinfo 或 Ghostscript 读取 PDF 页数，不安装额外软件。"""

    if shutil.which("pdfinfo") is not None:
        completed = subprocess.run(
            ["pdfinfo", str(path)], check=True, text=True, capture_output=True,
        )
        for line in completed.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
    if shutil.which("gs") is None:
        raise RuntimeError("neither pdfinfo nor gs is available")
    escaped = str(path).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    command = f"({escaped}) (r) file runpdfbegin pdfpagecount = quit"
    completed = subprocess.run(
        ["gs", "-q", "-dNOSAFER", "-dNODISPLAY", "-c", command],
        check=True, text=True, capture_output=True,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    """执行要求级独立审计。"""

    required = (
        run.MATRIX, run.MATRIX_AUDIT, run.P_THEORY, run.PREREGISTRATION,
        run.PROFILE_JSON, run.PROFILE_NPZ, run.RESULT_JSON, run.RESULT_NPZ,
        run.RESULT_CSV, run.REPORT, run.MARGINAL_PDF, run.RECOVERY_PDF,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    result = json.loads(run.RESULT_JSON.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise RuntimeError("result status not complete")
    unified = load_npz(run.MATRIX)
    theory = load_npz(run.P_THEORY)
    combined = load_npz(run.RESULT_NPZ)
    originals = original_samples()
    profiles = run.load_profiles()
    result_rows = {(row["method"], row["target"]): row for row in result["rows"]}

    checks: dict[str, Any] = {}
    sample_checks = []
    target_checks = []
    covariance_checks = []
    for method in run.METHODS:
        fiducial, lcp = originals[method]
        sample_checks.extend((
            compare(f"{method}_fiducial", unified[f"{method}__fiducial_P0"], fiducial),
            compare(f"{method}_LCp", unified[f"{method}__LC_p_P0"], lcp),
        ))
        target_checks.extend((
            compare(
                f"{method}_target_fNL0",
                unified[f"{method}__target_fNL0_matched100"], fiducial[:100].mean(axis=0), atol=2e-11,
            ),
            compare(
                f"{method}_target_fNL100",
                unified[f"{method}__target_fNL100_matched100"], lcp.mean(axis=0), atol=2e-11,
            ),
        ))
        covariance_checks.append(compare(
            f"{method}_covariance",
            unified[f"{method}__covariance_fNL0_fid500_one_box"],
            np.cov(fiducial, rowvar=False, ddof=1), atol=2e-8,
        ))
    checks["original_sample_parity"] = {"rows": sample_checks, "pass": all(row["pass"] for row in sample_checks)}
    checks["matched100_target_parity"] = {"rows": target_checks, "pass": all(row["pass"] for row in target_checks)}
    checks["fid500_covariance_parity"] = {"rows": covariance_checks, "pass": all(row["pass"] for row in covariance_checks)}

    case_checks = []
    convergence_checks = []
    quantile_checks = []
    comparison_checks = []
    for target_name in run.TARGETS:
        pre_width = result_rows[("pre", target_name)]["fNL"]["half_width"]
        for method in run.METHODS:
            name = run.key(method, target_name)
            posterior = np.asarray(combined[f"{name}__posterior"], dtype=float)
            saved_logp = np.asarray(combined[f"{name}__log_probability"], dtype=float)
            indices = np.linspace(0, posterior.shape[0] - 1, 101, dtype=int)
            physical = posterior[indices]
            prediction = independent_prediction(theory, physical)
            target = originals[method][0][:100].mean(axis=0) if target_name == "fNL0" else originals[method][1].mean(axis=0)
            covariance = np.cov(originals[method][0], rowvar=False, ddof=1) / (run.VOLUME * run.HARTLAP)
            residual = prediction - target[None, :]
            white = np.linalg.solve(np.linalg.cholesky(covariance), residual.T).T
            independent_logp = -0.5 * np.sum(white**2, axis=1)
            case_checks.append({
                "case": name,
                "target_max_abs": float(np.max(np.abs(combined[f"{name}__target"] - target))),
                "covariance_max_abs": float(np.max(np.abs(combined[f"{name}__covariance"] - covariance))),
                "logp_max_abs": float(np.max(np.abs(saved_logp[indices] - independent_logp))),
                "pass": bool(
                    np.max(np.abs(combined[f"{name}__target"] - target)) <= 2e-11
                    and np.max(np.abs(combined[f"{name}__covariance"] - covariance)) <= 2e-8
                    and np.max(np.abs(saved_logp[indices] - independent_logp)) <= 1e-9
                ),
            })

            q = np.quantile(posterior[:, 0], (0.025, 0.16, 0.5, 0.84, 0.975))
            row = result_rows[(method, target_name)]
            saved = np.asarray([
                row["fNL"]["q2p5"], row["fNL"]["q16"], row["fNL"]["median"],
                row["fNL"]["q84"], row["fNL"]["q97p5"],
            ])
            quantile_checks.append({
                "case": name, "max_abs": float(np.max(np.abs(q - saved))),
                "pass": bool(np.max(np.abs(q - saved)) <= 1e-12),
            })

            task = json.loads(run.task_path(method, target_name).read_text(encoding="utf-8"))
            context = run.build_context(method, target_name, profiles)
            coordinates, accepted = run.load_chain(method, target_name)
            recalculated, _ = run.diagnose(context, coordinates, accepted)
            convergence_checks.append({
                "case": name,
                "task_status": task["status"],
                "recalculated_pass": recalculated["pass"],
                "rhat_max_abs": float(np.max(np.abs(
                    np.asarray(task["diagnostics"]["rank_normalized_split_rhat"])
                    - np.asarray(recalculated["rank_normalized_split_rhat"])
                ))),
                "pass": bool(
                    task["status"] == "complete" and task["diagnostics"]["pass"]
                    and recalculated["pass"]
                ),
            })

            ratio = row["fNL"]["half_width"] / pre_width
            stored = result["comparison_to_pre"][target_name][method]
            comparison_checks.append({
                "case": name,
                "ratio_abs": abs(ratio - float(stored["half_width_over_pre"])),
                "information_abs": abs(1.0 / ratio**2 - float(stored["information_over_pre"])),
                "pass": bool(
                    abs(ratio - float(stored["half_width_over_pre"])) <= 1e-12
                    and abs(1.0 / ratio**2 - float(stored["information_over_pre"])) <= 1e-12
                ),
            })
    checks["independent_likelihood"] = {"rows": case_checks, "pass": all(row["pass"] for row in case_checks)}
    checks["posterior_quantiles"] = {"rows": quantile_checks, "pass": all(row["pass"] for row in quantile_checks)}
    checks["chain_convergence"] = {"rows": convergence_checks, "pass": all(row["pass"] for row in convergence_checks)}
    checks["comparison_to_pre"] = {"rows": comparison_checks, "pass": all(row["pass"] for row in comparison_checks)}

    png_files = sorted(str(path) for path in FIGURES.glob("*.png"))
    pages = {str(run.MARGINAL_PDF): pdf_pages(run.MARGINAL_PDF), str(run.RECOVERY_PDF): pdf_pages(run.RECOVERY_PDF)}
    checks["figures"] = {
        "pdf_pages": pages, "png_files": png_files,
        "pass": bool(pages[str(run.MARGINAL_PDF)] == 3 and pages[str(run.RECOVERY_PDF)] == 1 and not png_files),
    }
    checks["result_hashes"] = {
        "posteriors": sha256(run.RESULT_NPZ) == result["outputs"]["posteriors"]["sha256"],
        "marginal_pdf": sha256(run.MARGINAL_PDF) == result["outputs"]["marginal_pdf"]["sha256"],
        "recovery_pdf": sha256(run.RECOVERY_PDF) == result["outputs"]["recovery_pdf"]["sha256"],
    }
    checks["result_hashes"]["pass"] = bool(all(checks["result_hashes"].values()))

    overall = bool(all(item.get("pass", False) for item in checks.values()))
    payload = {
        "schema": "quijote-z0p5-four-method-p0only-independent-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if overall else "fail",
        "checks": checks,
        "scope": {
            "methods": run.METHODS, "targets": run.TARGETS,
            "requirements": "matched100 targets; method-specific fNL0 Fid500 covariance; eight converged MCMCs; PDF-only",
        },
    }
    atomic_json(OUTPUT, payload)
    REPORT.write_text(
        "# 四方法 P0-only MCMC 独立审计\n\n"
        f"Status: `{payload['status']}`。\n\n"
        + "\n".join(f"- {name}: `{item.get('pass')}`" for name, item in checks.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable({"status": payload["status"], "audit": str(OUTPUT)}), indent=2))
    if not overall:
        raise RuntimeError("independent audit failed")


if __name__ == "__main__":
    main()
