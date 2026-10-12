#!/usr/bin/env python3
"""绘制matched100 pre/STO P-only三参数MCMC contours，仅输出PDF。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

import run_quijote_halo_z0p5_pre_sto_marisa_p100matched100_pk0p08_bkmax0p20_joint_mcmc as base
import plot_quijote_halo_z0p5_pre_sto_marisa_p100matched100_mcmc_contours as contour


OUTPUT_PDF = base.FIGURES / "ponly_mcmc_contours.pdf"
OUTPUT_JSON = base.ANALYSIS / "PONLY_CONTOUR_MANIFEST.json"
METHODS = base.METHODS
COLORS = {"pre": "#4E79A7", "sto_marisa": "#59A14F"}
LABELS = (r"$f_{\rm NL}$", r"$b_1$", r"$P_{\rm shot,res}$")


def main() -> None:
    """生成两页P-only marginal/triangle PDF和coverage manifest。"""

    result = json.loads(base.RESULT_JSON.read_text(encoding="utf-8"))
    audit_path = base.ANALYSIS / "PONLY_JOINT_MCMC_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if result["status"] != "complete" or not audit["status"].startswith("pass"):
        raise ValueError("上游P-only/joint MCMC或audit未通过")
    arrays = base.load_npz(base.RESULT_NPZ)
    samples = {
        method: np.asarray(arrays[f"{base.key(method, 'P_only')}__posterior"], dtype=float)
        for method in METHODS
    }
    for method, sample in samples.items():
        if sample.shape != (base.POSTERIOR_DRAWS, 3) or not np.all(np.isfinite(sample)):
            raise ValueError(f"{method} P-only posterior shape/finite错误")
    ranges = [
        contour.parameter_range(np.concatenate([samples[method][:, i] for method in METHODS]))
        for i in range(3)
    ]
    checks = {}
    base.FIGURES.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for method in METHODS:
            contour.draw_marginal(axes[0], samples[method][:, 0], ranges[0], COLORS[method])
            checks[f"direct__{method}"] = contour.draw_contours(
                axes[1], samples[method][:, 0], samples[method][:, 1],
                ranges[0], ranges[1], COLORS[method], fill=False,
            )
        axes[0].axvline(base.TRUTH, color="#E15759", ls=":", lw=1.2)
        axes[1].axvline(base.TRUTH, color="#E15759", ls=":", lw=1.2)
        axes[0].set_xlabel(LABELS[0]); axes[0].set_ylabel("marginal density"); axes[0].set_yticks([])
        axes[1].set_xlabel(LABELS[0]); axes[1].set_ylabel(LABELS[1])
        handles = [
            Line2D([0], [0], color=COLORS[method], lw=1.8, label=base.METHODS[base.METHODS.index(method)])
            for method in METHODS
        ] + [
            Line2D([0], [0], color="0.25", lw=1.5, ls="-", label="68% HPD"),
            Line2D([0], [0], color="0.25", lw=1.0, ls="--", label="95% HPD"),
        ]
        axes[0].legend(handles=handles[:2], frameon=False)
        axes[1].legend(handles=handles, frameon=False, fontsize=8)
        for axis in axes: axis.grid(alpha=.15)
        figure.suptitle(r"P-only matched LCp100, $k_{\max}^P=0.08$")
        figure.tight_layout(); pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

        figure, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))
        for row in range(3):
            for column in range(3):
                axis = axes[row, column]
                if column > row:
                    axis.axis("off"); continue
                for method in METHODS:
                    sample = samples[method]
                    if row == column:
                        contour.draw_marginal(axis, sample[:, column], ranges[column], COLORS[method])
                    else:
                        checks[f"triangle__{method}__{column}_{row}"] = contour.draw_contours(
                            axis, sample[:, column], sample[:, row],
                            ranges[column], ranges[row], COLORS[method], fill=False,
                        )
                if row == 2: axis.set_xlabel(LABELS[column])
                else: axis.set_xticklabels([])
                if column == 0 and row > 0: axis.set_ylabel(LABELS[row])
                elif column > 0: axis.set_yticklabels([])
                axis.tick_params(labelsize=7); axis.grid(alpha=.1)
        figure.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
        figure.suptitle("pre vs STO+Marisa P-only posterior", y=.995)
        figure.tight_layout(rect=(0, 0, .98, .98)); pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)

    coverage_gate = bool(
        all(.68 <= item["coverage_68"] <= .72 for item in checks.values())
        and all(.95 <= item["coverage_95"] <= .98 for item in checks.values())
    )
    payload = {
        "schema": "quijote-z0p5-pre-sto-matched100-ponly-contours-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if coverage_gate else "coverage_fail",
        "scope": {
            "methods": METHODS, "parameters": base.PARAMETERS["P_only"],
            "draws_per_method": base.POSTERIOR_DRAWS, "pages": 2,
            "policy": "pre and STO+Marisa contours overlaid on identical axes",
        },
        "coverage_gate": coverage_gate, "contour_checks": checks,
        "inputs": {
            "result": base.artifact(base.RESULT_JSON),
            "posterior": base.artifact(base.RESULT_NPZ),
            "audit": base.artifact(audit_path),
            "runner": base.artifact(Path(__file__)),
        },
        "output": base.artifact(OUTPUT_PDF),
    }
    base.atomic_json(OUTPUT_JSON, payload)
    if not coverage_gate:
        raise RuntimeError("P-only contour coverage gate失败")
    print(json.dumps(base.jsonable({
        "status": payload["status"], "pdf": str(OUTPUT_PDF), "pages": 2,
    }), indent=2))


if __name__ == "__main__":
    main()
