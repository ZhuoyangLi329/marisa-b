#!/usr/bin/env python3
"""Phase 2+4：生成 marisa only（unit 输入）v2 重产 manifest（600 套全量）。

源：quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100（recon_class=marisa-plane、
input_mode=unit、png_p=1.0、bias=1.9327）。改：
- png_p: 1.0 -> 1.1217（P0 freep 实测 p_marisa）
- bias: 1.9327 -> 1.9364（P0 freep 拟合 b1），造新 bias json
- 输出 -> v2_p1217 新 root（不覆盖旧数据）
用途：Phase 3 的 marisa P 线重校准 + marisa B 扫描；Phase 4 的 Slurm 全量生产。
"""
import json
from pathlib import Path

OLD_MANIFEST = Path("/pscratch/sd/l/lzy/reconstruction-png/manifests/"
                    "quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100/tasks_all.jsonl")
NEW_TAG = "quijote_halo_z0p5_r10_stovar_unit_fid500_lcp100_marisa_b000_pk_v2_p1217"
NEW_MANIFEST_DIR = Path("/pscratch/sd/l/lzy/reconstruction-png/manifests_v2_paware") / NEW_TAG
NEW_RESULT_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png") / ("result_" + NEW_TAG)
BIAS_JSON = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/analysis/"
                 "v2_paware_bias_marisa_b1_1p9364.json")
P_NEW = 1.1217
B1_NEW = 1.9364310827111566


def main():
    NEW_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    BIAS_JSON.parent.mkdir(parents=True, exist_ok=True)

    bias_payload = {
        "created_utc": "2026-09-18T14:05:00+00:00",
        "fit_model": "P_ww = b1^2 P_lin + c0",
        "fit_space": "realspace",
        "free_sn": {"b1": B1_NEW, "b1_err_from_cov_of_mean": 0.002,
                    "b1_squared": B1_NEW ** 2, "ndof": 13},
        "kmax_fit": 0.1, "kmin_fit": 0.0,
        "legacy_desilike_bias": None,
        "n_realizations": 500,
        "recommended_bias_source": "v2_paware_freep_b1_marisa",
        "recommended_bias_value": B1_NEW,
        "redshift": 0.5,
        "note": ("v2 p-aware re-production (marisa only): b1 from P0 freep; "
                 "png_p=1.1217 (measured p_marisa)"),
    }
    BIAS_JSON.write_text(json.dumps(bias_payload, indent=2) + "\n", encoding="utf-8")

    counts = {"fiducial": 0, "LC_p": 0}
    rows = []
    with OLD_MANIFEST.open() as fh:
        for line in fh:
            row = json.loads(line)
            cosmo = row.get("cosmology")
            if cosmo not in counts or row.get("recon_class") != "marisa-plane":
                continue
            row["png_p"] = P_NEW
            row["bias"] = B1_NEW
            row["bias_json_path"] = str(BIAS_JSON)
            row["bias_source"] = "bias_json"
            row["result_root"] = str(NEW_RESULT_ROOT)
            row["pipeline_name"] = NEW_TAG
            real = int(row["realization"])
            row["output_dir"] = str(NEW_RESULT_ROOT / "full" / cosmo / f"real{real:03d}")
            row["completed"] = False
            row["runnable"] = True
            row["overwrite"] = False
            counts[cosmo] += 1
            rows.append(row)

    out = NEW_MANIFEST_DIR / "tasks_all.jsonl"
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"manifest rows: {len(rows)} ({counts}) -> {out}")
    print(f"result root:   {NEW_RESULT_ROOT}")
    print(f"bias json:     {BIAS_JSON}")


if __name__ == "__main__":
    main()
