#!/usr/bin/env python3
"""Phase 1a：生成 sto_marisa LC_p 重产新 manifest（png_p=1.2822、bias=2.2281）。

从旧生产 manifest 复制 LC_p 100 行，改：
- png_p: 1.0 -> 1.2822（P0 实测校准 p_sto_marisa）
- bias/bias_json: 2.1951 -> 2.2281（五方法 MCMC b1 median），造新 bias json
- output_dir/result_root/pipeline_name -> 新 v2_p1282 root（不覆盖旧数据）
其余字段（sto 输入、几何、随机数、fnl_loc=truth）原样保留。
"""
import json
from pathlib import Path

OLD_MANIFEST = Path("/pscratch/sd/l/lzy/reconstruction-png/manifests/"
                    "quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100/"
                    "sto_marisa_b000_pk/tasks_all.jsonl")
NEW_TAG = "quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_sto_marisa_b000_pk_v2_p1282"
NEW_MANIFEST_DIR = Path("/pscratch/sd/l/lzy/reconstruction-png/manifests_v2_paware") / NEW_TAG
NEW_RESULT_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png") / ("result_" + NEW_TAG)
BIAS_JSON = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/analysis/"
                 "v2_paware_bias_sto_marisa_b1_2p2281.json")
P_NEW = 1.2822
B1_NEW = 2.2280546353635335


def main():
    NEW_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    BIAS_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 新 bias json：结构沿用 canonical（recommended_bias_value + free_sn.b1 都置新值）
    bias_payload = {
        "created_utc": "2026-09-18T13:45:00+00:00",
        "fit_model": "P_ww = b1^2 P_lin + c0",
        "fit_space": "realspace",
        "free_sn": {"b1": B1_NEW, "b1_err_from_cov_of_mean": 0.002,
                    "b1_squared": B1_NEW ** 2, "ndof": 13},
        "kmax_fit": 0.1, "kmin_fit": 0.0,
        "legacy_desilike_bias": None,
        "n_realizations": 500,
        "recommended_bias_source": "v2_paware_calibration_b1_median",
        "recommended_bias_value": B1_NEW,
        "redshift": 0.5,
        "note": ("v2 p-aware re-production: b1 from five-method MCMC sto_marisa median; "
                 "paired with png_p=1.2822 (P0 freep measured p_sto_marisa)"),
    }
    BIAS_JSON.write_text(json.dumps(bias_payload, indent=2) + "\n", encoding="utf-8")

    rows = []
    with OLD_MANIFEST.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("cosmology") != "LC_p":
                continue
            row["png_p"] = P_NEW
            row["bias"] = B1_NEW
            row["bias_json_path"] = str(BIAS_JSON)
            row["bias_source"] = "bias_json"
            row["result_root"] = str(NEW_RESULT_ROOT)
            row["pipeline_name"] = NEW_TAG
            real = int(row["realization"])
            row["output_dir"] = str(NEW_RESULT_ROOT / "full" / "LC_p" / f"real{real:03d}")
            row["completed"] = False
            row["runnable"] = True
            row["overwrite"] = False
            rows.append(row)

    out = NEW_MANIFEST_DIR / "tasks_all.jsonl"
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"manifest rows: {len(rows)} -> {out}")
    print(f"result root:   {NEW_RESULT_ROOT}")
    print(f"bias json:     {BIAS_JSON}")


if __name__ == "__main__":
    main()
