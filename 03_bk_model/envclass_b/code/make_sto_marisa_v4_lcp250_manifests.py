#!/usr/bin/env python3
"""生成并审计 sto_marisa v4 LC_p real100--249 扩产 manifest。

旧仓库保持只读：real000--099 继续从既有 p-aware 生产读取，新增尾部的
STO source 与 MARISA P+B 都写入新仓库第一部分 outputs。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
PART = ROOT / "01_reconstruction_methods"
HERE = ROOT / "03_bk_model/envclass_b"
WORK = HERE / "work/sto_marisa_v4_lcp250"
SOURCE_ROOT = PART / "outputs/sto_marisa_v4_lcp250_source"
MEASUREMENT_ROOT = PART / "outputs/sto_marisa_v4_lcp250"
OLD_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
OLD_SOURCE_MANIFEST = (
    OLD_ROOT
    / "manifests/quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100/tasks_all.jsonl"
)
OLD_MEASUREMENT_MANIFEST = (
    OLD_ROOT
    / "manifests_v2_paware/quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_"
    "sto_marisa_b000_pk_v2_p1282/tasks_all.jsonl"
)
OLD_HEAD = (
    OLD_ROOT
    / "result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_"
    "sto_marisa_b000_pk_v2_p1282/full/LC_p"
)
HALO_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions")
DM_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/dm")
LABEL = "env_lam1_gsr2_thr15"
PIPELINE_NAME = (
    "quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp250_"
    "sto_marisa_b000_pk_v4_p1282"
)
FIRST_REAL = 100
LAST_REAL = 249


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_first_lcp(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cosmology") == "LC_p":
                return row
    raise RuntimeError(f"没有在模板 manifest 中找到 LC_p 行: {path}")


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_inputs() -> None:
    missing: list[str] = []
    for real in range(FIRST_REAL, LAST_REAL + 1):
        halo = HALO_ROOT / "LC_p" / f"real{real:03d}" / "positions_mpc_h.npy"
        dm = DM_ROOT / "LC_p" / str(real) / "snapdir_003"
        if not halo.is_file():
            missing.append(str(halo))
        if not dm.is_dir() or not any(dm.glob("*.hdf5")):
            missing.append(str(dm))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"扩产输入缺失 {len(missing)} 项:\n{preview}")


def require_complete_head() -> None:
    required = ("DONE", "metadata.json", "post_pk.npz", "post_b000.h5")
    missing = [
        str(OLD_HEAD / f"real{real:03d}" / name)
        for real in range(FIRST_REAL)
        for name in required
        if not (OLD_HEAD / f"real{real:03d}" / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"既有 p-aware head100 不完整，首个缺失文件: {missing[0]}")


def source_row(template: dict, real: int) -> dict:
    row = dict(template)
    row.update(
        {
            "cosmology": "LC_p",
            "realization": real,
            "task_id": real,
            "result_root": str(SOURCE_ROOT),
            "output_dir": str(SOURCE_ROOT / "LC_p" / f"real{real:03d}" / LABEL),
        }
    )
    return row


def measurement_row(template: dict, real: int) -> dict:
    row = dict(template)
    out = MEASUREMENT_ROOT / "full" / "LC_p" / f"real{real:03d}"
    sto = SOURCE_ROOT / "LC_p" / f"real{real:03d}" / LABEL / "post_sto_halo.npz"
    row.update(
        {
            "cosmology": "LC_p",
            "realization": real,
            "task_id": 500 + real,
            "pipeline_name": PIPELINE_NAME,
            "marisa_recon_root": str(PART / "marisa_gra"),
            "result_root": str(MEASUREMENT_ROOT),
            "output_dir": str(out),
            "sto_result_root": str(SOURCE_ROOT),
            "sto_post_halo": str(sto),
            "sto_exists": False,
            "completed": False,
            "runnable": True,
            "overwrite": False,
        }
    )
    return row


def assert_contract(source_rows: list[dict], measurement_rows: list[dict]) -> None:
    expected = list(range(FIRST_REAL, LAST_REAL + 1))
    for name, rows in (("source", source_rows), ("measurement", measurement_rows)):
        actual = [int(row["realization"]) for row in rows]
        if actual != expected or len(set(actual)) != len(expected):
            raise RuntimeError(f"{name} realization 集合不等于 real100--249")
    frozen = {
        "input_mode": "sto-weighted",
        "recon_class": "marisa-plane",
        "png_p": 1.2822,
        "fnl_loc": 100.0,
        "bias": 2.2280546353635335,
        "measure_pk": True,
        "meshsize": 256,
        "pk_meshsize": 256,
        "expected_nbins": 120,
        "marisa_recon_root": str(PART / "marisa_gra"),
    }
    for row in measurement_rows:
        for key, expected_value in frozen.items():
            if row.get(key) != expected_value:
                raise RuntimeError(f"measurement contract 失配: {key}={row.get(key)!r}, 预期 {expected_value!r}")
    for row in source_rows:
        expected_source = {
            "method": "nenv",
            "env_source": "dm",
            "weight_source": "dm",
            "base_weight": "number",
            "env_mesh": 128,
            "recon_mesh": 128,
            "gsr": 2.0,
            "thr": 15.0,
            "lambda_th": 1.0,
        }
        for key, expected_value in expected_source.items():
            if row.get(key) != expected_value:
                raise RuntimeError(f"source contract 失配: {key}={row.get(key)!r}, 预期 {expected_value!r}")


def main() -> None:
    require_inputs()
    require_complete_head()
    source_template = load_first_lcp(OLD_SOURCE_MANIFEST)
    measurement_template = load_first_lcp(OLD_MEASUREMENT_MANIFEST)
    source_rows = [source_row(source_template, real) for real in range(FIRST_REAL, LAST_REAL + 1)]
    measurement_rows = [measurement_row(measurement_template, real) for real in range(FIRST_REAL, LAST_REAL + 1)]
    assert_contract(source_rows, measurement_rows)

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    MEASUREMENT_ROOT.mkdir(parents=True, exist_ok=True)
    source_path = WORK / "source_tail_real100_249.jsonl"
    measurement_path = WORK / "measurement_tail_real100_249.jsonl"
    atomic_jsonl(source_path, source_rows)
    atomic_jsonl(measurement_path, measurement_rows)
    summary = {
        "schema": "sto-marisa-v4-lcp250-expansion-v1",
        "created_utc": utc_now(),
        "realizations": {"first": FIRST_REAL, "last": LAST_REAL, "count": len(source_rows)},
        "head_read_only": {"root": str(OLD_HEAD), "realizations": "000--099"},
        "tail_outputs": {"source_root": str(SOURCE_ROOT), "measurement_root": str(MEASUREMENT_ROOT)},
        "manifests": {
            "source": str(source_path),
            "source_sha256": sha256(source_path),
            "measurement": str(measurement_path),
            "measurement_sha256": sha256(measurement_path),
        },
        "templates_read_only": {
            "source": str(OLD_SOURCE_MANIFEST),
            "measurement": str(OLD_MEASUREMENT_MANIFEST),
        },
        "contract": {
            "png_p": 1.2822,
            "bias": 2.2280546353635335,
            "fnl_loc": 100.0,
            "statistics": ["post_pk", "post_b000"],
            "policy": "single p-aware P250+B250 product; no legacy-P branch",
        },
    }
    atomic_json(WORK / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
