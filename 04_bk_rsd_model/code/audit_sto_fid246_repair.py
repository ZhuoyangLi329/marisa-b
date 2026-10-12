"""模块大纲：同realization/snapshot的FoF恢复验收；原reader/质量阈值/行身份。

不覆盖旧FoF，不修改STO，不重算N-body；来源SHA与真实RSD转换单独保存。
文件传输完成本身不等于checksum验收，后者以独立任务实际状态为准。
"""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import numpy as np
import audit_sto_post_fid500_payloads as shared

ROOT = shared.ROOT
OUT = ROOT/'input_repair_fid246'
RAW = OUT/'raw/fiducial/real246/groups_003/group_tab_003.0'


def main():
    if len(os.sched_getaffinity(0)) != 1: raise ValueError('single core validation only')
    if (OUT/'AUDIT.json').exists(): raise ValueError('existing repair audit must not be overwritten')
    raw = shared.read_blob(RAW)
    if len(raw) != 25985172: raise ValueError('source-listed byte count mismatch')
    source = shared.SOURCE/'fiducial/real246/env_lam1_gsr2_thr15'
    payload = shared.read_blob(source/'post_sto_halo.npz')
    diagnostics = shared.read_blob(source/'diagnostics.json')
    diag = json.loads(diagnostics)
    if (any(diag['config'].get(k) != v for k,v in shared.RECIPE.items())
        or diag['component_config']['method'] != 'nenv'
        or diag['component_config']['base_weight'] != 'number'):
        raise ValueError('original source recipe mismatch')
    rsd, positions, meta = shared.catalog_from_blobs('fiducial',246,raw,raw_source=RAW)
    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
        identity = shared.validate_periodic_identity(data['positions_mpc_h'],positions)
        weights = np.asarray(data['weights'],dtype=float)
    if weights.shape != (len(positions),) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('invalid original weights')
    report = dict(status='SAME_REAL246_FOF_FORMAT_RECIPE_AND_STO_ROW_IDENTITY_PASS',
        created_UTC=datetime.now(timezone.utc).isoformat(), cosmology='fiducial', realization=246,
        snapshot=3, original_raw_path=str(shared.adapter.original.raw_path('fiducial',246)),
        consumed_raw_path=str(RAW), raw_bytes=len(raw), raw_sha256=shared.sha(raw),
        source_STO=str(source), source_STO_sha256=shared.sha(payload),
        source_diagnostics_sha256=shared.sha(diagnostics), source_position_identity=identity,
        RSD_input_domain=shared.validate_rsd_before_painting(rsd), raw_fof=meta,
        n_halo=len(positions), N2_data_self=float(1e9*np.sum(weights**2)/weights.sum()**2),
        N3_data_self=float(1e18*np.sum(weights**3)/weights.sum()**3),
        original_FoF_mass_threshold_reader_and_RSD_conversion_unchanged=True,
        original_FoF_or_STO_modified=False, new_Nbody_or_DM_generated=False,
        covariance_or_MCMC_released=False, CPU_affinity=sorted(os.sched_getaffinity(0)),
        entry_sha256=shared.sha(shared.read_blob(Path(__file__))))
    shared.save(OUT/'AUDIT.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
