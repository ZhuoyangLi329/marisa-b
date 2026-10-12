"""模块大纲：原target/cov数值入口，仅输出路径隔离；完成后原字节原子发布。

30分钟作业若超时，只保留独立staging，不留下半成品canonical pilot。
只路由ROOT/pilots，所有谱/alpha/self审计输入仍读原ROOT；随机种子不变。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from sto_post_compute8_common import PART, ROOT, OUT, directory, read, sha, save, row_for, target_rows


class OutputOnlyRoot:
    def __init__(self, original, staged):
        self.original, self.staged = Path(original), Path(staged)

    def __truediv__(self, name):
        return self.staged/'pilots' if name == 'pilots' else self.original/name


def validate_publication(staged, final, row, key):
    if final.exists():
        raise ValueError('canonical target already exists; never overwrite')
    done = read(staged/'DONE.json'); contract = read(staged/'CONTRACT.json')
    if (done['status'] != 'PILOT_COMPLETE_NOT_SCIENTIFIC_VALIDATION'
            or done['entry_sha256'] != sha(PART/'code/run_sto_post_rsd_pilot.py')
            or contract['source_STO_sha256'] != row['source_sha256']
            or contract['source_diagnostics_sha256'] != row['diagnostics_sha256']
            or contract['raw_fof']['source_raw_sha256'] != row['raw_fof']['source_raw_sha256']
            or (contract['arguments']['cosmology'], contract['arguments']['realization']) != key
            or len(done['CPU_affinity']) != 1):
        raise ValueError('completed staged box does not match original source/numerics')
    for field, filename in (('contract_sha256', 'CONTRACT.json'), ('measurement_sha256', 'MEASUREMENTS.npz'),
                            ('post_halos_sha256', 'POST_HALOS.npz')):
        if sha(staged/filename) != done[field]:
            raise ValueError('completed staging artifact hash drift')
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cosmology', choices=('fiducial', 'LC_p', 'LC_m'), required=True)
    ap.add_argument('--realization', type=int, required=True); ap.add_argument('--source-row')
    args = ap.parse_args(); key = args.cosmology, args.realization
    if len(os.sched_getaffinity(0)) != 1 or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('single-core compute-node atomic worker required')
    if not 0 <= args.realization < 500 or (args.realization >= 100 and args.cosmology != 'fiducial'):
        raise ValueError('original700 identity only')
    final = directory(key)
    if final.exists():
        raise ValueError('dispatcher must not assign an existing canonical output')
    staged_root = OUT/('job'+os.environ['SLURM_JOB_ID'])/'staging'/f'{key[0]}_{key[1]:03d}'
    if staged_root.exists():
        raise ValueError('preserve existing attempt staging')
    staged_root.mkdir(parents=True)
    import run_sto_post_rsd_pilot as pilot
    original_root = pilot.ROOT; pilot.ROOT = OutputOnlyRoot(ROOT, staged_root)
    try:
        if args.realization < 100:
            import run_sto_post_target_box as worker
            sys.argv = [worker.__file__, '--cosmology', args.cosmology, '--realization', str(args.realization)]
        else:
            import run_sto_post_covariance_box as worker
            if not args.source_row:
                raise ValueError('original admitted covariance row required')
            sys.argv = [worker.__file__, '--realization', str(args.realization), '--source-row', args.source_row]
        worker.main()
    finally:
        pilot.ROOT = original_root
    staged = staged_root/'pilots'/final.name
    _, row = row_for(key, target_rows()); done = validate_publication(staged, final, row, key)
    # 独占暂停授权+单个compute8 series lock保证没有另一个本任务producer写同一目录。
    if final.exists():
        raise ValueError('canonical output appeared during staging; preserve both')
    staged.rename(final)
    save(OUT/('job'+os.environ['SLURM_JOB_ID'])/'publications'/f'{key[0]}_{key[1]:03d}.json',
        dict(status='COMPLETE_ORIGINAL_BYTES_ATOMICALLY_PUBLISHED_NOT_MODEL_RELEASE', key=key,
            created_UTC=datetime.now(timezone.utc).isoformat(), canonical=str(final), staged_origin=str(staged),
            original_DONE=done, output_router_sha256=sha(Path(__file__)), original_numeric_sources_modified=False))


if __name__ == '__main__':
    main()
