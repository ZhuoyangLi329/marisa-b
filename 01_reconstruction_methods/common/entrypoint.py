"""v2 方法示例公共入口。

大纲：配置线程与导入路径 -> 选择方法 -> 读取单条示例 manifest -> 复用生产数值流程。
输出只允许落入第一部分 outputs；输入可以读取共享 Quijote 数据。
"""
from pathlib import Path
import os
import sys

PART = Path(__file__).resolve().parents[1]

def validate_output_path(path):
    """返回解析后的输出路径；拒绝第一部分 outputs 以外的写入目录。"""
    resolved = Path(path).expanduser().resolve()
    root = (PART / 'outputs').resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f'v2 示例输出必须在 {root} 下：{resolved}')
    return resolved

def validate_gra_paths(args):
    """在 manifest 覆盖配置后核对最终输出及方法，避免读取旧配置导致写回旧仓库。"""
    target = Path(args.result_root) / args.stage / args.cosmology / f'real{args.realization:03d}'
    if args.label:
        target /= args.label
    validate_output_path(target)
    if args.recon_class != os.environ.get('RECON_V2_EXPECTED_CLASS', args.recon_class):
        raise ValueError('manifest 方法与所选入口不一致')
    if args.input_mode != os.environ.get('RECON_V2_EXPECTED_INPUT', args.input_mode):
        raise ValueError('manifest 输入类型与所选入口不一致')

def run_gra(method):
    """运行 standard_gra、marisa_gra、sto_standard 或 sto_marisa 的一个示例。

    method 为上述稳定标识，返回无；数值结果由共享 pipeline 写出。
    默认 manifest 仅一条 fiducial 示例，CLI/manifest 参数仍沿用原 runner。
    """
    for key, value in {'OMP_NUM_THREADS':'8', 'OPENBLAS_NUM_THREADS':'1',
                      'MKL_NUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
                      'PYTHONDONTWRITEBYTECODE':'1', 'JAX_ENABLE_X64':'1'}.items():
        os.environ[key] = value
    sys.path.insert(0, str(PART / 'marisa_gra'))
    expected = 'marisa-plane' if 'marisa' in method else 'plane'
    input_mode = 'sto-weighted' if method.startswith('sto_') else 'positions'
    os.environ['RECON_V2_EXPECTED_CLASS'] = expected
    os.environ['RECON_V2_EXPECTED_INPUT'] = input_mode
    if '--manifest' not in sys.argv and '--help' not in sys.argv and '-h' not in sys.argv:
        sys.argv += ['--manifest', str(PART / 'examples' / f'{method}.jsonl'), '--task-index', '0']
    if '--recon-class' not in sys.argv:
        sys.argv += ['--recon-class', expected]
    import halo_b000_pipeline as pipeline
    # 复用仓库现存 API 兼容层；这里只修复摘要读取，不修改 estimator。
    import stovar_jaxpower_compat as compat
    compat.install_spectrum_summary_compat(pipeline)
    pipeline.main()
