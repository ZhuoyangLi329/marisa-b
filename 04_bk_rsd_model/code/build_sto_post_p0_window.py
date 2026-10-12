"""模块大纲：调用唯一实际 binner/window 导出器；仅重定向新 post 任务输出。

不修改旧 P+B 工具源文件/窗口；两个 OUT 仅在本进程中暂时绑定新目录。
"""
import json
from pathlib import Path
import numpy as np
import rsd_pb_common as common
import build_rsd_p0_window as original
from rsd_b000_contract import PART,digest


def main():
    out=PART/'outputs/sto_marisa_post_rsd_mean100_v1/calibration_holdout_p0'
    if not (out/'rows/LC_p_real100/DONE.json').exists():raise ValueError('measure the first independent holdout before exporting window')
    if (out/'P0_WINDOW_AUDIT.json').exists():raise ValueError('window already exists; inspect instead of overwriting')
    old_common,old_original=common.OUT,original.OUT
    try:
        common.OUT=out;original.OUT=out;original.main()
    finally:
        common.OUT=old_common;original.OUT=old_original
    with np.load(out/'P0_WINDOW.npz') as z,np.load(out/'rows/LC_p_real100/P0.npz') as p:
        np.testing.assert_array_equal(z['nmodes'],p['P0_nmodes'])
        np.testing.assert_array_equal(z['edges'],p['edges'])
        np.testing.assert_allclose(z['k'],p['P0_k'],rtol=3e-6,atol=2e-8)
    audit=json.loads((out/'P0_WINDOW_AUDIT.json').read_text())
    audit.update(post_wrapper_sha256=digest(Path(__file__)),holdout_measurement_nmodes_match=True,
                 holdout_measurement_sha256=digest(out/'rows/LC_p_real100/P0.npz'),
                 output_redirect_only=True,old_pre_window_modified=False)
    common.save_json(out/'POST_P0_WINDOW_AUDIT.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
