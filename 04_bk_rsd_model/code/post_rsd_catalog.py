"""模块大纲：复用原 FoF/RSD 数值读取；显式处理仅缺可选 mass cache 的旧导出。

原 catalog 把 positions cache 存在等同于 masses cache 存在；独立 LC_p tail
只存位置。此 adapter 不改原函数/源文件或 raw FoF；临时解绑其可选缓存后
重用原数值流程，再独立严格核对旧位置。绝不伪造 mass cache 或放宽质量选择。
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import run_rsd_bonly_workflow as original
from rsd_b000_contract import PART,digest
from post_rsd_catalog_identity import validate_periodic_identity


def catalog(cosmology,realization):
    root=original.OLD_POS
    cache=root/cosmology/f'real{realization:03d}'
    positions=cache/'positions_mpc_h.npy';masses=cache/'masses_msun_h.npy'
    incomplete_optional_cache=positions.exists() and not masses.exists()
    if not incomplete_optional_cache:
        return original.catalog(cosmology,realization)
    # 仅此单进程内的可选 cache 路径变化；原质量阈值/排序/速度/RSD 计算仍只执行原函数。
    with TemporaryDirectory(prefix='post_rsd_optional_cache_',dir=PART/'work') as empty:
        try:
            original.OLD_POS=Path(empty)
            rsd,real,meta=original.catalog(cosmology,realization)
        finally:
            original.OLD_POS=root
    identity=validate_periodic_identity(np.load(positions,allow_pickle=False),real)
    meta.update(optional_position_cache_sha256=digest(positions),optional_position_identity=identity,
                optional_mass_cache_exists=False,optional_mass_cache_independently_validated=False,
                raw_FoF_mass_selection_unchanged=True,
                optional_cache_adapter='missing mass cache only; original catalog numeric path; old positions checked separately')
    return rsd,real,meta
