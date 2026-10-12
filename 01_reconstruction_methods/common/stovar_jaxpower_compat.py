#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""jaxpower 2026-09-08 版本回归的运行时兼容 shim（STO 加权扫描专用）。

背景
====
adematti 于 2026-09-08 03:43 原地更新了共享 jaxpower（cosmodesiconda
20260321-1.0.0/code/jaxpower/main）。新版 ``jaxpower.mesh.get_mesh_attrs``
不再对 ``mpicomm=None`` 容忍（旧版可用），而冻结的
``/pscratch/sd/lzy/marisa-recon/clustering_statistics/spectrum2_tools.py``
在 ``prepare_jaxpower_particles`` 中调用它时没有显式传 mpicomm，导致
``AttributeError: 'NoneType' object has no attribute 'allgather'``
（2026-09-07 15:20 起 stovar_v0pk element 7-11 与全部 sto-only 元素失败）。

修复原则
========
1. 不修改任何冻结代码（marisa-recon、归档 estimator、jaxpower 安装）；
2. 在我们自己的入口（statistics batch 脚本 import runner 之前）调用
   ``install()``，把 ``jaxpower.mesh.get_mesh_attrs`` 包一层：缺省注入
   ``mpicomm=MPI.COMM_WORLD``；
3. 该 shim 只影响通信参数的默认值（单 rank 下 allgather 恒等），
   **不改变任何数值结果**——新旧 jaxpower 的数值一致性由单独的
   bitwise 复跑验证（real350 对照）另行确认。
"""

from __future__ import annotations

import sys
from typing import Any

_installed = False


def install() -> dict[str, Any]:
    """给 jaxpower.mesh.get_mesh_attrs 注入默认 mpicomm；幂等。

    返回
    ----
    dict
        记录是否打补丁、目标函数来源等审计信息。
    """

    global _installed
    if _installed:
        return {"already_installed": True}

    try:
        from mpi4py import MPI
    except ImportError:  # pragma: no cover - cosmodesi 环境必有 mpi4py
        return {"installed": False, "reason": "mpi4py unavailable"}

    try:
        import jaxpower.mesh as jaxpower_mesh
    except ImportError:  # pragma: no cover
        return {"installed": False, "reason": "jaxpower.mesh unavailable"}

    original = getattr(jaxpower_mesh, "get_mesh_attrs", None)
    if original is None:  # pragma: no cover
        return {"installed": False, "reason": "jaxpower.mesh.get_mesh_attrs missing"}

    def get_mesh_attrs_with_default_mpicomm(*args: Any, **kwargs: Any) -> Any:
        """缺省注入 mpicomm=MPI.COMM_WORLD 后调用原版 get_mesh_attrs。"""

        kwargs.setdefault("mpicomm", MPI.COMM_WORLD)
        return original(*args, **kwargs)

    # 必须在 clustering_statistics 等下游模块 import get_mesh_attrs 之前
    # 替换模块属性，它们 ``from jaxpower.mesh import get_mesh_attrs`` 时
    # 拿到的就是补丁版。
    jaxpower_mesh.get_mesh_attrs = get_mesh_attrs_with_default_mpicomm
    # jaxpower 顶层 re-export 同步替换（若有）。
    try:
        import jaxpower

        if getattr(jaxpower, "get_mesh_attrs", None) is original:
            jaxpower.get_mesh_attrs = get_mesh_attrs_with_default_mpicomm
    except ImportError:  # pragma: no cover
        pass

    _installed = True
    return {
        "installed": True,
        "patched": f"{jaxpower_mesh.__name__}.get_mesh_attrs",
        "jaxpower_file": str(sys.modules.get("jaxpower", jaxpower_mesh).__file__),
    }


def install_spectrum_summary_compat(module: Any) -> bool:
    """把模块对象上的 ``spectrum_bin_summary`` 替换为新 API 兼容版。

    lsstypes 2026-09-08 更新移除了 ``pole.get().k`` 旧访问路径；冻结的
    ``spectrum_bin_summary`` 因此 AttributeError。替换实现：把 spectrum
    写到临时 h5 再读回 ``k/k_edges/nmodes``——与 write_and_validate_spectrum
    的读法同源，数值逐位一致。幂等（重复调用无副作用）。
    """

    if getattr(module, "_stovar_spectrum_summary_patched", False):
        return False

    def _spectrum_bin_summary_compat(spectrum: Any) -> dict[str, Any]:
        import os
        import tempfile

        import h5py
        import numpy as np

        fd, tmp_path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        try:
            spectrum.write(tmp_path)
            with h5py.File(tmp_path, "r") as handle:
                group = handle["0_0_0"] if "0_0_0" in handle else handle["data"]["0_0_0"]
                return {
                    "k": np.asarray(group["k"]),
                    "k_edges": np.asarray(group["k_edges"]),
                    "nmodes": np.asarray(group["nmodes"]),
                }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    module.spectrum_bin_summary = _spectrum_bin_summary_compat
    module._stovar_spectrum_summary_patched = True
    return True


if __name__ == "__main__":
    print(install())
    print("spectrum compat ready")
