"""模块大纲：旧IR球谐的显式进程内API兼容；不改旧源码或安装包。

旧sph_harm(m,l,azimuth,polar)与新sph_harm_y(l,m,polar,azimuth)
次序不同；仅旧名字缺失时补桥接，保留Condon-Shortley相位和归一化。
"""
import numpy as np
import scipy
from scipy import special

_OLD = getattr(special, 'sph_harm', None)


def legacy_sph_harm(m, ell, azimuth, polar, out=None):
    if hasattr(special, 'sph_harm_y'):
        value = special.sph_harm_y(ell, m, polar, np.mod(azimuth, 2*np.pi))
    elif _OLD is not None:
        value = _OLD(m, ell, azimuth, polar)
    else:
        raise RuntimeError('existing SciPy has neither spherical-harmonic API')
    if out is not None:
        np.copyto(out, value)
        return out
    return value


def install_legacy_sph_harm():
    installed = not hasattr(special, 'sph_harm')
    if installed:
        if not hasattr(special, 'sph_harm_y'):
            raise RuntimeError('existing SciPy has no supported harmonic API')
        special.sph_harm = legacy_sph_harm
    return dict(process_only=True, original_source_modified=False, packages_installed=False,
        bridge_installed=installed, scipy_version=scipy.__version__,
        mapping='legacy(m,l,azimuth,polar) -> sph_harm_y(l,m,polar,azimuth mod2pi)')
