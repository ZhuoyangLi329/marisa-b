"""模块大纲：保持 halo 行序的严格周期位置身份核验；只允许确切的 0=L 端点。

不引入浮点 tolerance，不改变数组或权重，不接受其他偏移/重排/越界坐标。
"""
import numpy as np


def validate_periodic_identity(source,raw,boxsize=1000.):
    a=np.asarray(source);b=np.asarray(raw)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:raise ValueError('halo coordinate shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite halo coordinates')
    if not np.isfinite(boxsize) or boxsize<=0:raise ValueError('invalid periodic box size')
    if np.any(a<0) or np.any(a>boxsize) or np.any(b<0) or np.any(b>boxsize):
        raise ValueError('coordinate outside closed [0,L] input domain')
    mismatch=a!=b
    endpoint=((a==0)&(b==boxsize))|((a==boxsize)&(b==0))
    bad=np.argwhere(mismatch&~endpoint)
    if len(bad):raise ValueError('non-endpoint coordinate mismatch: '+str(bad[:8].tolist()))
    np.testing.assert_array_equal(np.mod(a,boxsize),np.mod(b,boxsize))
    index=np.argwhere(mismatch)
    return {'policy':'exact array equality modulo only the periodic 0=L endpoint; no tolerance',
            'array_equal_before_periodic_identification':bool(not mismatch.any()),
            'endpoint_equivalent_components':len(index),'endpoint_indices_first8':index[:8].tolist(),
            'halo_order_changed':False,'coordinates_or_weights_modified':False}
