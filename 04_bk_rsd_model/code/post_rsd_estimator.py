"""模块大纲：B000 S113 的 exact discrete-shell 投影；只作进程内显式适配。

共享 estimator 的 raw、S111、S122、S121、normalization 均复用原实现。
旧 S113 使用 j0(kmean*r)j0(kmean*r)；这里改用实际 bin masks 的归一化
逆 FFT 窗口。只支持固定 LOS B000；退出 context 后恢复原函数，不改安装文件。
"""
from contextlib import contextmanager
import numpy as np
import jax
import jax.numpy as jnp


def exact_b000_s113(rmesh,cmesh,s111=None,cmesh_s111=1.,bin=None,ells=(),los='z',diagnostics=None):
    if tuple(ells)!=((0,0,0),) or bin is None or 'sugiyama' not in bin.basis:
        raise NotImplementedError('exact S113 adapter is scoped to Sugiyama B000')
    if isinstance(los,str) and los=='local':raise NotImplementedError('global LOS only')
    rmesh=rmesh-rmesh.mean()
    cmesh=cmesh.clone(value=cmesh.value.at[(0,)*cmesh.ndim].set(0.))
    cross_k=rmesh.r2c()*cmesh.conj()-s111[(0,0)]*cmesh_s111
    windows={}
    shape=cmesh.value.shape
    labels_by_axis=[np.asarray(x).reshape(shape) for x in bin.ibin1d]
    same_axes=np.array_equal(labels_by_axis[0],labels_by_axis[1])
    if diagnostics is not None:
        if not same_axes:raise NotImplementedError('zero-atom diagnostic requires common radial bins')
        labels=labels_by_axis[0];counts=np.asarray(bin.nmodes1d[0],dtype=float)
        cval=np.asarray(cross_k.value).real;zero=(0,)*cmesh.ndim;zero_bin=int(labels[zero])
        selected=(labels>=0)&(labels<len(counts))
        sums=np.bincount(labels[selected].astype(int),weights=cval[selected],minlength=len(counts))
        nonzero_counts=counts.copy()
        if 0<=zero_bin<len(counts):sums[zero_bin]-=cval[zero];nonzero_counts[zero_bin]-=1
        zero_shot=[];zero_fraction=[]
        for i,j in np.asarray(bin._iedges):
            i,j=int(i),int(j);value=0.;number=0.
            if i==zero_bin:value+=2*sums[j];number+=nonzero_counts[j]
            if j==zero_bin:value+=2*sums[i];number+=nonzero_counts[i]
            if i==j:value+=2*sums[i];number+=nonzero_counts[i]
            if i==j==zero_bin:
                value+=3*cval[zero]+float(np.asarray(s111[(0,0)]).real);number+=1
            denominator=counts[i]*counts[j]
            zero_shot.append(value/denominator);zero_fraction.append(number/denominator)
        diagnostics.update(zero_leg_num_shotnoise=np.asarray(zero_shot),zero_leg_pair_fraction=np.asarray(zero_fraction),
                           zero_leg_policy='diagnostic only; original full-shell shot result is unchanged')
    cross=cross_k.c2r()
    for axis in (0,1):
        labels=labels_by_axis[axis]
        counts=np.asarray(bin.nmodes1d[axis])
        for shell in np.unique(np.asarray(bin._iedges)[:,axis]):
            shell=int(shell);mask=labels==shell
            if int(mask.sum())!=int(counts[shell]):
                raise ValueError(('bin mask count mismatch',axis,shell,int(mask.sum()),int(counts[shell])))
            if axis==1 and same_axes and (0,shell) in windows:
                windows[axis,shell]=windows[0,shell]
                continue
            transformed=cmesh.clone(value=jnp.asarray(mask,dtype=cmesh.value.dtype)).c2r().value
            origin=transformed[(0,)*cmesh.ndim]
            if abs(complex(origin))==0.:raise ValueError('empty shell')
            # origin=Nmode/Ncell（或等价 FFT normalization），因此窗口在 r=0 归一。
            # B000 的完整 radial mask 是偶函数，窗口严格为实；不存两份复数组。
            windows[(axis,shell)]=(transformed/origin).real
    values=[]
    for i,j in np.asarray(bin._iedges):
        value=jnp.sum(cross.value*windows[(0,int(i))]*windows[(1,int(j))])
        values.append(value)
    return [jnp.asarray(values,dtype=cmesh.value.dtype)]


@contextmanager
def matched_b000_self_contractions(record=None,*,matched_paint=False):
    """record 可同时保留原 S113，便于同一次 raw 测量输出 legacy/exact 两种扣除。"""
    import jaxpower.mesh3 as module
    import jaxpower
    original=module._compute_sugiyama_spectrum_S113
    original_shot=jaxpower.compute_fkp3_shotnoise
    def adapter(rmesh,cmesh,s111=None,cmesh_s111=1.,bin=None,ells=(),los='z'):
        legacy=None
        if record is not None:
            # 原 JIT donate 输入；独立副本避免破坏随后 exact 投影的原数据。
            rr=rmesh.clone(value=jnp.array(rmesh.value,copy=True));cc=cmesh.clone(value=jnp.array(cmesh.value,copy=True))
            ss=jax.tree_util.tree_map(lambda x:jnp.array(x,copy=True),s111)
            legacy=original(rr,cc,s111=ss,cmesh_s111=cmesh_s111,bin=bin,ells=ells,los=los)
            jax.block_until_ready(legacy)
        diagnostics={} if record is not None else None
        result=exact_b000_s113(rmesh,cmesh,s111=s111,cmesh_s111=cmesh_s111,bin=bin,ells=ells,los=los,diagnostics=diagnostics)
        if record is not None:
            record.append({'legacy_num_S113':np.asarray(legacy[0]),'exact_num_S113':np.asarray(result[0]),
                           'difference':np.asarray(result[0])-np.asarray(legacy[0]),**diagnostics})
        return result
    module._compute_sugiyama_spectrum_S113=adapter
    if matched_paint:
        def shot_adapter(*args,**kwargs):
            for name,value in {'resampler':'tsc','interlacing':3,'compensate':True}.items():
                if name in kwargs and kwargs[name]!=value:raise ValueError('conflicting matched shot paint setting: '+name)
                kwargs[name]=value
            return original_shot(*args,**kwargs)
        jaxpower.compute_fkp3_shotnoise=shot_adapter
    try:yield
    finally:
        module._compute_sugiyama_spectrum_S113=original
        jaxpower.compute_fkp3_shotnoise=original_shot
