"""模块大纲：只读复用已冻结 EH-ratio smooth split；保留原 RSD 物理 P/M。

只消费第三部分的共享谱分解，不复用其实空间投影、阻尼或拟合结果。
重构 alpha_rec 与物理 1/M 明确分离；不允许静默替换谱或外推。
"""
import json
import numpy as np
from scipy.interpolate import CubicSpline
from rsd_b000_contract import PART,TABLE,TABLE_HASH,load_spectrum,digest
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_displacement import GaussianLongDisplacement

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
SPLIT=PART.parent/'03_bk_model/halo/outputs/local_png_tree_ir_20260930/spectrum_split_s0.25.npz'
SPLIT_SHA='df105acc326fd4cbe365a06902151063af8a5bfd757396ed7bfab8717a38922f'
SPLIT_META=SPLIT.parent/'tree_ir_templates_r6_m120_s0.25_rbao110.0.json'
ALPHA=ROOT/'RECON_ALPHA_COMPARISON.npz'
ALPHA_SHA='d1a846fa8ac71fe5eee626d0020f591d0cdcfdb82a53da97250799d858fcff83'


class FrozenIRSpectra:
    def __init__(self):
        if digest(TABLE)!=TABLE_HASH or digest(SPLIT)!=SPLIT_SHA:raise ValueError('frozen physical/smooth spectrum changed')
        meta=json.loads(SPLIT_META.read_text())
        if meta['inputs']['png_table']['sha256']!=TABLE_HASH or meta['redshift']!=.5:raise ValueError('smooth split provenance differs')
        with np.load(SPLIT) as z:
            self.k=np.array(z['k']);self.p_snapshot=np.array(z['plin']);self.m_snapshot=np.array(z['transfer']);self.pnw=np.array(z['pnw'])
        if not np.isfinite(self.pnw).all() or np.any(self.pnw<=0):raise ValueError('invalid frozen no-wiggle spectrum')
        self.physical=load_spectrum();self._nw=CubicSpline(np.log(self.k),self.pnw,extrapolate=False)
        p,m=self.physical(self.k)
        self.snapshot_relative_errors={'power':float(np.max(np.abs(self.p_snapshot/p-1))),
                                       'transfer':float(np.max(np.abs(self.m_snapshot/m-1)))}
        if max(self.snapshot_relative_errors.values())>3e-12:raise ValueError('physical source and smooth split are not matched')
        self.provenance={'physical_table':str(TABLE),'physical_sha256':TABLE_HASH,'no_wiggle_split':str(SPLIT),
                         'no_wiggle_sha256':SPLIT_SHA,'no_wiggle_metadata_sha256':digest(SPLIT_META),
                         'recipe':meta['split'],'smoothing_lambda_ln_k':meta['smoothing_lambda_ln_k'],
                         'redshift':.5,'physical_P_M_unchanged':True}

    def nowiggle(self,k):
        k=np.asarray(k,dtype=float);active=k>0
        if not np.isfinite(k).all() or np.any(k<0):raise ValueError('invalid wavenumber')
        if np.any(k[active]<self.k[0]) or np.any(k[active]>self.k[-1]):raise ValueError('no-wiggle extrapolation forbidden')
        out=np.zeros_like(k);out[active]=self._nw(np.log(k[active]))
        return out

    def long_modes(self,nq=513,*,n_mu=32,n_phi=64,qmin=2*np.pi/1000,k_ir=.15):
        q=np.geomspace(qmin,k_ir,nq);p,m=self.physical(q);pn=self.nowiggle(q)
        return GaussianLongDisplacement(q,pn,m*np.sqrt(pn/p),n_mu,n_phi)


def frozen_reconstruction(meshsize=256):
    if digest(ALPHA)!=ALPHA_SHA:raise ValueError('frozen reconstruction alpha changed')
    with np.load(ALPHA) as z:ak=np.array(z['alpha_k_2048']);av=np.array(z['alpha_values_2048'])
    return FixedReconstruction(10.,2.2280546353635335,1.2822,100.,1000./meshsize,4,alpha_table=(ak,av))
