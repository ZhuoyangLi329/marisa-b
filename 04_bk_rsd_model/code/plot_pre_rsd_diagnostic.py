"""模块大纲：真实 P0/P2 和 strict B0；单盒误差标签；只写两页矢量 PDF。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PART=Path(__file__).resolve().parents[1]
OUT=PART/'outputs/pre_rsd_single_box'

def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'savefig.format':'pdf','axes.grid':True,'grid.alpha':.18})
    path=OUT/'PRE_RSD_DIAGNOSTIC.pdf'
    with PdfPages(path) as pdf:
        for name in ('fiducial','LC_p'):
            stats=summary['samples'][name];data=np.load(OUT/(name+'.npz'))
            fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
            pk=data['pk'];k=pk[:,0]
            for ell,ax,col in [(0,axes[0,0],2),(2,axes[0,1],3)]:
                ax.errorbar(k,pk[:,col]/1e4,yerr=pk[:,col+2]/1e4,fmt='o',color='#2563a6',label='pre-RSD data')
                ax.plot(k,data['pmodel'][:,col-2]/1e4,'-',color='#c75128',label='tree RSD, truth fixed')
                ax.plot(k,data['pkreal'][:,col]/1e4,'s--',color='#6e7781',alpha=.75,label='real-space catalog control')
                ax.set(xlabel=r'$k\ [h\,\mathrm{Mpc}^{-1}]$',ylabel=rf'$P_{ell}\ [10^4(h^{{-1}}\mathrm{{Mpc}})^3]$',title=f'Power multipole ell={ell}')
                ax.legend(fontsize=8)
            indices=np.arange(len(data['triangles']))
            ax=axes[1,0]
            ax.errorbar(indices,data['B_factorial']/1e9,yerr=data['B_sigma_diag']/1e9,fmt='o',color='#2563a6',label='self-term-subtracted data')
            ax.plot(indices,data['B_fitted']/1e9,'o-',color='#c75128',label='tree RSD + profiled b2, bK2')
            labels=[''.join(str(int(x)+1) for x in row) for row in data['triangles']]
            ax.set(xticks=indices,xticklabels=labels,xlabel='Strict three-leg shell labels (1-based)',
                   ylabel=r'$B_0\ [10^9(h^{-1}\mathrm{Mpc})^6]$',title='All external legs <= 0.055 h/Mpc')
            ax.legend(fontsize=8)
            ax=axes[1,1]
            residual=(data['B_factorial']-data['B_fitted'])/data['B_sigma_diag']
            ax.axhline(0,color='#6e7781',lw=1);ax.axhspan(-1,1,color='#cbd5e1',alpha=.35)
            ax.plot(indices,residual,'o',color='#2563a6')
            ax.set(xticks=indices,xticklabels=labels,xlabel='Strict three-leg shell labels (1-based)',
                ylabel='Residual / approximate diagonal error',title='Diagnostic only: no frozen RSD covariance')
            truth=stats['fNL_truth_fixed']
            fig.suptitle(f'MARISA-B pre-reconstruction RSD | {name} real000 | fNL={truth} fixed\n'
                         'z=0.5 | LOS=z | V=1 (Gpc/h)^3 | strict tree: IR, FoG and loops OFF',fontsize=14)
            fig.supxlabel(f'b1 from P0/P2={stats["b1_from_P0_P2"]:.4f}; '
                          f'B profile: b2_native={stats["b2_native_profile"]:.3f}, bK2={stats["bK2_profile"]:.3f}. '
                          'Gaussian diagonal error bars are approximate; not an fNL recovery/MCMC test.',fontsize=9)
            # 保存前核对 Matplotlib 的文字 bbox；不生成任何 PNG。
            fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
            ignored=set()
            # Matplotlib 可能保留视区外 ticks 的 Text 对象；这些不会写入实际 PDF。
            for ax in axes.flat:
                for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
                    for tick in (*axis.get_major_ticks(),*axis.get_minor_ticks()):
                        if not min(limits)<=tick.get_loc()<=max(limits):
                            ignored.update((tick.label1,tick.label2))
            for artist in fig.findobj(matplotlib.text.Text):
                if artist not in ignored and artist.get_visible() and artist.get_text():
                    bb=artist.get_window_extent(renderer)
                    assert bb.x0>=bounds.x0-1 and bb.y0>=bounds.y0-1 and bb.x1<=bounds.x1+1 and bb.y1<=bounds.y1+1,(artist.get_text(),bb)
            pdf.savefig(fig);plt.close(fig)
    print(path)

if __name__=='__main__':main()
