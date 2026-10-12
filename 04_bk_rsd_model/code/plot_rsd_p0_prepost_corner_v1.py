"""模块大纲：三种真值的pre/post RSD P0-only三参数叠加corner PDF。

只读取六组已完成的随机thin后验，不重采样或加权；噪声转为共同物理单位。
每页同一真值共用坐标范围，覆盖全部保存样本；二维68/95%分箱HPD轮廓。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter

PART=Path('/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model')
PRE=PART/'outputs/rsd_all_available_p_recalibration_20261010_v2/pre_P0_only_all_available_p'
POST=PART/'outputs/sto_marisa_realspace_p_rsd_20261011_v1/cases'
TRUTHS=('fiducial','LC_p','LC_m')
NAMES=['fNL','b1','Pshot_residual_nbar']
LABELS=[r'$f_{\rm NL}$',r'$b_1$',r'$\Delta P_{\rm shot}\ [10^3\,(\mathrm{Mpc}/h)^3]$']
COLORS={'pre':'#2875a8','post':'#d05b36'}

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(stage,truth,pre,post):
    d=pre/'cases'/(truth+'_pre_P0_all_available_p') if stage=='pre' else post/(truth+'_post_P0_realspaceP')
    m=read(d/'contract.json');s=read(d/'SUMMARY.json')
    proof=next(r for r in read(pre/'COMPLETION_AUDIT.json')['checks'] if r['name']==d.name) if stage=='pre' else read(d/'COMPLETION_AUDIT.json')
    if not proof['converged'] or sha(d/'SUMMARY.json')!=proof['summary_sha256']:
        raise ValueError('实际完成证据与摘要不一致')
    if (m['parameters']!=NAMES or not s['converged'] or m['n_covariance']!=500 or m['n_target']!=100
        or m['volume_Gpc_h3']!=1 or m['covariance_is_of_mean'] or m['covariance_multiplier']!=1
        or m['kmax_P']!=.08 or m['IR'] or m['B_likelihood']):raise ValueError('实际P0-only设置改变')
    want=1.1551560180012217 if stage=='pre' else 1.2822
    if m['p_fixed_physical']!=want:raise ValueError('p版本不符')
    with np.load(d/'POSTERIOR_THIN.npz',allow_pickle=False) as z:
        a=z['samples'].copy();names=z['parameter_names'].tolist()
    if a.shape!=(30000,3) or names!=NAMES or not np.isfinite(a).all():raise ValueError('后验样本格式改变')
    bounds=np.asarray(m['bounds'])
    if np.any(a<=bounds[:,0]) or np.any(a>=bounds[:,1]):raise ValueError('样本超出原先验')
    q=np.asarray(s['quantiles_2p5_16_50_84_97p5']).copy()
    conversion=float(m['Pshot_unit_Mpc_h3'])/1000.
    a[:,2]*=conversion;q[:,2]*=conversion
    return dict(samples=a,quantiles=q,meta=m,summary=s,
        source=str(d),source_sha256={f:sha(d/f) for f in ('POSTERIOR_THIN.npz','SUMMARY.json','contract.json')},
        physical_noise_conversion=conversion)

def levels(hist):
    order=np.sort(hist.ravel())[::-1];mass=np.cumsum(order)/order.sum()
    return [float(order[min(np.searchsorted(mass,p),len(order)-1)]) for p in (.95,.68)]

def page(truth,data,pdf):
    pooled=np.concatenate([data[s]['samples'] for s in ('pre','post')])
    low=pooled.min(axis=0);high=pooled.max(axis=0);pad=(high-low)*.035
    ranges=list(zip(low-pad,high+pad))
    for row in pooled:
        if np.any(row<low-pad) or np.any(row>high+pad):raise ValueError('坐标范围未覆盖全部样本')
    fig,axes=plt.subplots(3,3,figsize=(9.5,10.2))
    fig.subplots_adjust(left=.145,right=.965,bottom=.17,top=.775,wspace=.11,hspace=.13)
    truth_fnl=data['pre']['meta']['truth_fNL']
    pair_records=[]
    for i in range(3):
        for j in range(3):
            ax=axes[i,j]
            if j>i:ax.set_visible(False);continue
            ax.set_xlim(ranges[j]);ax.xaxis.set_major_locator(MaxNLocator(4,prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
            if i==j:
                maximum=0.
                for stage in ('pre','post'):
                    counts,edges=np.histogram(data[stage]['samples'][:,j],bins=70,range=ranges[j],density=True)
                    maximum=max(maximum,float(counts.max()))
                    ax.stairs(counts,edges,color=COLORS[stage],fill=True,alpha=.10,lw=.7)
                    ax.stairs(counts,edges,color=COLORS[stage],lw=1.35)
                    ax.axvline(data[stage]['quantiles'][2,j],color=COLORS[stage],ls='--',lw=.9)
                ax.set_ylim(0,maximum*1.55);ax.set_yticks([])
                for stage,y in [('pre',.96),('post',.86)]:
                    q=data[stage]['quantiles'];digits=3 if j==1 else (1 if j==0 else 2)
                    text=f"{stage.capitalize()}: {q[2,j]:.{digits}f}  (-{q[2,j]-q[1,j]:.{digits}f}, +{q[3,j]-q[2,j]:.{digits}f})"
                    ax.text(.03,y,text,transform=ax.transAxes,ha='left',va='top',fontsize=8.4,color=COLORS[stage],
                        bbox=dict(facecolor='white',edgecolor='none',pad=.7,alpha=.97),zorder=5)
            else:
                ax.set_ylim(ranges[i])
                for stage in ('pre','post'):
                    a=data[stage]['samples'];h,ex,ey=np.histogram2d(a[:,j],a[:,i],bins=65,range=[ranges[j],ranges[i]])
                    if int(h.sum())!=len(a):raise ValueError('分箱丢失样本')
                    h=gaussian_filter(h,1.0,mode='reflect');threshold=levels(h)
                    if not 0<threshold[0]<threshold[1]<h.max():raise ValueError('HPD轮廓不可区分')
                    centers_x=(ex[:-1]+ex[1:])/2;centers_y=(ey[:-1]+ey[1:])/2
                    ax.contourf(centers_x,centers_y,h.T,levels=[*threshold,h.max()*1.001],
                        colors=[COLORS[stage],COLORS[stage]],alpha=.105)
                    ax.contour(centers_x,centers_y,h.T,levels=threshold,colors=COLORS[stage],linewidths=[1.,1.5])
                    pair_records.append(dict(stage=stage,x_parameter=NAMES[j],y_parameter=NAMES[i],
                        mass_at_threshold=[float(h[h>=t].sum()/h.sum()) for t in threshold]))
            if j==0:ax.axvline(truth_fnl,color='#333333',ls=':',lw=1.15)
            ax.tick_params(direction='in',labelsize=9,length=3,pad=4)
            if i==2:ax.set_xlabel(LABELS[j],fontsize=11,labelpad=8)
            else:ax.tick_params(labelbottom=False)
            if j==0 and i>0:ax.set_ylabel(LABELS[i],fontsize=11,labelpad=10)
            else:ax.tick_params(labelleft=False)
    fig.suptitle(f'RSD P0-only | pre + STO+Marisa post | true fNL = {truth_fnl:+d}',
        x=.5,y=.952,fontsize=18,weight='bold',color='#183d52')
    fig.text(.5,.906,'Three-parameter posteriors from completed MCMC chains',ha='center',fontsize=11)
    fig.text(.5,.878,'mean100 data | C_single500 | V = 1 (Gpc/h)^3 | kP = 0.08 | no IR',ha='center',fontsize=10)
    fig.legend([Line2D([0],[0],color=COLORS['pre'],lw=2),Line2D([0],[0],color=COLORS['post'],lw=2),
        Line2D([0],[0],color='#333333',ls=':',lw=1.2)],
        ['Pre: p = 1.155156','STO+Marisa post: p = 1.282200','fNL truth'],
        loc='upper left',bbox_to_anchor=(.45,.765),frameon=False,fontsize=10)
    fig.text(.465,.656,'Solid contours: 68% and 95% binned HPD\nDashed lines: posterior medians\nNoise shown in common physical units\n30,000 saved draws per stage',
        va='top',fontsize=9.3,linespacing=1.6,color='#46545d')
    fig.text(.145,.081,'Diagonal annotations: full-chain median and central 68% errors. p is fixed.',fontsize=9)
    fig.text(.145,.056,'2D histograms: 65 bins, Gaussian smoothing = 1 bin; axes contain all displayed draws.',fontsize=8.5)
    fig.text(.145,.031,'Pre C: fid1000-1499; post C: fid0-499. Accepted post target/C overlap retained.',fontsize=8.5)
    fig.text(.963,.031,f'{TRUTHS.index(truth)+1} / 3',ha='right',fontsize=9)
    pdf.savefig(fig);plt.close(fig)
    return dict(truth=truth,truth_fNL=truth_fnl,common_ranges=ranges,all_draws_in_display=True,
        contours=pair_records,stages={s:dict(source=data[s]['source'],source_sha256=data[s]['source_sha256'],
            p=data[s]['meta']['p_fixed_physical'],sample_count=len(data[s]['samples']),
            noise_conversion_to_1e3_physical=data[s]['physical_noise_conversion'],
            full_chain_display_quantiles=data[s]['quantiles'].tolist(),
            plotted_draws_medians=np.median(data[s]['samples'],axis=0).tolist()) for s in ('pre','post')})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pre-root',type=Path,default=PRE)
    ap.add_argument('--post-root',type=Path,default=POST)
    ap.add_argument('--output-dir',type=Path,default=PART/'outputs/rsd_p0_prepost_corner_20261011_v1')
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/'RSD_P0_PRE_POST_OVERLAID_CORNER.pdf'
    if target.exists():
        manifest_path=args.output_dir/'PLOT_MANIFEST.json'
        if not manifest_path.exists() or sha(target)!=read(manifest_path)['output_sha256']:
            raise ValueError('已有PDF并非本次已记录的绘图产物，不覆盖')
    datasets={t:{s:load(s,t,args.pre_root,args.post_root) for s in ('pre','post')} for t in TRUTHS}
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':10,'axes.linewidth':.7})
    with PdfPages(target) as pdf:records=[page(t,datasets[t],pdf) for t in TRUTHS]
    manifest=dict(created_UTC=datetime.now(timezone.utc).isoformat(),output=str(target),output_sha256=sha(target),
        records=records,parameters=NAMES,third_parameter_display='Pshot_residual / [1e3 (Mpc/h)^3]',
        posterior_resampled=False,posterior_reweighted=False,source_chains_modified=False,HDF_read=False,
        generated_formats=['pdf'],script_sha256=sha(__file__))
    (args.output_dir/'PLOT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('THREE_PAGE_OVERLAID_CORNER_READY',target,flush=True)

if __name__=='__main__':main()
