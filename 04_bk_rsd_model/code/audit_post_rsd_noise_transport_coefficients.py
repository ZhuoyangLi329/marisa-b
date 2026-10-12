"""模块大纲：独立回放44-bin噪声transport系数；396节点/88探针/NN结构。

不调用生产fit/evaluate；只认证数值表示及原geometry，不指定STO幅度或放行MCMC。
另有标量闭合恒等式：K_same=-sum_i D_i R_i，K_NN=-sum_i R_i，独立于有向vector分解。
实际job仍运行/排队、数据不全或观察不确定时不写PASS，不自动重启任何job。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

PART=Path(__file__).resolve().parents[1]
OUT=PART/'outputs/sto_marisa_post_rsd_mean100_v1/noise_transport_templates_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_job(text,job_id):
    matches=[]
    for line in text.splitlines():
        cells=line.strip().split('|')
        if len(cells)>=3 and cells[0]==str(job_id):matches.append(cells)
    if len(matches)!=1:raise ValueError('one exact accounting job record required; absence is not termination')
    cells=matches[0]
    return dict(job_id=str(job_id),state=cells[1].split()[0].rstrip('+'),exit_code=cells[2])


def evaluate_independent(c,b,p,coordinates):
    c=np.asarray(c,dtype=float)
    if c.shape!=(3,3,3,3) or not np.isfinite(c).all():raise ValueError('finite per-bin [b,p,operator,fNL] tensor required')
    bc,br=coordinates['b1'];pc,pr=coordinates['p'];x=(b-bc)/br;y=(p-pc)/pr
    if not all(np.isfinite(a) for a in (x,y)):raise ValueError('finite diagnostic coordinates required')
    return np.einsum('i,j,ijaf->af',[1,x,2*x*x-1],[1,y,2*y*y-1],c)


def error(predicted,direct,powers):
    predicted=np.asarray(predicted,dtype=float);direct=np.asarray(direct,dtype=float)
    if predicted.shape!=(3,3) or direct.shape!=(3,3) or not np.isfinite(direct).all():
        raise ValueError('finite three-operator G/L/Q design required')
    scale=np.maximum(1.,np.max(np.abs(direct)*powers,axis=1,keepdims=True))
    return float(np.max(np.abs(predicted-direct)*powers/scale))


def replay_bin(c,nodes,probes,contract):
    nodes=np.asarray(nodes,dtype=float)
    if nodes.shape!=(3,3,3,3) or not np.isfinite(nodes).all():raise ValueError('all nine finite nodes required')
    powers=np.asarray(contract['fNL_coefficient_scaling'])[None,:];coords=contract['coordinate_centers_and_scales']
    bn,pn=contract['b1_nodes'],contract['physical_p_nodes']
    if len(bn)!=3 or len(pn)!=3 or len(probes)!=2:raise ValueError('original node/probe counts required')
    node_errors=[];probe_errors=[]
    for ib,b in enumerate(bn):
        for ip,p in enumerate(pn):
            node_errors.append(error(evaluate_independent(c,b,p,coords),nodes[ib,ip],powers))
    for record,(b,p) in zip(probes,contract['probes']):
        if record['b1']!=b or record['p_halo']!=p:raise ValueError('probe coordinate drift')
        predicted=evaluate_independent(c,b,p,coords);direct=np.asarray(record['direct'])
        np.testing.assert_allclose(predicted,record['predicted'],rtol=5e-13,atol=2e-9)
        probe_errors.append(error(predicted,direct,powers))
    # NN是固定response的几何项，不可带任何物理fNL；节点也必须与b/p无关。
    np.testing.assert_array_equal(nodes[:,:,2,1:],0.)
    np.testing.assert_array_equal(np.asarray(c)[:,:,2,1:],0.)
    np.testing.assert_array_equal(nodes[:,:,2,0],np.full((3,3),nodes[0,0,2,0]))
    # 高阶b/p的NN系数仅能有浮点solve舍入；保留原范数，不截断系数。
    nn=np.array(c[:,:,2,0],copy=True);nn[0,0]=0.
    structural=float(np.max(np.abs(nn))/max(1.,abs(c[0,0,2,0])))
    result=dict(max_node_scaled_error=max(node_errors),max_fine_probe_scaled_error=max(probe_errors),
        NN_nonconstant_roundoff_scaled=structural)
    if max(result['max_node_scaled_error'],result['max_fine_probe_scaled_error'])>contract['scaled_tolerance']:
        raise ValueError('original combined fNL-scaled gate failed')
    if structural>1e-13:raise ValueError('NN acquired physical b/p dependence')
    return result


def scalar_oracle(fields):
    """用闭合波矢的和恒等式，不使用生产的三轴vector contractions。"""
    from rsd_b000_fft_projection import Field,ONE
    same=np.array([-fields.integral(Field('DR'),d,ONE,0,ONE,0) for d in range(3)])
    nn=np.zeros_like(same);nn[0]=-fields.integral(Field('SR'),0,ONE,0,ONE,0)
    return np.stack((same,nn))


def integer_count_oracle(fields):
    """独立整数闭合计数；不调用production transform/integral或field.values。

输入两腿排除源零模，closing零模另删；分母仍是原full-shell Ni*Nj。
只对整数pair multiplicity做round-to-nearest并严格检查FFT舍入残差，
不截断任何科学系数、谱或协方差。D与R逐格点直接按物理定义求值。
"""
    if not hasattr(fields,'masks'):raise TypeError('lattice field required for integer count oracle')
    v=np.asarray(fields.v);k2=np.sum(v*v,axis=-1);active=k2>0
    i,j=fields.slots[:2];full_i=np.asarray(fields.masks[i]);full_j=np.asarray(fields.masks[j])
    n=fields.n;neg=(-np.arange(n))%n
    for mask in (full_i,full_j):
        np.testing.assert_array_equal(mask,mask[np.ix_(neg,neg,neg)])
    a=(full_i&active).astype(float);b=(full_j&active).astype(float)
    denominator=int(full_i.sum())*int(full_j.sum())
    if denominator!=fields.denominator or denominator<=0:raise ValueError('original full shell denominator required')
    raw=np.fft.ifftn(np.fft.fftn(a)*np.fft.fftn(b)).real
    pairs=np.rint(raw);roundoff=float(np.max(np.abs(raw-pairs)))
    if roundoff>1e-5 or np.min(pairs)<0:raise ValueError('integer closure count roundoff gate failed')
    if int(pairs.sum())!=int(a.sum())*int(b.sum()):raise ValueError('not all input pairs counted')
    pairs[~active]=0.
    # 从i/j腿出发，合法partner数要减去q=-k导致closing为0的那个格点。
    multiplicity=a*(b.sum()-b)+b*(a.sum()-a)+pairs
    response=np.zeros_like(k2)
    if fields.reconstruction is not None:
        response[active]=fields.reconstruction.response(v[active],np.asarray(fields.m)[active])
    inv=np.divide(1.,k2,out=np.zeros_like(k2),where=active)
    z0=fields.b1+fields.f*v[...,2]**2*inv
    zphi=2*1.686*(fields.b1-fields.p_halo)/np.asarray(fields.m)
    power=np.asarray(fields.p);design=np.stack((power*z0*z0,2*power*z0*zphi,power*zphi*zphi))
    same=-np.sum(design*(response*multiplicity)[None,...],axis=(1,2,3))/denominator
    nn=np.zeros(3);nn[0]=-np.sum(response*multiplicity)/denominator
    return np.stack((same,nn)),dict(full_denominator=denominator,nonzero_input_pairs=int(a.sum())*int(b.sum()),
        nonzero_closing_pairs=int(pairs.sum()),maximum_integer_FFT_roundoff=roundoff)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--job-id',type=int,default=59594709)
    ap.add_argument('--check-readiness',action='store_true');args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1:raise ValueError('one logical CPU per independent replay')
    observed=subprocess.run(['sacct','-X','-n','-P','-j',str(args.job_id),'-o','JobIDRaw,State,ExitCode'],
        check=True,text=True,stdout=subprocess.PIPE,timeout=20)
    job=actual_job(observed.stdout,args.job_id)
    if job['state']!='COMPLETED':
        print(json.dumps(dict(status='WAITING_FOR_SPECIFIC_OPERATOR_JOB_NOT_REPLAY_PASS',actual_job=job,
            created_UTC=datetime.now(timezone.utc).isoformat())),flush=True);return
    if job['exit_code']!='0:0':raise ValueError('successful actual batch exit required')
    target=OUT/'INDEPENDENT_COEFFICIENT_AUDIT.json'
    if target.exists():raise RuntimeError('preserve prior independent replay; verify existing hashes instead')
    cp=OUT/'CONTRACT.json';contract=json.loads(cp.read_text());csha=sha(cp)
    status=json.loads((OUT/'COMPUTE_STATUS.json').read_text())
    report=json.loads((OUT/'COMPLETION_AUDIT.json').read_text())
    indices=contract['indices']
    if (status['state']!='COMPLETE' or status['active_workers'] or status['job_id']!=str(args.job_id)
        or sorted(status['completed_bins'])!=sorted(indices) or status['contract_sha256']!=csha
        or report['status']!='ALL44_NOISE_TRANSPORT_OPERATOR_TEMPLATES_COMPLETE_NOT_STO_CLOSURE'
        or report['bins']!=44 or report['nodes']!=396 or report['fine_probes']!=88
        or len(report['rows'])!=44 or len(indices)!=44 or len(set(indices))!=44
        or report['contract_sha256']!=csha):raise ValueError('all44 completed compute/contract identity required')
    for path,want in dict(contract['immutable_sources'],**status['queue_sources']).items():
        if sha(path)!=want:raise ValueError('frozen scientific/compute source drift: '+path)
    aggregate=OUT/'OPERATOR_COEFFICIENTS.npz'
    if sha(aggregate)!=report['coefficients_sha256']:raise ValueError('aggregate bytes drift')
    with np.load(aggregate,allow_pickle=False) as z:
        coefficients=np.array(z['coefficients']);np.testing.assert_array_equal(z['indices'],indices)
        np.testing.assert_array_equal(z['b1_nodes'],contract['b1_nodes'])
        np.testing.assert_array_equal(z['physical_p_nodes'],contract['physical_p_nodes'])
        if str(z['contract_sha256'])!=csha:raise ValueError('aggregate contract marker differs')
    if coefficients.shape!=(3,3,3,3,44) or not np.isfinite(coefficients).all():raise ValueError('full operator tensor required')
    report_rows={r['index']:r for r in report['rows']}
    if len(report_rows)!=44:raise ValueError('duplicate report bin indices')
    rows=[]
    for position,index in enumerate(indices):
        file=OUT/'bins'/f'bin{index:03d}.npz';meta=json.loads(file.with_suffix('.json').read_text())
        if meta!=report_rows[index] or meta['coefficients_sha256']!=sha(file) or meta['contract_sha256']!=csha:
            raise ValueError('original bin metadata/bytes drift')
        with np.load(file,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['coefficients'],coefficients[...,position]);nodes=np.array(z['node_designs'])
            if str(z['contract_sha256'])!=csha:raise ValueError('individual bin marker differs')
        for ib,b in enumerate(contract['b1_nodes']):
            for ip,p in enumerate(contract['physical_p_nodes']):
                node=json.loads((OUT/'nodes'/f'bin{index:03d}_b{ib}_p{ip}.json').read_text())
                if node['contract_sha256']!=csha or node['b1']!=b or node['p_halo']!=p:raise ValueError('node identity differs')
                np.testing.assert_array_equal(node['design'],nodes[ib,ip])
        if any(p['mesh']!=contract['meshes'][str(index)][1] for p in meta['probes']):raise ValueError('wrong fine grid')
        row=replay_bin(coefficients[...,position],nodes,meta['probes'],contract);row['bin_index']=index;rows.append(row)
    result=dict(status='ALL44_NOISE_OPERATOR_NODE_AND_FINE_GRID_REPLAY_PASS_NOT_STO_CLOSURE',
        created_UTC=datetime.now(timezone.utc).isoformat(),actual_job=job,bins=44,nodes_replayed=396,fine_probes_replayed=88,
        rows=rows,contract_sha256=csha,coefficients_sha256=sha(aggregate),
        max_node_scaled_error=max(r['max_node_scaled_error'] for r in rows),
        max_fine_probe_scaled_error=max(r['max_fine_probe_scaled_error'] for r in rows),
        original_scaled_tolerance=contract['scaled_tolerance'],source_sha256=sha(__file__),
        physical_p_calibrated=False,actual_STO_noise_amplitudes_validated=False,
        formal_estimator_choice_validated=False,scalar_grid_oracle_all44_executed=False,MCMC_released=False,IR=False)
    if not args.check_readiness:
        with target.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
