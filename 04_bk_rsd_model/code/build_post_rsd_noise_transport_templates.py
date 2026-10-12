"""模块大纲：原44-bin有向噪声transport算子；物理b/p二次张量与细网格核验。

只是有向算子模板，不将其三个分量认作三个nuisance，也不指定实际STO幅度。
同一原B窗口/分箱/冻结alpha；逐节点checkpoint；保护旧输出；不放行主MCMC。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
from rsd_b000_contract import PART,TABLE,REFERENCE,F_GROWTH,digest,reference,selected,CUTS,save_json
from post_rsd_ir_inputs import ROOT,ALPHA,ALPHA_SHA,frozen_reconstruction
from post_rsd_poisson_projection import PoissonShiftFFTProjection
from post_rsd_noise_transport import NAMES,integrate_noise_transport

OUT=ROOT/'noise_transport_templates_v1'
BC,BR=2.25,1.75;PC,PR=1.3,.5
BN=BC+BR*np.cos((np.arange(3)+.5)*np.pi/3)
PN=PC+PR*np.cos((np.arange(3)+.5)*np.pi/3)
PROBES=((1.83641,1.287),(3.613,1.513))


def fit(nodes):
    b,p=np.meshgrid((BN-BC)/BR,(PN-PC)/PR,indexing='ij')
    matrix=np.polynomial.chebyshev.chebvander2d(b.ravel(),p.ravel(),(2,2))
    return np.linalg.solve(matrix,np.asarray(nodes).reshape(9,-1)).reshape(3,3,3,3)


def evaluate(coefficients,b,p):
    bb=np.array([1.,(b-BC)/BR,2*((b-BC)/BR)**2-1])
    pp=np.array([1.,(p-PC)/PR,2*((p-PC)/PR)**2-1])
    return np.einsum('i,j,ijaf->af',bb,pp,coefficients)


def scaled_error(predicted,direct):
    powers=np.array([1.,500.,250000.])[None,:]
    scale=np.maximum(1.,np.max(np.abs(direct)*powers,axis=1,keepdims=True))
    return float(np.max(np.abs(predicted-direct)*powers/scale))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--plan-only',action='store_true')
    ap.add_argument('--bin-index',type=int);args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('hard single-core affinity required')
    if args.detach:
        log=ROOT/'noise_transport_templates_v1.log';launch=ROOT/'noise_transport_templates_v1.launch.json'
        if log.exists() or launch.exists():raise RuntimeError('inspect prior launch; do not overwrite or duplicate')
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve())]+[s for s in sys.argv[1:] if s!='--detach']
        with log.open('xb') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(host=socket.gethostname(),pid=proc.pid,CPU_affinity=sorted(os.sched_getaffinity(0)),
            command=command,log=str(log),launched_UTC=datetime.now(timezone.utc).isoformat())
        with launch.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps(record),flush=True);return
    if (OUT/'COMPLETION_AUDIT.json').exists():raise RuntimeError('completed operator templates exist; preserve them')
    status_path=OUT/'STATUS.json' if args.bin_index is None else OUT/'workers'/f'bin{args.bin_index:03d}_STATUS.json'
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if not args.resume or old['host']!=socket.gethostname():raise RuntimeError('explicit original-host recovery required')
        try:os.kill(old['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('prior process exists; observation timeout is not termination')
    for file,status in (('CLUSTERED_NONPOISSON_RESPONSE_AUDIT.json','CORRELATED_CLUSTER_OPERATOR_RESPONSE_PASS_NOT_STO_LIKELIHOOD'),
        ('estimator_B_window/AUDIT.json','PRODUCTION_B000_WINDOW_MATCHES_EXISTING_FFT_ALL44_BINS')):
        if json.loads((ROOT/file).read_text())['status']!=status:raise ValueError('operator/window prerequisite failed')
    for name,want in json.loads((ROOT/'CLUSTERED_NONPOISSON_RESPONSE_AUDIT.json').read_text())['source_sha256'].items():
        if digest(PART/'code'/name)!=want:raise ValueError('correlated operator audit source drift')
    ref=reference();indices=selected(.20).tolist();recon=frozen_reconstruction(256)
    # 直接复用已验收no-IR合同的网格映射，避免为纯几何helper导入旧IR sph_harm。
    mesh_contract_path=ROOT/'noir_paware_templates/CONTRACT.json'
    mesh_contract=json.loads(mesh_contract_path.read_text())
    if mesh_contract['indices']!=indices or mesh_contract['immutable_sources'].get(str(REFERENCE))!=digest(REFERENCE):
        raise ValueError('original independently validated no-IR registry/grid contract required')
    mesh_audit_path=ROOT/'noir_paware_templates/INDEPENDENT_COEFFICIENT_AUDIT.json'
    mesh_audit=json.loads(mesh_audit_path.read_text())
    if mesh_audit['status']!='ALL44_NOIR_COEFFICIENT_NODE_AND_FINE_GRID_REPLAY_PASS' or mesh_audit['contract_sha256']!=digest(mesh_contract_path):
        raise ValueError('verified original no-IR grid contract hash differs')
    meshes=mesh_contract['meshes']
    for index in indices:
        levels=meshes[str(index)]
        if len(levels)!=2 or levels[1]!=levels[0]+16:raise ValueError('original dual-grid mapping changed')
    window_path=ROOT/'estimator_B_window/B000_WINDOW.npz'
    if len(indices)!=44 or digest(window_path)!=json.loads((ROOT/'estimator_B_window/AUDIT.json').read_text())['window_sha256']:
        raise ValueError('original all44 B window required')
    if args.bin_index is not None and args.bin_index not in indices:raise ValueError('bin is not in the original44 registry')
    with np.load(window_path,allow_pickle=False) as z:
        actual_counts=np.array(z['measured_pair_counts']);np.testing.assert_array_equal(z['fit_indices'],indices)
    paths=[Path(__file__),TABLE,REFERENCE,ALPHA,window_path,mesh_contract_path,mesh_audit_path,ROOT/'estimator_B_window/AUDIT.json',
        ROOT/'CLUSTERED_NONPOISSON_RESPONSE_AUDIT.json']
    paths+=[PART/'code'/name for name in ('post_rsd_noise_transport.py','test_post_rsd_noise_transport.py',
        'post_rsd_poisson_projection.py','post_rsd_fft_projection.py','post_rsd_fixedrec.py',
        'post_rsd_ir_inputs.py','rsd_b000_fft_projection.py','rsd_bonly_common.py','rsd_b000_contract.py',
        'post_rsd_noir_polynomial.py','post_rsd_noir_likelihood.py',
        'vendor/quijote_strict_triangle_fft_estimator.py')]
    contract=dict(scope='all44 fixed-catalog no-IR unit-amplitude noise transport operators, not actual STO stochastic closure',
        operator_names=list(NAMES),axes=['b1_chebyshev3','physical_p_chebyshev3','operator','G_L_Q','bin'],
        indices=indices,cuts={str(c):selected(c).tolist() for c in CUTS},edges=ref['edges'].tolist(),
        meshes=meshes,
        b1_nodes=BN.tolist(),physical_p_nodes=PN.tolist(),coordinate_centers_and_scales={'b1':[BC,BR],'p':[PC,PR]},
        exact_degree_proof='D=P*((b+f*mu2)+fNL*2*dc*(b-p)/M)^2; fixed R contains no physical b,p,fNL',
        recon_mesh=256,alpha_sha256=ALPHA_SHA,f_growth=F_GROWTH,
        probes=[list(x) for x in PROBES],scaled_tolerance=1e-9,fNL_coefficient_scaling=[1.,500.,250000.],
        zero_source_leg_value=0,closing_leg_kmax_cut=False,denominator='original full two-shell mode counts',
        formal_estimator_definition_chosen=False,STO_amplitudes_or_nuisance_mapping_validated=False,
        physical_p_calibrated=False,MCMC_released=False,IR=False,immutable_sources={str(p):digest(p) for p in paths})
    contract=json.loads(json.dumps(contract))
    OUT.mkdir(exist_ok=True);(OUT/'nodes').mkdir(exist_ok=True);(OUT/'bins').mkdir(exist_ok=True);(OUT/'workers').mkdir(exist_ok=True)
    cp=OUT/'CONTRACT.json'
    if cp.exists():
        if json.loads(cp.read_text())!=contract:raise ValueError('operator template contract drift')
    else:save_json(cp,contract)
    csha=digest(cp)
    if args.plan_only:
        print(json.dumps(dict(state='OPERATOR_TEMPLATE_PLAN_VALIDATED',bins=44,nodes=396,
            fine_probes=88,contract_sha256=csha)),flush=True);return
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_noise_transport'))
    if not tests.wasSuccessful() or tests.testsRun!=3:raise RuntimeError('independent operator/source tests failed')
    state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        started_UTC=datetime.now(timezone.utc).isoformat(),contract_sha256=csha,completed_bins=[])
    start=time.time()
    def update(stage,**values):
        state.update(stage=stage,updated_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start,**values)
        save_json(status_path,state);print(json.dumps(state),flush=True)
    def guard():
        if digest(cp)!=csha:raise ValueError('contract changed while projecting')
        for file,want in contract['immutable_sources'].items():
            if digest(file)!=want:raise ValueError('frozen noise operator source changed: '+file)
    def project(index,b,p,mesh):
        field=PoissonShiftFFTProjection(mesh,ref['edges'],ref['shell_pairs'][index],b,F_GROWTH,
            p_halo=p,reconstruction=recon)
        if field.denominator!=actual_counts[index]:raise ValueError('actual B pair count mismatch')
        design=integrate_noise_transport(field);peak=field.cache_peak_bytes
        del field;gc.collect()
        if design.shape!=(3,3) or not np.isfinite(design).all():raise ValueError('invalid operator design')
        np.testing.assert_array_equal(design[2,1:],0.)
        return design,peak
    try:
        rows=[];all_coefficients=[];update('begin')
        for index in (indices if args.bin_index is None else [args.bin_index]):
            guard();target=OUT/'bins'/f'bin{index:03d}.npz';meta_path=target.with_suffix('.json')
            if target.exists():
                meta=json.loads(meta_path.read_text())
                if meta['contract_sha256']!=csha or meta['coefficients_sha256']!=digest(target):raise ValueError('bin checkpoint drift')
                with np.load(target,allow_pickle=False) as z:coefficients=np.array(z['coefficients'])
            else:
                nodes=np.zeros((3,3,3,3));peak=0
                for ib,b in enumerate(BN):
                    for ip,p in enumerate(PN):
                        node_path=OUT/'nodes'/f'bin{index:03d}_b{ib}_p{ip}.json'
                        if node_path.exists():
                            node=json.loads(node_path.read_text())
                            if node['contract_sha256']!=csha or node['b1']!=b or node['p_halo']!=p:raise ValueError('node checkpoint drift')
                        else:
                            update('project_node',bin_index=index,b_node=ib,p_node=ip)
                            design,cache=project(index,b,p,contract['meshes'][str(index)][0])
                            node=dict(contract_sha256=csha,b1=b,p_halo=p,design=design.tolist(),cache_peak_bytes=cache)
                            save_json(node_path,node)
                        nodes[ib,ip]=node['design'];peak=max(peak,node['cache_peak_bytes'])
                coefficients=fit(nodes);probes=[]
                for b,p in PROBES:
                    update('fine_probe',bin_index=index,b1=b,p_halo=p)
                    direct,cache=project(index,b,p,contract['meshes'][str(index)][1]);peak=max(peak,cache)
                    predicted=evaluate(coefficients,b,p);error=scaled_error(predicted,direct)
                    probe=dict(b1=b,p_halo=p,mesh=contract['meshes'][str(index)][1],direct=direct.tolist(),
                        predicted=predicted.tolist(),scaled_error=error)
                    probes.append(probe)
                    if error>contract['scaled_tolerance']:
                        save_json(OUT/f'FAILED_BIN{index:03d}.json',probe);raise ValueError('original operator fine-grid gate failed')
                np.savez_compressed(target,coefficients=coefficients,node_designs=nodes,
                    contract_sha256=np.array(csha),b1_nodes=BN,physical_p_nodes=PN)
                meta=dict(index=index,contract_sha256=csha,coefficients_sha256=digest(target),probes=probes,
                    max_probe_scaled_error=max(p['scaled_error'] for p in probes),cache_peak_bytes=peak)
                save_json(meta_path,meta)
            rows.append(meta);all_coefficients.append(coefficients);state['completed_bins'].append(index)
            update('bin_complete',bin_index=index)
        if args.bin_index is not None:
            state['state']='COMPLETE';update('single_bin_complete_not_full44');return
        guard();aggregate=OUT/'OPERATOR_COEFFICIENTS.npz'
        np.savez_compressed(aggregate,coefficients=np.stack(all_coefficients,axis=-1),indices=indices,
            contract_sha256=np.array(csha),b1_nodes=BN,physical_p_nodes=PN)
        report=dict(status='ALL44_NOISE_TRANSPORT_OPERATOR_TEMPLATES_COMPLETE_NOT_STO_CLOSURE',
            created_UTC=datetime.now(timezone.utc).isoformat(),bins=44,nodes=396,fine_probes=88,rows=rows,
            contract_sha256=csha,coefficients_sha256=digest(aggregate),
            max_probe_scaled_error=max(row['max_probe_scaled_error'] for row in rows),
            maximum_cache_bytes=max(row['cache_peak_bytes'] for row in rows),
            actual_STO_noise_amplitudes_validated=False,formal_estimator_choice_validated=False,
            MCMC_released=False,IR=False)
        save_json(OUT/'COMPLETION_AUDIT.json',report);state['state']='COMPLETE';update('complete')
    except Exception:
        state['state']='FAILED';state['error']=traceback.format_exc();update('failed');raise


if __name__=='__main__':main()
