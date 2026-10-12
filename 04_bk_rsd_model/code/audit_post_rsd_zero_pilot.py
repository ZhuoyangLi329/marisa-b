"""模块大纲：真实 fid000 的零腿 self 诊断；核对旧 matched 数值完全不变。"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
from rsd_b000_contract import PART,CUTS,selected,digest


def main():
    root=PART/'outputs/sto_marisa_post_rsd_mean100_v1';parent=root/'pilots'
    a=parent/'fiducial_real000_rec256_meas256_ran30_selfmatched';b=parent/(a.name+'_zeroaudit')
    done=json.loads((b/'DONE.json').read_text());prior=json.loads((a/'DONE.json').read_text())
    if done['measurement_sha256']!=digest(b/'MEASUREMENTS.npz'):raise ValueError('new pilot changed')
    if prior['measurement_sha256']!=digest(a/'MEASUREMENTS.npz'):raise ValueError('old pilot changed')
    if done['post_halos_sha256']!=prior['post_halos_sha256']:raise ValueError('post positions changed')
    with np.load(a/'MEASUREMENTS.npz') as aa,np.load(b/'MEASUREMENTS.npz') as bb:
        exact={k:bool(np.array_equal(aa[k],bb[k])) for k in aa.files}
        if not all(exact.values()):raise ValueError('diagnostic extension changed a preexisting output array')
        zero=np.asarray(bb['zero_leg_shot_contribution']);base=np.asarray(bb['B_post']);cf=np.asarray(bb['B_post_nonzero_only_counterfactual'])
        np.testing.assert_array_equal(cf,base+zero)
        rows=[]
        for cut in CUTS:
            ix=selected(cut)
            rows.append({'kmax_B':cut,'n_B':len(ix),'zero_counterfactual_L2_over_B_L2':float(np.linalg.norm(zero[ix])/np.linalg.norm(base[ix])),
                         'max_abs_zero_contribution':float(np.max(np.abs(zero[ix]))),
                         'maximum_zero_pair_fraction':float(np.max(bb['zero_leg_pair_fraction'][ix])),
                         'changed_bin_indices':ix[np.flatnonzero(zero[ix])].tolist()})
        largest=int(np.argmax(np.abs(zero[selected(.20)])));largest=int(selected(.20)[largest])
        extreme={'bin_index':largest,'k':bb['B_post_k'][largest].tolist(),'B_post':float(base[largest]),
                 'zero_shot_contribution':float(zero[largest]),'counterfactual':float(cf[largest])}
    out={'status':'REAL_PILOT_ZERO_DIAGNOSTIC_RECORDED_NOT_ESTIMATOR_RELEASE','created_UTC':datetime.now(timezone.utc).isoformat(),
         'new_pilot':str(b),'old_matched_pilot':str(a),'all_original_arrays_bitwise_equal':exact,
         'post_halos_identical':True,'cuts':rows,'largest_kmax020_entry':extreme,
         'old_and_primary_B_changed':False,'binning_and_denominator_changed':False,
         'whitened_error':None,'p_or_fNL_fit_run':False,'production_estimator_choice_finalized':False,
         'entry_sha256':digest(Path(__file__)),'new_measurement_sha256':done['measurement_sha256']}
    target=root/'ZERO_ATOM_REAL_PILOT_AUDIT.json';target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
