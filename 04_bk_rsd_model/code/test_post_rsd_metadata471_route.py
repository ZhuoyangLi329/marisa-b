"""模块大纲：471完整重放证明、只路由元数据、不替代原权重与原FoF。"""
import copy
from pathlib import Path
import unittest
import audit_sto_post_fid500_routed471 as route


class MetadataRouteTest(unittest.TestCase):
    def report(self):
        return dict(status='EXISTING_STO_PAYLOAD_REPRODUCED_FROM_SAME_RECIPE_AND_EXISTING_DM',
            cosmology='fiducial', realization=471, all_arrays_exactly_equal={k:True for k in route.ARRAYS},
            original_payload_sha256='same', regenerated_payload_sha256='same',
            original_inputs_modified=False, new_DM_downloaded_or_Nbody_generated=False)

    def test_full_proof_and_identity_required(self):
        route.validate_replay_report(self.report())
        for key, value in (('realization', 470), ('cosmology', 'LC_p'),
                           ('regenerated_payload_sha256', 'other'), ('original_inputs_modified', True)):
            report=self.report();report[key]=value
            with self.assertRaises(ValueError):route.validate_replay_report(report)

    def test_partial_array_proof_never_accepted(self):
        for field in route.ARRAYS:
            report=self.report();report['all_arrays_exactly_equal'].pop(field)
            with self.assertRaises(ValueError):route.validate_replay_report(report)
            report=self.report();report['all_arrays_exactly_equal'][field]=False
            with self.assertRaises(ValueError):route.validate_replay_report(report)

    def test_routing_is_exact_path_and_metadata_only(self):
        old=Path('/original/fiducial/real471/diagnostics.json');fresh=Path('/replayed/diagnostics.json')
        calls=[]
        def read(path,**kwargs):calls.append((Path(path),kwargs));return b'actual'
        for path in (old, old.with_name('post_sto_halo.npz'), Path('/raw/groups_003/group_tab_003.0'),
                     Path('/original/fiducial/real470/diagnostics.json')):
            self.assertEqual(route.route_read(read,old,fresh,path,max_bytes=8000),b'actual')
            self.assertEqual(calls[-1],(fresh if path==old else path,dict(max_bytes=8000)))


if __name__=='__main__':unittest.main()
