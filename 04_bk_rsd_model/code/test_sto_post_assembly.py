"""模块大纲：实际已批准样本选择；拒绝平均cov和少盒；不齐时不创建产物。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import assemble_sto_post_measurements as assembler


class AssemblyTest(unittest.TestCase):
    def selection(self):
        return json.loads((assembler.ROOT/'ENSEMBLE_SELECTION.json').read_text())

    def test_actual_confirmed_selection(self):
        chosen=assembler.selection_from_record(self.selection())
        self.assertEqual(len(chosen.measurement_keys),700)
        self.assertEqual(chosen.covariance_ids,tuple(range(500)))
        self.assertEqual(chosen.overlap,tuple(range(100)))
        self.assertFalse(chosen.metadata()['covariance_is_of_mean'])

    def test_no_covariance_average_or_smaller_covariance(self):
        for key,value in (('covariance_multiplier',.01),('volume_Gpc_h3',100.),
                          ('n_covariance',20),('covariance_is_of_mean',True)):
            record=self.selection();record[key]=value
            with self.assertRaises(ValueError):assembler.selection_from_record(record)

    def test_missing_rows_never_ready(self):
        selection=assembler.selection_from_record(self.selection())
        with TemporaryDirectory(prefix='sto_assembly_') as temporary:
            root=Path(temporary)
            with patch.object(assembler,'ROOT',root):
                ready=assembler.readiness(selection)
                self.assertFalse(ready['artifact_ready']);self.assertEqual(ready['required'],700)
                self.assertEqual(ready['complete_measurements'],0)
                self.assertFalse((root/'ensemble_measurements').exists())

    def test_compute_completion_requires_completed_migration(self):
        selection=assembler.selection_from_record(self.selection())
        with TemporaryDirectory(prefix='sto_assembly_') as temporary:
            root=Path(temporary);pilot=root/'fixture';pilot.mkdir();(pilot/'DONE.json').write_text('{}')
            queue=root/'compute32_measurement_queue';queue.mkdir()
            (queue/'STATUS.json').write_text(json.dumps(dict(state='COMPLETE',completed_count=700,
                workers=[],host='nid000',pid=123)))
            migration=root/'compute32_migration';migration.mkdir()
            with patch.object(assembler,'ROOT',root),patch.object(assembler,'directory',lambda key:pilot):
                self.assertFalse(assembler.readiness(selection)['artifact_ready'])
                (migration/'MIGRATION.json').write_text(json.dumps(dict(
                    state='LOGIN_MEASUREMENT_PARENTS_TERMINATED_WORKERS_COMPLETE',
                    original_parent_processes_confirmed_absent=True)))
                ready=assembler.readiness(selection)
                self.assertTrue(ready['artifact_ready']);self.assertTrue(ready['via_compute_migration'])
                self.assertFalse(ready['process_termination_verified_by_readiness'])


if __name__=='__main__':unittest.main()
