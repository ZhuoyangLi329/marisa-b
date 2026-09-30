"""四方法教学示例的参数分离、极限、快照和命令行回归；不依赖模拟数据。"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/examples/reconstruction_current_p.py"
SPEC = importlib.util.spec_from_file_location("reconstruction_current_p_example", SCRIPT)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class ReconstructionCurrentPTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(EXAMPLE.DEFAULT_CONFIG.read_text())

    def test_snapshot_values_and_method_mapping(self):
        rows = EXAMPLE.summarize(self.config)["methods"]
        self.assertEqual(set(rows), {"pre", "standard_gra", "marisa_gra", "sto_recon", "sto_marisa"})
        self.assertEqual(rows["pre"]["bias"]["p_halo"], 1.1788659191136683)
        self.assertEqual(rows["sto_recon"]["bias"]["p_halo"], 1.3394637946618602)
        self.assertAlmostEqual(rows["pre"]["bias"]["bphi"], 2 * 1.686 * (1.93 - 1.1788659191136683))
        self.assertIsNone(rows["sto_recon"]["reconstruction"])
        self.assertEqual(rows["sto_marisa"]["kernel"], "adaptive_brec")
        self.assertEqual(rows["standard_gra"]["reconstruction"]["b_rec"], 1.9326526728794324)
        self.assertNotEqual(rows["standard_gra"]["bias"]["b1"], rows["standard_gra"]["reconstruction"]["b_rec"])

    def test_tracer_p_does_not_change_reconstruction_p_or_higher_bias_anchor(self):
        changed = copy.deepcopy(self.config)
        changed["tracers"]["halo"]["p_halo"] = 1.12
        current = EXAMPLE.summarize(self.config)["methods"]["marisa_gra"]
        legacy = EXAMPLE.summarize(changed)["methods"]["marisa_gra"]
        self.assertNotEqual(current["bias"]["bphi"], legacy["bias"]["bphi"])
        for key in ("bphidelta", "bphi2"):
            self.assertEqual(current["bias"][key], legacy["bias"][key])
        for key in ("p_rec", "bphi_rec", "denominator_at_supplied_M"):
            self.assertEqual(current["reconstruction"][key], legacy["reconstruction"][key])

    def test_reconstruction_p_changes_only_adaptive_branch(self):
        old = EXAMPLE.summarize(self.config)["methods"]
        new = EXAMPLE.summarize(self.config, p_rec_override=1.2)["methods"]
        for name in ("pre", "standard_gra", "sto_recon"):
            self.assertEqual(old[name], new[name])
        for name in ("marisa_gra", "sto_marisa"):
            self.assertEqual(old[name]["bias"], new[name]["bias"])
            self.assertNotEqual(old[name]["reconstruction"]["denominator_at_supplied_M"], new[name]["reconstruction"]["denominator_at_supplied_M"])

    def test_fnl_and_fnl_rec_are_independent(self):
        a = EXAMPLE.summarize(self.config, fnl=0, fnl_rec=100)["methods"]["marisa_gra"]
        b = EXAMPLE.summarize(self.config, fnl=100, fnl_rec=100)["methods"]["marisa_gra"]
        c = EXAMPLE.summarize(self.config, fnl=100, fnl_rec=0)["methods"]["marisa_gra"]
        self.assertNotEqual(a["Z1_at_supplied_M"], b["Z1_at_supplied_M"])
        self.assertEqual(a["reconstruction"], b["reconstruction"])
        self.assertEqual(b["Z1_at_supplied_M"], c["Z1_at_supplied_M"])
        self.assertEqual(c["reconstruction"]["denominator_at_supplied_M"], c["reconstruction"]["b_rec"])

    def test_native_adaptive_normalizations(self):
        row = EXAMPLE.summarize(self.config)["methods"]["marisa_gra"]
        rec = row["reconstruction"]; b1 = row["bias"]["b1"]
        ratio = rec["bphi_rec"] / rec["b_rec"]
        coefficients = rec["adaptive_raw_basis_prefactors"]
        self.assertAlmostEqual(coefficients["gaussian_linear"], b1**4 * ratio)
        self.assertAlmostEqual(coefficients["gaussian_quadratic"], b1**4 * ratio**2)
        self.assertAlmostEqual(coefficients["png_linear_cross"], b1**3 * row["bias"]["bphi"] * ratio)

    def test_invalid_denominator_and_nonfinite_values_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(transfer_m=value), self.assertRaises(ValueError):
                EXAMPLE.summarize(self.config, transfer_m=value)
        with self.assertRaises(ValueError):
            EXAMPLE.reconstruction_denominator(2, 1, -100, 1)
        with self.assertRaises(ValueError):
            EXAMPLE.summarize(self.config, fnl=float("nan"))

    def test_config_is_not_mutated(self):
        before = copy.deepcopy(self.config)
        EXAMPLE.summarize(self.config, p_rec_override=1.2)
        self.assertEqual(self.config, before)

    def test_cli_runs_without_package_installation(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--fnl-rec", "0"],
                                text=True, capture_output=True, check=True, cwd=ROOT)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "illustration_only_not_a_fit_or_IR_prediction")
        self.assertEqual(payload["fNL_rec_example"], 0)
        self.assertEqual(len(payload["methods"]), 5)


if __name__ == "__main__":
    unittest.main()
