"""
A test where we can modify different parts of the modeling_options to ensure
they work with a top-to-bottom WISDEM run.
"""

import os
import unittest

from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
    "examples",
    "02_reference_turbines",
)
fname_analysis_options = os.path.join(test_dir, "analysis_options.yaml")
this_dir = os.path.dirname(os.path.realpath(__file__))
fname_modeling_options = os.path.join(this_dir, "modified_modeling_options.yaml")


class TestRegression(unittest.TestCase):
    def test15MW(self):
        ## IEA 15MW
        fname_wt_input = os.path.join(test_dir, "IEA-15-240-RWT.yaml")
        wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 68233.0936092383, -1) # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 78.17441255915259, 1)
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 75.80451104905833, 0)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 25.981457969698813, 1)
        self.assertAlmostEqual(wt_opt["towerse.z_param"][-1], 144.386, 3)


if __name__ == "__main__":
    unittest.main()
