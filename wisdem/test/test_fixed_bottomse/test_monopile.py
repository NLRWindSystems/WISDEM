import copy
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt

import wisdem.fixed_bottomse.monopile as mon
from wisdem.commonse import gravity as g

npts = 100
myones = np.ones((npts,))
RIGID = mon.RIGID


class TestPreDiscretization(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        inputs["monopile_height"] = 60.0
        inputs["monopile_foundation_height"] = -50.0
        inputs["tower_foundation_height"] = 10.0
        inputs["tower_base_diameter"] = 10.0
        inputs["monopile_top_diameter"] = 10.0
        inputs["water_depth"] = 30.0

        mydis = mon.PreDiscretization()
        mydis.compute(inputs, outputs)
        self.assertEqual(outputs["transition_piece_height"], 10.0)
        self.assertEqual(outputs["z_start"], -50.0)
        npt.assert_array_equal(outputs["joint1"], np.array([0.0, 0.0, -50.0]))
        npt.assert_array_equal(outputs["joint2"], np.array([0.0, 0.0, 10.0]))
        self.assertEqual(outputs["suctionpile_depth"], 20.0)
        self.assertEqual(outputs["s_const1"], 20.0 / 60.0)
        self.assertEqual(outputs["bending_height"], 40.0)
        self.assertEqual(outputs["constr_diam_consistency"], 1.0)

    def testBadHeight(self):
        inputs = {}
        outputs = {}
        inputs["monopile_height"] = 70.0
        inputs["monopile_foundation_height"] = -50.0
        inputs["tower_foundation_height"] = 10.0
        inputs["tower_base_diameter"] = 5.0
        inputs["monopile_top_diameter"] = 10.0
        inputs["water_depth"] = 30.0

        mydis = mon.PreDiscretization()
        mydis.compute(inputs, outputs)
        self.assertEqual(outputs["transition_piece_height"], 10.0)
        self.assertEqual(outputs["z_start"], -60.0)
        npt.assert_array_equal(outputs["joint1"], np.array([0.0, 0.0, -60.0]))
        npt.assert_array_equal(outputs["joint2"], np.array([0.0, 0.0, 10.0]))
        self.assertEqual(outputs["suctionpile_depth"], 30.0)
        self.assertEqual(outputs["s_const1"], 30.0 / 70.0)
        self.assertEqual(outputs["bending_height"], 40.0)
        self.assertEqual(outputs["constr_diam_consistency"], 0.5)


class TestMass(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        inputs["cylinder_mass"] = 10.0
        inputs["cylinder_cost"] = 100.0
        inputs["cylinder_z_cg"] = 50.0
        inputs["cylinder_I_base"] = 20 * np.ones(6)
        inputs["transition_piece_mass"] = 2.0
        inputs["transition_piece_cost"] = 5.0
        inputs["gravity_foundation_mass"] = 3.0
        inputs["tower_mass"] = 30.0
        inputs["tower_cost"] = 300.0
        inputs["z_full"] = np.linspace(-30, 10, 5)
        inputs["outer_diameter_full"] = 10.0 * np.ones(5)

        mydis = mon.MonopileMass(npts=5)
        mydis.compute(inputs, outputs)
        self.assertEqual(outputs["monopile_mass"], 10.0 + 2 + 3)
        self.assertEqual(outputs["monopile_cost"], 100 + 5)
        self.assertEqual(outputs["monopile_z_cg"], (10 * 50 + 2 * 10 - 3 * 30) / 15.0)
        self.assertEqual(outputs["structural_mass"], 10.0 + 2 + 3 + 30)
        self.assertEqual(outputs["structural_cost"], 100.0 + 5 + 300)
        I_root = np.r_[0.5 * 25, 0.5 * 25, 25, np.zeros(3)]
        npt.assert_equal(outputs["transition_piece_I"], 2 * I_root)
        npt.assert_equal(outputs["gravity_foundation_I"], 0.5 * 3 * I_root)
        npt.assert_equal(
            outputs["monopile_I_base"],
            20 * np.ones(6) + 0.5 * 3 * I_root + 2 * I_root + 2 * 40**2 * np.r_[1, 1, np.zeros(4)],
        )


class TestMonopileSE(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        # Store analysis options
        self.modeling_options = {}
        self.modeling_options["materials"] = {}
        self.modeling_options["materials"]["n_mat"] = 1

        self.modeling_options["WISDEM"] = {}
        self.modeling_options["WISDEM"]["n_dlc"] = 1
        self.modeling_options["WISDEM"]["FixedBottomSE"] = {}
        self.modeling_options["WISDEM"]["FixedBottomSE"]["buckling_method"] = "eurocode"
        self.modeling_options["WISDEM"]["FixedBottomSE"]["buckling_length"] = 30.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = 0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = 0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = 3
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_refine"] = 3
        self.modeling_options["WISDEM"]["FixedBottomSE"]["wind"] = "PowerWind"

        self.modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"] = False
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gravity_foundation"] = False

        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_fatigue"] = 1.0

        # Simplified the options available to the user
        self.modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
        self.modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = True
        self.modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = True
        self.modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["modal_method"] = 1
        self.modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["tol"] = 1e-9

        self.modeling_options["flags"] = {}
        self.modeling_options["flags"]["tower"] = False
        self.modeling_options["WISDEM"]["TowerSE"] = {}
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 0

    def testProblemFixedPile(self):
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = 3
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = 1
        self.modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"] = True
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gravity_foundation"] = False

        prob = om.Problem(reports=False)
        prob.model = mon.MonopileSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["water_depth"] = 30.0
        prob["transition_piece_mass"] = 1e2
        prob["transition_piece_cost"] = 1e3
        prob["gravity_foundation_mass"] = 0.0  # 1e4

        hval = np.array([15.0, 30.0])
        prob["monopile_s"] = np.cumsum(np.r_[0, hval]) / hval.sum()
        prob["monopile_foundation_height"] = -45.0
        prob["monopile_height"] = hval.sum()
        prob["monopile_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["monopile_top_diameter"] = 10.0
        prob["monopile_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["outfitting_factor_in"] = 1.0
        prob["monopile_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["sigma_ult_mat"] = 1e8 * np.ones((1, 3))
        prob["wohler_exp_mat"] = 1e1
        prob["wohler_A_mat"] = 1e1

        prob["yaw"] = 0.0
        prob["G_soil"] = 1e7
        prob["nu_soil"] = 0.5
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["ca_usr"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.3351e-3
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["Hsig_wave"] = 0.0
        prob["Tsig_wave"] = 1e3
        prob["env.Uref"] = 15.0
        prob["turbine_F"] = 1e3 * np.arange(2, 5)
        prob["turbine_M"] = 1e4 * np.arange(2, 5)
        prob["turbine_mass"] = 1e5
        prob["turbine_cg"] = 1e1 * np.ones(3)
        prob["turbine_I"] = 2e1 * np.ones(6)
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4 * (5.0**2 - 4.9**2) * np.pi
        npt.assert_equal(prob["z_start"], -45.0)
        npt.assert_equal(prob["transition_piece_height"], 0.0)
        npt.assert_equal(prob["suctionpile_depth"], 15.0)
        npt.assert_equal(prob["z_param"], np.array([-45.0, -30.0, 0.0]))
        npt.assert_almost_equal(prob["monopile_mass"], mass_dens * 45.0 + 1e2)

        npt.assert_equal(prob.model.perf.monopile.frame.rnode, 1 + np.arange(4, dtype=np.int_))
        npt.assert_array_less(prob.model.perf.monopile.frame.rKx, RIGID)
        npt.assert_array_less(prob.model.perf.monopile.frame.rKy, RIGID)
        npt.assert_array_less(prob.model.perf.monopile.frame.rKz[0], RIGID)
        npt.assert_array_less(prob.model.perf.monopile.frame.rKtx, RIGID)
        npt.assert_array_less(prob.model.perf.monopile.frame.rKty, RIGID)
        npt.assert_array_less(prob.model.perf.monopile.frame.rKtz, RIGID)
        npt.assert_array_less(0.0, prob.model.perf.monopile.frame.rKx, RIGID)
        npt.assert_array_less(0.0, prob.model.perf.monopile.frame.rKy, RIGID)
        npt.assert_array_less(0.0, prob.model.perf.monopile.frame.rKz[0], RIGID)
        npt.assert_array_less(0.0, prob.model.perf.monopile.frame.rKtx, RIGID)
        npt.assert_array_less(0.0, prob.model.perf.monopile.frame.rKty, RIGID)
        npt.assert_array_less(0.0, prob.model.perf.monopile.frame.rKtz, RIGID)
        npt.assert_equal(prob.model.perf.monopile.frame.rKz[1:], 0.0)

        npt.assert_equal(prob.model.perf.monopile.frame.ENMnode, np.array([6, 5, 1], dtype=np.int_))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMmass, np.array([1e5, 1e2, 0]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMrhox, np.r_[10.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMrhoy, np.r_[10.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMrhoz, np.r_[10.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIxx, np.array([20, 1e2 * 25 * 0.5, 0.0]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIyy, np.array([20, 1e2 * 25 * 0.5, 0.0]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIzz, np.array([20, 1e2 * 25, 0.0]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIxy, np.r_[20.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIxz, np.r_[20.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIyz, np.r_[20.0, 0, 0])
        self.assertEqual(prob.model.perf.monopile.frame.addGravityLoadForExtraNodeMass, [False, True, True])

        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].NF, np.array([6]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Fx, np.array([2e3]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Fy, np.array([3e3]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Fz, np.array([4e3]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Mxx, np.array([2e4]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Myy, np.array([3e4]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Mzz, np.array([4e4]))

        # npt.assert_almost_equal(prob["monopile.mudline_F"], [4.61183362e04, 1.59353875e03, -2.94077236e07], 0)
        # npt.assert_almost_equal(prob["monopile.mudline_M"], [-248566.38259147, -3286049.81237828, 40000.0], 0)

    def testProblemFixedPile_GBF(self):
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = 3
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = 1
        self.modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"] = False
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gravity_foundation"] = True

        prob = om.Problem(reports=False)
        prob.model = mon.MonopileSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["water_depth"] = 30.0
        prob["transition_piece_mass"] = 1e2
        prob["transition_piece_cost"] = 1e3
        prob["gravity_foundation_mass"] = 1e4

        hval = np.array([15.0, 30.0])
        prob["monopile_s"] = np.cumsum(np.r_[0, hval]) / hval.sum()
        prob["monopile_foundation_height"] = -45.0
        prob["monopile_height"] = hval.sum()
        prob["monopile_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["monopile_top_diameter"] = 10.0
        prob["monopile_layer_thickness"] = 0.1 * np.ones(3).reshape((1, 3))
        prob["outfitting_factor_in"] = 1.0
        prob["monopile_layer_materials"] = prob["monopile_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["sigma_ult_mat"] = 1e8 * np.ones((1, 3))
        prob["wohler_exp_mat"] = 1e1
        prob["wohler_A_mat"] = 1e1

        prob["yaw"] = 0.0
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.3351e-3
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["Hsig_wave"] = 0.0
        prob["Tsig_wave"] = 1e3
        prob["env.Uref"] = 15.0
        prob["turbine_F"] = 1e3 * np.arange(2, 5)
        prob["turbine_M"] = 1e4 * np.arange(2, 5)
        prob["turbine_mass"] = 1e5
        prob["turbine_cg"] = 1e1 * np.ones(3)
        prob["turbine_I"] = 2e1 * np.ones(6)
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4 * (5.0**2 - 4.9**2) * np.pi
        npt.assert_equal(prob["z_start"], -45.0)
        npt.assert_equal(prob["transition_piece_height"], 0.0)
        npt.assert_equal(prob["suctionpile_depth"], 15.0)
        npt.assert_equal(prob["z_param"], np.array([-45.0, -30.0, 0.0]))
        npt.assert_almost_equal(prob["monopile_mass"], mass_dens * 45.0 + 1e2 + 1e4)

        npt.assert_equal(prob.model.perf.monopile.frame.rnode, np.array([1], dtype=np.int_))
        npt.assert_equal(prob.model.perf.monopile.frame.rKx, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.monopile.frame.rKy, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.monopile.frame.rKz, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.monopile.frame.rKtx, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.monopile.frame.rKty, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.monopile.frame.rKtz, np.array([RIGID]))

        npt.assert_equal(prob.model.perf.monopile.frame.ENMnode, np.array([6, 5, 1], dtype=np.int_))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMmass, np.array([1e5, 1e2, 1e4]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMrhox, np.r_[10.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMrhoy, np.r_[10.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMrhoz, np.r_[10.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIxx, np.array([20, 1e2 * 25 * 0.5, 1e4 * 25 * 0.25]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIyy, np.array([20, 1e2 * 25 * 0.5, 1e4 * 25 * 0.25]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIzz, np.array([20, 1e2 * 25, 1e4 * 25 * 0.5]))
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIxy, np.r_[20.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIxz, np.r_[20.0, 0, 0])
        npt.assert_equal(prob.model.perf.monopile.frame.ENMIyz, np.r_[20.0, 0, 0])
        self.assertEqual(prob.model.perf.monopile.frame.addGravityLoadForExtraNodeMass, [False, True, True])

        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].NF, np.array([6]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Fx, np.array([2e3]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Fy, np.array([3e3]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Fz, np.array([4e3]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Mxx, np.array([2e4]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Myy, np.array([3e4]))
        npt.assert_equal(prob.model.perf.monopile.frame.loadCases[0].Mzz, np.array([4e4]))

        # npt.assert_almost_equal(prob["monopile.mudline_F"], [3.74393291e04, 1.84264671e03, -3.39826364e07], 0)
        # npt.assert_almost_equal(prob["monopile.mudline_M"], [-294477.83027742, -2732413.3684215, 40000.0], 0)

    def testAddedMassForces(self):
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = 3
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = 1
        self.modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"] = False
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gravity_foundation"] = False

        prob = om.Problem(reports=False)
        prob.model = mon.MonopileSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["water_depth"] = 30.0
        prob["transition_piece_mass"] = 0.0
        prob["transition_piece_cost"] = 0.0
        prob["gravity_foundation_mass"] = 0.0

        hval = np.array([15.0, 30.0])
        prob["monopile_s"] = np.cumsum(np.r_[0, hval]) / hval.sum()
        prob["monopile_foundation_height"] = -45.0
        prob["monopile_height"] = hval.sum()
        prob["monopile_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["monopile_top_diameter"] = 10.0
        prob["monopile_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["outfitting_factor_in"] = 1.0
        prob["monopile_layer_materials"] = prob["monopile_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["sigma_ult_mat"] = 1e8 * np.ones((1, 3))
        prob["wohler_exp_mat"] = 1e1
        prob["wohler_A_mat"] = 1e1

        prob["yaw"] = 0.0
        # prob["G_soil"] = 1e7
        # prob["nu_soil"] = 0.5
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["ca_usr"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["rho_water"] = 1025.0
        prob["mu_water"] = 1.3351e-3
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["Hsig_wave"] = 0.0
        prob["Tsig_wave"] = 1e3
        prob["env.Uref"] = 15.0
        prob["turbine_F"] = 1e3 * np.arange(2, 5)
        prob["turbine_M"] = 1e4 * np.arange(2, 5)
        prob.run_model()

        myFz = copy.copy(prob["monopile.monopile_Fz"])

        prob["transition_piece_mass"] = 1e2
        prob.run_model()
        myFz[3:-2] -= 1e2 * g
        npt.assert_almost_equal(prob["monopile.monopile_Fz"], myFz)

        prob["gravity_foundation_mass"] = 1e3
        prob.run_model()
        # myFz[0] -= 1e3*g
        npt.assert_almost_equal(prob["monopile.monopile_Fz"], myFz)

    def testExampleRegression(self):
        # --- geometry ----
        h_param = np.diff(np.array([-60.0, -30.0, 10.0]))
        d_param = np.array([6.0, 6.0, 5.5])
        t_param = 1.3 * np.array([0.027, 0.023, 0.019])
        z_foundation = -60.0
        yaw = 0.0
        Koutfitting = 1.07

        # --- material props ---
        E = 210e9
        G = 80.8e9
        rho = 8500.0
        sigma_y = 450.0e6
        sigma_ult = wohler_A = 500e6
        wohler_exp = 4.0

        # --- wind ---
        wind_zref = 90.0
        wind_z0 = 0.0
        shearExp = 0.2
        cd_usr = -1.0
        # ---------------

        # --- wave ---
        water_depth = 30.0
        soilG = 140e6
        soilnu = 0.4
        Hsig = 4.0
        Tsig = 10.0
        # ---------------

        # --- costs ---
        material_cost = 5.0
        labor_cost = 100.0 / 60.0
        painting_cost = 30.0
        # ---------------

        # two load cases.  TODO: use a case iterator

        # # --- loading case 1: max Thrust ---
        wind_Uref1 = 11.73732
        Fx1 = 1284744.19620519
        Fy1 = 0.0
        Fz1 = -2914124.84400512
        Mxx1 = 3963732.76208099
        Myy1 = -2275104.79420872
        Mzz1 = -346781.68192839
        # # ---------------

        # # --- loading case 2: max wind speed ---
        wind_Uref2 = 70.0
        Fx2 = 930198.60063279
        Fy2 = 0.0
        Fz2 = -2883106.12368949
        Mxx2 = -1683669.22411597
        Myy2 = -2522475.34625363
        Mzz2 = 147301.97023764
        # # ---------------

        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = len(d_param)
        self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = 1
        self.modeling_options["WISDEM"]["n_dlc"] = 2
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.35
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.3
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.1
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gamma_fatigue"] = 1.35 * 1.3 * 1.0
        self.modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"] = True
        self.modeling_options["WISDEM"]["FixedBottomSE"]["gravity_foundation"] = False

        def fill_prob():
            prob = om.Problem(reports=False)
            prob.model = mon.MonopileSE(modeling_options=self.modeling_options)
            prob.setup()

            if self.modeling_options["WISDEM"]["FixedBottomSE"]["wind"] == "PowerWind":
                prob["shearExp"] = shearExp

            # assign values to params

            # --- geometry ----
            prob["water_depth"] = water_depth
            prob["monopile_s"] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
            prob["monopile_foundation_height"] = z_foundation
            prob["tower_foundation_height"] = z_foundation + h_param.sum()
            prob["monopile_height"] = h_param.sum()
            prob["monopile_outer_diameter_in"] = d_param
            prob["monopile_top_diameter"] = d_param[-1]
            prob["monopile_layer_thickness"] = t_param.reshape((1, len(t_param)))
            prob["outfitting_factor_in"] = Koutfitting
            prob["monopile_layer_materials"] = ["steel"]
            prob["material_names"] = ["steel"]
            prob["yaw"] = yaw
            prob["G_soil"] = soilG
            prob["nu_soil"] = soilnu
            prob["Hsig_wave"] = Hsig
            prob["Tsig_wave"] = Tsig
            # --- material props ---
            prob["E_mat"] = E * np.ones((1, 3))
            prob["G_mat"] = G * np.ones((1, 3))
            prob["rho_mat"] = rho
            prob["sigma_y_mat"] = sigma_y
            prob["sigma_ult_mat"] = sigma_ult * np.ones((1, 3))
            prob["wohler_exp_mat"] = wohler_exp
            prob["wohler_A_mat"] = wohler_A

            # --- costs ---
            prob["unit_cost_mat"] = [material_cost]
            prob["labor_cost_rate"] = labor_cost
            prob["painting_cost_rate"] = painting_cost
            # -----------

            # --- wind & wave ---
            prob["wind_reference_height"] = wind_zref
            prob["z0"] = wind_z0
            prob["cd_usr"] = cd_usr
            prob["ca_usr"] = 0.0 # Default was zero before, but now the default is -1
            prob["rho_air"] = 1.225
            prob["mu_air"] = 1.7934e-5

            # # --- loading case 1: max Thrust ---
            # # --- loading case 2: max Wind Speed ---
            prob["env1.Uref"] = wind_Uref1
            prob["env2.Uref"] = wind_Uref2

            prob["turbine_F"] = np.c_[[Fx1, Fy1, Fz1], [Fx2, Fy2, Fz2]]
            prob["turbine_M"] = np.c_[[Mxx1, Myy1, Mzz1], [Mxx2, Myy2, Mzz2]]

            return prob

        # # --- run ---
        prob = fill_prob()
        prob.run_model()
        npt.assert_almost_equal(prob["z_full"], [0.0, 10.0, 20.0, 30.0, 43.33333333, 56.66666667, 70.0])
        npt.assert_almost_equal(prob["outer_diameter_full"], [6.0, 6.0, 6.0, 6.0, 5.833333333, 5.6666667, 5.5])
        npt.assert_almost_equal(prob["t_full"], [0.0325, 0.0325, 0.0325, 0.0273, 0.0273, 0.0273])

        npt.assert_almost_equal(prob["monopile_mass"], [344802.1109282])
        npt.assert_almost_equal(prob["monopile_z_cg"], [32.9740276])
        npt.assert_almost_equal(prob["constr_d_to_t"], [184.6153846, 210.6227106])
        npt.assert_almost_equal(prob["constr_taper"], [1.0, 0.9166667])
        npt.assert_almost_equal(prob["env1.Uref"], [11.73732])
        npt.assert_almost_equal(prob["env2.Uref"], [70.0])
        npt.assert_almost_equal(prob["f1"], [2.76468844], 3)
        npt.assert_almost_equal(prob["monopile.top_deflection"], [0.0479862, 0.03513], 3)
        npt.assert_almost_equal(
            prob["post.constr_stress"].T,
            [
                [0.0378804, 0.0534528, 0.1838693, 0.1924264, 0.0922121, 0.0103132],
                [0.037407 , 0.0479956, 0.1538032, 0.1701437, 0.0811193, 0.0105535],
           ],
           3,
        )
        npt.assert_almost_equal(
            prob["post.constr_global_buckling"].T,
            [
                [0.111556 , 0.1234476, 0.2021098, 0.1781971, 0.129298 , 0.0862711],
                [0.1111474, 0.119174 , 0.1765608, 0.157615 , 0.1206368, 0.0862711],
            ],
            2,
        )
        npt.assert_almost_equal(
            prob["post.constr_shell_buckling"].T,
            [
                [0.0037742, 0.0073766, 0.619166 , 1.7880485, 0.6482763, 0.0886264],
                [0.0036865, 0.0059793, 0.6026997, 1.7732243, 0.6422529, 0.0912751],
            ],
            3,
        )
        npt.assert_almost_equal(prob["monopile.mudline_F"][0, :], [1352463.503616674, 1031332.0852743523], 2)
        npt.assert_array_less(np.abs(prob["monopile.mudline_F"][1, :]), 1e2, 2)
        npt.assert_almost_equal(prob["monopile.mudline_F"][2, :], [-4665110.903069687, -4634092.182754055], 2)
        npt.assert_almost_equal(
            prob["monopile.mudline_M"].T,
            [
                [ 3984005.677708 , 33542440.2236337,  -346821.3257299],
                [-1692224.1838883, 24992789.5098108,   147318.6977002],
            ],
            1,
        )

        # Now regression on DNV-GL C202 methods
        self.modeling_options["WISDEM"]["FixedBottomSE"]["buckling_method"] = "dnvgl"
        prob = fill_prob()
        prob.run_model()

        npt.assert_almost_equal(prob["z_full"], [0.0, 10.0, 20.0, 30.0, 43.33333333, 56.66666667, 70.0])
        npt.assert_almost_equal(prob["outer_diameter_full"], [6.0, 6.0, 6.0, 6.0, 5.833333333, 5.6666667, 5.5])
        npt.assert_almost_equal(prob["t_full"], [0.0325, 0.0325, 0.0325, 0.0273, 0.0273, 0.0273])

        npt.assert_almost_equal(prob["monopile_mass"], [344802.1109282])
        npt.assert_almost_equal(prob["monopile_z_cg"], [32.9740276])
        npt.assert_almost_equal(prob["constr_d_to_t"], [184.6153846, 210.6227106])
        npt.assert_almost_equal(prob["constr_taper"], [1.0, 0.9166667])
        npt.assert_almost_equal(prob["env1.Uref"], [11.73732])
        npt.assert_almost_equal(prob["env2.Uref"], [70.0])
        npt.assert_almost_equal(prob["f1"], [2.76468844], 3)
        npt.assert_almost_equal(prob["monopile.top_deflection"], [0.0479862, 0.03513], 3)
        npt.assert_almost_equal(
            prob["post.constr_stress"].T,
            [
                [0.0382954, 0.053019 , 0.17944  , 0.1898312, 0.0924179, 0.0103132],
                [0.0380006, 0.0470939, 0.1456076, 0.1645079, 0.0826036, 0.0105535],
            ],
            2,
        )
        npt.assert_almost_equal(
            prob["post.constr_global_buckling"].T,
            [
                [3.2956754e-02, 3.7180672e-02, 7.3347527e-02, 6.2425147e-02, 3.7038800e-02, 1.2614017e-10],
                [3.2843563e-02, 3.4863560e-02, 5.8869295e-02, 5.0591031e-02, 3.3466536e-02, 9.1721922e-11],
            ],
            1,
        )
        npt.assert_almost_equal(
            prob["post.constr_shell_buckling"].T,
            [
                [0.0421064, 0.0561218, 0.6648091, 1.8606197, 0.8695418, 0.1664218],
                [0.0416873, 0.049646 , 0.6302573, 1.8213539, 0.8470644, 0.1702952],
            ],
            2,
        )
        npt.assert_almost_equal(prob["monopile.mudline_F"][0, :], [1352463.503616674, 1031332.0852743523], 2)
        npt.assert_array_less(np.abs(prob["monopile.mudline_F"][1, :]), 1e2, 2)
        npt.assert_almost_equal(prob["monopile.mudline_F"][2, :], [-4665110.903069687, -4634092.182754055], 2)
        npt.assert_almost_equal(
            prob["monopile.mudline_M"].T,
            [
                [ 3984005.677708 , 33542440.2236337,  -346821.3257299],
                [-1692224.1838883, 24992789.5098108,   147318.6977002],
            ],
            1,
        )


def suite():
    suite = [
        unittest.TestLoader().loadTestsFromTestCase(TestPreDiscretization),
        unittest.TestLoader().loadTestsFromTestCase(TestMass),
        unittest.TestLoader().loadTestsFromTestCase(TestMonopileSE),
    ]
    return unittest.TestSuite(suite)


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
