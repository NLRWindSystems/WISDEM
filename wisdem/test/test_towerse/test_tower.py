import copy
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt

import wisdem.towerse.tower as tow
from wisdem.commonse import gravity as g

npts = 100
myones = np.ones((npts,))
RIGID = tow.RIGID


class TestPreDiscretization(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        inputs["hub_height"] = np.array([ 125.0 ])
        inputs["tower_height"] = np.array([ 100.0 ])
        inputs["foundation_height"] = np.array([ 10.0 ])

        mydis = tow.PreDiscretization()
        mydis.compute(inputs, outputs)
        self.assertEqual(outputs["transition_piece_height"], 10.0)
        self.assertEqual(outputs["z_start"], 10.0)
        npt.assert_array_equal(outputs["joint1"], np.array([0.0, 0.0, 10.0]))
        npt.assert_array_equal(outputs["joint2"], np.array([0.0, 0.0, 110.0]))
        self.assertEqual(outputs["height_constraint"], 15.0)


class TestMass(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        inputs["joint2"] = np.array([0.0, 0.0, 125.0])
        inputs["rna_mass"] = np.array([ 1000.0 ])
        inputs["rna_I"] = 3e4 * np.ones(6)
        inputs["rna_cg"] = 20.0 * np.ones(3)
        inputs["tower_mass"] = np.array([ 3000.0 ])
        inputs["tower_center_of_mass"] = 0.5 * inputs["joint2"][-1]
        inputs["tower_I_base"] = 2e4 * np.ones(6)

        myobj = tow.TurbineMass()
        myobj.compute(inputs, outputs)

        self.assertEqual(outputs["turbine_mass"], 4e3)
        h = np.r_[0.0, 0.0, 125.0]
        npt.assert_equal(outputs["turbine_center_of_mass"], (1e3 * (inputs["rna_cg"] + h) + 3e3 * 0.5 * h) / 4e3)
        # npt.assert_array_less(5e4, np.abs(outputs["turbine_I_base"]))

        
class TestTowerSE(unittest.TestCase):
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
        self.modeling_options["WISDEM"]["TowerSE"] = {}
        self.modeling_options["WISDEM"]["TowerSE"]["buckling_method"] = "eurocode"
        self.modeling_options["WISDEM"]["TowerSE"]["buckling_length"] = 30.0
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["n_refine"] = 3
        self.modeling_options["WISDEM"]["TowerSE"]["wind"] = "PowerWind"

        self.modeling_options["WISDEM"]["TowerSE"]["gamma_f"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_m"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_n"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_b"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_fatigue"] = 1.0

        # Simplified the options available to the user
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"] = {}
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["shear"] = True
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["geom"] = True
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["modal_method"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["tol"] = 1e-9

    def testProblemLand(self):
        prob = om.Problem(reports=False)
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["hub_height"] = 80.0

        prob["tower_s"] = np.linspace(0, 1, 3)
        prob["foundation_height"] = 0.0
        prob["tower_height"] = 80.0
        # prob['tower_section_height'] = 40.0*np.ones(2)
        prob["tower_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["tower_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["outfitting_factor_in"] = 1.0
        prob["tower_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 1e9 * np.ones((1, 3))
        prob["G_mat"] = 1e8 * np.ones((1, 3))
        prob["rho_mat"] = 1e4
        prob["sigma_y_mat"] = 1e8
        prob["sigma_ult_mat"] = 1e8 * np.ones((1, 3))
        prob["wohler_exp_mat"] = 1e1
        prob["wohler_A_mat"] = 1e1

        prob["yaw"] = 0.0
        prob["rna_mass"] = 2e5
        prob["rna_I"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob["rna_cg"] = np.array([-3.0, 0.0, 1.0])
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["env.Uref"] = 15.0
        prob["tower.rna_F"] = 1e3 * np.arange(2, 5)
        prob["tower.rna_M"] = 1e4 * np.arange(2, 5)
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4 * (5.0**2 - 4.9**2) * np.pi
        npt.assert_equal(prob["z_start"], 0.0)
        npt.assert_equal(prob["transition_piece_height"], 0.0)
        npt.assert_equal(prob["z_param"], np.array([0.0, 40.0, 80.0]))

        self.assertEqual(prob["height_constraint"], 0.0)
        npt.assert_almost_equal(prob["tower_center_of_mass"], 40.0)
        npt.assert_almost_equal(prob["tower_mass"], mass_dens * 80.0)

        npt.assert_equal(prob.model.perf.tower.frame.rnode, np.array([1], dtype=np.int_))
        npt.assert_equal(prob.model.perf.tower.frame.rKx, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.tower.frame.rKy, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.tower.frame.rKz, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.tower.frame.rKtx, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.tower.frame.rKty, np.array([RIGID]))
        npt.assert_equal(prob.model.perf.tower.frame.rKtz, np.array([RIGID]))

        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].NF, np.array([7]))
        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].Fx, np.array([2e3]))
        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].Fy, np.array([3e3]))
        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].Fz, np.array([4e3]))
        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].Mxx, np.array([2e4]))
        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].Myy, np.array([3e4]))
        npt.assert_equal(prob.model.perf.tower.frame.loadCases[0].Mzz, np.array([4e4]))

    def testAddedMassForces(self):
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = 3

        prob = om.Problem(reports=False)
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        prob["hub_height"] = 80.0

        prob["tower_s"] = np.linspace(0, 1, 3)
        prob["foundation_height"] = 0.0
        prob["tower_height"] = 60.0
        prob["tower_outer_diameter_in"] = 10.0 * np.ones(3)
        prob["tower_layer_thickness"] = 0.1 * np.ones((1, 3))
        prob["outfitting_factor_in"] = 1.0
        hval = np.array([15.0, 30.0])
        prob["tower_layer_materials"] = ["steel"]
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
        prob["rna_mass"] = 0.0
        prob["rna_I"] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob["rna_cg"] = np.array([-3.0, 0.0, 1.0])
        prob["wind_reference_height"] = 80.0
        prob["z0"] = 0.0
        prob["cd_usr"] = -1.0
        prob["rho_air"] = 1.225
        prob["mu_air"] = 1.7934e-5
        prob["shearExp"] = 0.2
        prob["beta_wind"] = 0.0
        prob["env.Uref"] = 15.0
        prob["tower.rna_F"] = 1e3 * np.arange(2, 5)
        prob["tower.rna_M"] = 1e4 * np.arange(2, 5)
        prob.run_model()
        myFz = copy.copy(prob["tower.tower_Fz"])

        prob["tower.rna_F"][-1] += 1e4
        prob.run_model()
        myFz += 1e4
        npt.assert_almost_equal(prob["tower.tower_Fz"], myFz)

    def test15MWmode_shapes(self):
        # --- geometry ----
        h_param = np.array(
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 14.1679]
        )
        d_param = np.array(
            [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                9.92647687,
                9.44319282,
                8.83283769,
                8.15148167,
                7.38976138,
                6.90908962,
                6.74803581,
                6.57231775,
                6.5,
            ]
        )
        t_param = np.array(
            [
                0.05534138,
                0.05534138,
                0.05344902,
                0.05150928,
                0.04952705,
                0.04751736,
                0.04551709,
                0.0435267,
                0.04224176,
                0.04105759,
                0.0394965,
                0.03645589,
                0.03377851,
                0.03219233,
                0.03070819,
                0.02910109,
                0.02721289,
                0.02400931,
                0.0208264,
                0.02399756,
            ]
        )

        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = len(d_param)

        prob = om.Problem(reports=False)
        prob.model = tow.TowerSE(modeling_options=self.modeling_options)
        prob.setup()

        # Set common and then customized parameters
        prob["hub_height"] = prob["wind_reference_height"] = 30 + 146.1679
        # prob["foundation_height"] = 0.0  # -30.0

        prob["tower_s"] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
        prob["foundation_height"] = 0.0  # 15.0
        prob["tower_height"] = h_param.sum()
        prob["tower_outer_diameter_in"] = d_param
        prob["tower_layer_thickness"] = t_param.reshape((1, len(t_param)))
        prob["outfitting_factor_in"] = 1.0
        prob["tower_layer_materials"] = ["steel"]
        prob["material_names"] = ["steel"]
        prob["E_mat"] = 210e9 * np.ones((1, 3))
        prob["G_mat"] = 79.3e9 * np.ones((1, 3))
        prob["rho_mat"] = 7850.0
        prob["sigma_y_mat"] = 345e6
        prob["sigma_ult_mat"] = 500e6 * np.ones((1, 3))
        prob["wohler_exp_mat"] = 4.0
        prob["wohler_A_mat"] = 500e6

        prob["yaw"] = 0.0
        # prob['G_soil'] = 140e6
        # prob['nu_soil'] = 0.4
        prob["shearExp"] = 0.11
        prob["rho_air"] = 1.225
        prob["z0"] = 0.0
        prob["mu_air"] = 1.7934e-5

        mIxx = 379640227.0
        mIyy = 224477294.0
        mIzz = 182971949.0
        mIxy = 0.0
        mIxz = -7259625.38
        mIyz = 0.0
        prob["rna_mass"] = 1007537.0
        prob["rna_I"] = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
        prob["rna_cg"] = np.array([-5.019, 0.0, 0.0])

        prob["env.Uref"] = 0.0  # 20.00138038
        prob["tower.rna_F"] = np.zeros(3)  # np.array([3569257.70891496, -22787.83765441, -404483.54819059])
        prob["tower.rna_M"] = np.zeros(3)  # np.array([68746553.1515807, 16045443.38557568, 1811078.988995])

        # # --- run ---
        prob.run_model()
        """
        Natural Frequencies (Hz): [ 0.2161   0.21842  1.1091   1.167    1.2745   2.3611   2.5877   5.1233  5.2111   9.9725  10.007   10.151   16.388   16.4     18.092   21.813 23.955   23.958   30.184   33.706  ]

        Polynomial fit coefficients to modal displacements (x^2, x^3, x^4, x^5, x^6)
        1st Fore-aft    = [1.11422342, -2.73438505, 6.84397071, -5.97959674, 1.75578766]
        2nd Fore-aft    = [-48.86125831, 82.74454067, -156.79260263, 208.53125496, -84.62193469]
        1st Side-side   = [1.10492357, -2.71587869, 6.80247339, -5.93612744, 1.74460918]
        2nd Side-side   = [48.9719383, -89.25323746, 183.04839183, -226.34534799, 84.57825533]
        """

    def testExampleRegression(self):
        # --- geometry ----
        h_param = np.diff(np.array([0.0, 43.8, 87.6]))
        d_param = np.array([6.0, 4.935, 3.87])
        t_param = 1.3 * np.array([0.027, 0.023, 0.019])
        z_foundation = 0.0
        yaw = 0.0
        Koutfitting = 1.07

        # --- material props ---
        E = 210e9
        G = 80.8e9
        rho = 8500.0
        sigma_y = 450.0e6
        sigma_ult = wohler_A = 500e6
        wohler_exp = 4.0

        # --- extra mass ----
        m = np.array([285598.8])
        mIxx = 1.14930678e08
        mIyy = 2.20354030e07
        mIzz = 1.87597425e07
        mIxy = 0.0
        mIxz = 5.03710467e05
        mIyz = 0.0
        mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
        mrho = np.array([-1.13197635, 0.0, 0.50875268])
        # -----------

        # --- wind ---
        wind_zref = 90.0
        wind_z0 = 0.0
        shearExp = 0.2
        cd_usr = -1.0
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
        Fz1 = -2914124.84400512 + m * g
        Mxx1 = 3963732.76208099
        Myy1 = -2275104.79420872
        Mzz1 = -346781.68192839
        # # ---------------

        # # --- loading case 2: max wind speed ---
        wind_Uref2 = 70.0
        Fx2 = 930198.60063279
        Fy2 = 0.0
        Fz2 = -2883106.12368949 + m * g
        Mxx2 = -1683669.22411597
        Myy2 = -2522475.34625363
        Mzz2 = 147301.97023764
        # # ---------------

        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = len(d_param)
        self.modeling_options["WISDEM"]["TowerSE"]["n_layers"] = 1
        self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = len(d_param)
        self.modeling_options["WISDEM"]["n_dlc"] = 2
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_f"] = 1.35
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_m"] = 1.3
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_n"] = 1.0
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_b"] = 1.1
        self.modeling_options["WISDEM"]["TowerSE"]["gamma_fatigue"] = 1.35 * 1.3 * 1.0

        def fill_prob():
            prob = om.Problem(reports=False)
            prob.model = tow.TowerSE(modeling_options=self.modeling_options)
            prob.setup()

            if self.modeling_options["WISDEM"]["TowerSE"]["wind"] == "PowerWind":
                prob["shearExp"] = shearExp

            # assign values to params

            # --- geometry ----
            prob["hub_height"] = h_param.sum()
            prob["tower_s"] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
            prob["foundation_height"] = z_foundation
            prob["tower_height"] = h_param.sum()
            prob["tower_outer_diameter_in"] = d_param
            prob["tower_layer_thickness"] = t_param.reshape((1, len(t_param)))
            prob["outfitting_factor_in"] = Koutfitting
            prob["tower_layer_materials"] = ["steel"]
            prob["material_names"] = ["steel"]
            prob["yaw"] = yaw
            # --- material props ---
            prob["E_mat"] = E * np.ones((1, 3))
            prob["G_mat"] = G * np.ones((1, 3))
            prob["rho_mat"] = rho
            prob["sigma_y_mat"] = sigma_y
            prob["sigma_ult_mat"] = sigma_ult * np.ones((1, 3))
            prob["wohler_exp_mat"] = wohler_exp
            prob["wohler_A_mat"] = wohler_A

            # --- extra mass ----
            prob["rna_mass"] = m
            prob["rna_I"] = mI
            prob["rna_cg"] = mrho
            # -----------

            # --- costs ---
            prob["unit_cost_mat"] = [material_cost]
            prob["labor_cost_rate"] = labor_cost
            prob["painting_cost_rate"] = painting_cost
            # -----------

            # --- wind & wave ---
            prob["wind_reference_height"] = wind_zref
            prob["z0"] = wind_z0
            prob["cd_usr"] = cd_usr
            prob["rho_air"] = 1.225
            prob["mu_air"] = 1.7934e-5

            # # --- loading case 1: max Thrust ---
            prob["env1.Uref"] = wind_Uref1
            prob["env2.Uref"] = wind_Uref2

            prob["tower.rna_F"] = np.c_[np.r_[Fx1, Fy1, Fz1], np.r_[Fx2, Fy2, Fz2]]
            prob["tower.rna_M"] = np.c_[np.r_[Mxx1, Myy1, Mzz1], np.r_[Mxx2, Myy2, Mzz2]]
            # # ---------------

            return prob

        # # --- run ---
        prob = fill_prob()
        prob.run_model()

        npt.assert_almost_equal(prob["z_full"], [0.0, 14.6, 29.2, 43.8, 58.4, 73.0, 87.6])
        npt.assert_almost_equal(prob["outer_diameter_full"], [6.0, 5.645, 5.29, 4.935, 4.58, 4.225, 3.87])
        npt.assert_almost_equal(prob["t_full"], [0.0325, 0.0325, 0.0325, 0.0273, 0.0273, 0.0273])

        npt.assert_almost_equal(prob["tower_mass"], [370541.14008246])
        npt.assert_almost_equal(prob["tower_center_of_mass"], [38.78441074])
        npt.assert_almost_equal(prob["constr_d_to_t"], [168.23076923, 161.26373626])
        npt.assert_almost_equal(prob["constr_taper"], [0.8225, 0.78419453])
        npt.assert_almost_equal(prob["env1.Uref"], [11.73732])
        npt.assert_almost_equal(prob["env2.Uref"], [70.0])
        npt.assert_almost_equal(prob["tower.f1"], [0.33444], 5)
        npt.assert_almost_equal(prob["tower.top_deflection"], [0.7186541, 0.6634745])
        npt.assert_almost_equal(
            prob["post.constr_stress"].T,
            [
                [0.4469963, 0.4016846, 0.3400776, 0.3055749, 0.1741112, 0.0694753],
                [0.4343727, 0.3710877, 0.2961703, 0.2480675, 0.1270603, 0.0475901],
            ],
        )
        npt.assert_almost_equal(
            prob["post.constr_global_buckling"].T,
            [
                [0.4907594, 0.451831 , 0.3981311, 0.3695584, 0.252773 , 0.1473065],
                [0.4797098, 0.4257058, 0.3609528, 0.3209128, 0.2132003, 0.1316666],
            ],
        )
        npt.assert_almost_equal(
            prob["post.constr_shell_buckling"].T,
            [
                [0.3101001, 0.2460605, 0.1751842, 0.1600565, 0.0587418, 0.0147917],
                [0.2958948, 0.2151103, 0.1381308, 0.1134728, 0.0364789, 0.0101659],
            ],
        )
        x = prob["tower.turbine_F"].T
        x[np.abs(x)<1e-8] = 0.0
        npt.assert_allclose(x,
                            [
                                [1299652.107673, 0.0, -3747097.344505],
                                [1609815.917488, 0.0, -3716078.62419 ],
                            ], 1e-6
                            )
        npt.assert_almost_equal(
            prob["tower.turbine_M"].T,
            [
                [ 4020089.690650426,  112049899.66314322, -3.46781682e+05],
                [-1707006.7270627208, 113622909.0698837,  1.47301970e+05],
            ], -2,
        )

        # Now regression on DNV-GL C202 methods
        self.modeling_options["WISDEM"]["TowerSE"]["buckling_method"] = "dnvgl"
        prob = fill_prob()
        prob.run_model()

        npt.assert_almost_equal(prob["z_full"], [0.0, 14.6, 29.2, 43.8, 58.4, 73.0, 87.6])
        npt.assert_almost_equal(prob["outer_diameter_full"], [6.0, 5.645, 5.29, 4.935, 4.58, 4.225, 3.87])
        npt.assert_almost_equal(prob["t_full"], [0.0325, 0.0325, 0.0325, 0.0273, 0.0273, 0.0273])

        npt.assert_almost_equal(prob["tower_mass"], [370541.14008246])
        npt.assert_almost_equal(prob["tower_center_of_mass"], [38.78441074])
        npt.assert_almost_equal(prob["constr_d_to_t"], [168.23076923, 161.26373626])
        npt.assert_almost_equal(prob["constr_taper"], [0.8225, 0.78419453])
        npt.assert_almost_equal(prob["env1.Uref"], [11.73732])
        npt.assert_almost_equal(prob["env2.Uref"], [70.0])
        npt.assert_almost_equal(prob["tower.f1"], [0.33444], 5)
        npt.assert_almost_equal(prob["tower.top_deflection"], [0.7186541, 0.6634745])

        npt.assert_almost_equal(
            prob["post.constr_stress"].T,
            [
                [0.4469963, 0.4016846, 0.3400776, 0.3055749, 0.1741112, 0.0694753],
                [0.4343727, 0.3710877, 0.2961703, 0.2480675, 0.1270603, 0.0475901],
            ],
        )
        npt.assert_almost_equal(
            prob["post.constr_global_buckling"].T,
            [
                [0.8343938, 0.7498092, 0.6338816, 0.5685886, 0.3172955, 0.0987612],
                [0.7988185, 0.6823171, 0.5438744, 0.4547193, 0.2274408, 0.0648491],
            ],
        )
        npt.assert_almost_equal(
            prob["post.constr_shell_buckling"].T,
            [
                [0.3409915, 0.2915678, 0.2374476, 0.2597997, 0.1651283, 0.0979941],
                [0.3417898, 0.2825076, 0.219439 , 0.2283339, 0.1367243, 0.0791142],
            ],
        )
        x = prob["tower.turbine_F"].T
        x[np.abs(x)<1e-8] = 0.0
        npt.assert_allclose(x,
                            [
                                [1299652.107673, 0.0, -3747097.344505],
                                [1609815.917488, 0.0, -3716078.62419 ],
                            ], 1e-6
        )
        npt.assert_almost_equal(
            prob["tower.turbine_M"].T,
            [
                [ 4020089.690650426,  112049899.66314322, -3.46781682e+05],
                [-1707006.7270627208, 113622909.0698837,  1.47301970e+05],
            ], -2,
        )


def suite():
    suite = [
        unittest.TestLoader().loadTestsFromTestCase(TestPreDiscretization),
        unittest.TestLoader().loadTestsFromTestCase(TestMass),
        unittest.TestLoader().loadTestsFromTestCase(TestTowerSE),
    ]
    return unittest.TestSuite(suite)

if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
