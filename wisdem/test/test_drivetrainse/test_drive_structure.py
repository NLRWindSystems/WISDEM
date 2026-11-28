import unittest

import numpy as np
import numpy.testing as npt

import wisdem.drivetrainse.layout as lay
import wisdem.drivetrainse.drive_structure as ds
from wisdem.commonse import gravity

npts = 12


class TestDirectStructure(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}
        self.opt = {}

        self.discrete_inputs["upwind"] = True

        self.inputs["L_12"] = np.array([2.0])
        self.inputs["L_h1"] = np.array([1.0])
        self.inputs["L_generator"] = np.array([3.25])
        # self.inputs['L_2n'] = np.array([1.5])
        # self.inputs['L_grs'] = np.array([1.1])
        # self.inputs['L_gsn'] = np.array([1.1])
        self.inputs["L_hss"] = np.array([0.75])
        self.inputs["L_gearbox"] = np.array([1.2])
        self.inputs["overhang"] = np.array([6.25])
        self.inputs["drive_height"] = np.array([4.875])
        self.inputs["tilt"] = np.array([4.0])
        self.inputs["access_diameter"] = np.array([0.9])

        myones = np.ones(5)
        self.inputs["lss_diameter"] = 3.3 * myones
        self.inputs["lss_wall_thickness"] = 0.45 * myones
        self.inputs["hss_diameter"] = 1.6 * np.ones(3)
        self.inputs["hss_wall_thickness"] = 0.25 * np.ones(3)
        self.inputs["nose_diameter"] = 2.2 * myones
        self.inputs["nose_wall_thickness"] = 0.1 * myones
        self.inputs["bedplate_wall_thickness"] = 0.06 * np.ones(npts)

        self.inputs["bedplate_flange_width"] = np.array([1.5])
        self.inputs["bedplate_flange_thickness"] = np.array([0.05])
        # self.inputs['bedplate_web_height'] = np.array([1.0])
        self.inputs["bedplate_web_thickness"] = np.array([0.05])
        self.inputs["bedplate_mass_user"] = np.array([0.0])

        self.inputs["D_top"] = np.array([6.5])
        self.inputs["hub_diameter"] = np.array([4.0])

        self.inputs["other_mass"] = np.array([200e3])
        self.inputs["mb1_mass"] = np.array([10e3])
        self.inputs["mb1_I"] = 10e3 * 0.5 * 2**2 * np.ones(3)
        self.inputs["mb2_mass"] = np.array([10e3])
        self.inputs["mb2_I"] = 10e3 * 0.5 * 1.5**2 * np.ones(3)
        self.inputs["mb1_max_defl_ang"] = np.array([0.008])
        self.inputs["mb2_max_defl_ang"] = np.array([0.008])

        self.inputs["m_stator"] = np.array([100e3])
        self.inputs["cm_stator"] = np.array([-0.3])
        self.inputs["I_stator"] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])

        self.inputs["generator_rotor_mass"] = np.array([100e3])
        self.inputs["cm_rotor"] = np.array([-0.3])
        self.inputs["generator_rotor_I"] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])

        self.inputs["generator_stator_mass"] = np.array([100e3])
        self.inputs["cm_rotor"] = np.array([-0.3])
        self.inputs["generator_stator_I"] = np.array([1e6, 5e5, 5e5, 0.0, 0.0, 0.0])

        self.inputs["generator_mass"] = np.array([200e3])
        self.inputs["generator_I"] = np.array([2e6, 1e6, 1e6, 0.0, 0.0, 0.0])

        self.inputs["gearbox_mass"] = np.array([100e3])
        self.inputs["gearbox_I"] = np.array([1e6, 5e5, 5e5])

        self.inputs["brake_mass"] = np.array([10e3])
        self.inputs["brake_I"] = np.array([1e4, 5e3, 5e3])

        self.inputs["carrier_mass"] = np.array([10e3])
        self.inputs["carrier_I"] = np.array([1e4, 5e3, 5e3])

        self.inputs["gear_ratio"] = np.array([1.0])

        self.inputs["F_mb1"] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([2409.750e3, -1716.429e3, 74.3529e3]).reshape((3, 1))
        self.inputs["M_mb1"] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3, 1))
        self.inputs["M_mb2"] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3]).reshape((3, 1))

        self.inputs["hub_system_mass"] = np.array([100e3])
        self.inputs["hub_system_cm"] = np.array([2.0])
        self.inputs["hub_system_I"] = np.array([2409.750e3, -1716.429e3, 74.3529e3, 0.0, 0.0, 0.0])
        self.inputs["blades_mass"] = np.array([200e3])
        self.inputs["blades_cm"] = np.array([3.0])
        self.inputs["blades_I"] = 2*np.array([2409.750e3, -1716.429e3, 74.3529e3, 0.0, 0.0, 0.0])
        self.inputs["F_aero_hub"] = np.array([2409.750e3, 0.0, 74.3529e2]).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.array([-1.83291e4, 6171.7324e2, 5785.82946e2]).reshape((3, 1))

        self.inputs["lss_E"] = self.inputs["hss_E"] = self.inputs["bedplate_E"] = 210e9
        self.inputs["lss_G"] = self.inputs["hss_G"] = self.inputs["bedplate_G"] = 80.8e9
        self.inputs["lss_rho"] = self.inputs["hss_rho"] = self.inputs["bedplate_rho"] = 7850.0
        self.inputs["lss_Xy"] = self.inputs["hss_Xy"] = self.inputs["bedplate_Xy"] = 250e6
        self.inputs["lss_mass_user"] = 0.

        self.inputs["shaft_deflection_allowable"] = np.array([1e-4])
        self.inputs["shaft_angle_allowable"] = np.array([1e-3])
        self.inputs["stator_deflection_allowable"] = np.array([1e-4])
        self.inputs["stator_angle_allowable"] = np.array([1e-3])

        self.opt["gamma_f"] = 1.35
        self.opt["gamma_m"] = 1.3
        self.opt["gamma_n"] = 1.0


        for k in self.inputs:
            if type(self.inputs[k]) == type(np.array([])): continue
            self.inputs[k] = np.array([self.inputs[k]])
        
    def compute_layout(self, direct=True):
        myobj = lay.DirectLayout() if direct else lay.GearedLayout()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs.keys():
            if type(self.outputs[k]) == type(np.array([])):
                self.inputs[k] = self.outputs[k]
            else:
                self.inputs[k] = np.array([ self.outputs[k] ])

    def testBaseF_BaseM(self):
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"] + self.inputs["M_mb2"], decimal=-1)

        self.inputs["M_torq"] = 30e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"] + self.inputs["M_mb2"] + self.inputs["M_torq"], decimal=-1)

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 2 * self.inputs["F_mb2"][:2])
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity + 2 * 50e2)

    def testBaseF_BaseM_withTilt(self):
        self.inputs["tilt"] = np.array([5.0])
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(
            self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1] + self.inputs["M_mb2"][1], decimal=-1
        )

        self.inputs["M_torq"] = 30e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(
            self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1] + self.inputs["M_mb2"][1] + self.inputs["M_torq"][1], decimal=-1
        )

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][1], 2 * self.inputs["F_mb2"][1])

    def testBaseF_BaseM_Downwind(self):
        self.inputs["tilt"] = np.array([0.0])
        self.discrete_inputs["upwind"] = False
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"] + self.inputs["M_mb2"], decimal=-1)

        self.inputs["M_torq"] = 30e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"], M0 + self.inputs["M_mb1"] + self.inputs["M_mb2"] + self.inputs["M_torq"], decimal=-1)

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 2 * self.inputs["F_mb2"][:2])
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity + 2 * 50e2)

    def testBaseF_BaseM_withTilt_Downwind(self):
        self.inputs["tilt"] = np.array([5.0])
        self.discrete_inputs["upwind"] = False
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.compute_layout()
        myobj = ds.Nose_Stator_Bedplate_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][-1], 0.0)
        F0 = self.outputs["base_F"]
        M0 = self.outputs["base_M"]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][0], 0.0)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])
        npt.assert_almost_equal(self.outputs["base_M"][2], 0.0)

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1], decimal=0)

        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(
            self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1] + self.inputs["M_mb2"][1], decimal=-1
        )

        self.inputs["M_torq"] = 30e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(
            self.outputs["base_M"][1], M0[1] + self.inputs["M_mb1"][1] + self.inputs["M_mb2"][1] + self.inputs["M_torq"][1], decimal=-1
        )

        self.inputs["F_mb1"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        self.inputs["F_mb2"] = np.array([30e2, 40e2, 50e2]).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][1], 2 * self.inputs["F_mb2"][1])

    def testBaseF_BaseM_Geared(self):
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_generator"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_generator"] = np.zeros(3).reshape((3, 1))
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout(False)
        myobj = ds.Bedplate_IBeam_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        F0 = self.outputs["base_F"][:, 0]
        M0 = self.outputs["base_M"][:, 0]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity, decimal=0)
        # npt.assert_almost_equal(self.outputs['base_M'], M0+self.inputs['M_mb1']+self.inputs['M_mb2'], decimal=-1)

        self.inputs["F_mb1"] = self.inputs["F_mb2"] = self.inputs["F_generator"] = self.inputs["F_torq"] = np.array(
            [30e2, 40e2, 50e2]
        ).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 4 * self.inputs["F_mb1"][:2, 0], decimal=1)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity + 4 * 50e2, decimal=0)

    def testBaseF_BaseM_withTilt_Geared(self):
        self.inputs["tilt"] = np.array([5.0])
        self.inputs["F_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["F_generator"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb1"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_mb2"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_torq"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_generator"] = np.zeros(3).reshape((3, 1))
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout(False)
        myobj = ds.Bedplate_IBeam_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        F0 = self.outputs["base_F"][:, 0]
        M0 = self.outputs["base_M"][:, 0]

        self.inputs["other_mass"] += 500e3
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=1)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity)
        npt.assert_almost_equal(self.outputs["base_M"][[0, 2], 0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["base_M"][1], M0[1])

        self.inputs["M_mb1"] = 10e3 * np.arange(1, 4).reshape((3, 1))
        self.inputs["M_mb2"] = 20e3 * np.arange(1, 4).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][:2, 0], 0.0, decimal=1)
        npt.assert_almost_equal(self.outputs["base_F"][2, 0], F0[2] - 500e3 * gravity, decimal=0)
        # npt.assert_almost_equal(self.outputs['base_M'], M0+self.inputs['M_mb1']+self.inputs['M_mb2'], decimal=-1)

        self.inputs["F_mb1"] = self.inputs["F_mb2"] = self.inputs["F_generator"] = self.inputs["F_torq"] = np.array(
            [30e2, 40e2, 50e2]
        ).reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["base_F"][1, 0], 4 * self.inputs["F_mb1"][1, 0], decimal=1)

    def testRunRotatingDirect_noTilt(self):
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3**4 - 2.4**4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_aero_hub"] = g.reshape((3, 1))
        self.inputs["M_aero_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        t = 0.45
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r**2 - (r - t) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:3], 1.0 / (2*t*self.inputs["L_lss"][0]))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][3], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r**4 - (r - t) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r**4 - (r - t) ** 4))
        )

    def testRunRotatingDirect_withTilt(self):
        self.inputs["tilt"] = np.array([5.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[0])
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][1], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3**4 - 2.4**4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_aero_hub"] = g.reshape((3, 1))
        self.inputs["M_aero_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=1)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        t = 0.45
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r**2 - (r - t) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:3], 1.0 / (2*t*self.inputs["L_lss"][0]))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][3], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r**4 - (r - t) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r**4 - (r - t) ** 4))
        )

    def testRotorMasses(self):
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        npt.assert_almost_equal(self.outputs["F_mb1"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)

        self.inputs["hub_system_mass"] = 1e3*np.array([100.0])
        self.inputs["hub_system_cm"] = np.array([2.0])
        self.inputs["hub_system_I"] = 1e3*np.array([1., 2., 3.])
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F1 = self.outputs["F_mb1"].flatten()
        M1 = self.outputs["M_mb2"].flatten()
        npt.assert_almost_equal(F0[:2], F1[:2], decimal=2)
        npt.assert_almost_equal(F0[2] - 100e3*9.801, F1[2], decimal=-4)
        npt.assert_almost_equal(M0[[0,2]], M1[[0,2]], decimal=2)
        npt.assert_almost_equal(M0[1] - (2+1)*100e3*9.801, M1[1], decimal=-4)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)

        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = 1e3*np.array([100.0])
        self.inputs["blades_cm"] = np.array([2.0])
        self.inputs["blades_I"] = 1e3*np.array([1., 2., 3.])
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F1 = self.outputs["F_mb1"].flatten()
        M1 = self.outputs["M_mb2"].flatten()
        npt.assert_almost_equal(F0[:2], F1[:2], decimal=2)
        npt.assert_almost_equal(F0[2] - 100e3*9.801, F1[2], decimal=-4)
        npt.assert_almost_equal(M0[[0,2]], M1[[0,2]], decimal=2)
        npt.assert_almost_equal(M0[1] - (2+1)*100e3*9.801, M1[1], decimal=-4) #Lh1=1
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)

    def testRunRotatingGeared_noTilt(self):
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["gear_ratio"] = np.array([50.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        self.inputs["lss_mass_user"] = [0.]
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout(False)
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=False)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        self.assertAlmostEqual(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3**4 - 2.4**4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_aero_hub"] = g.reshape((3, 1))
        self.inputs["M_aero_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        t = 0.45
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r**2 - (r - t) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:3], 1.0 / (2*t*self.inputs["L_lss"][0]))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][3], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r**4 - (r - t) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r**4 - (r - t) ** 4))
        )

    def testRunRotatingGeared_withTilt(self):
        self.inputs["tilt"] = np.array([5.0])
        self.inputs["gear_ratio"] = np.array([50.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        self.inputs["lss_mass_user"] = [0.]
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout(False)
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=False)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        F0 = self.outputs["F_mb1"].flatten()
        M0 = self.outputs["M_mb2"].flatten()
        self.assertGreater(0.0, F0[0])
        self.assertGreater(0.0, F0[-1])
        # self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_mb1"][1], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"][[0, 2]], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_torq"], 0.0, decimal=2)
        self.assertAlmostEqual(
            self.outputs["lss_spring_constant"], 80.8e9 * np.pi * (3.3**4 - 2.4**4) / 32 / self.inputs["L_lss"], 4
        )

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_aero_hub"] = g.reshape((3, 1))
        self.inputs["M_aero_hub"] = 2 * g.reshape((3, 1))
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs["F_mb1"].flatten(), g + F0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_mb2"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["F_torq"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb1"], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[0], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[1], g[-1] * 1 + 2 * g[1] + M0[1], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_mb2"].flatten()[2], -g[1] * 1 + 2 * g[2], decimal=2)  # *1=*L_h1
        npt.assert_almost_equal(self.outputs["M_torq"].flatten(), np.r_[2 * g[0], 0.0, 0.0], decimal=2)

        r = 0.5 * 3.3
        t = 0.45
        self.assertAlmostEqual(self.outputs["lss_axial_load2stress"][0], 1.0 / (np.pi * (r**2 - (r - t) ** 2)))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][1:3], 1.0 / (2*t*self.inputs["L_lss"][0]))
        npt.assert_almost_equal(self.outputs["lss_axial_load2stress"][3], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_axial_load2stress"][4:], r / (0.25 * np.pi * (r**4 - (r - t) ** 4))
        )
        self.assertAlmostEqual(self.outputs["lss_shear_load2stress"][0], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][1], 0.0)
        self.assertGreater(self.outputs["lss_shear_load2stress"][2], 0.0)
        npt.assert_almost_equal(self.outputs["lss_shear_load2stress"][4:], 0.0)
        npt.assert_almost_equal(
            self.outputs["lss_shear_load2stress"][3], r / (0.5 * np.pi * (r**4 - (r - t) ** 4))
        )

    def testHSS_noTilt(self):
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["gear_ratio"] = np.array([50.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        F0 = self.outputs["F_generator"].flatten()
        M0 = self.outputs["M_generator"].flatten()
        self.assertGreater(0.0, F0[-1])
        self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_generator"].flatten()[:2], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten()[[0, 2]], 0.0, decimal=2)

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_aero_hub"] = g.reshape((3, 1))
        self.inputs["M_aero_hub"] = 2 * g.reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_almost_equal(self.outputs["F_generator"].flatten(), F0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten(), np.r_[2 * g[0] / 50.0, M0[1], 0.0], decimal=2)

    def testHSS_withTilt(self):
        self.inputs["tilt"] = np.array([5.0])
        self.inputs["gear_ratio"] = np.array([50.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        F0 = self.outputs["F_generator"].flatten()
        M0 = self.outputs["M_generator"].flatten()
        self.assertGreater(0.0, F0[0])
        self.assertGreater(0.0, F0[-1])
        self.assertGreater(0.0, M0[1])
        npt.assert_almost_equal(self.outputs["F_generator"].flatten()[1], 0.0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten()[[0, 2]], 0.0, decimal=2)

        g = np.array([30e2, 40e2, 50e2])
        self.inputs["F_aero_hub"] = g.reshape((3, 1))
        self.inputs["M_aero_hub"] = 2 * g.reshape((3, 1))
        self.compute_layout(False)
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_almost_equal(self.outputs["F_generator"].flatten(), F0, decimal=2)
        npt.assert_almost_equal(self.outputs["M_generator"].flatten(), np.r_[2 * g[0] / 50.0, M0[1], 0.0], decimal=2)

    def testShaftTheoryLSS(self):
        # https://www.engineersedge.com/calculators/torsional-stress-calculator.htm
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["F_aero_hub"] = np.zeros(3).reshape((3, 1))
        self.inputs["M_aero_hub"] = np.array([1e5, 0.0, 0.0]).reshape((3, 1))
        self.inputs["brake_mass"] = np.array([0.0])
        self.inputs["brake_I"] = np.zeros(3)
        self.inputs["generator_rotor_mass"] = np.array([0.0])
        self.inputs["cm_rotor"] = np.array([0.0])
        self.inputs["generator_rotor_I"] = np.zeros(6)
        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        myones = np.ones(5)
        self.inputs["lss_diameter"] = 5 * myones
        self.inputs["lss_wall_thickness"] = 0.5 * myones
        self.inputs["G"] = np.array([100e9])
        self.inputs["lss_rho"] = np.array([1e-6])
        self.inputs["lss_mass_user"] = [0.]
        self.inputs["hss_mass_user"] = [0.]
        self.compute_layout()
        myobj = ds.Hub_Rotor_LSS_Frame(n_dlcs=1, modeling_options=self.opt, direct_drive=True)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        J = 0.5 * np.pi * (2.5**4 - 2**4)
        sigma = 1e5 / J * 2.5
        npt.assert_almost_equal(self.outputs["lss_axial_stress"], 0.0, decimal=4)
        npt.assert_almost_equal(self.outputs["lss_shear_stress"].flatten(), np.r_[np.zeros(3), sigma], decimal=4)

    def testShaftTheoryHSS(self):
        # https://www.engineersedge.com/calculators/torsional-stress-calculator.htm
        self.inputs["tilt"] = np.array([0.0])
        self.inputs["gear_ratio"] = np.array([50.0])
        self.inputs["s_hss"] = np.array([0.0, 0.5, 1.0])
        self.inputs["M_aero_hub"] = np.array([1e5, 0.0, 0.0]).reshape((3, 1))
        self.inputs["s_generator"] = np.array([0.0])
        self.inputs["generator_mass"] = np.array([0.0])
        self.inputs["generator_I"] = np.zeros(3)
        self.inputs["brake_mass"] = np.array([0.0])
        self.inputs["brake_I"] = np.zeros(3)
        self.inputs["hub_system_mass"] = np.array([0.0])
        self.inputs["hub_system_cm"] = np.array([0.0])
        self.inputs["hub_system_I"] = np.zeros(6)
        self.inputs["blades_mass"] = np.array([0.0])
        self.inputs["blades_cm"] = np.array([0.0])
        self.inputs["blades_I"] = np.zeros(6)
        myones = np.ones(3)
        self.inputs["hss_diameter"] = 5 * myones
        self.inputs["hss_wall_thickness"] = 0.5 * myones
        self.inputs["G"] = np.array([100e9])
        self.inputs["hss_rho"] = np.array([1e-6])
        self.inputs["lss_mass_user"] = [0.]
        self.inputs["hss_mass_user"] = [0.]

        self.compute_layout()
        myobj = ds.HSS_Frame(modeling_options=self.opt, n_dlcs=1)
        myobj.compute(self.inputs, self.outputs)
        J = 0.5 * np.pi * (2.5**4 - 2**4)
        sigma = 1e5 / 50.0 / J * 2.5
        npt.assert_almost_equal(self.outputs["hss_axial_stress"], 0.0, decimal=4)
        npt.assert_almost_equal(self.outputs["hss_bending_stress"], 0.0, decimal=4)
        npt.assert_almost_equal(self.outputs["hss_shear_stress"].flatten(), sigma * np.ones(2), decimal=4)


if __name__ == "__main__":
    unittest.main()
