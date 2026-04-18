#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import openmdao.api as om

import wisdem.pyframe3dd.pyframe3dd as frame3dd
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector
from wisdem.commonse.utilities import find_nearest, nodal2sectional
from wisdem.commonse.cross_sections import Tube, IBeam
from wisdem.commonse.utilization_constraints import TubevonMisesStressUtilization

RIGID = 1
FREE = 0


def tube_prop(s, Di, ti):
    L = s.max() - s.min()

    def equal_pts(xi):
        if len(xi) < len(s) and len(xi) == 2:
            x = np.interp((s - s.min()) / L, [0, 1], xi)
        elif len(xi) == len(s):
            x = xi
        else:
            raise ValueError("Unknown grid of input", str(xi))
        return x

    D = equal_pts(Di)
    t = equal_pts(ti)
    return Tube(nodal2sectional(D)[0], nodal2sectional(t)[0])


class Hub_Rotor_LSS_Frame(om.ExplicitComponent):
    """
    Run structural analysis of hub system with the generator rotor and main (LSS) shaft.

    Parameters
    ----------
    tilt : float, [deg]
        Shaft tilt
    s_lss : numpy array[5], [m]
        Discretized s-coordinates along drivetrain, measured from bedplate (direct) or tower center (geared)
    lss_diameter : numpy array[2], [m]
        LSS outer diameter from hub to bearing 2
    lss_wall_thickness : numpy array[2], [m]
        LSS wall thickness
    hub_system_mass : float, [kg]
        Hub system mass
    hub_system_cm : float, [m]
        Hub system center of mass distance from hub flange
    hub_system_I : numpy array[6], [kg*m**2]
        Hub system moment of inertia
    F_aero_hub : numpy array[3, n_dlcs], [N]
        Aero-only force vector applied to the hub
    M_aero_hub : numpy array[3, n_dlcs], [N*m]
        Aero-only moment vector applied to the hub
    blades_mass : float, [kg]
        Mass of all blades
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate (direct) or tower center (geared)
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate (direct) or tower center (geared)
    s_rotor : float, [m]
        Generator rotor attachment to lss s-coordinate measured from bedplate (direct) or tower center (geared)
    generator_rotor_mass : float, [kg]
        Generator rotor mass
    generator_rotor_I : numpy array[3], [kg*m**2]
        Generator rotor moment of inertia (measured about its cm)
    gearbox_mass : float, [kg]
        Gearbox rotor mass
    gearbox_I : numpy array[3], [kg*m**2]
        Gearbox moment of inertia (measured about its cm)
    lss_E : float, [Pa]
        modulus of elasticity
    lss_G : float, [Pa]
        shear modulus
    lss_rho : float, [kg/m**3]
        material density
    lss_Xy : float, [Pa]
        yield stress

    Returns
    -------
    lss_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the low speed shaft froom T=(G*J/L)*theta
    torq_deflection : float, [m]
        Maximum deflection distance at rotor (direct) or gearbox (geared) attachment
    torq_angle : float, [rad]
        Maximum rotation angle at rotor (direct) or gearbox (geared) attachment
    torq_axial_stress : numpy array[5, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    torq_shear_stress : numpy array[5, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    torq_bending_stress : numpy array[5, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_lss_vonmises : numpy array[5, n_dlcs]
        Sigma_y/Von_Mises
    F_mb1 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 1 in hub c.s.
    F_mb2 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 2 in hub c.s.
    F_torq : numpy array[3, n_dlcs], [N]
        Force vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    M_mb1 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 1 in hub c.s.
    M_mb2 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 2 in hub c.s.
    M_torq : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    lss_axial_load2stress : numpy array[nFull-1,6], [m**2]
        Linear conversion factors between loads [Fx-z; Mx-z] and axial stress
    lss_shear_load2stress : numpy array[nFull-1,6], [m**2]
        Linear conversion factors between loads [Fx-z; Mx-z] and shear stress

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("direct_drive", default=True)
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_lss", val=np.zeros(5), units="m")
        self.add_input("lss_diameter", val=np.zeros(2), units="m")
        self.add_input("lss_wall_thickness", val=np.zeros(2), units="m")
        self.add_input("hub_system_mass", 0.0, units="kg")
        self.add_input("hub_system_cm", 0.0, units="m")
        self.add_input("hub_system_I", np.zeros(6), units="kg*m**2")
        self.add_input("F_aero_hub", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_aero_hub", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("blades_mass", 0.0, units="kg")
        self.add_input("blades_cm", 0.0, units="m")
        self.add_input("blades_I", np.zeros(6), units="kg*m**2")
        self.add_input("s_mb1", val=0.0, units="m")
        self.add_input("s_mb2", val=0.0, units="m")
        self.add_input("s_rotor", val=0.0, units="m")
        self.add_input("generator_rotor_mass", val=0.0, units="kg")
        self.add_input("generator_rotor_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("gearbox_mass", val=0.0, units="kg")
        self.add_input("gearbox_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("brake_mass", val=0.0, units="kg")
        self.add_input("brake_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("carrier_mass", val=0.0, units="kg")
        self.add_input("carrier_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("lss_E", val=0.0, units="Pa")
        self.add_input("lss_G", val=0.0, units="Pa")
        self.add_input("lss_rho", val=0.0, units="kg/m**3")
        self.add_input("lss_Xy", val=0.0, units="Pa")

        self.add_input("shaft_deflection_allowable", val=1.0, units="m")
        self.add_input("shaft_angle_allowable", val=1.0, units="rad")

        self.add_output("lss_spring_constant", 0.0, units="N*m/rad")
        self.add_output("torq_deflection", val=0.0, units="m")
        self.add_output("torq_angle", val=0.0, units="rad")
        self.add_output("lss_axial_stress", np.zeros((4, n_dlcs)), units="Pa")
        self.add_output("lss_shear_stress", np.zeros((4, n_dlcs)), units="Pa")
        self.add_output("constr_lss_vonmises", np.zeros((4, n_dlcs)))
        self.add_output("F_mb1", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("F_mb2", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("F_torq", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("M_mb1", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("M_mb2", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("M_torq", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("lss_axial_load2stress", val=np.zeros(6), units="m**2")
        self.add_output("lss_shear_load2stress", val=np.zeros(6), units="m**2")
        self.add_output("constr_shaft_deflection", 0.0)
        self.add_output("constr_shaft_angle", 0.0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0
        tilt = float(np.deg2rad(inputs["tilt"][0]))

        s_lss = inputs["s_lss"]
        D_lss = inputs["lss_diameter"]
        t_lss = inputs["lss_wall_thickness"]

        s_mb1 = float(inputs["s_mb1"][0])
        s_mb2 = float(inputs["s_mb2"][0])

        if self.options["direct_drive"]:
            s_rotor = float(inputs["s_rotor"][0])
            m_rotor = float(inputs["generator_rotor_mass"][0])
            I_rotor = inputs["generator_rotor_I"]

            m_brake = float(inputs["brake_mass"][0])
            I_brake = inputs["brake_I"]
        else:
            m_gearbox = float(inputs["gearbox_mass"][0])
            I_gearbox = inputs["gearbox_I"]

            m_carrier = float(inputs["carrier_mass"][0])
            I_carrier = inputs["carrier_I"]

        rho = float(inputs["lss_rho"][0])
        E = float(inputs["lss_E"][0])
        G = float(inputs["lss_G"][0])
        sigma_y = float(inputs["lss_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        m_hub = float(inputs["hub_system_mass"][0])
        cm_hub = Cup*float(inputs["hub_system_cm"][0])
        I_hub = inputs["hub_system_I"]
        m_blades = float(inputs['blades_mass'][0])
        cm_blades = Cup*float(inputs["blades_cm"][0])
        I_blades = inputs["blades_I"]
        m_blades_hub = m_blades + m_hub + 1e-10
        cm_blades_hub = (m_blades*cm_blades + m_hub*cm_hub) / m_blades_hub
        I_blades_hub = I_blades[:3] + I_hub[:3]
        F_hub = inputs["F_aero_hub"]
        M_hub = inputs["M_aero_hub"]

        torq_defl_allow = float(inputs["shaft_deflection_allowable"][0])
        torq_angle_allow = float(inputs["shaft_angle_allowable"][0])

        # ------- node data ----------------
        n = len(s_lss)
        inode = np.arange(1, n + 1)
        ynode = znode = rnode = np.zeros(n)
        xnode = Cup * s_lss.copy()
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        i1 = inode[find_nearest(xnode, Cup * s_mb1)]
        i2 = inode[find_nearest(xnode, Cup * s_mb2)]
        iadd = inode[1]
        # Differences between direct annd geared
        if self.options["direct_drive"]:
            itorq = inode[find_nearest(xnode, Cup * s_rotor)]
            m_torq = m_rotor
            I_torq = I_rotor

            m_add = m_brake
            I_add = I_brake
        else:
            itorq = inode[0]
            m_torq = m_gearbox - m_carrier
            I_torq = I_gearbox - I_carrier

            m_add = m_carrier
            I_add = I_carrier
        # ------------------------------------

        # ------ reaction data ------------
        # Reactions at main bearings
        rnode = np.r_[i1, i2, itorq]
        Rx = np.array([RIGID, FREE, FREE])  # Upwind bearing restricts translational
        Ry = np.array([RIGID, FREE, FREE])  # Upwind bearing restricts translational
        Rz = np.array([RIGID, FREE, FREE])  # Upwind bearing restricts translational
        Rxx = np.array([FREE, FREE, RIGID])  # Torque is absorbed by stator, so this is the best way to capture that
        Ryy = np.array([FREE, RIGID, FREE])  # downwind bearing carry moments
        Rzz = np.array([FREE, RIGID, FREE])  # downwind bearing carry moments
        reactions = frame3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        # s_lss goes from gearbox/tower to hub, D_lss and t_lss start at hub and go to gearbox
        lsscyl = tube_prop(s_lss, np.flip(D_lss), np.flip(t_lss))
        ielement = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)
        myones = np.ones(n - 1)
        Ax = lsscyl.Area
        As = lsscyl.Asx
        S = lsscyl.S
        C = lsscyl.C
        J0 = lsscyl.J0
        Jx = lsscyl.Ixx

        elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E * myones, G * myones, roll, rho * myones
        )

        lss_stiff = G * J0 / np.diff(s_lss)
        outputs["lss_spring_constant"] = 1.0 / np.sum(1.0 / lss_stiff)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add hub and generator rotor (direct) or gearbox (geared) extra mass ------------
        three0 = np.zeros(3).tolist()
        myframe.changeExtraNodeMass(
            np.r_[inode[-1], itorq, iadd],
            [m_blades_hub, m_torq, m_add],
            [I_blades_hub[0], I_torq[0], I_add[0]],
            [I_blades_hub[1], I_torq[1], I_add[1]],
            [I_blades_hub[2], I_torq[2], I_add[2]],
            three0,
            three0,
            three0,
            [cm_blades_hub, 0.0, 0.0],
            three0,
            three0,
            True,
        )
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        # myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gy = 0.0
        gx = -gravity * np.sin(tilt)
        gz = -gravity * np.cos(tilt)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads (should be in shaft c.s.)
            load.changePointLoads(
                [inode[-1]], [F_hub[0, k]], [F_hub[1, k]], [F_hub[2, k]], [M_hub[0, k]], [M_hub[1, k]], [M_hub[2, k]]
            )
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        # myframe.write('myframe1.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # Loop over DLCs and append to outputs
        rotor_gearbox_deflection = np.zeros(n_dlcs)
        rotor_gearbox_angle = np.zeros(n_dlcs)
        outputs["F_mb1"] = np.zeros((3, n_dlcs))
        outputs["F_mb2"] = np.zeros((3, n_dlcs))
        outputs["F_torq"] = np.zeros((3, n_dlcs))
        outputs["M_mb1"] = np.zeros((3, n_dlcs))
        outputs["M_mb2"] = np.zeros((3, n_dlcs))
        outputs["M_torq"] = np.zeros((3, n_dlcs))
        outputs["lss_axial_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["lss_shear_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["constr_lss_vonmises"] = np.zeros((n - 1, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at torq attachment
            rotor_gearbox_deflection[k] = np.sqrt(
                displacements.dx[k, itorq - 1] ** 2
                + displacements.dy[k, itorq - 1] ** 2
                + displacements.dz[k, itorq - 1] ** 2
            )
            rotor_gearbox_angle[k] = (
                displacements.dxrot[k, itorq - 1]
                + displacements.dyrot[k, itorq - 1]
                + displacements.dzrot[k, itorq - 1]
            )

            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            F = np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            M = np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments
            outputs["F_mb1"][:, k] = -1.0 * np.array([reactions.Fx[k, 0], reactions.Fy[k, 0], reactions.Fz[k, 0]])
            outputs["F_mb2"][:, k] = -1.0 * np.array([reactions.Fx[k, 1], reactions.Fy[k, 1], reactions.Fz[k, 1]])
            outputs["F_torq"][:, k] = -1.0 * np.array([reactions.Fx[k, 2], reactions.Fy[k, 2], reactions.Fz[k, 2]])
            outputs["M_mb1"][:, k] = -1.0 * np.array([reactions.Mxx[k, 0], reactions.Myy[k, 0], reactions.Mzz[k, 0]])
            outputs["M_mb2"][:, k] = -1.0 * np.array([reactions.Mxx[k, 1], reactions.Myy[k, 1], reactions.Mzz[k, 1]])
            outputs["M_torq"][:, k] = -1.0 * np.array([reactions.Mxx[k, 2], reactions.Myy[k, 2], reactions.Mzz[k, 2]])
            outputs["lss_axial_stress"][:, k] = np.abs(Fx) / Ax + M / S
            outputs["lss_shear_stress"][:, k] = 2.0 * F / As + np.abs(Mxx) / C
            hoop = np.zeros(F.shape)

            outputs["constr_lss_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["lss_axial_stress"][:, k],
                hoop,
                outputs["lss_shear_stress"][:, k],
                gamma,
                sigma_y,
            )
        outputs["torq_deflection"] = rotor_gearbox_deflection.max()
        outputs["torq_angle"] = rotor_gearbox_angle.max()
        outputs["constr_shaft_deflection"] = gamma * outputs["torq_deflection"] / torq_defl_allow
        outputs["constr_shaft_angle"] = gamma * outputs["torq_angle"] / torq_angle_allow

        # Load->stress conversion for fatigue
        try: # Numpy v1/2 clash
            cross_section_xy = 2.0 * np.trapezoid(t_lss, s_lss)
        except AttributeError:
            cross_section_xy = 2.0 * np.trapz(t_lss, s_lss)
        ax_load2stress = np.zeros(6)
        ax_load2stress[0] = 1.0 / Ax[0]
        ax_load2stress[1] = 1.0 / cross_section_xy
        ax_load2stress[2] = 1.0 / cross_section_xy
        ax_load2stress[4] = 1.0 / S[0]
        ax_load2stress[5] = 1.0 / S[0]
        sh_load2stress = np.zeros(6)
        sh_load2stress[1] = 1.0 / As[0]
        sh_load2stress[2] = 1.0 / As[0]
        sh_load2stress[3] = 1.0 / C[0]
        outputs["lss_axial_load2stress"] = ax_load2stress
        outputs["lss_shear_load2stress"] = sh_load2stress


class HSS_Frame(om.ExplicitComponent):
    """
    Run structural analysis of high speed shaft (HSS) between gearbox and generator (only for geared configurations).

    Parameters
    ----------
    tilt : float, [deg]
        Shaft tilt
    s_hss : numpy array[3], [m]
        Discretized s-coordinates along drivetrain, measured from bedplate (direct) or tower center (geared)
    hss_diameter : numpy array[2], [m]
        Lss discretized diameter values at coordinates
    hss_wall_thickness : numpy array[2], [m]
        Lss discretized thickness values at coordinates
    M_aero_hub : numpy array[3, n_dlcs], [N*m]
        Aero-only moment vector applied to the hub
    m_generator : float, [kg]
        Gearbox rotor mass
    cm_generator : float, [kg]
        Gearbox center of mass (measured from tower center)
    I_generator : numpy array[3], [kg*m**2]
        Gearbox moment of inertia (measured about its cm)
    hss_E : float, [Pa]
        modulus of elasticity
    hss_G : float, [Pa]
        shear modulus
    hss_rho : float, [kg/m**3]
        material density
    hss_Xy : float, [Pa]
        yield stress

    Returns
    -------
    hss_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the high speed shaft froom T=(G*J/L)*theta
    hss_axial_stress : numpy array[5, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    hss_shear_stress : numpy array[5, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    hss_bending_stress : numpy array[5, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_hss_vonmises : numpy array[5, n_dlcs]
        Sigma_y/Von_Mises
    F_generator : numpy array[3, n_dlcs], [N]
        Force vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    M_generator : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_hss", val=np.zeros(3), units="m")
        self.add_input("hss_diameter", val=np.zeros(2), units="m")
        self.add_input("hss_wall_thickness", val=np.zeros(2), units="m")
        self.add_input("M_aero_hub", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("gear_ratio", val=1.0)
        self.add_input("s_generator", val=0.0, units="m")
        self.add_input("generator_mass", val=0.0, units="kg")
        self.add_input("generator_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("brake_mass", val=0.0, units="kg")
        self.add_input("brake_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("hss_E", val=0.0, units="Pa")
        self.add_input("hss_G", val=0.0, units="Pa")
        self.add_input("hss_rho", val=0.0, units="kg/m**3")
        self.add_input("hss_Xy", val=0.0, units="Pa")

        self.add_output("hss_spring_constant", 0.0, units="N*m/rad")
        self.add_output("hss_axial_stress", np.zeros((2, n_dlcs)), units="Pa")
        self.add_output("hss_shear_stress", np.zeros((2, n_dlcs)), units="Pa")
        self.add_output("hss_bending_stress", np.zeros((2, n_dlcs)), units="Pa")
        self.add_output("constr_hss_vonmises", np.zeros((2, n_dlcs)))
        self.add_output("F_generator", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("M_generator", val=np.zeros((3, n_dlcs)), units="N*m")

    def compute(self, inputs, outputs):
        # Unpack inputs
        tilt = float(np.deg2rad(inputs["tilt"][0]))

        s_hss = inputs["s_hss"]
        D_hss = inputs["hss_diameter"]
        t_hss = inputs["hss_wall_thickness"]

        s_generator = float(inputs["s_generator"][0])
        m_generator = float(inputs["generator_mass"][0])
        I_generator = inputs["generator_I"]

        m_brake = float(inputs["brake_mass"][0])
        I_brake = inputs["brake_I"]

        rho = float(inputs["hss_rho"][0])
        E = float(inputs["hss_E"][0])
        G = float(inputs["hss_G"][0])
        sigma_y = float(inputs["hss_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        M_hub = inputs["M_aero_hub"]
        gear_ratio = float(inputs["gear_ratio"][0])

        # ------- node data ----------------
        n = len(s_hss)
        inode = np.arange(1, n + 1)
        ynode = znode = rnode = np.zeros(n)
        xnode = s_hss.copy()
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # ------------------------------------

        # ------ reaction data ------------
        # Reaction at generator attachment
        rnode = [inode[0]]
        Rx = Ry = Rz = Rxx = Ryy = Rzz = np.array([RIGID])
        reactions = frame3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        hsscyl = tube_prop(s_hss, np.flip(D_hss), np.flip(t_hss))
        ielement = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)
        myones = np.ones(n - 1)
        Ax = hsscyl.Area
        As = hsscyl.Asx
        S = hsscyl.S
        C = hsscyl.C
        J0 = hsscyl.J0
        Jx = hsscyl.Ixx

        elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E * myones, G * myones, roll, rho * myones
        )

        hss_stiff = G * J0 / np.diff(s_hss)
        outputs["hss_spring_constant"] = 1.0 / np.sum(1.0 / hss_stiff)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add brake hub and generator rotor (direct) or generator (geared) extra mass ------------
        myframe.changeExtraNodeMass(
            np.r_[inode[1], inode[0]],
            [m_brake, m_generator],
            [I_brake[0], I_generator[0]],
            [I_brake[1], I_generator[1]],
            [I_brake[2], I_generator[2]],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, s_generator - s_hss[-1]],
            [0.0, 0.0],
            [0.0, 0.0],
            True,
        )
        # ------------------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gy = 0.0
        gx = -gravity * np.sin(tilt)
        gz = -gravity * np.cos(tilt)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            Fx = Fy = Fz = My = Mz = np.zeros(1)
            Mx = M_hub[0] / gear_ratio
            load.changePointLoads([inode[-1]], Fx, Fy, Fz, Mx, My, Mz)
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        # myframe.write('myframe2.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # Loop over DLCs and append to outputs
        outputs["F_generator"] = np.zeros((3, n_dlcs))
        outputs["M_generator"] = np.zeros((3, n_dlcs))
        outputs["hss_axial_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["hss_shear_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["hss_bending_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["constr_hss_vonmises"] = np.zeros((n - 1, n_dlcs))
        for k in range(n_dlcs):
            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            F = np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            M = np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments
            outputs["F_generator"][:, k] = -1.0 * np.array([reactions.Fx[k, 0], reactions.Fy[k, 0], reactions.Fz[k, 0]])
            outputs["M_generator"][:, k] = -1.0 * np.array(
                [reactions.Mxx[k, 0], reactions.Myy[k, 0], reactions.Mzz[k, 0]]
            )
            outputs["hss_axial_stress"][:, k] = np.abs(Fx) / Ax + M / S
            outputs["hss_shear_stress"][:, k] = 2.0 * F / As + np.abs(Mxx) / C
            hoop = np.zeros(F.shape)

            outputs["constr_hss_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["hss_axial_stress"][:, k],
                hoop,
                outputs["hss_shear_stress"][:, k],
                gamma,
                sigma_y,
            )


class Nose_Stator_Bedplate_Frame(om.ExplicitComponent):
    """
    Run structural analysis of nose/turret with the generator stator and bedplate

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    tilt : float, [deg]
        Lss tilt
    s_nose : numpy array[5], [m]
        Discretized s-coordinates along drivetrain, measured from bedplate
    nose_diameter : numpy array[2], [m]
        Nose outer diameter from bearing 1 to bedplate
    nose_wall_thickness : numpy array[2], [m]
        Nose wall thickness
    x_bedplate : numpy array[12], [m]
        Bedplate centerline x-coordinates
    z_bedplate : numpy array[12], [m]
        Bedplate centerline z-coordinates
    x_bedplate_inner : numpy array[12], [m]
        Bedplate lower curve x-coordinates
    z_bedplate_inner : numpy array[12], [m]
        Bedplate lower curve z-coordinates
    x_bedplate_outer : numpy array[12], [m]
        Bedplate outer curve x-coordinates
    z_bedplate_outer : numpy array[12], [m]
        Bedplate outer curve z-coordinates
    D_bedplate : numpy array[12], [m]
        Bedplate diameters
    t_bedplate : numpy array[12], [m]
        Bedplate wall thickness (mirrors input)
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate
    mb1_mass : float, [kg]
        component mass
    mb1_I : numpy array[3], [kg*m**2]
        component I
    mb1_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    mb2_mass : float, [kg]
        component mass
    mb2_I : numpy array[3], [kg*m**2]
        component I
    mb2_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    s_stator : float, [m]
        Generator stator attachment to lss s-coordinate measured from bedplate
    generator_stator_mass : float, [kg]
        Generator stator mass
    generator_stator_I : numpy array[3], [kg*m**2]
        Generator stator moment of inertia (measured about cm)
    F_mb1 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 1 in hub c.s.
    F_mb2 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 2 in hub c.s.
    M_mb1 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 1 in hub c.s.
    M_mb2 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 2 in hub c.s.
    other_mass : float, [kg]
        Mass of other nacelle components that rest on mainplate
    bedplate_E : float, [Pa]
        modulus of elasticity
    bedplate_G : float, [Pa]
        shear modulus
    bedplate_rho : float, [kg/m**3]
        material density
    bedplate_Xy : float, [Pa]
        yield stress

    Returns
    -------
    mb1_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 1
    mb2_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 2
    stator_deflection : float, [m]
        Maximum deflection distance at stator attachment
    mb1_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 1
    mb2_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 2
    stator_angle : float, [rad]
        Maximum rotation angle at stator attachment
    base_F : numpy array[3, n_dlcs], [N]
        Total reaction force at bedplate base in tower top coordinate system
    base_M : numpy array[3, n_dlcs], [N*m]
        Total reaction moment at bedplate base in tower top coordinate system
    bedplate_nose_axial_stress : numpy array[12+3, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    bedplate_nose_shear_stress : numpy array[12+3, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    bedplate_nose_bending_stress : numpy array[12+3, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_bedplate_vonmises : numpy array[12+3, n_dlcs]
        Sigma_y/Von_Mises
    constr_mb1_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 1 (should be <1)
    constr_mb2_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 2 (should be <1)

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_nose", val=np.zeros(5), units="m")
        self.add_input("nose_diameter", np.zeros(2), units="m")
        self.add_input("nose_wall_thickness", np.zeros(2), units="m")
        self.add_input("x_bedplate", val=np.zeros(12), units="m")
        self.add_input("z_bedplate", val=np.zeros(12), units="m")
        self.add_input("x_bedplate_inner", val=np.zeros(12), units="m")
        self.add_input("z_bedplate_inner", val=np.zeros(12), units="m")
        self.add_input("x_bedplate_outer", val=np.zeros(12), units="m")
        self.add_input("z_bedplate_outer", val=np.zeros(12), units="m")
        self.add_input("D_bedplate", val=np.zeros(12), units="m")
        self.add_input("t_bedplate", val=np.zeros(12), units="m")
        self.add_input("s_mb1", val=0.0, units="m")
        self.add_input("s_mb2", val=0.0, units="m")
        self.add_input("mb1_mass", 0.0, units="kg")
        self.add_input("mb1_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb1_max_defl_ang", 0.0, units="rad")
        self.add_input("mb2_mass", 0.0, units="kg")
        self.add_input("mb2_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb2_max_defl_ang", 0.0, units="rad")
        self.add_input("s_stator", val=0.0, units="m")
        self.add_input("generator_stator_mass", val=0.0, units="kg")
        self.add_input("generator_stator_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("F_mb1", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_mb2", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_mb1", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_mb2", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("other_mass", val=0.0, units="kg")
        self.add_input("bedplate_E", val=0.0, units="Pa")
        self.add_input("bedplate_G", val=0.0, units="Pa")
        self.add_input("bedplate_rho", val=0.0, units="kg/m**3")
        self.add_input("bedplate_Xy", val=0.0, units="Pa")

        self.add_input("stator_deflection_allowable", val=1.0, units="m")
        self.add_input("stator_angle_allowable", val=1.0, units="rad")

        self.add_output("mb1_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("mb2_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("stator_deflection", val=0.0, units="m")
        self.add_output("mb1_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("mb2_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("stator_angle", val=0.0, units="rad")
        self.add_output("base_F", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("base_M", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("bedplate_nose_axial_stress", np.zeros((12 + 3, n_dlcs)), units="Pa")
        self.add_output("bedplate_nose_shear_stress", np.zeros((12 + 3, n_dlcs)), units="Pa")
        self.add_output("bedplate_nose_bending_stress", np.zeros((12 + 3, n_dlcs)), units="Pa")
        self.add_output("constr_bedplate_vonmises", np.zeros((12 + 3, n_dlcs)))
        self.add_output("constr_mb1_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_mb2_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_stator_deflection", 0.0)
        self.add_output("constr_stator_angle", 0.0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0
        tiltD = float(inputs["tilt"][0])
        tiltR = np.deg2rad(tiltD)

        x_c = inputs["x_bedplate"]
        z_c = inputs["z_bedplate"]
        x_inner = inputs["x_bedplate_inner"]
        z_inner = inputs["z_bedplate_inner"]
        x_outer = inputs["x_bedplate_outer"]
        z_outer = inputs["z_bedplate_outer"]
        D_bed = inputs["D_bedplate"]
        t_bed = inputs["t_bedplate"]

        s_nose = inputs["s_nose"][1:]  # First point duplicated with bedplate
        D_nose = inputs["nose_diameter"]
        t_nose = inputs["nose_wall_thickness"]
        x_nose = s_nose.copy()
        x_nose *= Cup
        x_nose += x_c[-1]

        s_mb1 = float(inputs["s_mb1"][0])
        s_mb2 = float(inputs["s_mb2"][0])
        m_mb1 = float(inputs["mb1_mass"][0])
        m_mb2 = float(inputs["mb2_mass"][0])
        I_mb1 = inputs["mb1_I"]
        I_mb2 = inputs["mb2_I"]

        s_stator = float(inputs["s_stator"][0])
        m_stator = float(inputs["generator_stator_mass"][0])
        I_stator = inputs["generator_stator_I"]

        rho = float(inputs["bedplate_rho"][0])
        E = float(inputs["bedplate_E"][0])
        G = float(inputs["bedplate_G"][0])
        sigma_y = float(inputs["bedplate_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        F_mb1 = inputs["F_mb1"]
        F_mb2 = inputs["F_mb2"]
        M_mb1 = inputs["M_mb1"]
        M_mb2 = inputs["M_mb2"]

        m_other = float(inputs["other_mass"][0])
        stator_defl_allow = float(inputs["stator_deflection_allowable"][0])
        stator_angle_allow = float(inputs["stator_angle_allowable"][0])

        # ------- node data ----------------
        n = len(x_c) + len(x_nose)
        inode = np.arange(1, n + 1)
        ynode = rnode = np.zeros(n)
        xnode = np.r_[x_c, x_nose]
        znode = np.r_[z_c, z_c[-1] * np.ones(x_nose.shape)]
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        inose = len(x_c)
        istator = inode[find_nearest(xnode, Cup * s_stator + x_c[-1])]
        i1 = inode[find_nearest(xnode, Cup * s_mb1 + x_c[-1])]
        i2 = inode[find_nearest(xnode, Cup * s_mb2 + x_c[-1])]
        # ------------------------------------

        # ------ reaction data ------------
        # Rigid base
        rnode = [int(inode[0])]
        rk = np.array([RIGID])
        reactions = frame3dd.ReactionData(rnode, rk, rk, rk, rk, rk, rk, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        bedcyl = tube_prop(x_c, D_bed, t_bed)
        nosecyl = tube_prop(inputs["s_nose"], np.flip(D_nose), np.flip(t_nose))
        ielement = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)
        myones = np.ones(n - 1)
        Ax = np.r_[bedcyl.Area, nosecyl.Area]
        As = np.r_[bedcyl.Asx, nosecyl.Asx]
        S = np.r_[bedcyl.S, nosecyl.S]
        C = np.r_[bedcyl.C, nosecyl.C]
        J0 = np.r_[bedcyl.J0, nosecyl.J0]
        Ixx = np.r_[bedcyl.Ixx, nosecyl.Ixx]

        elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Ixx, Ixx, E * myones, G * myones, roll, rho * myones
        )
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add misc nacelle components at base and stator extra mass ------------
        myframe.changeExtraNodeMass(
            np.r_[inode[0], istator, i1, i2],
            [m_other, m_stator, m_mb1, m_mb2],
            [0.0, I_stator[0], I_mb1[0], I_mb2[0]],
            [0.0, I_stator[1], I_mb1[1], I_mb2[1]],
            [0.0, I_stator[2], I_mb1[2], I_mb2[2]],
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
            [0.0, 0.0, 0.0, 0.0],
            np.zeros(4),
            np.zeros(4),
            True,
        )
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        # myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gy = 0.0
        gx = -gravity * np.sin(tiltR)
        gz = -gravity * np.cos(tiltR)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            F_12 = np.c_[F_mb2[:, k], F_mb1[:, k]]
            M_12 = np.c_[M_mb2[:, k], M_mb1[:, k]]
            load.changePointLoads(np.r_[i2, i1], F_12[0, :], F_12[1, :], F_12[2, :], M_12[0, :], M_12[1, :], M_12[2, :])
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        # myframe.write('myframe3.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # ------------ Bedplate "curved beam" geometry for post-processing -------------
        # Need to compute neutral axis, so shift points such that bedplate top is at x=0
        R_c = np.sqrt((x_c - x_c[-1]) ** 2 + z_c**2)
        Ro = np.sqrt((x_outer - x_c[-1]) ** 2 + z_outer**2)
        Ri = np.sqrt((x_inner - x_c[-1]) ** 2 + z_inner**2)
        r_bed_o = 0.5 * D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed = np.pi * (r_bed_o**2 - r_bed_i**2)

        # Radius of the neutral axis
        # http://faculty.fairfield.edu/wdornfeld/ME311/BasicStressEqns-DBWallace.pdf
        R_n = A_bed / (2 * np.pi * (np.sqrt(R_c**2 - r_bed_i**2) - np.sqrt(R_c**2 - r_bed_o**2)))
        e_cn = R_c - R_n
        # ------------------------------------

        # Loop over DLCs and append to outputs
        outputs["mb1_deflection"] = np.zeros(n_dlcs)
        outputs["mb2_deflection"] = np.zeros(n_dlcs)
        stator_deflection = np.zeros(n_dlcs)
        outputs["mb1_angle"] = np.zeros(n_dlcs)
        outputs["mb2_angle"] = np.zeros(n_dlcs)
        stator_angle = np.zeros(n_dlcs)
        outputs["base_F"] = np.zeros((3, n_dlcs))
        outputs["base_M"] = np.zeros((3, n_dlcs))
        outputs["bedplate_nose_axial_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["bedplate_nose_shear_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["bedplate_nose_bending_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["constr_bedplate_vonmises"] = np.zeros((n - 1, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at bearings- how to sum up rotation angles?
            outputs["mb1_deflection"][k] = np.sqrt(
                displacements.dx[k, i1 - 1] ** 2 + displacements.dy[k, i1 - 1] ** 2 + displacements.dz[k, i1 - 1] ** 2
            )
            outputs["mb2_deflection"][k] = np.sqrt(
                displacements.dx[k, i2 - 1] ** 2 + displacements.dy[k, i2 - 1] ** 2 + displacements.dz[k, i2 - 1] ** 2
            )
            stator_deflection[k] = np.sqrt(
                displacements.dx[k, istator - 1] ** 2
                + displacements.dy[k, istator - 1] ** 2
                + displacements.dz[k, istator - 1] ** 2
            )
            outputs["mb1_angle"][k] = (
                displacements.dxrot[k, i1 - 1] + displacements.dyrot[k, i1 - 1] + displacements.dzrot[k, i1 - 1]
            )
            outputs["mb2_angle"][k] = (
                displacements.dxrot[k, i2 - 1] + displacements.dyrot[k, i2 - 1] + displacements.dzrot[k, i2 - 1]
            )
            stator_angle[k] = (
                displacements.dxrot[k, istator - 1]
                + displacements.dyrot[k, istator - 1]
                + displacements.dzrot[k, istator - 1]
            )

            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            F = np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            M = np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments
            F_base_k = DirectionVector(-reactions.Fx[k, :].sum(), -reactions.Fy[k, :].sum(), -reactions.Fz[k, :].sum())
            M_base_k = DirectionVector(
                -reactions.Mxx[k, :].sum(), -reactions.Myy[k, :].sum(), -reactions.Mzz[k, :].sum()
            )

            # Rotate vector from tilt axes to yaw/tower axes
            outputs["base_F"][:, k] = F_base_k.hubToYaw(-tiltD).toArray()
            outputs["base_M"][:, k] = M_base_k.hubToYaw(-tiltD).toArray()

            outputs["bedplate_nose_axial_stress"][:, k] = np.abs(Fx) / Ax + M / S
            outputs["bedplate_nose_shear_stress"][:, k] = 2.0 * F / As + np.abs(Mxx) / C

            Bending_stress_outer = M[: (inose - 1)] * nodal2sectional((Ro - R_n) / (A_bed * e_cn * Ro))[0]
            # Bending_stress_inner = M[:(inose-1)] * nodal2sectional( (R_n-Ri) / (A_bed*e_cn*Ri) )[0]
            outputs["bedplate_nose_bending_stress"][: (inose - 1), k] = Bending_stress_outer

            outputs["constr_bedplate_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["bedplate_nose_axial_stress"][:, k],
                outputs["bedplate_nose_bending_stress"][:, k],
                outputs["bedplate_nose_shear_stress"][:, k],
                gamma,
                sigma_y,
            )

        # Evaluate bearing limits
        outputs["constr_mb1_defl"] = np.abs(outputs["mb1_angle"] / inputs["mb1_max_defl_ang"])
        outputs["constr_mb2_defl"] = np.abs(outputs["mb2_angle"] / inputs["mb2_max_defl_ang"])
        outputs["stator_deflection"] = stator_deflection.max()
        outputs["stator_angle"] = np.abs(stator_angle).max()
        outputs["constr_stator_deflection"] = gamma * outputs["stator_deflection"] / stator_defl_allow
        outputs["constr_stator_angle"] = gamma * outputs["stator_angle"] / stator_angle_allow


class Bedplate_IBeam_Frame(om.ExplicitComponent):
    """
    Run structural analysis of bedplate in geared configuration with 2 X 2 parallel I beams providing support

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    tilt : float, [deg]
        Lss tilt
    bedplate_flange_width : float, [m]
        Bedplate is two parallel I beams, this is the flange width
    bedplate_flange_thickness : float, [m]
        Bedplate is two parallel I beams, this is the flange thickness
    bedplate_web_thickness : float, [m]
        Bedplate is two parallel I beams, this is the web thickness
    bedplate_web_height : float, [m]
        Bedplate is two parallel I beams, this is the web height
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate
    mb1_mass : float, [kg]
        component mass
    mb1_I : numpy array[3], [kg*m**2]
        component I
    mb1_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    mb2_mass : float, [kg]
        component mass
    mb2_I : numpy array[3], [kg*m**2]
        component I
    mb2_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    F_mb1 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 1 in hub c.s.
    F_mb2 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 2 in hub c.s.
    M_mb1 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 1 in hub c.s.
    M_mb2 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 2 in hub c.s.
    other_mass : float, [kg]
        Mass of other nacelle components that rest on mainplate
    E : float, [Pa]
        modulus of elasticity
    G : float, [Pa]
        shear modulus
    rho : float, [kg/m**3]
        material density
    sigma_y : float, [Pa]
        yield stress

    Returns
    -------
    mb1_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 1
    mb2_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 2
    bedplate_deflection : float, [m]
        Maximum deflection distance at bedplate I-beam tip
    mb1_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 1
    mb2_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 2
    bedplate_angle : float, [rad]
        Maximum rotation angle at bedplate I-beam tip
    base_F : numpy array[3, n_dlcs], [N]
        Total reaction force at bedplate base in tower top coordinate system
    base_M : numpy array[3, n_dlcs], [N*m]
        Total reaction moment at bedplate base in tower top coordinate system
    bedplate_axial_stress : numpy array[22, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    bedplate_shear_stress : numpy array[22, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    bedplate_bending_stress : numpy array[22, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_bedplate_vonmises : numpy array[22, n_dlcs]
        Sigma_y/Von_Mises
    constr_mb1_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 1 (should be <1)
    constr_mb2_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 2 (should be <1)

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("D_top", 0.0, units="m")
        self.add_input("s_drive", val=np.zeros(12), units="m")
        self.add_input("bedplate_flange_width", val=0.0, units="m")
        self.add_input("bedplate_flange_thickness", val=0.0, units="m")
        self.add_input("bedplate_web_height", val=0.0, units="m")
        self.add_input("bedplate_web_thickness", val=0.0, units="m")
        self.add_input("s_mb1", val=0.0, units="m")
        self.add_input("s_mb2", val=0.0, units="m")
        self.add_input("mb1_mass", 0.0, units="kg")
        self.add_input("mb1_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb1_max_defl_ang", 0.0, units="rad")
        self.add_input("mb2_mass", 0.0, units="kg")
        self.add_input("mb2_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb2_max_defl_ang", 0.0, units="rad")
        self.add_input("s_gearbox", val=0.0, units="m")
        self.add_input("s_generator", val=0.0, units="m")
        self.add_input("F_mb1", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_mb2", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_torq", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_generator", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_mb1", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_mb2", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_torq", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_generator", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("other_mass", val=0.0, units="kg")
        self.add_input("bedplate_E", val=0.0, units="Pa")
        self.add_input("bedplate_G", val=0.0, units="Pa")
        self.add_input("bedplate_rho", val=0.0, units="kg/m**3")
        self.add_input("bedplate_Xy", val=0.0, units="Pa")

        self.add_input("stator_deflection_allowable", val=1.0, units="m")
        self.add_input("stator_angle_allowable", val=1.0, units="rad")

        self.add_output("mb1_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("mb2_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("stator_deflection", val=0.0, units="m")
        self.add_output("mb1_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("mb2_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("stator_angle", val=0.0, units="rad")
        self.add_output("base_F", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("base_M", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("bedplate_axial_stress", np.zeros((22, n_dlcs)), units="Pa")
        self.add_output("bedplate_shear_stress", np.zeros((22, n_dlcs)), units="Pa")
        self.add_output("bedplate_bending_stress", np.zeros((22, n_dlcs)), units="Pa")
        self.add_output("constr_bedplate_vonmises", np.zeros((22, n_dlcs)))
        self.add_output("constr_mb1_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_mb2_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_stator_deflection", 0.0)
        self.add_output("constr_stator_angle", 0.0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0
        D_top = float(inputs["D_top"][0])
        tiltD = float(inputs["tilt"][0])
        tiltR = np.deg2rad(tiltD)
        s_drive = inputs["s_drive"]

        s_gear = float(inputs["s_gearbox"][0])
        s_gen = float(inputs["s_generator"][0])

        s_mb1 = float(inputs["s_mb1"][0])
        s_mb2 = float(inputs["s_mb2"][0])
        m_mb1 = float(inputs["mb1_mass"][0])
        m_mb2 = float(inputs["mb2_mass"][0])
        I_mb1 = inputs["mb1_I"]
        I_mb2 = inputs["mb2_I"]

        bed_w_flange = float(inputs["bedplate_flange_width"][0])
        bed_t_flange = float(inputs["bedplate_flange_thickness"][0])
        bed_h_web = float(inputs["bedplate_web_height"][0])
        bed_t_web = float(inputs["bedplate_web_thickness"][0])

        rho = float(inputs["bedplate_rho"][0])
        E = float(inputs["bedplate_E"][0])
        G = float(inputs["bedplate_G"][0])
        sigma_y = float(inputs["bedplate_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        F_mb1 = inputs["F_mb1"]
        F_mb2 = inputs["F_mb2"]
        F_gear = inputs["F_torq"]
        F_gen = inputs["F_generator"]
        M_mb1 = inputs["M_mb1"]
        M_mb2 = inputs["M_mb2"]
        M_gear = inputs["M_torq"]
        M_gen = inputs["M_generator"]

        m_other = float(inputs["other_mass"][0])
        stator_defl_allow = float(inputs["stator_deflection_allowable"][0])
        stator_angle_allow = float(inputs["stator_angle_allowable"][0])

        # ------- node data ----------------
        n = len(s_drive)
        inode = np.arange(1, 3 * n + 1)
        ynode = 0.25 * D_top * np.r_[np.zeros(n), np.ones(n), -np.ones(n)]
        xnode = s_drive * np.cos(tiltR)
        xnode = Cup * np.r_[xnode, xnode, xnode]
        znode = rnode = np.zeros(3 * n)
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        igenerator = inode[find_nearest(s_drive, s_gen)]
        igearbox = inode[find_nearest(s_drive, s_gear)]
        itower = inode[find_nearest(s_drive, 0.0)]
        i1 = inode[find_nearest(s_drive, s_mb1)]
        i2 = inode[find_nearest(s_drive, s_mb2)]
        # ------------------------------------

        # ------ reaction data ------------
        # Rigid base
        rnode = np.int_(np.r_[itower + n, itower + 2 * n])
        rk = np.array([RIGID, RIGID])
        reactions = frame3dd.ReactionData(rnode, rk, rk, rk, rk, rk, rk, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        # 2 parallel I-beams, with inner line of points to receive loads & distribute
        # No connections on ghost nodes, just connections to I-beams at same xval
        # Elements connect the lines, then connect the I-beams to the center nodes
        myI = IBeam(bed_w_flange, bed_t_flange, bed_h_web, bed_t_web)

        N1i = np.arange(1, n)
        N2i = np.arange(2, n + 1)
        Ni = np.arange(1, n + 1)
        N1 = np.r_[N1i + n, N1i + 2 * n, Ni + n, Ni + 2 * n]
        N2 = np.r_[N2i + n, N2i + 2 * n, Ni, Ni]
        ielement = np.arange(1, N1.size + 1)
        roll = np.zeros(N1.size)
        myones = np.ones(N1i.size)
        plate1s = np.ones(Ni.size)

        A_plate = As_plate = S_plate = C_plate = I_plate = 1e3 * plate1s
        rho_plate = 1e-6 * plate1s
        E_plate = G_plate = 1e16 * plate1s

        Ax = np.r_[myI.Area * myones, myI.Area * myones, A_plate, A_plate]
        Asy = np.r_[myI.Asx * myones, myI.Asx * myones, As_plate, As_plate]
        Asz = np.r_[myI.Asy * myones, myI.Asy * myones, As_plate, As_plate]
        Sy = np.r_[myI.Sx * myones, myI.Sx * myones, S_plate, S_plate]
        Sz = np.r_[myI.Sy * myones, myI.Sy * myones, S_plate, S_plate]
        C = np.r_[myI.C * myones, myI.C * myones, C_plate, C_plate]
        J0 = np.r_[myI.J0 * myones, myI.J0 * myones, I_plate, I_plate]
        Jy = np.r_[myI.Ixx * myones, myI.Ixx * myones, I_plate, I_plate]
        Jz = np.r_[myI.Iyy * myones, myI.Iyy * myones, I_plate, I_plate]
        myE = np.r_[E * myones, E * myones, E_plate, E_plate]
        myG = np.r_[G * myones, G * myones, G_plate, G_plate]
        myrho = np.r_[rho * myones, rho * myones, rho_plate, rho_plate]

        elements = frame3dd.ElementData(ielement, N1, N2, Ax, Asy, Asz, J0, Jy, Jz, myE, myG, roll, myrho)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add misc nacelle components at base and stator extra mass ------------
        myframe.changeExtraNodeMass(
            np.r_[itower, i1, i2],
            [m_other, m_mb1, m_mb2],
            [0.0, I_mb1[0], I_mb2[0]],
            [0.0, I_mb1[1], I_mb2[1]],
            [0.0, I_mb1[2], I_mb2[2]],
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            True,
        )
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        # myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gx = gy = 0.0
        gz = -gravity
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            F_ext = np.c_[F_mb2[:, k], F_mb1[:, k], F_gear[:, k], F_gen[:, k]]
            M_ext = np.c_[M_mb2[:, k], M_mb1[:, k], M_gear[:, k], M_gen[:, k]]
            F_rot = DirectionVector(F_ext[0, :], F_ext[1, :], F_ext[2, :]).hubToYaw(-tiltD).toArray()
            M_rot = DirectionVector(M_ext[0, :], M_ext[1, :], M_ext[2, :]).hubToYaw(-tiltD).toArray()
            load.changePointLoads(
                np.r_[i2, i1, igearbox, igenerator],
                F_rot[:, 0],
                F_rot[:, 1],
                F_rot[:, 2],
                M_rot[:, 0],
                M_rot[:, 1],
                M_rot[:, 2],
            )
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        # myframe.write('myframe4.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # Loop over DLCs and append to outputs
        outputs["mb1_deflection"] = np.zeros(n_dlcs)
        outputs["mb2_deflection"] = np.zeros(n_dlcs)
        outputs["mb1_angle"] = np.zeros(n_dlcs)
        outputs["mb2_angle"] = np.zeros(n_dlcs)
        outputs["base_F"] = np.zeros((3, n_dlcs))
        outputs["base_M"] = np.zeros((3, n_dlcs))
        outputs["bedplate_axial_stress"] = np.zeros((2 * n - 2, n_dlcs))
        outputs["bedplate_shear_stress"] = np.zeros((2 * n - 2, n_dlcs))
        outputs["bedplate_bending_stress"] = np.zeros((2 * n - 2, n_dlcs))
        outputs["constr_bedplate_vonmises"] = np.zeros((2 * n - 2, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at bearings- how to sum up rotation angles?
            outputs["mb1_deflection"][k] = np.sqrt(
                displacements.dx[k, i1 - 1] ** 2 + displacements.dy[k, i1 - 1] ** 2 + displacements.dz[k, i1 - 1] ** 2
            )
            outputs["mb2_deflection"][k] = np.sqrt(
                displacements.dx[k, i2 - 1] ** 2 + displacements.dy[k, i2 - 1] ** 2 + displacements.dz[k, i2 - 1] ** 2
            )
            bedplate_deflection = np.maximum(
                np.sqrt(displacements.dx[k, n] ** 2 + displacements.dy[k, n] ** 2 + displacements.dz[k, n] ** 2),
                np.sqrt(displacements.dx[k, -1] ** 2 + displacements.dy[k, -1] ** 2 + displacements.dz[k, -1] ** 2),
            )
            outputs["mb1_angle"][k] = (
                displacements.dxrot[k, i1 - 1] + displacements.dyrot[k, i1 - 1] + displacements.dzrot[k, i1 - 1]
            )
            outputs["mb2_angle"][k] = (
                displacements.dxrot[k, i2 - 1] + displacements.dyrot[k, i2 - 1] + displacements.dzrot[k, i2 - 1]
            )
            bedplate_angle = np.maximum(
                displacements.dxrot[k, n] + displacements.dyrot[k, n] + displacements.dzrot[k, n],
                displacements.dxrot[k, -1] + displacements.dyrot[k, -1] + displacements.dzrot[k, -1],
            )

            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            # F  =  np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            # M   =  np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments at base
            outputs["base_F"][:, k] = np.r_[
                -reactions.Fx[k, :].sum(), -reactions.Fy[k, :].sum(), -reactions.Fz[k, :].sum()
            ]
            outputs["base_M"][:, k] = np.r_[
                -reactions.Mxx[k, :].sum(), -reactions.Myy[k, :].sum(), -reactions.Mzz[k, :].sum()
            ]

            outputs["bedplate_axial_stress"][:, k] = (np.abs(Fx) / Ax + np.abs(Myy) / Sy + np.abs(Mzz) / Sz)[
                : (2 * n - 2)
            ]
            outputs["bedplate_shear_stress"][:, k] = (2.0 * (np.abs(Vy) / Asy + np.abs(Vz) / Asz) + np.abs(Mxx) / C)[
                : (2 * n - 2)
            ]
            hoop = np.zeros(2 * n - 2)

            outputs["constr_bedplate_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["bedplate_axial_stress"][:, k],
                hoop,
                outputs["bedplate_shear_stress"][:, k],
                gamma,
                sigma_y,
            )

        # Evaluate bearing limits
        outputs["constr_mb1_defl"] = outputs["mb1_angle"] / inputs["mb1_max_defl_ang"]
        outputs["constr_mb2_defl"] = outputs["mb2_angle"] / inputs["mb2_max_defl_ang"]
        outputs["stator_deflection"] = bedplate_deflection.max()
        outputs["stator_angle"] = bedplate_angle.max()
        outputs["constr_stator_deflection"] = gamma * outputs["stator_deflection"] / stator_defl_allow
        outputs["constr_stator_angle"] = gamma * outputs["stator_angle"] / stator_angle_allow
