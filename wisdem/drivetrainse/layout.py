#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import openmdao.api as om
from scipy.special import ellipeinc

import wisdem.commonse.utilities as util
from wisdem.commonse.cross_sections import IBeam


def rod_prop(s, Di, ti, rho):
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
    y = 0.25 * rho * np.pi * (D**2 - (D - 2 * t) ** 2)
    # Numpy v1/2 clash
    try:
        m = np.trapezoid(y, s)
        cm = np.trapezoid(y * s, s) / m
    except AttributeError:
        m = np.trapz(y, s)
        cm = np.trapz(y * s, s) / m
    Dm = D.mean()
    tm = t.mean()
    I = np.array(
        [
            0.5 * 0.25 * (Dm**2 + (Dm - 2 * tm) ** 2),
            (1.0 / 12.0) * (3 * 0.25 * (Dm**2 + (Dm - 2 * tm) ** 2) + L**2),
            (1.0 / 12.0) * (3 * 0.25 * (Dm**2 + (Dm - 2 * tm) ** 2) + L**2),
        ]
    )
    return m, cm, m * I, D, t


class Layout(om.ExplicitComponent):
    """
    Calculate lengths, heights, and diameters of key drivetrain components in a
    direct drive system (valid for upwind or downwind).

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    L_12 : float, [m]
        Length from bearing #1 to bearing #2
    L_h1 : float, [m]
        Length from hub / start of lss to bearing #1
    L_generator : float, [m]
        Generator stack width
    overhang : float, [m]
        Overhang of rotor from tower along x-axis in yaw-aligned c.s.
    drive_height : float, [m]
        Hub height above tower top
    tilt : float, [deg]
        Angle of drivetrain lss tilt
    lss_diameter : numpy array[2], [m]
        LSS outer diameter from hub to bearing 2
    lss_wall_thickness : numpy array[2], [m]
        LSS wall thickness
    lss_mass_user : float, [kg]
        User override of lss mass
    hub_diameter : float, [m]
        Diameter of hub
    D_top : float, [m]
        Tower top outer diameter
    lss_rho : float, [kg/m**3]
        material density
    bedplate_rho : float, [kg/m**3]
        material density
    bedplate_mass_user : float, [kg]
        User override of bedplate mass

    Returns
    -------
    L_lss : float, [m]
        Length of nose
    L_drive : float, [m]
        Length of drivetrain from bedplate to hub flang
    s_lss : numpy array[5], [m]
        LSS discretized s-coordinates
    lss_mass : float, [kg]
        LSS mass
    lss_cm : float, [m]
        LSS center of mass along lss axis from bedplate
    lss_I : numpy array[3], [kg*m**2]
        LSS moment of inertia around cm in axial (hub-aligned) c.s.
    L_bedplate : float, [m]
        Length of bedplate
    H_bedplate : float, [m]
        height of bedplate
    bedplate_mass : float, [kg]
        Bedplate mass
    bedplate_cm : numpy array[3], [m]
        Bedplate center of mass
    bedplate_I : numpy array[6], [kg*m**2]
        Bedplate mass moment of inertia about base
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate
    s_gearbox : float, [m]
        Overall gearbox cm
    s_generator : float, [m]
        Overall generator cm
    constr_length : float, [m]
        Margin for drivetrain length and desired overhang distance (should be > 0)
    constr_height : float, [m]
        Margin for drivetrain height and desired hub height (should be > 0)
    D_shaft_mb1 : float, [m]
        Diameter of shaft (nose/lss) at location of first main bearing
    D_shaft_mb2 : float, [m]
        Diameter of shaft (nose/lss) at location of second main bearing
    """

    def setup(self):
        self.add_discrete_input("upwind", True)
        self.add_input("L_12", 0.0, units="m")
        self.add_input("L_h1", 0.0, units="m")
        self.add_input("L_generator", 0.0, units="m")
        self.add_input("overhang", 0.0, units="m")
        self.add_input("drive_height", 0.0, units="m")
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("lss_diameter", np.zeros(2), units="m")
        self.add_input("lss_wall_thickness", np.zeros(2), units="m")
        self.add_input("lss_mass_user", val=0.0, units="kg")
        self.add_input("D_top", 0.0, units="m")
        self.add_input("hub_diameter", val=0.0, units="m")
        self.add_input("lss_rho", val=0.0, units="kg/m**3")
        self.add_input("bedplate_rho", val=0.0, units="kg/m**3")
        self.add_input("bedplate_mass_user", val=0.0, units="kg")

        self.add_output("L_lss", 0.0, units="m")
        self.add_output("L_drive", 0.0, units="m")
        self.add_output("s_lss", val=np.zeros(5), units="m")
        self.add_output("lss_mass", val=0.0, units="kg")
        self.add_output("lss_cm", val=0.0, units="m")
        self.add_output("lss_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("L_bedplate", 0.0, units="m")
        self.add_output("H_bedplate", 0.0, units="m")
        self.add_output("bedplate_mass", val=0.0, units="kg")
        self.add_output("bedplate_cm", val=np.zeros(3), units="m")
        self.add_output("bedplate_I", val=np.zeros(6), units="kg*m**2")
        self.add_output("s_mb1", val=0.0, units="m")
        self.add_output("s_mb2", val=0.0, units="m")
        self.add_output("s_stator", val=0.0, units="m")
        self.add_output("s_rotor", val=0.0, units="m")
        self.add_output("s_gearbox", val=0.0, units="m")
        self.add_output("s_generator", val=0.0, units="m")
        self.add_output("hss_mass", val=0.0, units="kg")
        self.add_output("hss_cm", val=0.0, units="m")
        self.add_output("hss_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("constr_length", 0.0, units="m")
        self.add_output("constr_height", 0.0, units="m")
        self.add_output("D_shaft_mb1", 0.0, units="m")
        self.add_output("D_shaft_mb2", 0.0, units="m")


class DirectLayout(Layout):
    """
    Calculate lengths, heights, and diameters of key drivetrain components in a
    direct drive system (valid for upwind or downwind).

    Parameters
    ----------
    access_diameter : float, [m]
        Minimum diameter required for maintenance access
    nose_diameter : numpy array[2], [m]
        Nose outer diameter from bearing 1 to bedplate
    nose_wall_thickness : numpy array[2], [m]
        Nose wall thickness
    bedplate_wall_thickness : numpy array[4], [m]
        Bedplate wall thickness

    Returns
    -------
    L_nose : float, [m]
        Length of nose
    D_bearing1 : float, [m]
        Diameter of bearing #1 (closer to hub)
    D_bearing2 : float, [m]
        Diameter of bearing #2 (closer to tower)
    s_nose : numpy array[5], [m]
        Nose discretized hub-aligned s-coordinates
    nose_mass : float, [kg]
        Nose mass
    nose_cm : float, [m]
        Nose center of mass along nose axis from bedplate
    nose_I : numpy array[3], [kg*m**2]
        Nose moment of inertia around cm in axial (hub-aligned) c.s.
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
    s_stator : float, [m]
        Generator stator attachment to nose s-coordinate
    s_rotor : float, [m]
        Generator rotor attachment to lss s-coordinate
    constr_access : numpy array[2], [m]
        Margin for allowing maintenance access (should be > 0)
    constr_ecc : float, [m]
        Margin for bedplate ellipse eccentricity (should be > 0)
    """

    def setup(self):
        super().setup()

        self.add_input("access_diameter", 0.0, units="m")
        self.add_input("nose_diameter", np.zeros(2), units="m")
        self.add_input("nose_wall_thickness", np.zeros(2), units="m")
        self.add_input("bedplate_wall_thickness", np.zeros(4), units="m")

        self.add_output("L_nose", 0.0, units="m")
        self.add_output("D_bearing1", 0.0, units="m")
        self.add_output("D_bearing2", 0.0, units="m")
        
        self.add_output("s_nose", val=np.zeros(5), units="m")
        self.add_output("nose_mass", val=0.0, units="kg")
        self.add_output("nose_cm", val=0.0, units="m")
        self.add_output("nose_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("x_bedplate", val=np.zeros(12), units="m")
        self.add_output("z_bedplate", val=np.zeros(12), units="m")
        self.add_output("x_bedplate_inner", val=np.zeros(12), units="m")
        self.add_output("z_bedplate_inner", val=np.zeros(12), units="m")
        self.add_output("x_bedplate_outer", val=np.zeros(12), units="m")
        self.add_output("z_bedplate_outer", val=np.zeros(12), units="m")
        self.add_output("D_bedplate", val=np.zeros(12), units="m")
        self.add_output("t_bedplate", val=np.zeros(12), units="m")
        self.add_output("constr_access", np.zeros((2, 2)), units="m")
        self.add_output("constr_ecc", 0.0, units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        L_12 = float(inputs["L_12"][0])
        L_h1 = float(inputs["L_h1"][0])
        L_generator = float(inputs["L_generator"][0])
        L_overhang = float(inputs["overhang"][0])
        H_drive = float(inputs["drive_height"][0])
        tilt = float(np.deg2rad(inputs["tilt"][0]))
        D_access = float(inputs["access_diameter"][0])
        D_nose = inputs["nose_diameter"]
        D_lss = inputs["lss_diameter"]
        D_top = float(inputs["D_top"][0])
        D_hub = float(inputs["hub_diameter"][0])
        t_nose = inputs["nose_wall_thickness"]
        t_lss = inputs["lss_wall_thickness"]
        t_bed = inputs["bedplate_wall_thickness"]
        upwind = discrete_inputs["upwind"]
        lss_rho = float(inputs["lss_rho"][0])
        bedplate_rho = float(inputs["bedplate_rho"][0])
        bedplate_mass_user = float(inputs["bedplate_mass_user"][0])
        lss_mass_user = float(inputs["lss_mass_user"][0])

        # ------- Discretization ----------------
        L_grs = 0.5 * L_h1
        L_gsn = L_generator - L_grs - L_12
        L_2n = 2.0 * L_gsn

        # Length of lss and nose
        L_lss = L_12 + L_h1
        L_nose = L_12 + L_2n
        outputs["L_lss"] = L_lss
        outputs["L_nose"] = L_nose

        # Total length from bedplate to hub flange
        ds = 0.5 * np.ones(2)
        s_drive = np.cumsum(np.r_[0.0, L_2n * ds, L_12 * ds, L_h1 * ds])
        L_drive = s_drive[-1]
        outputs["L_drive"] = L_drive

        # From Overhang input (dist from center of tower measured in yaw-aligned
        # c.s.-parallel to ground), compute bedplate length and height
        L_bedplate = L_overhang - (L_drive + 0.5 * D_hub) * np.cos(tilt)
        constr_Ldrive = L_bedplate - 0.5 * D_top  # Should be > 0
        if constr_Ldrive < 0:
            L_bedplate = 0.5 * D_top
        H_bedplate = H_drive - (L_drive + 0.5 * D_hub) * np.sin(tilt)  # Keep eccentricity under control
        outputs["L_bedplate"] = L_bedplate
        outputs["H_bedplate"] = H_bedplate

        # Discretize the drivetrain from bedplate end to hub
        # Note that the offset from the bedplate to the tower center line is taken into account in drive_components by "shaft0"
        s_mb1 = s_drive[4]
        s_mb2 = s_drive[2]
        s_rotor = s_drive[-2]
        s_stator = s_drive[1]
        s_nose = s_drive[:5]
        s_lss = s_drive[2:]

        # Store outputs
        # outputs['s_drive']     = np.sort(s_drive)
        outputs["s_rotor"] = s_rotor
        outputs["s_stator"] = s_stator
        outputs["s_nose"] = s_nose
        outputs["s_lss"] = s_lss
        outputs["s_generator"] = 0.5 * (s_rotor + s_stator)
        outputs["s_mb1"] = s_mb1
        outputs["s_mb2"] = s_mb2
        # ------------------------------------

        # ------------ Bedplate geometry and coordinates -------------
        # Define reference/centroidal axis
        # Origin currently set like standard ellipse eqns, but will shift below to being at tower top
        # The end point of 90 deg isn't exactly right for non-zero tilt, but leaving that for later
        n_points = 12
        if upwind:
            rad = np.linspace(0.0, 0.5 * np.pi, n_points)
            dw = 1
        else:
            rad = np.linspace(np.pi, 0.5 * np.pi, n_points)
            dw = -1

        # Make sure we have the right number of bedplate thickness points
        t_bed = np.interp(rad, np.linspace(rad[0], rad[-1], len(t_bed)), t_bed)

        # Centerline
        x_c = L_bedplate * np.cos(rad)
        z_c = H_bedplate * np.sin(rad)

        # Points on the outermost ellipse
        x_outer = (L_bedplate + 0.5 * D_top) * np.cos(rad)
        z_outer = (H_bedplate + 0.5 * D_nose[0]) * np.sin(rad)

        # Points on the innermost ellipse
        x_inner = (L_bedplate - 0.5 * D_top) * np.cos(rad)
        z_inner = (H_bedplate - 0.5 * D_nose[0]) * np.sin(rad)

        # Cross-sectional properties
        D_bed = np.sqrt((z_outer - z_inner) ** 2 + (x_outer - x_inner) ** 2)
        r_bed_o = 0.5 * D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed = np.pi * (r_bed_o**2 - r_bed_i**2)

        # This finds the central angle (rad2) given the parametric angle (rad)
        rad2 = np.arctan2(z_c/H_bedplate, dw * x_c/L_bedplate)

        # arc length from eccentricity of the centroidal ellipse using incomplete elliptic integral of the second kind
        if L_bedplate >= H_bedplate:
            ecc = np.sqrt(1 - (H_bedplate / L_bedplate) ** 2)
            arc = L_bedplate * np.diff(ellipeinc(rad2, ecc))
        else:
            ecc = np.sqrt(1 - (L_bedplate / H_bedplate) ** 2)
            arc = H_bedplate * np.diff(ellipeinc(rad2, ecc))

        # Mass and MoI properties
        x_c_sec = util.nodal2sectional(x_c)[0]
        z_c_sec = util.nodal2sectional(z_c)[0]
        # R_c_sec = np.sqrt( x_c_sec**2 + z_c_sec**2 ) # unnecesary
        mass = util.nodal2sectional(A_bed)[0] * arc * bedplate_rho
        mass_tot = mass.sum()
        cm = np.array([np.sum(mass * x_c_sec), 0.0, np.sum(mass * z_c_sec)]) / mass_tot
        # For I, could do integral over sectional I, rotate axes by rad2, and then parallel axis theorem
        # we simplify by assuming lumped point mass.  TODO: Find a good way to check this?  Torus shell?
        I_bed = util.assembleI(np.zeros(6))
        for k in range(len(mass)):
            r_bed_o_k = 0.5 * (r_bed_o[k] + r_bed_o[k + 1])
            r_bed_i_k = 0.5 * (r_bed_i[k] + r_bed_i[k + 1])
            I_sec = mass[k] * np.array(
                [
                    0.5 * (r_bed_o_k**2 + r_bed_i_k**2),
                    (1.0 / 12.0) * (3 * (r_bed_o_k**2 + r_bed_i_k**2) + arc[k] ** 2),
                    (1.0 / 12.0) * (3 * (r_bed_o_k**2 + r_bed_i_k**2) + arc[k] ** 2),
                ]
            )
            I_sec_rot = util.rotateI(I_sec, 0.5 * np.pi - rad2[k], axis="y")
            R_k = np.array([x_c_sec[k] - x_c[0], 0.0, z_c_sec[k]])
            I_bed += util.assembleI(I_sec_rot) + mass[k] * (np.dot(R_k, R_k) * np.eye(3) - np.outer(R_k, R_k))

        # Now shift origin to be at tower top
        cm[0] -= x_c[0]
        x_inner -= x_c[0]
        x_outer -= x_c[0]
        x_c -= x_c[0]

        coef_user = 1.0 if bedplate_mass_user == 0.0 else bedplate_mass_user / mass_tot
        outputs["bedplate_mass"] = coef_user * mass_tot
        outputs["bedplate_cm"] = cm
        outputs["bedplate_I"] = coef_user * util.unassembleI(I_bed)

        # Geometry outputs
        outputs["x_bedplate"] = x_c
        outputs["z_bedplate"] = z_c
        outputs["x_bedplate_inner"] = x_inner
        outputs["z_bedplate_inner"] = z_inner
        outputs["x_bedplate_outer"] = x_outer
        outputs["z_bedplate_outer"] = z_outer
        outputs["D_bedplate"] = D_bed
        outputs["t_bedplate"] = t_bed
        # ------------------------------------

        # ------- Constraints ----------------
        outputs["constr_access"] = np.c_[D_lss - 2 * t_lss - D_nose - 0.25 * D_access, D_nose - 2 * t_nose - D_access]
        outputs["constr_length"] = constr_Ldrive  # Should be > 0
        outputs["constr_height"] = H_bedplate  # Should be > 0
        outputs["constr_ecc"] = L_bedplate - H_bedplate  # Should be > 0
        # ------------------------------------

        # Compute center of mass based on area
        m_nose, cm_nose, I_nose, Ds_nose, ts_nose = rod_prop(s_nose, D_nose, t_nose, bedplate_rho)
        outputs["nose_mass"] = m_nose
        outputs["nose_cm"] = cm_nose
        outputs["nose_I"] = I_nose

        m_lss, cm_lss, I_lss, Ds_lss, ts_lss = rod_prop(s_lss, D_lss, t_lss, lss_rho)
        if lss_mass_user > 0.0:
            m_lss = lss_mass_user
        outputs["lss_mass"] = m_lss
        outputs["lss_cm"] = cm_lss
        outputs["lss_I"] = I_lss

        # ------- Nose, lss, and bearing properties ----------------
        # Now is a good time to set bearing diameters
        D_lss_mb = np.interp([s_mb1, s_mb2], s_lss, Ds_lss)
        t_lss_mb = np.interp([s_mb1, s_mb2], s_lss, ts_lss)
        D_nose_mb = np.interp([s_mb1, s_mb2], s_nose, Ds_nose)
        #t_nose_mb = np.interp(s_nose, ts_nose, [s_mb1, s_mb2])
        outputs["D_bearing1"] = 0.5 * D_lss_mb[0] - t_lss_mb[0] - 0.5 * D_nose_mb[0]
        outputs["D_bearing2"] = 0.5 * D_lss_mb[1] - t_lss_mb[1] - 0.5 * D_nose_mb[1]
        outputs["D_shaft_mb1"]  = D_nose_mb[0]
        outputs["D_shaft_mb2"]  = D_nose_mb[1]
        

class GearedLayout(Layout):
    """
    Calculate lengths, heights, and diameters of key drivetrain components in a
    geared drive system (valid for upwind or downwind).

    |_Lgen|_Lhss|Lgear|dl|_L12_|_Lh1_|
                      |_____Llss_____|
    |--|--|--|--|--|--|--|--|--|--|--|
    0  1  2  3  4  5  6  7  8  9  10 11 (indices)
                        mb2   mb1

    Parameters
    ----------
    hss_diameter : numpy array[2], [m]
        HSS outer diameter from hub to bearing 2
    hss_wall_thickness : numpy array[2], [m]
        HSS wall thickness
    bedplate_flange_width : float, [m]
        Bedplate is two parallel I beams, this is the flange width
    bedplate_flange_thickness : float, [m]
        Bedplate is two parallel I beams, this is the flange thickness
    bedplate_web_thickness : float, [m]
        Bedplate is two parallel I beams, this is the web thickness
    bedplate_web_height : float, [m]
        Bedplate is two parallel I beams, this is the web height
    hss_rho : float, [kg/m**3]
        material density
    hss_mass_user : float, [kg]
        User override of hss mass

    Returns
    -------
    s_drive : numpy array[12], [m]
        Discretized, hub-aligned s-coordinates of the drivetrain starting at
        generator and ending at hub flange
    s_hss : numpy array[5], [m]
        HSS discretized s-coordinates
    hss_mass : float, [kg]
        HSS mass
    hss_cm : float, [m]
        HSS center of mass along hss axis from bedplate
    hss_I : numpy array[3], [kg*m**2]
        HSS moment of inertia around cm in axial (hub-aligned) c.s.
    s_gearbox : float, [m]
        Gearbox (centroid) position in s-coordinates
    s_generator : float, [m]
        Generator (centroid) position in s-coordinates

    """

    def setup(self):
        super().setup()

        self.add_input("L_hss", 0.0, units="m")
        self.add_input("L_gearbox", 0.0, units="m")
        self.add_input("hss_diameter", np.zeros(2), units="m")
        self.add_input("hss_wall_thickness", np.zeros(2), units="m")
        self.add_input("hss_rho", val=0.0, units="kg/m**3")
        self.add_input("hss_mass_user", val=0.0, units="kg")
        self.add_input("bedplate_flange_width", val=0.0, units="m")
        self.add_input("bedplate_flange_thickness", val=0.0, units="m")
        self.add_input("bedplate_web_thickness", val=0.0, units="m")

        self.add_output("s_drive", val=np.zeros(12), units="m")
        self.add_output("s_hss", val=np.zeros(3), units="m")
        self.add_output("bedplate_web_height", val=0.0, units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0

        L_12 = float(inputs["L_12"][0])
        L_h1 = float(inputs["L_h1"][0])
        L_hss = float(inputs["L_hss"][0])
        L_gearbox = float(inputs["L_gearbox"][0])
        L_generator = float(inputs["L_generator"][0])
        L_overhang = float(inputs["overhang"][0])
        H_drive = float(inputs["drive_height"][0])

        tilt = float(np.deg2rad(inputs["tilt"][0]))

        D_lss = inputs["lss_diameter"]
        t_lss = inputs["lss_wall_thickness"]
        D_hss = inputs["hss_diameter"]
        t_hss = inputs["hss_wall_thickness"]
        lss_mass_user = float(inputs["lss_mass_user"][0])
        hss_mass_user = float(inputs["hss_mass_user"][0])

        D_top = float(inputs["D_top"][0])
        D_hub = float(inputs["hub_diameter"][0])

        bed_w_flange = float(inputs["bedplate_flange_width"][0])
        bed_t_flange = float(inputs["bedplate_flange_thickness"][0])
        # bed_h_web    = float(inputs['bedplate_web_height'][0])
        bed_t_web = float(inputs["bedplate_web_thickness"][0])

        lss_rho = float(inputs["lss_rho"][0])
        hss_rho = float(inputs["hss_rho"][0])
        bedplate_rho = float(inputs["bedplate_rho"][0])
        bedplate_mass_user = float(inputs["bedplate_mass_user"][0])

        # ------- Discretization ----------------
        # Length of lss and drivetrain length
        delta = 0.1  # separation between MB2 and gearbox attachment
        L_lss = L_12 + L_h1 + delta
        L_drive = L_lss + L_gearbox + L_hss + L_generator
        ds = 0.5 * np.ones(2)
        s_drive = np.cumsum(np.r_[0.0, L_generator * ds, L_hss * ds, L_gearbox * ds, delta, L_12 * ds, L_h1 * ds])
        L_drive = s_drive[-1] - s_drive[0]
        outputs["L_drive"] = L_drive
        outputs["L_lss"] = L_lss

        # Put tower at 0 position
        s_tower = s_drive[-1] + 0.5 * D_hub - L_overhang / np.cos(tilt)
        s_drive -= s_tower
        outputs["s_drive"] = s_drive

        # Discretize the drivetrain from generator to hub
        s_stator = s_drive[0]
        s_rotor = s_drive[1]
        s_generator = 0.5*(s_rotor + s_stator)
        s_mb1 = s_drive[9]
        s_mb2 = s_drive[7]
        s_gearbox = s_drive[5]
        s_lss = s_drive[6:]
        s_lss = np.r_[s_lss[:-2], s_lss[-1]]  # Need to stick to 5 points
        s_hss = s_drive[2:5]

        # Store outputs
        outputs["s_stator"] = s_stator
        outputs["s_rotor"] = s_rotor
        outputs["s_generator"] = s_generator
        outputs["s_gearbox"] = s_gearbox
        outputs["s_mb1"] = s_mb1
        outputs["s_mb2"] = s_mb2
        # ------------------------------------

        # ------- hss, lss, and bearing properties ----------------
        # Compute center of mass based on area
        m_hss, cm_hss, I_hss, _, _ = rod_prop(s_hss, D_hss, t_hss, hss_rho)
        if hss_mass_user > 0.0:
            m_hss = hss_mass_user
        outputs["hss_mass"] = m_hss
        outputs["hss_cm"] = cm_hss
        outputs["hss_I"] = I_hss
        outputs["s_hss"] = s_hss

        m_lss, cm_lss, I_lss, Ds_lss, ts_lss = rod_prop(s_lss, D_lss, t_lss, lss_rho)
        if lss_mass_user > 0.0:
            m_lss = lss_mass_user
        outputs["lss_mass"] = m_lss
        outputs["lss_cm"] = cm_lss
        outputs["lss_I"] = I_lss
        outputs["s_lss"] = s_lss

        D_lss_mb = np.interp([s_mb1, s_mb2], s_lss, Ds_lss)
        outputs["D_shaft_mb1"]  = D_lss_mb[0]
        outputs["D_shaft_mb2"]  = D_lss_mb[1]
        
        # ------- Bedplate I-beam properties ----------------
        L_bedplate = (L_drive + 0.5 * D_hub) * np.cos(tilt)
        H_bedplate = H_drive - (L_drive + 0.5 * D_hub) * np.sin(tilt)  # Subtract thickness of platform plate
        outputs["L_bedplate"] = L_bedplate
        outputs["H_bedplate"] = H_bedplate
        bed_h_web = H_bedplate - 2 * bed_t_flange - 0.05  # Leave some extra room for plate?

        yoff = 0.25 * D_top
        myI = IBeam(bed_w_flange, bed_t_flange, bed_h_web, bed_t_web)
        m_bedplate = myI.Area * L_bedplate * bedplate_rho
        cg_bedplate = np.r_[Cup * (L_overhang - 0.5 * L_bedplate), 0.0, myI.CG]  # from tower top
        I_bedplate = (
            bedplate_rho * L_bedplate * np.r_[myI.J0, myI.Ixx, myI.Iyy]
            + m_bedplate * L_bedplate**2 / 12.0 * np.r_[0.0, 1.0, 1.0]
            + m_bedplate * yoff**2 * np.r_[1.0, 0.0, 1.0]
        )
        coef_user = 1.0 if bedplate_mass_user == 0.0 else bedplate_mass_user / (2 * m_bedplate)
        outputs["bedplate_web_height"] = bed_h_web
        outputs["bedplate_mass"] = coef_user * 2 * m_bedplate
        outputs["bedplate_cm"] = cg_bedplate
        outputs["bedplate_I"] = coef_user * 2 * np.r_[I_bedplate, np.zeros(3)]

        # ------- Constraints ----------------
        outputs["constr_length"] = (L_drive + 0.5 * D_hub) * np.cos(tilt) - L_overhang - 0.5 * D_top  # Should be > 0
        outputs["constr_height"] = H_bedplate  # Should be > 0
        # ------------------------------------
