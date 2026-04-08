import numpy as np
import openmdao.api as om

import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.cylinder_member as mem
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.environment import TowerSoil

RIGID = 1e30
NPTS_SOIL = 10

class PreDiscretization(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.

    Parameters
    ----------
    tower_foundation_height : float, [m]
        Starting z-coordinate value of the tower with 0 at the water line
    monopile_foundation_height : float, [m]
        Starting z-coordinate value of the monopile with 0 at the water line
    monopile_height : float, [m]
        Scalar of the monopile height computed along the z axis.

    Returns
    -------
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint2 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member
    suctionpile_depth : float, [m]
        Depth of monopile below sea floor
    bending_height : float, [m]
        Length of monopile above mudline subject to bending

    """

    def setup(self):
        self.add_input("monopile_height", val=0.0, units="m")
        self.add_input("tower_foundation_height", val=0.0, units="m")
        self.add_input("monopile_foundation_height", val=0.0, units="m")
        self.add_input("tower_base_diameter", val=0.0, units="m")
        self.add_input("monopile_top_diameter", val=0.0, units="m")
        self.add_input("water_depth", val=0.0, units="m")

        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("z_start", 0.0, units="m")
        self.add_output("suctionpile_depth", 0.0, units="m")
        self.add_output("bending_height", 0.0, units="m")
        self.add_output("s_const1", 0.0)
        self.add_output("joint1", val=np.zeros(3), units="m")
        self.add_output("joint2", val=np.zeros(3), units="m")
        self.add_output("constr_diam_consistency", val=0.0)

    def compute(self, inputs, outputs):
        # Unpack values
        h_mon = inputs["monopile_height"]
        fh_tow = inputs["tower_foundation_height"]
        fh_mon = inputs["monopile_foundation_height"]
        water_depth = inputs["water_depth"]

        outputs["transition_piece_height"] = fh_tow

        if np.abs(fh_tow - fh_mon - h_mon) > 1.0:
            print(
                "WARNING: Monopile length is not consistent with transition piece height and monopile base height\n",
                "         Determining new base height value . . .",
            )
        outputs["z_start"] = fh_tow - h_mon

        pile = h_mon - fh_tow - water_depth
        outputs["suctionpile_depth"] = pile
        outputs["bending_height"] = h_mon - pile
        outputs["s_const1"] = pile / h_mon  # Ensure that we have only one segment for pile, a current limitation
        outputs["joint1"] = np.r_[0.0, 0.0, fh_tow - h_mon]
        outputs["joint2"] = np.r_[0.0, 0.0, fh_tow]
        outputs["constr_diam_consistency"] = inputs["tower_base_diameter"] / inputs["monopile_top_diameter"]


class MonopileMass(om.ExplicitComponent):
    """
    Compute the tower and monopile masses, raw cost, and CG properties.

    Parameters
    ----------
    shell_mass : numpy array[nFull-1], [kg]
        Total cylinder mass
    shell_cost : float, [USD]
        Total cylinder cost
    shell_z_cg : float, [m]
        z position of center of mass of cylinder
    shell_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of cylinder about base [xx yy zz xy xz yz]
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    transition_piece_mass : float, [kg]
        Point mass of transition piece
    transition_piece_cost : float, [USD]
        Cost of transition piece
    gravity_foundation_mass : float, [kg]
        Extra mass of gravity foundation
    z_full : numpy array[nFull], [m]
        Parameterized locations along tower, linear lofting between
    outer_diameter_full : numpy array[nFull], [m]
        diameter along tower

    Returns
    -------
    structural_cost : float, [USD]
        Total structural cost (tower+monopile)
    structural_mass : float, [kg]
        Total structural mass (tower+monopile)
    monopile_mass : float, [kg]
        Mass of monopile from bottom of suction pile through transition piece
    monopile_cost : float, [USD]
        Total monopile cost

    """

    def initialize(self):
        self.options.declare("npts")

    def setup(self):
        npts = self.options["npts"]

        self.add_input("cylinder_mass", val=0.0, units="kg")
        self.add_input("cylinder_cost", val=0.0, units="USD")
        self.add_input("cylinder_z_cg", val=0.0, units="m")
        self.add_input("cylinder_I_base", np.zeros(6), units="kg*m**2")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_cost", 0.0, units="USD")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("z_full", val=np.zeros(npts), units="m")
        self.add_input("outer_diameter_full", val=np.zeros(npts), units="m")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("tower_cost", val=0.0, units="USD")

        self.add_output("monopile_mass", val=0.0, units="kg")
        self.add_output("monopile_cost", val=0.0, units="USD")
        self.add_output("monopile_z_cg", val=0.0, units="m")
        self.add_output("monopile_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_output("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_output("structural_mass", val=0.0, units="kg")
        self.add_output("structural_cost", val=0.0, units="USD")

    def compute(self, inputs, outputs):
        # Unpack inputs
        z = inputs["z_full"]
        d = inputs["outer_diameter_full"]
        m_trans = inputs["transition_piece_mass"]
        c_trans = inputs["transition_piece_cost"]
        m_grav = inputs["gravity_foundation_mass"]
        m_cyl = inputs["cylinder_mass"]
        c_cyl = inputs["cylinder_cost"]

        outputs["monopile_mass"] = m_mono = m_cyl + m_trans + m_grav
        outputs["monopile_cost"] = c_cyl + c_trans
        outputs["structural_mass"] = outputs["monopile_mass"] + inputs["tower_mass"]
        outputs["structural_cost"] = outputs["monopile_cost"] + inputs["tower_cost"]

        outputs["monopile_z_cg"] = (m_cyl * inputs["cylinder_z_cg"] + m_trans * z[-1] + m_grav * z[0]) / m_mono

        # Mass properties for transition piece and gravity foundation
        r_trans = 0.5 * d[-1]
        r_grav = 0.5 * d[0]
        I_trans = m_trans * r_trans**2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]  # shell
        I_grav = m_grav * r_grav**2.0 * np.r_[0.25, 0.25, 0.5, np.zeros(3)]  # disk
        outputs["transition_piece_I"] = I_trans
        outputs["gravity_foundation_I"] = I_grav

        I_mono = util.assembleI(inputs["cylinder_I_base"] + I_grav)
        R = np.array([0.0, 0.0, z[-1] - z[0]])
        I_mono += util.assembleI(I_trans) + m_trans * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["monopile_I_base"] = util.unassembleI(I_mono)


class MonopileFrame(om.ExplicitComponent):
    """
    Run Frame3DD on the tower

    Parameters
    ----------
    z_full : numpy array[npts], [m]
        location along cylinder. start at bottom and go to top
    outer_diameter_full : numpy array[npts], [m]
        effective cylinder diameter for section
    t_full : numpy array[npts-1], [m]
        effective shell thickness for section
    E_full : numpy array[npts-1], [N/m**2]
        modulus of elasticity
    G_full : numpy array[npts-1], [N/m**2]
        shear modulus
    rho_full : numpy array[npts-1], [kg/m**3]
        material density
    turbine_F : numpy array[3], [N]
        Total turbine force at tower base / transition piece
    turbine_M : numpy array[3], [N*m]
        Total turbine moment at tower base / transition piece
    Px : numpy array[n_full], [N/m]
        force per unit length in x-direction
    Py : numpy array[n_full], [N/m]
        force per unit length in y-direction
    Pz : numpy array[n_full], [N/m]
        force per unit length in z-direction

    Returns
    -------
    f1 : float, [Hz]
        First natural frequency
    f2 : float, [Hz]
        Second natural frequency
    structural_frequencies : numpy array[NFREQ], [Hz]
        First and second natural frequency
    fore_aft_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower fore-aft direction
    side_side_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower side-side direction
    torsion_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower torsion direction
    fore_aft_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower fore-aft direction
        (without constant term)
    side_side_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower side-side direction
        (without constant term)
    torsion_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower torsion direction
        (without constant term)
    monopile_deflection : numpy array[n_full], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    monopile_Fz : numpy array[n_full-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    monopile_Vx : numpy array[n_full-1], [N]
        Shear force in x-direction in cylinder structure.
    monopile_Vy : numpy array[n_full-1], [N]
        Shear force in y-direction in cylinder structure.
    monopile_Mxx : numpy array[n_full-1], [N*m]
        Moment about x-axis in cylinder structure.
    monopile_Myy : numpy array[n_full-1], [N*m]
        Moment about y-axis in cylinder structure.
    monopile_Mzz : numpy array[n_full-1], [N*m]
        Moment about z-axis in cylinder structure.
    base_F : numpy array[3], [N]
        Total force on cylinder
    base_M : numpy array[3], [N*m]
        Total moment on cylinder measured at base
    """

    def initialize(self):
        self.options.declare("n_refine", default=3)
        self.options.declare("n_full")
        self.options.declare("n_full_tow")
        self.options.declare("nLC")
        self.options.declare("frame3dd_opt")
        self.options.declare("soil_springs", default=False)
        self.options.declare("gravity_foundation", default=False)

    def setup(self):
        n_full = self.options["n_full"]
        n_full_tow = self.options["n_full_tow"]
        ntow_sec = max(n_full_tow - 1, 0)
        nLC = self.options["nLC"]
        self.frame = None

        # Monopile handling
        self.add_input("z_soil", np.zeros(NPTS_SOIL), units="N/m")
        self.add_input("k_soil", np.zeros((NPTS_SOIL, 6)), units="N/m")

        # cross-sectional data along monopile
        self.add_input("nodes_xyz", np.zeros((n_full, 3)), units="m")
        self.add_input("section_A", np.zeros(n_full - 1), units="m**2")
        self.add_input("section_Asx", np.zeros(n_full - 1), units="m**2")
        self.add_input("section_Asy", np.zeros(n_full - 1), units="m**2")
        self.add_input("section_Ixx", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_Iyy", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_J0", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_rho", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("section_E", np.zeros(n_full - 1), units="Pa")
        self.add_input("section_G", np.zeros(n_full - 1), units="Pa")
        self.add_output("section_L", np.zeros(n_full - 1 + ntow_sec), units="m")
        self.add_input("outer_diameter_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")
        self.add_input("sigma_y_full", np.zeros(n_full -1), units="Pa")
        self.add_input("qdyn", np.zeros((n_full,nLC)), units="Pa")
        self.add_input("bending_height", 0.0, units="m")

        # cross-sectional data along tower
        self.add_input("tower_xyz", np.zeros((n_full_tow, 3)), units="m")
        self.add_input("tower_A", np.zeros(ntow_sec), units="m**2")
        self.add_input("tower_Asx", np.zeros(ntow_sec), units="m**2")
        self.add_input("tower_Asy", np.zeros(ntow_sec), units="m**2")
        self.add_input("tower_Ixx", np.zeros(ntow_sec), units="kg*m**2")
        self.add_input("tower_Iyy", np.zeros(ntow_sec), units="kg*m**2")
        self.add_input("tower_J0", np.zeros(ntow_sec), units="kg*m**2")
        self.add_input("tower_rho", np.zeros(ntow_sec), units="kg/m**3")
        self.add_input("tower_E", np.zeros(ntow_sec), units="Pa")
        self.add_input("tower_G", np.zeros(ntow_sec), units="Pa")
        self.add_input("tower_outer_diameter_full", np.zeros(n_full_tow), units="m")
        self.add_input("tower_t_full", np.zeros(ntow_sec), units="m")
        self.add_input("tower_sigma_y_full", np.zeros(ntow_sec), units="Pa")
        self.add_input("tower_qdyn", np.zeros((n_full_tow,nLC)), units="Pa")
        self.add_input("tower_bending_height", 0.0, units="m")

        # point loads
        self.add_input("rna_F", np.zeros((3, nLC)), units="N")
        self.add_input("rna_M", np.zeros((3, nLC)), units="N*m")
        self.add_input("turbine_F", np.zeros((3, nLC)), units="N")
        self.add_input("turbine_M", np.zeros((3, nLC)), units="N*m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_input("transition_piece_height", 0.0, units="m")
        self.add_input("suctionpile_depth", 0.0, units="m")
        self.add_input("water_depth", val=0.0, units="m")

        # For modal analysis only (loads captured in turbine_F & turbine_M)
        self.add_input("turbine_mass", val=0.0, units="kg")
        self.add_input("turbine_cg", val=np.zeros(3), units="m")
        self.add_input("turbine_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_mass", val=0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")

        # combined wind-water distributed loads
        self.add_input("Px", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("Py", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("Pz", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("tower_Px", val=np.zeros((n_full_tow, nLC)), units="N/m")
        self.add_input("tower_Py", val=np.zeros((n_full_tow, nLC)), units="N/m")
        self.add_input("tower_Pz", val=np.zeros((n_full_tow, nLC)), units="N/m")

        # Frequencies
        NFREQ2 = int(NFREQ / 2)
        self.add_output("f1", val=0.0, units="Hz")
        self.add_output("f2", val=0.0, units="Hz")
        self.add_output("structural_frequencies", np.zeros(NFREQ), units="Hz")
        self.add_output("fore_aft_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("side_side_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("torsion_freqs", np.zeros(NFREQ2), units="Hz")

        self.add_output("fore_aft_modes", np.zeros((NFREQ2, 5)))
        self.add_output("side_side_modes", np.zeros((NFREQ2, 5)))
        self.add_output("torsion_modes", np.zeros((NFREQ2, 5)))
        self.add_output("tower_fore_aft_modes", np.zeros((NFREQ2, 5)))
        self.add_output("tower_side_side_modes", np.zeros((NFREQ2, 5)))
        self.add_output("tower_torsion_modes", np.zeros((NFREQ2, 5)))

        self.add_output("monopile_deflection", np.zeros((n_full + ntow_sec, nLC)), units="m")
        self.add_output("top_deflection", np.zeros(nLC), units="m")
        self.add_output("monopile_Fz", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("monopile_Vx", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("monopile_Vy", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("monopile_Mxx", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("monopile_Myy", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("monopile_Mzz", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("mudline_F", val=np.zeros((3, nLC)), units="N")
        self.add_output("mudline_M", val=np.zeros((3, nLC)), units="N*m")


        # Tower and monopile combined
        self.add_output("monopile_tower_z_full", val=np.zeros(n_full + n_full_tow - 1), units="m")
        self.add_output("monopile_tower_outer_diameter_full", val=np.zeros(n_full + n_full_tow - 1), units="m")
        self.add_output("monopile_tower_t_full", val=np.zeros(n_full + ntow_sec - 1), units="m")
        self.add_output("monopile_tower_rho_sec", val=np.zeros(n_full + ntow_sec -1), units="kg/m**3")
        self.add_output("monopile_tower_E_sec", val=np.zeros(n_full + ntow_sec -1), units="Pa")
        self.add_output("monopile_tower_G_sec", val=np.zeros(n_full + ntow_sec -1), units="Pa")
        self.add_output("monopile_tower_A_sec", val=np.zeros(n_full + ntow_sec -1), units="m*m")
        self.add_output("monopile_tower_Asx_sec", val=np.zeros(n_full + ntow_sec -1), units="m*m")
        self.add_output("monopile_tower_Asy_sec", val=np.zeros(n_full + ntow_sec -1), units="m*m")
        self.add_output("monopile_tower_J0_sec", val=np.zeros(n_full + ntow_sec -1), units="kg*m*m")
        self.add_output("monopile_tower_Ixx_sec", val=np.zeros(n_full + ntow_sec -1), units="kg*m*m")
        self.add_output("monopile_tower_Iyy_sec", val=np.zeros(n_full + ntow_sec -1), units="kg*m*m")
        self.add_output("monopile_tower_sigma_y_full", val=np.zeros(n_full + ntow_sec -1), units="Pa")
        self.add_output("monopile_tower_bending_height", val=0.0, units="m")
        self.add_output("monopile_tower_qdyn", val=np.zeros((n_full + n_full_tow - 1, nLC)), units="Pa")
        self.add_output("monopile_tower_Fz", val=np.zeros((n_full + ntow_sec -1, nLC)), units="N")
        self.add_output("monopile_tower_Vx", val=np.zeros((n_full + ntow_sec -1, nLC)), units="N")
        self.add_output("monopile_tower_Vy", val=np.zeros((n_full + ntow_sec -1, nLC)), units="N")
        self.add_output("monopile_tower_Mxx", val=np.zeros((n_full + ntow_sec -1, nLC)), units="N*m")
        self.add_output("monopile_tower_Myy", val=np.zeros((n_full + ntow_sec -1, nLC)), units="N*m")
        self.add_output("monopile_tower_Mzz", val=np.zeros((n_full + ntow_sec -1, nLC)), units="N*m")

    def compute(self, inputs, outputs):
        frame3dd_opt = self.options["frame3dd_opt"]
        nLC          = self.options["nLC"]
        tower_flag   = self.options["n_full_tow"] > 0
        nref         = self.options["n_refine"]
        
        # ------- node data ----------------
        if tower_flag:
            xyz = np.vstack((inputs["nodes_xyz"][:-1, :], inputs["tower_xyz"]))
            n_tow = inputs["tower_xyz"].shape[0]
        else:
            xyz = inputs["nodes_xyz"]
            n_tow = 0

        n = xyz.shape[0]
        n_mono = inputs["nodes_xyz"].shape[0]
        z = xyz[:, 2]
        node = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node, xyz[:, 0], xyz[:, 1], z, r)
        # -----------------------------------

        # ------ reaction data ------------
        if not self.options["gravity_foundation"]:
            if self.options["soil_springs"]:
                z_soil = inputs["z_soil"]
                k_soil = inputs["k_soil"]
                depth = float(inputs["water_depth"][0])
                z_pile = z[z <= (z[0] + 1e-1 + np.abs(z_soil[0]))]
                if z_pile.size != nref + 1:
                    print(z)
                    print(z_soil)
                    print(z_pile)
                    raise ValueError("Please use only one section for submerged pile for now")
                k_mono = np.zeros((z_pile.size, 6))
                for k in range(6):
                    k_mono[:, k] = np.interp(z_pile + np.abs(depth), z_soil, k_soil[:, k])
                kidx = np.arange(len(z_pile), dtype=np.int_)
                kx = k_mono[:, 0]
                ky = k_mono[:, 2]
                kz = np.zeros(k_mono.shape[0])
                kz[0] = k_mono[0, 4]
                ktx = k_mono[:, 1]
                kty = k_mono[:, 3]
                ktz = k_mono[:, 5]

            else:
                z_pile = z[z <= (z[0] + 1e-1 + inputs["suctionpile_depth"])]
                npile = z_pile.size
                if npile != nref + 1:
                    print(z)
                    print(z_pile)
                    print(inputs["suctionpile_depth"])
                    raise ValueError("Please use only one section for submerged pile for now")
                kidx = np.arange(npile, dtype=np.int_)
                kx = ky = kz = RIGID * np.ones(npile)
                ktx = kty = ktz = RIGID * np.ones(npile)

        else:
            kidx = np.array([0], dtype=np.int_)
            kx = ky = kz = np.array([RIGID])
            ktx = kty = ktz = np.array([RIGID])

        rnode = kidx + 1  # 1-based indexing
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)

        # Element properties
        if tower_flag:
            Area = np.r_[inputs["section_A"], inputs["tower_A"]]
            Asx = np.r_[inputs["section_Asx"], inputs["tower_Asx"]]
            Asy = np.r_[inputs["section_Asy"], inputs["tower_Asy"]]
            J0 = np.r_[inputs["section_J0"], inputs["tower_J0"]]
            Ixx = np.r_[inputs["section_Ixx"], inputs["tower_Ixx"]]
            Iyy = np.r_[inputs["section_Iyy"], inputs["tower_Iyy"]]
            E = np.r_[inputs["section_E"], inputs["tower_E"]]
            G = np.r_[inputs["section_G"], inputs["tower_G"]]
            rho = np.r_[inputs["section_rho"], inputs["tower_rho"]]
            d = np.r_[inputs["outer_diameter_full"][:-1], inputs["tower_outer_diameter_full"]]
            t = np.r_[inputs["t_full"], inputs["tower_t_full"]]
            sigma_y = np.r_[inputs["sigma_y_full"], inputs["tower_sigma_y_full"]]

            outputs["monopile_tower_z_full"] = z
            outputs["monopile_tower_outer_diameter_full"] = d
            outputs["monopile_tower_t_full"] = t
            outputs["monopile_tower_A_sec"] = Area
            outputs["monopile_tower_Asx_sec"] = Asx
            outputs["monopile_tower_Asy_sec"] = Asy
            outputs["monopile_tower_J0_sec"] = J0
            outputs["monopile_tower_Ixx_sec"] = Ixx
            outputs["monopile_tower_Iyy_sec"] = Iyy
            outputs["monopile_tower_rho_sec"] = rho
            outputs["monopile_tower_E_sec"] = E
            outputs["monopile_tower_G_sec"] = G
            outputs["monopile_tower_sigma_y_full"] = sigma_y
            outputs["monopile_tower_bending_height"] = inputs['bending_height'] + inputs['tower_bending_height']
            for ic in range(nLC):
                qdyn = np.r_[inputs["qdyn"][:-1,ic], inputs["tower_qdyn"][:,ic]]
                outputs["monopile_tower_qdyn"][:, ic] = qdyn


        else:
            Area = inputs["section_A"]
            Asx = inputs["section_Asx"]
            Asy = inputs["section_Asy"]
            J0 = inputs["section_J0"]
            Ixx = inputs["section_Ixx"]
            Iyy = inputs["section_Iyy"]
            E = inputs["section_E"]
            G = inputs["section_G"]
            rho = inputs["section_rho"]

        outputs["section_L"] = L = np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], dx)
        # -----------------------------------

        # initialize frame3dd object
        self.frame = pyframe3dd.Frame(nodes, reactions, elements, options)

        # ------- enable dynamic analysis ----------
        lump = 0
        shift = 1e2 if self.options["soil_springs"] else 0.0
        # Run twice the number of modes to ensure that we can ignore the torsional modes and still get the desired number of fore-aft, side-side modes
        self.frame.enableDynamics(2 * NFREQ, frame3dd_opt["modal_method"], lump, frame3dd_opt["tol"], shift)
        # ----------------------------

        # ------ static load case 1 ------------
        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        for k in range(nLC):
            load = pyframe3dd.StaticLoadCase(gx, gy, gz)

            # Prepare point forces at transition node or rna
            if tower_flag:
                rna_F = inputs["rna_F"][:, k]
                rna_M = inputs["rna_M"][:, k]
                load.changePointLoads(
                    np.array([n-1], dtype=np.int_),  # -1 b/c same reason as above
                    np.array([rna_F[0]]).flatten(),
                    np.array([rna_F[1]]).flatten(),
                    np.array([rna_F[2]]).flatten(),
                    np.array([rna_M[0]]).flatten(),
                    np.array([rna_M[1]]).flatten(),
                    np.array([rna_M[2]]).flatten(),
                )

                # distributed loads: switch to local c.s.
                Px = np.r_[inputs["Pz"][:, k], inputs["tower_Pz"][:, k]]
                Py = np.r_[inputs["Py"][:, k], inputs["tower_Py"][:, k]]
                Pz = -1.0 * np.r_[inputs["Px"][:, k], inputs["tower_Px"][:, k]]

            else:
                turb_F = inputs["turbine_F"][:, k]
                turb_M = inputs["turbine_M"][:, k]
                load.changePointLoads(
                    np.array([n-1], dtype=np.int_),  # -1 b/c same reason as above
                    np.array([turb_F[0]]).flatten(),
                    np.array([turb_F[1]]).flatten(),
                    np.array([turb_F[2]]).flatten(),
                    np.array([turb_M[0]]).flatten(),
                    np.array([turb_M[1]]).flatten(),
                    np.array([turb_M[2]]).flatten(),
                )

                # distributed loads: switch to local c.s.
                Px, Py, Pz = inputs["Pz"][:, k], inputs["Py"][:, k], -inputs["Px"][:, k]

            # trapezoidally distributed loads
            EL = element
            xx1 = xy1 = xz1 = np.zeros(EL.shape)
            xx2 = xy2 = xz2 = 0.99 * L  # subtract small number b.c. of precision
            wx1 = Px[:-1]
            wx2 = Px[1:]
            wy1 = Py[:-1]
            wy2 = Py[1:]
            wz1 = Pz[:-1]
            wz2 = Pz[1:]

            load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            self.frame.addLoadCase(load)

        # ------ add extra mass ------------
        # Prepare transition piece, and gravity foundation (if any applicable) for "extra node mass"
        # Turbine mass added for modal analysis only- gravity loads accounted for in point force
        m_trans = float(inputs["transition_piece_mass"][0])
        I_trans = inputs["transition_piece_I"].flatten()

        m_grav = float(inputs["gravity_foundation_mass"][0])
        I_grav = inputs["gravity_foundation_I"].flatten()

        m_turb = float(inputs["turbine_mass"][0])
        cg_turb = inputs["turbine_cg"].flatten()
        I_turb = inputs["turbine_I"].flatten()

        m_rna = float(inputs["rna_mass"][0])
        cg_rna = inputs["rna_cg"].flatten()
        I_rna = inputs["rna_I"].flatten()

        # Note, need len()-1 because Frame3DD crashes if mass add at end
        if tower_flag:
            midx = np.array([n-1, n_mono, 1], dtype=np.int_)
            m_add = np.r_[m_rna, m_trans, m_grav]
            mI = np.c_[I_rna, I_trans, I_grav]
            mrho = np.c_[cg_rna, np.zeros(3), np.zeros(3)]
            add_gravity = [False, True, True]
        else:
            midx = np.array([n_mono - 1, n_mono - 2, 1], dtype=np.int_)
            m_add = np.r_[m_turb, m_trans, m_grav]
            mI = np.c_[I_turb, I_trans, I_grav]
            mrho = np.c_[cg_turb, np.zeros(3), np.zeros(3)]
            add_gravity = [False, True, True]
        self.frame.changeExtraNodeMass(
            midx,
            m_add,
            mI[0, :],
            mI[1, :],
            mI[2, :],
            mI[3, :],
            mI[4, :],
            mI[5, :],
            mrho[0, :],
            mrho[1, :],
            mrho[2, :],
            add_gravity,
        )

        # ------------------------------------
        # Debugging
        # self.frame.write('monopile_debug.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, rxns, internalForces, mass, modal = self.frame.run()

        # natural frequncies
        outputs["f1"] = modal.freq[0]
        outputs["f2"] = modal.freq[1]
        outputs["structural_frequencies"] = modal.freq[:NFREQ]

        # Get all mode shapes in batch
        NFREQ2 = int(NFREQ / 2)
        freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
            z,
            modal.freq,
            modal.xdsp,
            modal.ydsp,
            modal.zdsp,
            modal.xmpf,
            modal.ympf,
            modal.zmpf,
            idx0=nref,
            base_slope0=(not self.options["soil_springs"]),
        )
        outputs["fore_aft_freqs"] = freq_x[:NFREQ2]
        outputs["side_side_freqs"] = freq_y[:NFREQ2]
        outputs["torsion_freqs"] = freq_z[:NFREQ2]
        outputs["fore_aft_modes"] = mshapes_x[:NFREQ2, :]
        outputs["side_side_modes"] = mshapes_y[:NFREQ2, :]
        outputs["torsion_modes"] = mshapes_z[:NFREQ2, :]

        if tower_flag:
            # Evaluate full mode shape coefficients at tower points only
            zn = z[-n_tow:] / (z[-1] - z[0])
            cmat_high = np.vstack( (mshapes_x[:NFREQ2, :],
                                    mshapes_y[:NFREQ2, :],
                                    mshapes_z[:NFREQ2, :]) )
            cmat_low = np.zeros((3*NFREQ2, 2))
            ymat = np.polynomial.polynomial.polyval(zn, np.hstack((cmat_low, cmat_high)).T)
            p6, _ = util.get_modal_coefficients(zn, ymat.T, base_slope0=False)

            outputs["tower_fore_aft_modes"] = p6[:, :NFREQ2].T
            outputs["tower_side_side_modes"] = p6[:, NFREQ2:(2*NFREQ2)].T
            outputs["tower_torsion_modes"] = p6[:, (2*NFREQ2):].T

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs["monopile_deflection"] = np.sqrt(displacements.dx**2 + displacements.dy**2).T
        outputs["top_deflection"] = outputs["monopile_deflection"][-1, :]

        # Record total forces and moments
        ibase = 2 * int(kidx.max())
        outputs["mudline_F"] = -np.c_[-forces.Vz[:, ibase], forces.Vy[:, ibase], forces.Nx[:, ibase]].T
        outputs["mudline_M"] = -np.c_[-forces.Mzz[:, ibase], forces.Myy[:, ibase], forces.Txx[:, ibase]].T

        Fz = np.zeros((len(forces.Nx[0, 1::2]), nLC))
        Vx = np.zeros(Fz.shape)
        Vy = np.zeros(Fz.shape)
        Mxx = np.zeros(Fz.shape)
        Myy = np.zeros(Fz.shape)
        Mzz = np.zeros(Fz.shape)
        for ic in range(nLC):
            # Forces and moments along the structure
            Fz[:, ic] = forces.Nx[ic, 1::2]
            Vx[:, ic] = -forces.Vz[ic, 1::2]
            Vy[:, ic] = forces.Vy[ic, 1::2]
            Mxx[:, ic] = -forces.Mzz[ic, 1::2]
            Myy[:, ic] = forces.Myy[ic, 1::2]
            Mzz[:, ic] = forces.Txx[ic, 1::2]
        outputs["monopile_Fz"] = Fz[: (n_mono - 1)]
        outputs["monopile_Vx"] = Vx[: (n_mono - 1)]
        outputs["monopile_Vy"] = Vy[: (n_mono - 1)]
        outputs["monopile_Mxx"] = Mxx[: (n_mono - 1)]
        outputs["monopile_Myy"] = Myy[: (n_mono - 1)]
        outputs["monopile_Mzz"] = Mzz[: (n_mono - 1)]
    

        outputs["monopile_tower_Fz"] = Fz
        outputs["monopile_tower_Vx"] = Vx
        outputs["monopile_tower_Vy"] = Vy
        outputs["monopile_tower_Mxx"] = Mxx
        outputs["monopile_tower_Myy"] = Myy
        outputs["monopile_tower_Mzz"] = Mzz


class MonopileSEProp(om.Group):
    """
    This is the main MonopileSE group that performs analysis of the monopile.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["FixedBottomSE"]
        n_mat = self.options["modeling_options"]["materials"]["n_mat"]
        nref = mod_opt["n_refine"]
        
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_mono = mod_opt["n_height_monopile"]
            n_height = mod_opt["n_height"] = n_height_mono
        n_full = mem.get_nfull(n_height, nref=nref)

        self.add_subsystem("predis", PreDiscretization(), promotes=["*"])

        promlist = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "sigma_ult_mat",
            "wohler_exp_mat",
            "wohler_A_mat",
            "rho_mat",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
            "z_global",
            "z_param",
            "z_full",
            "s_full",
            "outer_diameter_full",
            "t_full",
            "rho_full",
            "E_full",
            "G_full",
            "sigma_y_full",
            "outfitting_full",
            "s_const1",
            "s_const2",
            "joint1",
            "joint2",
            "outfitting_factor_in",
            "E_user",
            "constr_taper",
            "constr_d_to_t",
            "slope",
            "nodes_xyz",
            "section_A",
            "section_Asx",
            "section_Asy",
            "section_Ixx",
            "section_Iyy",
            "section_J0",
            "section_rho",
            "section_E",
            "section_G",
            "rho_water",
            ("s_in", "monopile_s"),
            ("layer_materials", "monopile_layer_materials"),
            ("layer_thickness", "monopile_layer_thickness"),
            ("section_height", "monopile_section_height"),
            ("outer_diameter_in", "monopile_outer_diameter_in"),
            ("outer_diameter", "monopile_outer_diameter"),
            ("wall_thickness", "monopile_wall_thickness"),
            ("shell_mass_user", "monopile_mass_user"),
        ]
        if n_height > 2:
            promlist += ["thickness_slope"]

        temp_opt = mod_opt.copy()
        temp_opt["n_height"] = [n_height]
        temp_opt["n_layers"] = [mod_opt["n_layers"]]
        temp_opt["n_ballasts"] = [0]
        self.add_subsystem(
            "member",
            mem.MemberStandard(column_options=temp_opt, idx=0, n_mat=n_mat, n_refine=nref, member_shape = "circular"),
            promotes=promlist,
        )

        self.add_subsystem(
            "mono",
            MonopileMass(npts=n_full),
            promotes=[
                "transition_piece_mass",
                "transition_piece_cost",
                "gravity_foundation_mass",
                "z_full",
                "outer_diameter_full",
                "tower_mass",
                "tower_cost",
                "monopile_mass",
                "monopile_cost",
                "monopile_z_cg",
                "monopile_I_base",
                "transition_piece_I",
                "gravity_foundation_I",
                "structural_mass",
                "structural_cost",
            ],
        )

        self.connect("member.shell_mass", "mono.cylinder_mass")
        self.connect("member.shell_cost", "mono.cylinder_cost")
        self.connect("member.shell_z_cg", "mono.cylinder_z_cg")
        self.connect("member.shell_I_base", "mono.cylinder_I_base")

        if mod_opt["soil_springs"]:
            self.add_subsystem(
                "soil",
                TowerSoil(npts=NPTS_SOIL),
                promotes=[("G", "G_soil"), ("nu", "nu_soil"), ("depth", "suctionpile_depth")],
            )
            self.connect("outer_diameter_full", "soil.d0", src_indices=[0])



class MonopileSEPerf(om.Group):
    """
    This is the main MonopileSE group that performs analysis of the monopile.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["FixedBottomSE"]
        nLC = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        wind = mod_opt["wind"]  # not yet supported
        frame3dd_opt = mod_opt["frame3dd"]
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_mono = mod_opt["n_height_monopile"]
            n_height = mod_opt["n_height"] = n_height_mono
        nref = mod_opt["n_refine"]
        n_full = mem.get_nfull(n_height, nref=nref)

        self.add_subsystem("loads", mem.MemberLoads(n_full=n_full, n_lc=nLC, wind=wind, hydro=True), promotes=["*"])

        n_full_tow = 0
        if self.options["modeling_options"]["flags"]["tower"]:
            tow_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
            n_height = tow_opt["n_height"]
            n_full_tow = mem.get_nfull(n_height, nref=tow_opt["n_refine"])

        self.add_subsystem(
            "monopile",
            MonopileFrame(
                n_refine=nref,
                n_full=n_full,
                n_full_tow=n_full_tow,
                frame3dd_opt=frame3dd_opt,
                soil_springs=mod_opt["soil_springs"],
                gravity_foundation=mod_opt["gravity_foundation"],
                nLC=nLC,
            ),
            promotes=[
                "nodes_xyz",
                "tower_xyz",
                "section_A",
                "section_Asx",
                "section_Asy",
                "section_Ixx",
                "section_Iyy",
                "section_J0",
                "section_rho",
                "section_E",
                "section_G",
                "outer_diameter_full",
                "t_full",
                "sigma_y_full",
                "tower_outer_diameter_full",
                "tower_t_full",
                "tower_sigma_y_full",
                "bending_height",
                "tower_bending_height",
                "qdyn",
                "tower_qdyn",
                "tower_A",
                "tower_Asx",
                "tower_Asy",
                "tower_Ixx",
                "tower_Iyy",
                "tower_J0",
                "tower_rho",
                "tower_E",
                "tower_G",
                "transition_piece_height",
                "transition_piece_mass",
                "transition_piece_I",
                "gravity_foundation_mass",
                "gravity_foundation_I",
                "suctionpile_depth",
                "Px",
                "Py",
                "Pz",
                "tower_Px",
                "tower_Py",
                "tower_Pz",
                "turbine_mass",
                "turbine_cg",
                "turbine_I",
                "turbine_F",
                "turbine_M",
                "rna_mass",
                "rna_cg",
                "rna_I",
                "structural_frequencies",
                "fore_aft_freqs",
                "side_side_freqs",
                "torsion_freqs",
                "fore_aft_modes",
                "side_side_modes",
                "torsion_modes",
                "tower_fore_aft_modes",
                "tower_side_side_modes",
                "tower_torsion_modes",
                "f1",
                "f2",
                "water_depth"
            ],
        )

        # Compute stress and buckling constraints for monopile alone
        self.add_subsystem(
            "post",
            mem.CylinderPostFrame(modeling_options=mod_opt, n_dlc=nLC, n_full = n_full),
            promotes=[
                "z_full",
                "outer_diameter_full",
                "t_full",
                "rho_full",
                "E_full",
                "G_full",
                "sigma_y_full",
                "bending_height",
                "section_A",
                "section_Asx",
                "section_Asy",
                "section_Ixx",
                "section_Iyy",
                "section_J0",
                "section_rho",
                "section_E",
                "section_G",
                "qdyn",
            ],
        )

        self.connect("monopile.monopile_Fz", "post.cylinder_Fz")
        self.connect("monopile.monopile_Vx", "post.cylinder_Vx")
        self.connect("monopile.monopile_Vy", "post.cylinder_Vy")
        self.connect("monopile.monopile_Mxx", "post.cylinder_Mxx")
        self.connect("monopile.monopile_Myy", "post.cylinder_Myy")
        self.connect("monopile.monopile_Mzz", "post.cylinder_Mzz")

        if self.options["modeling_options"]["flags"]["tower"]:
            # Compute stress and buckling constraints for monopile and tower combined
            self.add_subsystem(
                "post_monopile_tower",
                mem.CylinderPostFrame(modeling_options=mod_opt, n_dlc=nLC, n_full = n_full + n_full_tow - 1),
            )

            self.connect("monopile.monopile_tower_z_full", "post_monopile_tower.z_full")
            self.connect("monopile.monopile_tower_outer_diameter_full", "post_monopile_tower.outer_diameter_full")
            self.connect("monopile.monopile_tower_t_full", "post_monopile_tower.t_full")
            self.connect("monopile.monopile_tower_rho_sec", "post_monopile_tower.rho_full")
            self.connect("monopile.monopile_tower_E_sec", "post_monopile_tower.E_full")
            self.connect("monopile.monopile_tower_G_sec", "post_monopile_tower.G_full")
            self.connect("monopile.monopile_tower_sigma_y_full", "post_monopile_tower.sigma_y_full")
            self.connect("monopile.monopile_tower_A_sec", "post_monopile_tower.section_A")
            self.connect("monopile.monopile_tower_Asx_sec", "post_monopile_tower.section_Asx")
            self.connect("monopile.monopile_tower_Asy_sec", "post_monopile_tower.section_Asy")
            self.connect("monopile.monopile_tower_J0_sec", "post_monopile_tower.section_J0")
            self.connect("monopile.monopile_tower_Ixx_sec", "post_monopile_tower.section_Ixx")
            self.connect("monopile.monopile_tower_Iyy_sec", "post_monopile_tower.section_Iyy")
            self.connect("monopile.monopile_tower_bending_height", "post_monopile_tower.bending_height")
            self.connect("monopile.monopile_tower_qdyn", "post_monopile_tower.qdyn")
            self.connect("monopile.monopile_tower_Fz", "post_monopile_tower.cylinder_Fz")
            self.connect("monopile.monopile_tower_Vx", "post_monopile_tower.cylinder_Vx")
            self.connect("monopile.monopile_tower_Vy", "post_monopile_tower.cylinder_Vy")
            self.connect("monopile.monopile_tower_Mxx", "post_monopile_tower.cylinder_Mxx")
            self.connect("monopile.monopile_tower_Myy", "post_monopile_tower.cylinder_Myy")
            self.connect("monopile.monopile_tower_Mzz", "post_monopile_tower.cylinder_Mzz")



class MonopileSE(om.Group):
    """
    This is the main MonopileSE group that performs analysis of the monopile.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["FixedBottomSE"]

        # Resolve ambiguity: both prop and perf promote rho_water, but only
        # perf's CylinderEnvironment sets a non-zero default (1025).
        self.set_input_defaults("rho_water", 1025.0, units="kg/m**3")

        self.add_subsystem("prop", MonopileSEProp(modeling_options=self.options["modeling_options"]), promotes=["*"])
        self.add_subsystem("perf", MonopileSEPerf(modeling_options=self.options["modeling_options"]), promotes=["*"])

        if mod_opt["soil_springs"]:
            self.connect("soil.z_k", "monopile.z_soil")
            self.connect("soil.k", "monopile.k_soil")
            
