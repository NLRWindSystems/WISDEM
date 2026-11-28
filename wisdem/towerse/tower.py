import numpy as np
import openmdao.api as om

import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.cylinder_member as mem
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.cylinder_member import get_nfull
import copy

RIGID = 1e30

class PreDiscretization(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.

    Parameters
    ----------
    hub_height : float, [m]
        Scalar of the rotor apex height computed along the z axis.
    tower_height : float, [m]
        Scalar of the tower height computed along the z axis.
    foundation_height : float, [m]
        starting height of tower

    Returns
    -------
    height_constraint : float, [m]
        mismatch between tower height and desired hub_height
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint2 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member

    """

    def setup(self):
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("tower_height", val=0.0, units="m")
        self.add_input("foundation_height", val=0.0, units="m")

        self.add_output("height_constraint", val=0.0, units="m")
        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("z_start", 0.0, units="m")
        self.add_output("joint1", val=np.zeros(3), units="m")
        self.add_output("joint2", val=np.zeros(3), units="m")

    def compute(self, inputs, outputs):
        # Unpack values
        h_tow = inputs["tower_height"]
        fh_tow = inputs["foundation_height"]

        outputs["transition_piece_height"] = outputs["z_start"] = fh_tow
        outputs["joint1"] = np.r_[0.0, 0.0, fh_tow]
        outputs["joint2"] = np.r_[0.0, 0.0, fh_tow + h_tow]
        outputs["height_constraint"] = inputs["hub_height"] - outputs["joint2"][-1]



class TurbineMass(om.ExplicitComponent):
    """
    Compute the turbine mass, center of mass, and mass moment of inertia.

    Parameters
    ----------
    hub_height : float, [m]
        Hub-height
    rna_mass : float, [kg]
        Total tower mass
    rna_I : numpy array[6], [kg*m**2]
        Mass moment of inertia of RNA about tower top [xx yy zz xy xz yz]
    rna_cg : numpy array[3], [m]
        xyz-location of RNA cg relative to tower top
    tower_mass : float, [kg]
        Total tower mass
    tower_center_of_mass : float, [m]
        z-position of center of mass of tower
    tower_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]

    Returns
    -------
    turbine_mass : float, [kg]
        Total mass of tower+rna
    turbine_center_of_mass : numpy array[3], [m]
        xyz-position of tower+rna center of mass
    turbine_I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of tower about base [xx yy zz xy xz yz]

    """

    def setup(self):
        self.add_input("joint2", val=np.zeros(3), units="m")
        self.add_input("rna_mass", val=0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("tower_center_of_mass", val=0.0, units="m")
        self.add_input("tower_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("turbine_mass", val=0.0, units="kg")
        self.add_output("turbine_center_of_mass", val=np.zeros(3), units="m")
        self.add_output("turbine_I_base", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack variables
        m_rna = inputs["rna_mass"]
        m_tow = inputs["tower_mass"]

        outputs["turbine_mass"] = m_turb = m_rna + m_tow

        cg_rna = inputs["rna_cg"] + inputs["joint2"]
        cg_tower = np.r_[0.0, 0.0, inputs["tower_center_of_mass"]]
        outputs["turbine_center_of_mass"] = (m_rna * cg_rna + m_tow * cg_tower) / m_turb

        R = inputs["joint2"]  # rna_I is already at tower top, so R goes to tower top, not rna_cg
        I_tower = util.assembleI(inputs["tower_I_base"])
        I_rna = util.assembleI(inputs["rna_I"]) + m_rna * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["turbine_I_base"] = util.unassembleI(I_tower + I_rna)
        

class TowerFrame(om.ExplicitComponent):
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
    rna_mass : float, [kg]
        Total tower mass
    rna_I : numpy array[6], [kg*m**2]
        Mass moment of inertia of RNA about tower top [xx yy zz xy xz yz]
    rna_cg : numpy array[3], [m]
        xyz-location of RNA cg relative to tower top
    rna_F : numpy array[3], [N]
        rna force at tower top from drivetrain analysis
    rna_M : numpy array[3], [N*m]
        rna moment at tower top from drivetrain analysis
    Px : numpy array[n_full], [N/m]
        x-force per unit length in member c.s. (x-direction along element length)
    Py : numpy array[n_full], [N/m]
        y-force per unit length in member c.s. (x-direction along element length)
    Pz : numpy array[n_full], [N/m]
        z-force per unit length in member c.s. (x-direction along element length)

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
    tower_deflection : numpy array[n_full], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    tower_Fz : numpy array[n_full-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    tower_Vx : numpy array[n_full-1], [N]
        Shear force in x-direction in cylinder structure.
    tower_Vy : numpy array[n_full-1], [N]
        Shear force in y-direction in cylinder structure.
    tower_Mxx : numpy array[n_full-1], [N*m]
        Moment about x-axis in cylinder structure.
    tower_Myy : numpy array[n_full-1], [N*m]
        Moment about y-axis in cylinder structure.
    tower_Mzz : numpy array[n_full-1], [N*m]
        Moment about z-axis in cylinder structure.
    base_F : numpy array[3], [N]
        Total force on cylinder
    base_M : numpy array[3], [N*m]
        Total moment on cylinder measured at base
    """

    def initialize(self):
        self.options.declare("n_full")
        self.options.declare("nLC")
        self.options.declare("frame3dd_opt")

    def setup(self):
        n_full = self.options["n_full"]
        nLC = self.options["nLC"]
        self.frame = None

        # cross-sectional data along cylinder.
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
        self.add_output("section_L", np.zeros(n_full - 1), units="m")

        # Add point masses
        self.add_input("lumped_mass", np.zeros(n_full), units="kg")

        # For modal analysis only
        self.add_input("rna_mass", val=0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")
        # point loads
        self.add_input("rna_F", np.zeros((3, nLC)), units="N")
        self.add_input("rna_M", np.zeros((3, nLC)), units="N*m")

        # combined wind-water distributed loads
        self.add_input("Px", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("Py", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("Pz", val=np.zeros((n_full, nLC)), units="N/m")

        # Frequencies
        NFREQ2 = int(NFREQ / 2)
        self.add_output("f1", val=0.0, units="Hz")
        self.add_output("f2", val=0.0, units="Hz")
        self.add_output("structural_frequencies", np.zeros(NFREQ), units="Hz")
        self.add_output("fore_aft_modes", np.zeros((NFREQ2, 5)))
        self.add_output("side_side_modes", np.zeros((NFREQ2, 5)))
        self.add_output("torsion_modes", np.zeros((NFREQ2, 5)))
        self.add_output("fore_aft_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("side_side_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("torsion_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("tower_deflection", np.zeros((n_full, nLC)), units="m")
        self.add_output("top_deflection", np.zeros(nLC), units="m")
        self.add_output("tower_Fz", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("tower_Vx", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("tower_Vy", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("tower_Mxx", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("tower_Myy", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("tower_Mzz", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("turbine_F", val=np.zeros((3, nLC)), units="N")
        self.add_output("turbine_M", val=np.zeros((3, nLC)), units="N*m")

    def compute(self, inputs, outputs):
        frame3dd_opt = self.options["frame3dd_opt"]
        nLC = self.options["nLC"]

        # ------- node data ----------------
        xyz = inputs["nodes_xyz"]
        n = xyz.shape[0]
        node = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node, xyz[:, 0], xyz[:, 1], xyz[:, 2], r)
        # -----------------------------------

        # ------ reaction data ------------
        # rigid base
        rnode = np.array([0], dtype=np.int_) + 1  # 1-based indexing
        kx = ky = kz = ktx = kty = ktz = np.array([RIGID])
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)

        # Element properties
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
        shift = 0.0
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

            # Prepare point forces at RNA node
            rna_F = inputs["rna_F"][:, k]
            rna_M = inputs["rna_M"][:, k]
            load.changePointLoads(
                np.array([n], dtype=np.int_),  # -1 b/c crash if added at final node
                np.array([rna_F[0]]),
                np.array([rna_F[1]]),
                np.array([rna_F[2]]),
                np.array([rna_M[0]]),
                np.array([rna_M[1]]),
                np.array([rna_M[2]]),
            )

            # distributed loads
            Px, Py, Pz = inputs["Px"][:, k], inputs["Py"][:, k], inputs["Pz"][:, k]

            # trapezoidally distributed loads
            EL = np.arange(1, n)
            xx1 = xy1 = xz1 = np.zeros(n - 1)
            xx2 = xy2 = xz2 = 0.99 * L  # subtract small number b.c. of precision
            wx1 = Px[:-1]
            wx2 = Px[1:]
            wy1 = Py[:-1]
            wy2 = Py[1:]
            wz1 = Pz[:-1]
            wz2 = Pz[1:]

            load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            self.frame.addLoadCase(load)

        # Add lumped masses
        # Add RNA mass to lumped mass
        total_lumped_mass = copy.copy(inputs["lumped_mass"])
        total_lumped_mass[-1] += inputs["rna_mass"][0] 
        
        # Add mass for modal analysis only (loads are captured in rna_F & rna_M)
        mID = np.arange(n, dtype=np.int_)+1 # 1-based indexing for frame3dd
        m_add = total_lumped_mass
        cg_add = np.zeros([3, n])
        if total_lumped_mass[-1] > 0:
            cg_add[:,n-1] = inputs["rna_cg"]/total_lumped_mass[-1]
        I_add = np.zeros([6,n])
        I_add[:,n-1] = inputs["rna_I"]
        add_gravity = False
        self.frame.changeExtraNodeMass(
            mID,
            m_add,
            I_add[0,:],
            I_add[1,:],
            I_add[2,:],
            I_add[3,:],
            I_add[4,:],
            I_add[5,:],
            cg_add[0,:],
            cg_add[1,:],
            cg_add[2,:],
            add_gravity,
        )

        # Debugging
        # self.frame.write('tower_debug.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = self.frame.run()

        # natural frequncies
        outputs["f1"] = modal.freq[0]
        outputs["f2"] = modal.freq[1]
        outputs["structural_frequencies"] = modal.freq[:NFREQ]

        # Get all mode shapes in batch
        NFREQ2 = int(NFREQ / 2)
        freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
            xyz[:, 2], modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf,
        )
        outputs["fore_aft_freqs"] = freq_x[:NFREQ2]
        outputs["side_side_freqs"] = freq_y[:NFREQ2]
        outputs["torsion_freqs"] = freq_z[:NFREQ2]
        outputs["fore_aft_modes"] = mshapes_x[:NFREQ2, :]
        outputs["side_side_modes"] = mshapes_y[:NFREQ2, :]
        outputs["torsion_modes"] = mshapes_z[:NFREQ2, :]

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs["tower_deflection"] = np.sqrt(displacements.dx**2 + displacements.dy**2).T
        outputs["top_deflection"] = outputs["tower_deflection"][-1, :]

        # Record total forces and moments at base
        outputs["turbine_F"] = -np.c_[reactions.Fx[:, 0], reactions.Fy[:, 0], reactions.Fz[:, 0]].T
        outputs["turbine_M"] = -np.c_[reactions.Mxx[:, 0], reactions.Myy[:, 0], reactions.Mzz[:, 0]].T

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
        outputs["tower_Fz"] = Fz
        outputs["tower_Vx"] = Vx
        outputs["tower_Vy"] = Vy
        outputs["tower_Mxx"] = Mxx
        outputs["tower_Myy"] = Myy
        outputs["tower_Mzz"] = Mzz


class TowerSEProp(om.Group):
    """
    This is the main TowerSE group that performs analysis of the tower.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
        n_mat = self.options["modeling_options"]["materials"]["n_mat"]

        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_tow = mod_opt["n_height_tower"]
            n_height = mod_opt["n_height"] = n_height_tow

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
            ("s_in", "tower_s"),
            ("layer_materials", "tower_layer_materials"),
            ("layer_thickness", "tower_layer_thickness"),
            ("section_height", "tower_section_height"),
            ("outer_diameter_in", "tower_outer_diameter_in"),
            ("outer_diameter", "tower_outer_diameter"),
            ("wall_thickness", "tower_wall_thickness"),
            ("shell_mass_user", "tower_mass_user"),
            ("shell_mass", "tower_mass"),
            ("shell_cost", "tower_cost"),
            ("shell_z_cg", "tower_center_of_mass"),
            ("shell_I_base", "tower_I_base"),
        ]
        if n_height > 2:
            promlist += ["thickness_slope"]

        temp_opt = mod_opt.copy()
        temp_opt["n_height"] = [n_height]
        temp_opt["n_layers"] = [mod_opt["n_layers"]]
        temp_opt["n_ballasts"] = [0]

        self.add_subsystem(
            "refine_lumped_mass",
            RefineLumpedMass(modeling_options=mod_opt),
            promotes=["lumped_mass_in", "lumped_mass"],
        )

        self.add_subsystem(
            "member",
            mem.MemberStandard(column_options=temp_opt, idx=0, n_mat=n_mat, n_refine=mod_opt["n_refine"], member_shape = "circular"),
            promotes=promlist
        )

        # Add lumped mass into mass_den
        self.add_subsystem(
            "distribute_lumped_mass",
            DistributeLumpedMass(modeling_options=mod_opt),
            promotes=["lumped_mass_in", "total_mass_den"],
        )
        self.connect("member.mass_den", "distribute_lumped_mass.mass_den")
        self.connect("tower_section_height", "distribute_lumped_mass.section_height")


class RefineLumpedMass(om.ExplicitComponent):
    """
    This is to refine the lumped mass considering the refinement

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]

        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_tow = mod_opt["n_height_tower"]
            n_height = mod_opt["n_height"] = n_height_tow
        n_full = get_nfull(n_height, nref=mod_opt["n_refine"])

        self.add_input("lumped_mass_in", np.zeros(n_height), units="kg")
        self.add_output("lumped_mass", np.zeros(n_full), units="kg")
    
    def compute(self, inputs, outputs):
        mod_opt = self.options["modeling_options"]
        n_height = mod_opt["n_height"]
        n_full = get_nfull(n_height, nref=mod_opt["n_refine"])

        # Refine the lumped mass
        lumped_mass_input = inputs["lumped_mass_in"]
        lumped_mass_refined = np.zeros(n_full)
        lumped_mass_refined[0] = lumped_mass_input[0]  # base mass
        for i in range(1, n_height):
            start_idx = (i - 1) * mod_opt["n_refine"]
            end_idx = i * mod_opt["n_refine"]
            lumped_mass_refined[start_idx:end_idx] = 0 # insert zeros to match the refined nodes
            lumped_mass_refined[end_idx - 1] = lumped_mass_input[i]  # last point of the segment
        
        outputs["lumped_mass"] = lumped_mass_refined

class DistributeLumpedMass(om.ExplicitComponent):
    """
    This is to distribute the lumped mass to mass density

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]

        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_tow = mod_opt["n_height_tower"]
            n_height = mod_opt["n_height"] = n_height_tow
        n_full = get_nfull(n_height, nref=mod_opt["n_refine"])

        self.add_input("mass_den", np.zeros(n_height-1), units="kg/m")
        self.add_input("section_height", np.zeros(n_height - 1), units="m")
        self.add_input("lumped_mass_in", np.zeros(n_height), units="kg")
        self.add_output("total_mass_den", np.zeros(n_height-1), units="kg/m")
    
    def compute(self, inputs, outputs):
        mod_opt = self.options["modeling_options"]
        n_height = mod_opt["n_height"]

        # Refine the lumped mass
        lumped_mass_input = inputs["lumped_mass_in"]
        total_mass_den = copy.copy(inputs["mass_den"])

        if lumped_mass_input[0] > 0:
            total_mass_den[0] += lumped_mass_input[0] / inputs["section_height"][0]

        for i in range(1, n_height-1):
            added_mass_den2 = lumped_mass_input[i]*inputs["section_height"][i-1] / (inputs["section_height"][i-1]*inputs["section_height"][i]+inputs["section_height"][i]**2)
            total_mass_den[i] += added_mass_den2 
            total_mass_den[i-1] += (lumped_mass_input[i] - added_mass_den2*inputs["section_height"][i]) / inputs["section_height"][i-1]

        if lumped_mass_input[-1] > 0:
            total_mass_den[-1] += lumped_mass_input[-1] / inputs["section_height"][-1]

        outputs["total_mass_den"] = total_mass_den

class TowerSEPerf(om.Group):
    """
    This is the main TowerSE group that performs analysis of the tower.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
        nLC = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        wind = mod_opt["wind"]  # not yet supported
        frame3dd_opt = mod_opt["frame3dd"]
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_tow = mod_opt["n_height_tower"]
            n_height = mod_opt["n_height"] = n_height_tow
        n_full = mem.get_nfull(n_height, nref=mod_opt["n_refine"])

        self.add_subsystem("turb", TurbineMass(), promotes=["*"])
        
        self.add_subsystem("loads", mem.MemberLoads(n_full=n_full, n_lc=nLC, wind=wind, hydro=False), promotes=["*"])

        self.add_subsystem(
            "tower",
            TowerFrame(n_full=n_full, frame3dd_opt=frame3dd_opt, nLC=nLC),
            promotes=[
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
                "section_L",
                "Px",
                "Py",
                "Pz",
                "rna_mass",
                "rna_cg",
                "rna_I",
                "lumped_mass",
            ],
        )

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
                "section_A",
                "section_Asx",
                "section_Asy",
                "section_Ixx",
                "section_Iyy",
                "section_J0",
                "section_rho",
                "section_E",
                "section_G",
                "section_L",
                "qdyn",
                ("bending_height", "tower_height"),
            ],
        )

        self.connect("tower.tower_Fz", "post.cylinder_Fz")
        self.connect("tower.tower_Vx", "post.cylinder_Vx")
        self.connect("tower.tower_Vy", "post.cylinder_Vy")
        self.connect("tower.tower_Mxx", "post.cylinder_Mxx")
        self.connect("tower.tower_Myy", "post.cylinder_Myy")
        self.connect("tower.tower_Mzz", "post.cylinder_Mzz")


class TowerSE(om.Group):
    """
    This is the main TowerSE group that performs analysis of the tower.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        self.add_subsystem("props", TowerSEProp(modeling_options=self.options["modeling_options"]), promotes=["*"])
        self.add_subsystem("perf", TowerSEPerf(modeling_options=self.options["modeling_options"]), promotes=["*"])


        
        
        
