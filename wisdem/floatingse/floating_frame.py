import numpy as np
import openmdao.api as om

import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.utilization_dnvgl as util_dnvgl
import wisdem.commonse.utilization_constraints as util_con
import wisdem.commonse.utilization_eurocode as util_euro
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.cylinder_member import NULL, MEMMAX, MemberLoads, get_nfull
from wisdem.floatingse.floating_system import RIGID, NELEM_MAX, NNODES_MAX


class PlatformLoads(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_dlc = opt["WISDEM"]["n_dlc"]
        n_member = opt["floating"]["members"]["n_members"]

        for k in range(n_member):
            kname = opt["floating"]["members"]["name"][k]
            self.add_input(f"member{k}_{kname}:Px", np.zeros((MEMMAX, n_dlc)), units="N/m")
            self.add_input(f"member{k}_{kname}:Py", np.zeros((MEMMAX, n_dlc)), units="N/m")
            self.add_input(f"member{k}_{kname}:Pz", np.zeros((MEMMAX, n_dlc)), units="N/m")
            self.add_input(f"member{k}_{kname}:qdyn", np.zeros((MEMMAX, n_dlc)), units="Pa")

        self.add_output("platform_elem_Px1", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_output("platform_elem_Px2", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_output("platform_elem_Py1", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_output("platform_elem_Py2", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_output("platform_elem_Pz1", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_output("platform_elem_Pz2", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_output("platform_elem_qdyn", NULL * np.ones((NELEM_MAX, n_dlc)), units="Pa")

    def compute(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_dlc = opt["WISDEM"]["n_dlc"]
        n_member = opt["floating"]["members"]["n_members"]

        # Initialize running lists across all members
        elem_Px1 = np.array([])
        elem_Px2 = np.array([])
        elem_Py1 = np.array([])
        elem_Py2 = np.array([])
        elem_Pz1 = np.array([])
        elem_Pz2 = np.array([])
        elem_qdyn = np.array([])

        # Append all member data
        for k in range(n_member):
            kname = opt["floating"]["members"]["name"][k]
            n = np.where(inputs[f"member{k}_{kname}:qdyn"][:, 0] == NULL)[0][0]
            mem_qdyn, _ = util.nodal2sectional(inputs[f"member{k}_{kname}:qdyn"][:n, :])

            # The loads should come in with length n+1
            if n > 0:
                # Avoid the case that has null starting from first element
                n -= 1
            if k == 0:
                elem_Px1 = inputs[f"member{k}_{kname}:Px"][:n, :]
                elem_Px2 = inputs[f"member{k}_{kname}:Px"][1 : (n + 1), :]
                elem_Py1 = inputs[f"member{k}_{kname}:Py"][:n, :]
                elem_Py2 = inputs[f"member{k}_{kname}:Py"][1 : (n + 1), :]
                elem_Pz1 = inputs[f"member{k}_{kname}:Pz"][:n, :]
                elem_Pz2 = inputs[f"member{k}_{kname}:Pz"][1 : (n + 1), :]
                elem_qdyn = mem_qdyn
            else:
                elem_Px1 = np.vstack((elem_Px1, inputs[f"member{k}_{kname}:Px"][:n, :]))
                elem_Px2 = np.vstack((elem_Px2, inputs[f"member{k}_{kname}:Px"][1 : (n + 1), :]))
                elem_Py1 = np.vstack((elem_Py1, inputs[f"member{k}_{kname}:Py"][:n, :]))
                elem_Py2 = np.vstack((elem_Py2, inputs[f"member{k}_{kname}:Py"][1 : (n + 1), :]))
                elem_Pz1 = np.vstack((elem_Pz1, inputs[f"member{k}_{kname}:Pz"][:n, :]))
                elem_Pz2 = np.vstack((elem_Pz2, inputs[f"member{k}_{kname}:Pz"][1 : (n + 1), :]))
                elem_qdyn = np.vstack((elem_qdyn, mem_qdyn))

        # Store outputs
        nelem = elem_qdyn.size
        outputs["platform_elem_Px1"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_elem_Px2"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_elem_Py1"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_elem_Py2"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_elem_Pz1"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_elem_Pz2"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_elem_qdyn"] = NULL * np.ones((NELEM_MAX, n_dlc))

        outputs["platform_elem_Px1"][:nelem, :] = elem_Px1
        outputs["platform_elem_Px2"][:nelem, :] = elem_Px2
        outputs["platform_elem_Py1"][:nelem, :] = elem_Py1
        outputs["platform_elem_Py2"][:nelem, :] = elem_Py2
        outputs["platform_elem_Pz1"][:nelem, :] = elem_Pz1
        outputs["platform_elem_Pz2"][:nelem, :] = elem_Pz2
        outputs["platform_elem_qdyn"][:nelem, :] = elem_qdyn


class FrameAnalysis(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_attach = opt["mooring"]["n_attach"]
        n_dlc = opt["WISDEM"]["n_dlc"]

        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_hull_center_of_mass", np.zeros(3), units="m")
        self.add_input("platform_added_mass", np.zeros(6), units="kg")

        self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_input("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_input("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        # self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m") # Not used
        self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_L", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_J0", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_TorsC", NULL * np.ones(NELEM_MAX), units="m**3")

        self.add_input("platform_elem_Px1", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_input("platform_elem_Px2", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_input("platform_elem_Py1", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_input("platform_elem_Py2", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_input("platform_elem_Pz1", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")
        self.add_input("platform_elem_Pz2", NULL * np.ones((NELEM_MAX, n_dlc)), units="N/m")

        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("turbine_F", np.zeros((3, n_dlc)), units="N")
        self.add_input("turbine_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_input("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_input("mooring_fairlead_joints", np.zeros((n_attach, 3)), units="m")
        self.add_input("mooring_stiffness", np.zeros((6, 6)), units="N/m")
        self.add_input("variable_ballast_mass", 0.0, units="kg")
        self.add_input("variable_center_of_mass", val=np.zeros(3), units="m")
        self.add_input("variable_I", np.zeros(6), units="kg*m**2")

        self.add_output("platform_base_F", np.zeros((3, n_dlc)), units="N")
        self.add_output("platform_base_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_output("platform_Fz", NULL * np.ones((NELEM_MAX, n_dlc)), units="N")
        self.add_output("platform_Vx", NULL * np.ones((NELEM_MAX, n_dlc)), units="N")
        self.add_output("platform_Vy", NULL * np.ones((NELEM_MAX, n_dlc)), units="N")
        self.add_output("platform_Mxx", NULL * np.ones((NELEM_MAX, n_dlc)), units="N*m")
        self.add_output("platform_Myy", NULL * np.ones((NELEM_MAX, n_dlc)), units="N*m")
        self.add_output("platform_Mzz", NULL * np.ones((NELEM_MAX, n_dlc)), units="N*m")

    def compute(self, inputs, outputs):
        # Unpack variables
        opt = self.options["options"]
        n_attach = opt["mooring"]["n_attach"]
        n_dlc = opt["WISDEM"]["n_dlc"]

        m_trans = float(inputs["transition_piece_mass"][0])
        I_trans = inputs["transition_piece_I"]
        m_variable = float(inputs["variable_ballast_mass"][0])
        cg_variable = inputs["variable_center_of_mass"]
        I_variable = inputs["variable_I"]

        fairlead_joints = inputs["mooring_fairlead_joints"]
        mooringF = inputs["mooring_neutral_load"]
        mooringK = np.abs(np.diag(inputs["mooring_stiffness"]))

        # Create frame3dd instance: nodes, elements, reactions, and options
        nodes = inputs["platform_nodes"]
        nnode = np.where(nodes[:, 0] == NULL)[0][0]
        nodes = nodes[:nnode, :]
        rnode = np.zeros(nnode)  # inputs["platform_Rnode"][:nnode]
        itrans = util.closest_node(nodes, inputs["transition_node"])
        ivariable = util.closest_node(nodes, cg_variable)

        N1 = np.int_(inputs["platform_elem_n1"])
        nelem = np.where(N1 == NULL)[0][0]
        N1 = N1[:nelem]
        N2 = np.int_(inputs["platform_elem_n2"][:nelem])
        A = inputs["platform_elem_A"][:nelem]
        Asx = inputs["platform_elem_Asx"][:nelem]
        Asy = inputs["platform_elem_Asy"][:nelem]
        Ixx = inputs["platform_elem_Ixx"][:nelem]
        Iyy = inputs["platform_elem_Iyy"][:nelem]
        J0 = inputs["platform_elem_J0"][:nelem]
        rho = inputs["platform_elem_rho"][:nelem]
        E = inputs["platform_elem_E"][:nelem]
        G = inputs["platform_elem_G"][:nelem]
        TorsC = inputs["platform_elem_TorsC"][:nelem]
        roll = np.zeros(nelem)
        L = inputs["platform_elem_L"][:nelem]  # np.sqrt(np.sum((nodes[N2, :] - nodes[N1, :]) ** 2, axis=1))

        inodes = np.arange(nnode) + 1
        node_obj = pyframe3dd.NodeData(inodes, nodes[:, 0], nodes[:, 1], nodes[:, 2], rnode)

        ielem = np.arange(nelem) + 1
        elem_obj = pyframe3dd.ElementData(ielem, N1 + 1, N2 + 1, A, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)

        # Use Mooring stiffness (TODO Hydro_K too)
        ind = []
        for k in range(n_attach):
            ind.append(util.closest_node(nodes, fairlead_joints[k, :]))
        rid = np.unique(np.array([ind]))  # np.array([np.argmin(nodes[:, 2])])

        Rx = Ry = Rz = Rxx = Ryy = Rzz = RIGID * np.ones(rid.size)
        # Rx, Ry, Rz = [mooringK[0]], [mooringK[1]], [mooringK[2]]
        # Only this solution works and there isn't much different with fully rigid
        # Rx, Ry, Rz = [RIGID], [RIGID], [mooringK[2]]
        # Rxx, Ryy, Rzz = [RIGID], [RIGID], [RIGID]
        react_obj = pyframe3dd.ReactionData(rid + 1, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)

        frame3dd_opt = opt["WISDEM"]["FloatingSE"]["frame3dd"]
        opt_obj = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], -1.0)

        myframe = pyframe3dd.Frame(node_obj, react_obj, elem_obj, opt_obj)

        # Added mass
        cg_add = np.zeros((3, 2))
        add_gravity = True
        mID = np.array([itrans, ivariable], dtype=np.int_)
        m_add = np.array([m_trans, m_variable])
        I_add = np.c_[I_trans, I_variable]
        myframe.changeExtraNodeMass(
            mID + 1,
            m_add,
            I_add[0, :],
            I_add[1, :],
            I_add[2, :],
            I_add[3, :],
            I_add[4, :],
            I_add[5, :],
            cg_add[0, :],
            cg_add[1, :],
            cg_add[2, :],
            add_gravity,
        )

        # Initialize loading with gravity, mooring line forces, and buoyancy (already in nodal forces)
        Mnode = np.zeros((nnode, 3))
        Fnode = inputs["platform_Fnode"][:nnode, :]
        for k in range(n_attach):
            ind = util.closest_node(nodes, fairlead_joints[k, :])
            Fnode[ind, :] += mooringF[k, :]

        gx = gy = 0.0
        gz = -gravity
        for k in range(n_dlc):
            load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

            Fnode2 = Fnode.copy()
            Fnode2[itrans, :] += inputs["turbine_F"][:, k]
            Mnode[itrans, :] = inputs["turbine_M"][:, k]
            nF = np.where(np.abs(Fnode2).sum(axis=1) > 0.0)[0]
            load_obj.changePointLoads(
                nF + 1, Fnode2[nF, 0], Fnode2[nF, 1], Fnode2[nF, 2], Mnode[nF, 0], Mnode[nF, 1], Mnode[nF, 2]
            )

            # trapezoidally distributed loads
            xx1 = xy1 = xz1 = np.zeros(ielem.size)
            xx2 = xy2 = xz2 = 0.99 * L  # multiply slightly less than unity b.c. of precision
            wx1 = inputs["platform_elem_Px1"][:nelem, k]
            wx2 = inputs["platform_elem_Px2"][:nelem, k]
            wy1 = inputs["platform_elem_Py1"][:nelem, k]
            wy2 = inputs["platform_elem_Py2"][:nelem, k]
            wz1 = inputs["platform_elem_Pz1"][:nelem, k]
            wz2 = inputs["platform_elem_Pz2"][:nelem, k]
            load_obj.changeTrapezoidalLoads(ielem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            # Add the load case and run
            myframe.addLoadCase(load_obj)

        # myframe.write("system.3dd")
        # myframe.draw()
        displacements, forces, reactions, internalForces, mass, modal = myframe.run()

        # Determine reaction forces
        outputs["platform_base_F"] = -np.c_[
            reactions.Fx.sum(axis=1), reactions.Fy.sum(axis=1), reactions.Fz.sum(axis=1)
        ].T
        outputs["platform_base_M"] = -np.c_[
            reactions.Mxx.sum(axis=1), reactions.Myy.sum(axis=1), reactions.Mzz.sum(axis=1)
        ].T

        # Forces and moments along the structure
        outputs["platform_Fz"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_Vx"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_Vy"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_Mxx"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_Myy"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["platform_Mzz"] = NULL * np.ones((NELEM_MAX, n_dlc))
        for k in range(n_dlc):
            outputs["platform_Fz"][:nelem, k] = forces.Nx[k, 1::2]
            outputs["platform_Vx"][:nelem, k] = -forces.Vz[k, 1::2]
            outputs["platform_Vy"][:nelem, k] = forces.Vy[k, 1::2]
            outputs["platform_Mxx"][:nelem, k] = -forces.Mzz[k, 1::2]
            outputs["platform_Myy"][:nelem, k] = forces.Myy[k, 1::2]
            outputs["platform_Mzz"][:nelem, k] = forces.Txx[k, 1::2]


class TowerModal(om.ExplicitComponent):
    """
    Run Frame3DD on the floating tower for frequencies and mode shapes only

    Parameters
    ----------
    z_full : numpy array[npts], [m]
        location along cylinder. start at bottom and go to top
    TODO: not needed?
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
    """

    def initialize(self):
        self.options.declare("n_full")
        self.options.declare("frame3dd_opt")

    def setup(self):
        n_full = self.options["n_full"]

        # cross-sectional data along cylinder.
        self.add_input("tower_xyz", np.zeros((n_full, 3)), units="m")
        self.add_input("tower_A", np.zeros(n_full - 1), units="m**2")
        self.add_input("tower_Asx", np.zeros(n_full - 1), units="m**2")
        self.add_input("tower_Asy", np.zeros(n_full - 1), units="m**2")
        self.add_input("tower_Ixx", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("tower_Iyy", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("tower_J0", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("tower_rho", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("tower_E", np.zeros(n_full - 1), units="Pa")
        self.add_input("tower_G", np.zeros(n_full - 1), units="Pa")
        self.add_output("tower_L", np.zeros(n_full - 1), units="m")

        self.add_input("rna_mass", val=0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_added_mass", np.zeros(6), units="kg")
        self.add_input("platform_total_center_of_mass", np.zeros(3), units="m")
        self.add_input("platform_I_total", np.zeros(6), units="kg*m**2")
        self.add_input("mooring_stiffness", np.zeros((6, 6)), units="N/m")

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

    def compute(self, inputs, outputs):
        frame3dd_opt = self.options["frame3dd_opt"]

        # ------- node data ----------------
        xyz = inputs["tower_xyz"]
        n = xyz.shape[0]
        node = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node, xyz[:, 0], xyz[:, 1], xyz[:, 2], r)
        # -----------------------------------

        # ------ reaction data ------------
        # free-free (no reactions)
        rnode = np.array([], dtype=np.int_)
        kx = ky = kz = ktx = kty = ktz = rnode
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)
        # rnode = np.array([1], dtype=np.int_)
        # moorK = np.abs(np.diag(inputs["mooring_stiffness"]))
        # reactions = pyframe3dd.ReactionData(
        #    rnode, [moorK[0]], [moorK[1]], [moorK[2]], [moorK[3]], [moorK[4]], [moorK[5]], rigid=RIGID
        # )
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)

        # Element properties
        Area = inputs["tower_A"]
        Asx = inputs["tower_Asx"]
        Asy = inputs["tower_Asy"]
        J0 = inputs["tower_J0"]
        Ixx = inputs["tower_Ixx"]
        Iyy = inputs["tower_Iyy"]
        E = inputs["tower_E"]
        G = inputs["tower_G"]
        rho = inputs["tower_rho"]

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], dx)
        # -----------------------------------

        # initialize frame3dd object
        myframe = pyframe3dd.Frame(nodes, reactions, elements, options)

        # ------- enable dynamic analysis ----------
        # Run extra freqs because could get 6 rigid body modes at zero-freq
        lump = 0
        myframe.enableDynamics(
            3 * NFREQ, frame3dd_opt["modal_method"], lump, frame3dd_opt["tol"], frame3dd_opt["shift"]
        )
        # ----------------------------

        # ------ static load case 1 ------------
        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity
        load = pyframe3dd.StaticLoadCase(gx, gy, gz)
        myframe.addLoadCase(load)

        # Added mass
        cg_add = np.c_[inputs["platform_total_center_of_mass"], inputs["rna_cg"]]
        add_gravity = False
        mID = np.array([1, n - 1], dtype=np.int_)
        # Scale the physical mass with the added mass
        m_fact1 = (inputs["platform_added_mass"][:3] / inputs["platform_mass"][0]).max()
        m_add = np.r_[(1 + m_fact1) * inputs["platform_mass"], inputs["rna_mass"]].flatten()
        # Scale the physical moment of inertia with the added mass
        m_fact2 = (inputs["platform_added_mass"][3:] / inputs["platform_I_total"][:3]).max()
        I_add = np.c_[(1 + m_fact2) * inputs["platform_I_total"], inputs["rna_I"]]
        myframe.changeExtraNodeMass(
            mID,
            m_add,
            I_add[0, :],
            I_add[1, :],
            I_add[2, :],
            I_add[3, :],
            I_add[4, :],
            I_add[5, :],
            cg_add[0, :],
            cg_add[1, :],
            cg_add[2, :],
            add_gravity,
        )

        # Debugging
        # myframe.write("floating_tower_debug.3dd")
        # -----------------------------------
        # run the analysis
        try:
            _, _, _, _, _, modal = myframe.run()

            # natural frequncies
            freq = modal.freq
            freq = freq[freq > 1e-1]
            if len(freq) >= NFREQ:
                outputs["f1"] = freq[0]
                outputs["f2"] = freq[1]
                outputs["structural_frequencies"] = freq[:NFREQ]

                # Get all mode shapes in batch
                NFREQ2 = int(NFREQ / 2)
                myzmpf = 1e-7*modal.zmpf # zmpf is buggy here, so suppressing it
                freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
                    xyz[:, 2], modal.freq,
                    modal.xdsp, modal.ydsp, modal.zdsp,
                    modal.xmpf, modal.ympf, myzmpf, #modal.zmpf,
                    base_slope0=False,
                    skip_duplicates=frame3dd_opt["skip_duplicate_modes"]
                )

                outputs["fore_aft_freqs"] = freq_x[:NFREQ2]
                outputs["side_side_freqs"] = freq_y[:NFREQ2]
                outputs["torsion_freqs"] = freq_z[:NFREQ2]
                outputs["fore_aft_modes"] = mshapes_x[:NFREQ2, :]
                outputs["side_side_modes"] = mshapes_y[:NFREQ2, :]
                outputs["torsion_modes"] = mshapes_z[:NFREQ2, :]
        except Exception:
            pass


class FloatingPost(om.ExplicitComponent):
    # Perform post structural design checks for the floating platform frame (all members)
    # Mainly vonMises stress criteria and buckling check
    # Every member has diameter and side lengths (a and b) variables but circular members have zeros for side-lengths
    # and rectangular have zeros for diameters.
    def initialize(self):
        self.options.declare("options")
        self.options.declare("shape")
        self.options.declare("n_dlc")

    def setup(self):
        n_dlc = self.options["n_dlc"]

        self.add_input("platform_elem_L", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_a", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_b", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_J0", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_TorsC", NULL * np.ones(NELEM_MAX), units="m**3")
        self.add_input("platform_elem_sigma_y", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_qdyn", NULL * np.ones((NELEM_MAX, n_dlc)), units="Pa")

        # Processed Frame3DD/OpenFAST outputs
        self.add_input("platform_Fz", NULL * np.ones((NELEM_MAX, n_dlc)), units="N")
        self.add_input("platform_Vx", NULL * np.ones((NELEM_MAX, n_dlc)), units="N")
        self.add_input("platform_Vy", NULL * np.ones((NELEM_MAX, n_dlc)), units="N")
        self.add_input("platform_Mxx", NULL * np.ones((NELEM_MAX, n_dlc)), units="N*m")
        self.add_input("platform_Myy", NULL * np.ones((NELEM_MAX, n_dlc)), units="N*m")
        self.add_input("platform_Mzz", NULL * np.ones((NELEM_MAX, n_dlc)), units="N*m")

        self.add_output("constr_platform_stress", NULL * np.ones((NELEM_MAX, n_dlc)))
        self.add_output("constr_platform_shell_buckling", NULL * np.ones((NELEM_MAX, n_dlc)))
        self.add_output("constr_platform_global_buckling", NULL * np.ones((NELEM_MAX, n_dlc)))

    def compute(self, inputs, outputs):
        # Unpack some variables
        opt = self.options["options"]
        n_dlc = self.options["n_dlc"]
        shape = self.options["shape"]
        gamma_f = opt["gamma_f"]
        gamma_m = opt["gamma_m"]
        gamma_n = opt["gamma_n"]
        gamma_b = opt["gamma_b"]

        d = inputs["platform_elem_D"]
        a = inputs["platform_elem_a"]
        b = inputs["platform_elem_b"]
        nelem = np.where(d == NULL)[0][0]
        d = d[:nelem]
        a = a[:nelem]
        b = b[:nelem]
        t = inputs["platform_elem_t"][:nelem]
        h = inputs["platform_elem_L"][:nelem]
        Az = inputs["platform_elem_A"][:nelem]
        Asx = inputs["platform_elem_Asx"][:nelem]
        Asy = inputs["platform_elem_Asy"][:nelem]
        Jz = inputs["platform_elem_J0"][:nelem]
        Ixx = inputs["platform_elem_Ixx"][:nelem]
        Iyy = inputs["platform_elem_Iyy"][:nelem]
        sigy = inputs["platform_elem_sigma_y"][:nelem]
        E = inputs["platform_elem_E"][:nelem]
        G = inputs["platform_elem_G"][:nelem]
        TorsC = inputs["platform_elem_TorsC"][:nelem]
        qdyn = inputs["platform_elem_qdyn"][:nelem, :]
        r = 0.5 * d

        # Get loads from Framee3dd/OpenFAST
        Fz = inputs["platform_Fz"][:nelem, :]
        Vx = inputs["platform_Vx"][:nelem, :]
        Vy = inputs["platform_Vy"][:nelem, :]
        Mxx = inputs["platform_Mxx"][:nelem, :]
        Myy = inputs["platform_Myy"][:nelem, :]
        Mzz = inputs["platform_Mzz"][:nelem, :]

        M = np.sqrt(Mxx**2 + Myy**2)
        V = np.sqrt(Vx**2 + Vy**2)

        # Initialize outputs
        outputs["constr_platform_stress"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["constr_platform_shell_buckling"] = NULL * np.ones((NELEM_MAX, n_dlc))
        outputs["constr_platform_global_buckling"] = NULL * np.ones((NELEM_MAX, n_dlc))

        # See http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        # print(Fz.shape, Az.shape, M.shape, r.shape, Iyy.shape)
                # Change to element-wise calculation
        # Get indices for circular member
        circ_idx = np.nonzero(d)[0]
        rect_idx = np.nonzero(a)[0]

        if circ_idx.any():
            # First do caculation for circular members
            circ_axial_stress = Fz[circ_idx] / Az[circ_idx].reshape(-1,1) + M[circ_idx] * (r[circ_idx] / Iyy[circ_idx]).reshape(-1,1)
            circ_shear_stress = np.abs(Mzz[circ_idx]) / (Jz[circ_idx] * r[circ_idx]).reshape(-1,1) + V[circ_idx] / Asx[circ_idx].reshape(-1,1)
            hoop_stress = -qdyn[circ_idx] * ((r[circ_idx] - 0.5 * t[circ_idx]) / t[circ_idx]).reshape(-1,1)  # util_con.hoopStress(d, t, qdyn)
            outputs["constr_platform_stress"][[circ_idx], :] = util_con.TubevonMisesStressUtilization(
                circ_axial_stress, hoop_stress, circ_shear_stress, gamma_f * gamma_m * gamma_n, sigy[circ_idx].reshape((-1, 1))
            )

            # Use DNV-GL CP202 Method
            circ_check = util_dnvgl.CylinderBuckling(h[circ_idx], d[circ_idx], t[circ_idx], E=E[circ_idx], G=G[circ_idx],
                                            sigma_y=sigy[circ_idx], gamma=gamma_f * gamma_b,
                                            A=Az[circ_idx], I=Iyy[circ_idx])
            for k in range(n_dlc):
                results = circ_check.run_buckling_checks(Fz[k, :], M[k, :], hoop_stress[k, :], circ_shear_stress[k, :])

                outputs["constr_platform_shell_buckling"][[circ_idx], k] = results["Shell"]
                outputs["constr_platform_global_buckling"][[circ_idx], k] = results["Global"]

        # Do calculation for rectangular members
        # Assuming a linear summation of bending on axial stress
        if rect_idx.any():
            rect_axial_stress = Fz[rect_idx] / Az[rect_idx, np.newaxis] + np.abs(Mxx[rect_idx]) * (a[rect_idx] / Ixx[rect_idx])[:, np.newaxis] + np.abs(Myy[rect_idx]) * (b[rect_idx] / Iyy[rect_idx])[:, np.newaxis]
            rect_shear_stress_x = np.abs(Mzz[rect_idx]) / TorsC[rect_idx][:, np.newaxis] + Vx[rect_idx] / Asx[rect_idx, np.newaxis]
            rect_shear_stress_y = np.abs(Mzz[rect_idx]) / TorsC[rect_idx][:, np.newaxis] + Vy[rect_idx] / Asy[rect_idx, np.newaxis]
            rect_shear_stress = np.sqrt(rect_shear_stress_x**2+rect_shear_stress_y**2)
            # Pick the maximum from sigma_y and sigma_z as the transverse sigma
            rect_transverse_stress = np.minimum(-qdyn[rect_idx,0]*a[rect_idx]/2*t[rect_idx], -qdyn[rect_idx,0]*b[rect_idx]/2*t[rect_idx]).reshape(-1,1)
            outputs["constr_platform_stress"][rect_idx, :] = util_euro.YieldCriterionEurocode(
                rect_axial_stress, rect_transverse_stress, rect_shear_stress, gamma_f * gamma_m * gamma_n, sigy[rect_idx].reshape((-1, 1))
            )

            # Use Eurocode Method
            rect_check = util_euro.memberBuckling(a[rect_idx], b[rect_idx], h[rect_idx], Az[rect_idx], Ixx[rect_idx], Iyy[rect_idx], Fz[rect_idx,0],
                                                Mxx[rect_idx,0], Myy[rect_idx,0], I_T=TorsC[rect_idx], E=E[rect_idx], G=G[rect_idx], sigma_y=sigy[rect_idx], gamma_m=gamma_m)
            for k in range(n_dlc):
                results = rect_check.run_buckling_checks(
                    Fz[k, :], Mxx[k, :], Myy[k, :])
                # No shell buckling for rectangular members
                outputs["constr_platform_global_buckling"][[rect_idx], k] = results        


class FloatingFrame(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]
        nLC = opt["WISDEM"]["n_dlc"]
        n_member = opt["floating"]["members"]["n_members"]
        frame3dd_opt = opt["WISDEM"]["FloatingSE"]["frame3dd"]

        mem_vars = ["Px", "Py", "Pz", "qdyn"]

        mem_prom_base = [
            "wind_reference_height",
            "z0",
            "shearExp",
            "rho_air",
            "rho_water",
            "mu_air",
            "mu_water",
            "beta_wind",
            "beta_wave",
            "Uc",
            "Hsig_wave",
            "Tsig_wave",
            "water_depth",
            "yaw",
        ]


        U_prom = []
        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)
            U_prom.append(f"env{lc}.Uref")

        for k in range(n_member):
            n_full = get_nfull(opt["floating"]["members"]["n_height"][k], nref=2)
            shape = opt["floating"]["members"]["outer_shape"][k]
            kname = opt['floating']['members']['name'][k]

            self.add_subsystem(
                f"memload{k}",
                MemberLoads(
                    n_full=n_full,
                    n_lc=nLC,
                    hydro=True,
                    memmax=True,
                    member_shape=shape,
                ),
                promotes= mem_prom_base + U_prom + [("joint1", f"member{k}_{kname}:joint1"), ("joint2", f"member{k}_{kname}:joint2")],
            )

        self.add_subsystem("loadsys", PlatformLoads(options=opt), promotes=["*"])

        self.add_subsystem("frame", FrameAnalysis(options=opt), promotes=["*"])

        if self.options["modeling_options"]["flags"]["tower"]:
            tow_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
            n_height = tow_opt["n_height"]
            n_full_tow = get_nfull(n_height, nref=tow_opt["n_refine"])
            self.add_subsystem(
                "tower",
                TowerModal(n_full=n_full_tow, frame3dd_opt=frame3dd_opt),
                promotes=["*"],
            )

        self.add_subsystem("post", FloatingPost(options=opt["WISDEM"]["FloatingSE"], shape = opt["floating"]["members"]["outer_shape"], n_dlc=nLC), promotes=["*"])

        for k in range(n_member):
            kname = opt['floating']['members']['name'][k]
            for var in mem_vars:
                self.connect(f"memload{k}.{var}", f"member{k}_{kname}:{var}")
