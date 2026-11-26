"""
This Jacket analysis and design module largely follows the work presented in:
"A systematic approach to offshore wind turbine jacket predesign and optimization:
geometry, cost, and surrogate structural code check models"
by Jan Häfele, Rick R. Damiani, Ryan N. King, Cristian G. Gebhardt, and Raimund Rolfes
Accessible via: https://wes.copernicus.org/articles/3/553/2018/
"""

import numpy as np
import openmdao.api as om

import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.manufacturing as manu
import wisdem.commonse.cross_sections as cs
import wisdem.commonse.utilization_dnvgl as util_dnvgl
import wisdem.commonse.utilization_constraints as util_con
from wisdem.commonse import NFREQ, RIGID, gravity


class PreDiscretization(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.

    Parameters
    ----------
    tower_foundation_height : float, [m]
        Starting z-coordinate value of the tower with 0 at the water line
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    transition_piece_mass : float, [kg]
        Point mass of transition piece
    transition_piece_cost : float, [USD]
        Cost of transition piece

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
        self.add_input("tower_foundation_height", val=0.0, units="m")
        self.add_input("tower_base_diameter", val=0.0, units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")

        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("transition_node", val=np.zeros(3), units="m")
        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack values
        fh_tow = inputs["tower_foundation_height"]
        m_trans = inputs["transition_piece_mass"]
        d_trans = inputs["tower_base_diameter"]

        outputs["transition_piece_height"] = fh_tow
        outputs["transition_node"] = np.r_[0.0, 0.0, fh_tow]

        # Mass properties for transition piece and gravity foundation
        r_trans = 0.5 * d_trans
        I_trans = m_trans * r_trans**2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]  # shell
        outputs["transition_piece_I"] = I_trans


class ComputeJacketNodes(om.ExplicitComponent):
    """"""

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        self.add_input("height", val=70.0, units="m")
        self.add_input("r_head", val=6.0, units="m")
        self.add_input("foot_head_ratio", val=1.5)
        self.add_input("bay_spacing", val=np.linspace(0.1, 0.9, n_bays + 1))
        self.add_input("tower_base_diameter", val=0.0, units="m")

        self.add_output("r_foot", val=10.0, units="m")
        self.add_output("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)), units="m")
        self.add_output("bay_nodes", val=np.zeros((n_legs, n_bays + 1, 3)), units="m")
        self.add_output("constr_diam_consistency", val=0.0)

    def compute(self, inputs, outputs):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        ratio = inputs["foot_head_ratio"]
        r_head = inputs["r_head"]
        outputs["r_foot"] = r_foot = r_head * ratio
        height = inputs["height"]

        leg_spacing = np.linspace(0, 1.0, n_bays + 2)
        z_leg = leg_spacing * height
        x_leg = leg_spacing * (r_head - r_foot) + r_foot
        for i in range(n_legs):
            outputs["leg_nodes"][i, :, 0] = x_leg * np.cos(i / n_legs * 2 * np.pi)
            outputs["leg_nodes"][i, :, 1] = x_leg * np.sin(i / n_legs * 2 * np.pi)
            outputs["leg_nodes"][i, :, 2] = z_leg

        z_bay = inputs["bay_spacing"] * height
        x_bay = inputs["bay_spacing"] * (r_head - r_foot) + r_foot
        for i in range(n_legs):
            outputs["bay_nodes"][i, :, 0] = x_bay * np.cos(i / n_legs * 2 * np.pi)
            outputs["bay_nodes"][i, :, 1] = x_bay * np.sin(i / n_legs * 2 * np.pi)
            outputs["bay_nodes"][i, :, 2] = z_bay

        outputs["constr_diam_consistency"] = inputs["tower_base_diameter"] / r_head


class ComputeFrame3DD(om.ExplicitComponent):
    """
    Now that we have modal information of the jacket structure, we construct the
    Frame3DD problem to solve for the structural properties of the jacket under
    the designated loading conditions.

    This is a lengthy process that requires creating a singular nodal array,
    member information for each brace and leg segment, and bringing in the
    loading from the tower.

    There are n_legs "ghost" members at the top of the jacket structure to connect
    to a ghost node that receives the turbine F and M values. These members are
    rigid and are needed to transmit the loads, but are not included in the mass,
    stress, and buckling calculations.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]
        n_mat = mod_opt["materials"]["n_mat"]

        self.add_input("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)), units="m")
        self.add_input("bay_nodes", val=np.zeros((n_legs, n_bays + 1, 3)), units="m")
        self.add_input("leg_diameter", val=1.4, units="m")
        self.add_input("leg_thickness", val=0.1, units="m")
        self.add_input("brace_diameters", val=np.ones((n_bays)), units="m")
        self.add_input("brace_thicknesses", val=np.ones((n_bays)) * 0.1, units="m")

        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("sigma_y_mat", val=np.zeros(n_mat), units="Pa")

        self.add_input("turbine_F", np.zeros((3, n_dlc)), units="N")
        self.add_input("turbine_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("jacket_mass_user", val=0.0, units="kg")

        # For modal analysis only (loads captured in turbine_F & turbine_M)
        self.add_input("turbine_mass", val=0.0, units="kg")
        self.add_input("turbine_cg", val=np.zeros(3), units="m")
        self.add_input("turbine_I", np.zeros(6), units="kg*m**2")

        n_node = 3 * n_legs * (n_bays + 1) + 1
        self.add_output("jacket_nodes", np.zeros((n_node, 3)), units="m")
        self.add_output("leg_indices", np.zeros(n_legs))

        n_elem = 2 * (n_legs * (n_bays + 1)) + 4 * (n_legs * n_bays) + int(x_mb) * n_legs + n_legs

        self.add_output("jacket_elem_N", np.zeros((n_elem, 2)))
        self.add_output("jacket_elem_L", np.zeros(n_elem), units="m")
        self.add_output("jacket_elem_D", np.zeros(n_elem), units="m")
        self.add_output("jacket_elem_t", np.zeros(n_elem), units="m")
        self.add_output("jacket_elem_A", np.zeros(n_elem), units="m**2")
        self.add_output("jacket_elem_Asx", np.zeros(n_elem), units="m**2")
        self.add_output("jacket_elem_Asy", np.zeros(n_elem), units="m**2")
        self.add_output("jacket_elem_Ixx", np.zeros(n_elem), units="kg*m**2")
        self.add_output("jacket_elem_Iyy", np.zeros(n_elem), units="kg*m**2")
        self.add_output("jacket_elem_J0", np.zeros(n_elem), units="kg*m**2")
        self.add_output("jacket_elem_E", np.zeros(n_elem), units="Pa")
        self.add_output("jacket_elem_G", np.zeros(n_elem), units="Pa")
        self.add_output("jacket_elem_rho", np.zeros(n_elem), units="kg/m**3")
        self.add_output("jacket_elem_sigma_y", np.zeros(n_elem), units="Pa")
        self.add_output("jacket_elem_qdyn", np.zeros((n_elem, n_dlc)), units="Pa")
        self.add_output("jacket_mass", 0.0, units="kg")
        self.add_output("structural_mass", val=0.0, units="kg")

        self.add_output("jacket_base_F", np.zeros((3, n_dlc)), units="N")
        self.add_output("jacket_base_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_output("jacket_Fz", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("jacket_Vx", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("jacket_Vy", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("jacket_Mxx", np.zeros((n_elem, n_dlc)), units="N*m")
        self.add_output("jacket_Myy", np.zeros((n_elem, n_dlc)), units="N*m")
        self.add_output("jacket_Mzz", np.zeros((n_elem, n_dlc)), units="N*m")

        # Frequencies
        self.add_output("f1", val=0.0, units="Hz")
        self.add_output("f2", val=0.0, units="Hz")
        self.add_output("structural_frequencies", np.zeros(NFREQ), units="Hz")

        self.add_output("jacket_deflection", np.zeros((n_node, n_dlc)), units="m")
        self.add_output("top_deflection", np.zeros(n_dlc), units="m")

        self.idx = 0

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]
        material_name = mod_opt["WISDEM"]["FixedBottomSE"]["material"]

        leg_nodes = inputs["leg_nodes"]
        bay_nodes = inputs["bay_nodes"]

        # Add center of x joint nodes, in between bay members.
        x_nodes = np.zeros((n_legs, n_bays, 3))
        for jdx in range(n_bays):
            for idx in range(n_legs):
                n1 = bay_nodes[idx, jdx]
                n2 = bay_nodes[(idx + 1) % n_legs, (jdx + 1) % (n_bays + 1)]

                n3 = bay_nodes[(idx + 1) % n_legs, jdx]
                n4 = bay_nodes[idx, (jdx + 1) % (n_bays + 1)]

                # Filter out division by 0 warnings
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = (n4 - n2) / ((n1 - n2) - (n3 - n4))
                alpha = alpha[0]
                new_node = n4 + alpha * (n3 - n4)
                x_nodes[idx, jdx, :] = new_node

        # Add ghost node for transition piece
        ghost_nodes = np.mean(leg_nodes[:, -1, :], axis=0)
        ghost_nodes[2] += 2.0  # add two meters in the z-direction
        ghost_nodes = ghost_nodes.reshape(1, 1, 3)

        # Concatenate all nodal information together
        xyz = np.vstack(
            (leg_nodes.reshape(-1, 3), bay_nodes.reshape(-1, 3), x_nodes.reshape(-1, 3), ghost_nodes.reshape(-1, 3))
        )
        n = xyz.shape[0]
        node_indices = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node_indices, xyz[:, 0], xyz[:, 1], xyz[:, 2], r)
        outputs["jacket_nodes"] = xyz

        # Create arrays to later reference the indices for each node. Needed because
        # Frame3DD expects a singular nodal array but we want to keep information
        # about which nodes belong to the legs, bays, or ghost nodes.
        leg_indices = node_indices[: leg_nodes.size // 3].reshape((n_legs, n_bays + 2))
        bay_indices = node_indices[leg_nodes.size // 3 : leg_nodes.size // 3 + bay_nodes.size // 3].reshape(
            (n_legs, n_bays + 1)
        )
        x_indices = node_indices[leg_nodes.size // 3 + bay_nodes.size // 3 : -1].reshape((n_legs, n_bays))
        ghost_indices = np.atleast_2d(node_indices[-1])

        rnode = np.array(leg_indices[:, 0], dtype=np.int_)
        kx = ky = kz = ktx = kty = ktz = np.array([RIGID] * len(rnode))
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)
        outputs["leg_indices"] = rnode
        # ------ frame element data ------------

        self.num_elements = 0
        self.N1 = []
        self.N2 = []
        self.L = []
        self.D = []
        self.t = []
        self.Area = []
        self.Asx = []
        self.Asy = []
        self.J0 = []
        self.Ixx = []
        self.Iyy = []
        self.vol = []

        # Helper function to add an element to all relevant lists.
        def add_element(n1_nodes, n1_indices, n2_nodes, n2_indices, itube, idx1, idx2, idx3, idx4):
            n1 = n1_nodes[idx1, idx2]
            n2 = n2_nodes[idx3, idx4]
            self.N1.append(n1_indices[idx1, idx2])
            self.N2.append(n2_indices[idx3, idx4])
            length = np.linalg.norm(n2 - n1)
            self.L.append(length)
            self.D.append(itube.D)
            self.t.append(itube.t)
            self.Area.append(itube.Area)
            self.Asx.append(itube.Asx)
            self.Asy.append(itube.Asy)
            self.J0.append(itube.J0)
            self.Ixx.append(itube.Ixx)
            self.Iyy.append(itube.Iyy)
            self.vol.append(itube.Area * length)
            self.num_elements += 1

        # Naive for loops to make sure we get indexing right.
        # Could vectorize later as needed.

        # Add leg members.
        for jdx in range(n_bays + 1):
            itube = cs.Tube(float(inputs["leg_diameter"][0]), float(inputs["leg_thickness"][0]))
            for idx in range(n_legs):
                add_element(leg_nodes, leg_indices, bay_nodes, bay_indices, itube, idx, jdx, idx, jdx)
                add_element(bay_nodes, bay_indices, leg_nodes, leg_indices, itube, idx, jdx, idx, jdx + 1)

        # Add brace (x-joint) members
        for jdx in range(n_bays):
            itube = cs.Tube(inputs["brace_diameters"][jdx], inputs["brace_thicknesses"][jdx])
            for idx in range(n_legs):
                add_element(bay_nodes, bay_indices, x_nodes, x_indices, itube, idx, jdx, idx, jdx)
                add_element(
                    x_nodes,
                    x_indices,
                    bay_nodes,
                    bay_indices,
                    itube,
                    idx,
                    jdx,
                    (idx + 1) % n_legs,
                    (jdx + 1) % (n_bays + 1),
                )

                add_element(bay_nodes, bay_indices, x_nodes, x_indices, itube, (idx + 1) % n_legs, jdx, idx, jdx)
                add_element(
                    x_nodes,
                    x_indices,
                    bay_nodes,
                    bay_indices,
                    itube,
                    idx,
                    jdx,
                    idx,
                    (jdx + 1) % (n_bays + 1),
                )

        # Add mud brace if boolean True
        if x_mb:
            itube = cs.Tube(inputs["brace_diameters"][0], inputs["brace_thicknesses"][0])
            for idx in range(n_legs):
                add_element(bay_nodes, bay_indices, bay_nodes, bay_indices, itube, idx, 0, (idx + 1) % n_legs, 0)

        # Add ghost point where we add the turbine_F and M as well as transition mass
        itube = cs.Tube(1.0e-2, 1.0e-3)
        for idx in range(n_legs):
            add_element(leg_nodes, leg_indices, ghost_nodes, ghost_indices, itube, idx, -1, 0, 0)

        # Grab material information; generally steel will be used
        imat = discrete_inputs["material_names"].index(material_name)
        E = [np.mean(inputs["E_mat"][imat])] * self.num_elements
        G = [np.mean(inputs["G_mat"][imat])] * self.num_elements
        rho = [inputs["rho_mat"][imat]] * self.num_elements

        # Convert all lists to arrays
        Area = np.squeeze(np.array(self.Area, dtype=np.float64))
        Asx = np.squeeze(np.array(self.Asx, dtype=np.float64))
        Asy = np.squeeze(np.array(self.Asy, dtype=np.float64))
        J0 = np.squeeze(np.array(self.J0, dtype=np.float64))
        Ixx = np.squeeze(np.array(self.Ixx, dtype=np.float64))
        Iyy = np.squeeze(np.array(self.Iyy, dtype=np.float64))
        L = np.squeeze(np.array(self.L, dtype=np.float64))
        D = np.squeeze(np.array(self.D, dtype=np.float64))
        t = np.squeeze(np.array(self.t, dtype=np.float64))
        E = np.squeeze(np.array(E, dtype=np.float64))
        G = np.squeeze(np.array(G, dtype=np.float64))
        rho = np.squeeze(np.array(rho, dtype=np.float64))
        N1 = self.N1
        N2 = self.N2

        # Modify last n_legs elements to make them rigid due to the ghost node
        E[-n_legs:] *= 1.0e8
        G[-n_legs:] *= 1.0e8
        rho[-n_legs:] = 1.0e-2

        outputs["jacket_elem_N"] = connect_mat = np.c_[N1, N2] - 1  # Storing as 0-indexed
        outputs["jacket_elem_L"] = L
        outputs["jacket_elem_D"] = D
        outputs["jacket_elem_t"] = t
        outputs["jacket_elem_A"] = Area
        outputs["jacket_elem_Asx"] = Asx
        outputs["jacket_elem_Asy"] = Asy
        outputs["jacket_elem_Ixx"] = Ixx
        outputs["jacket_elem_Iyy"] = Iyy
        outputs["jacket_elem_J0"] = J0
        outputs["jacket_elem_E"] = E
        outputs["jacket_elem_G"] = G
        outputs["jacket_elem_rho"] = rho
        outputs["jacket_elem_sigma_y"] = inputs["sigma_y_mat"][imat]
        outputs["jacket_elem_qdyn"] = 1.0e2  # hardcoded value for now
        outputs["jacket_elem_sigma_y"][-n_legs:] *= 1e6
        outputs["jacket_elem_qdyn"][-n_legs:] *= 1e4

        element = np.arange(1, self.num_elements + 1)
        roll = np.zeros(self.num_elements)

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)

        # Populate mass and cost outputs
        # TODO: Is there an "outfitting" factor for jackets.  Seems like it would be much smaller than monopiles
        m_jack_user = float(inputs["jacket_mass_user"][0])
        m_jack = np.sum(Area[:-n_legs] * rho[:-n_legs] * L[:-n_legs])
        outputs["jacket_mass"] = m_jack_user if m_jack_user > 0.0 else m_jack
        outputs["structural_mass"] = outputs["jacket_mass"] + inputs["tower_mass"]

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(
            mod_opt["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"],
            mod_opt["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"],
            dx,
        )
        # -----------------------------------

        # initialize frame3dd object
        self.frame = pyframe3dd.Frame(nodes, reactions, elements, options)

        if mod_opt["WISDEM"]["FixedBottomSE"]["save_truss_figures"]:
            if not self.under_approx:
                self.frame.draw(savefig=True, fig_idx=self.idx)
                self.idx += 1

        # ------- enable dynamic analysis ----------
        Mmethod = 1
        lump = 0
        shift = 0.0
        # Run twice the number of modes to ensure that we can ignore the torsional modes and still get the desired number of fore-aft, side-side modes
        self.frame.enableDynamics(2 * NFREQ, Mmethod, lump, 1e-4, shift)
        # ----------------------------

        # ------ static load case 1 ------------
        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        for k in range(n_dlc):
            load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

            # Prepare point forces at transition node
            turb_F = inputs["turbine_F"][:, k]
            turb_M = inputs["turbine_M"][:, k]
            load_obj.changePointLoads(
                np.array([n], dtype=np.int_),  # -1 b/c same reason as above
                np.array([turb_F[0]]).flatten(),
                np.array([turb_F[1]]).flatten(),
                np.array([turb_F[2]]).flatten(),
                np.array([turb_M[0]]).flatten(),
                np.array([turb_M[1]]).flatten(),
                np.array([turb_M[2]]).flatten(),
            )

            # # trapezoidally distributed loads
            # xx1 = xy1 = xz1 = np.zeros(ielem.size)
            # xx2 = xy2 = xz2 = 0.99 * L  # multiply slightly less than unity b.c. of precision
            # wx1 = inputs["jacket_elem_Px1"][:nelem, k]
            # wx2 = inputs["jacket_elem_Px2"][:nelem, k]
            # wy1 = inputs["jacket_elem_Py1"][:nelem, k]
            # wy2 = inputs["jacket_elem_Py2"][:nelem, k]
            # wz1 = inputs["jacket_elem_Pz1"][:nelem, k]
            # wz2 = inputs["jacket_elem_Pz2"][:nelem, k]
            # load_obj.changeTrapezoidalLoads(ielem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            # Add the load case and run
            self.frame.addLoadCase(load_obj)

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
        # Note, need len()-1 because Frame3DD crashes if mass add at end
        midx = np.array([n, n - 1, 1], dtype=np.int_)
        m_add = np.array([m_turb, m_trans, m_grav])
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

        # self.frame.write("jacket.3dd")
        # self.frame.draw()
        displacements, forces, reactions, internalForces, mass, modal = self.frame.run()

        # natural frequncies
        outputs["f1"] = modal.freq[0]
        outputs["f2"] = modal.freq[1]
        outputs["structural_frequencies"] = modal.freq[:NFREQ]

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs["jacket_deflection"] = np.sqrt(displacements.dx**2 + displacements.dy**2).T
        outputs["top_deflection"] = outputs["jacket_deflection"][-1, :]

        # Determine reaction forces
        outputs["jacket_base_F"] = -np.c_[
            reactions.Fx.sum(axis=1), reactions.Fy.sum(axis=1), reactions.Fz.sum(axis=1)
        ].T
        outputs["jacket_base_M"] = -np.c_[
            reactions.Mxx.sum(axis=1), reactions.Myy.sum(axis=1), reactions.Mzz.sum(axis=1)
        ].T

        for ic in range(n_dlc):
            # Forces and moments along the structure
            outputs["jacket_Fz"][:, ic] = forces.Nx[ic, 1::2]
            outputs["jacket_Vx"][:, ic] = -forces.Vz[ic, 1::2]
            outputs["jacket_Vy"][:, ic] = forces.Vy[ic, 1::2]
            outputs["jacket_Mxx"][:, ic] = -forces.Mzz[ic, 1::2]
            outputs["jacket_Myy"][:, ic] = forces.Myy[ic, 1::2]
            outputs["jacket_Mzz"][:, ic] = forces.Txx[ic, 1::2]


class JacketPost(om.ExplicitComponent):
    """
    This component computes the stress and buckling utilization values
    for the jacket structure based on the loading computed by Frame3DD.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]

        n_elem = 2 * (n_legs * (n_bays + 1)) + 4 * (n_legs * n_bays) + int(x_mb) * n_legs + n_legs

        self.add_input("jacket_elem_L", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_D", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_t", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_A", np.zeros(n_elem), units="m**2")
        self.add_input("jacket_elem_Asx", np.zeros(n_elem), units="m**2")
        self.add_input("jacket_elem_Asy", np.zeros(n_elem), units="m**2")
        self.add_input("jacket_elem_Ixx", np.zeros(n_elem), units="kg*m**2")
        self.add_input("jacket_elem_Iyy", np.zeros(n_elem), units="kg*m**2")
        self.add_input("jacket_elem_J0", np.zeros(n_elem), units="kg*m**2")
        self.add_input("jacket_elem_E", np.zeros(n_elem), units="Pa")
        self.add_input("jacket_elem_G", np.zeros(n_elem), units="Pa")
        self.add_input("jacket_elem_sigma_y", np.zeros(n_elem), units="Pa")
        self.add_input("jacket_elem_qdyn", np.zeros((n_elem, n_dlc)), units="Pa")

        # Processed Frame3DD/OpenFAST outputs
        self.add_input("jacket_Fz", np.ones((n_elem, n_dlc)), units="N")
        self.add_input("jacket_Vx", np.ones((n_elem, n_dlc)), units="N")
        self.add_input("jacket_Vy", np.ones((n_elem, n_dlc)), units="N")
        self.add_input("jacket_Mxx", np.ones((n_elem, n_dlc)), units="N*m")
        self.add_input("jacket_Myy", np.ones((n_elem, n_dlc)), units="N*m")
        self.add_input("jacket_Mzz", np.ones((n_elem, n_dlc)), units="N*m")

        # We don't care about the last n_leg members because they're connected
        # to the ghost node to transmit loads and masses.
        # Thus, the stress and buckling outputs are smaller dimensions
        # than the full n_elem.
        self.add_output("constr_stress", np.ones((n_elem - n_legs, n_dlc)))
        self.add_output("constr_shell_buckling", np.ones((n_elem - n_legs, n_dlc)))
        self.add_output("constr_global_buckling", np.ones((n_elem - n_legs, n_dlc)))

    def compute(self, inputs, outputs):
        # Unpack some variables
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        gamma_f = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_f"]
        gamma_m = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_m"]
        gamma_n = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_n"]
        gamma_b = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_b"]

        d = inputs["jacket_elem_D"]
        t = inputs["jacket_elem_t"]
        h = inputs["jacket_elem_L"]
        Az = inputs["jacket_elem_A"]
        Asx = inputs["jacket_elem_Asx"]
        Jz = inputs["jacket_elem_J0"]
        Iyy = inputs["jacket_elem_Iyy"]
        sigy = inputs["jacket_elem_sigma_y"]
        E = inputs["jacket_elem_E"]
        G = inputs["jacket_elem_G"]
        qdyn = inputs["jacket_elem_qdyn"]
        r = 0.5 * d

        # Get loads from Framee3dd/OpenFAST
        Fz = inputs["jacket_Fz"]
        Vx = inputs["jacket_Vx"]
        Vy = inputs["jacket_Vy"]
        Mxx = inputs["jacket_Mxx"]
        Myy = inputs["jacket_Myy"]
        Mzz = inputs["jacket_Mzz"]

        M = np.sqrt(Mxx**2 + Myy**2)
        V = np.sqrt(Vx**2 + Vy**2)

        # See http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        # print(Fz.shape, Az.shape, M.shape, r.shape, Iyy.shape)
        axial_stress = Fz / Az[:, np.newaxis] + M * (r / Iyy)[:, np.newaxis]
        shear_stress = np.abs(Mzz) / (Jz * r)[:, np.newaxis] + V / Asx[:, np.newaxis]
        hoop_stress = -qdyn * ((r - 0.5 * t) / t)[:, np.newaxis]  # util_con.hoopStress(d, t, qdyn)
        outputs["constr_stress"] = util_con.TubevonMisesStressUtilization(
            axial_stress, hoop_stress, shear_stress, gamma_f * gamma_m * gamma_n, sigy.reshape((-1, 1))
        )[:-n_legs]

        # Use DNV-GL CP202 Method
        check = util_dnvgl.CylinderBuckling(h, d, t, E=E, G=G, sigma_y=sigy, gamma=gamma_f * gamma_b)
        for k in range(n_dlc):
            results = check.run_buckling_checks(
                Fz[:, k], M[:, k], hoop_stress[:, k], shear_stress[:, k]
            )

            outputs["constr_shell_buckling"][:, k] = results["Shell"][:-n_legs]
            outputs["constr_global_buckling"][:, k] = results["Global"][:-n_legs]


class JacketCost(om.ExplicitComponent):
    """
    Compute the jacket costs using textbook relations, much like the monopile calcs.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]
        n_mat = mod_opt["materials"]["n_mat"]
        n_node = 3 * n_legs * (n_bays + 1) + 1
        n_elem = 2 * (n_legs * (n_bays + 1)) + 4 * (n_legs * n_bays) + int(x_mb) * n_legs + n_legs

        self.add_input("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)), units="m")

        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("unit_cost_mat", val=np.zeros(n_mat), units="USD/kg")
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")

        self.add_input("jacket_nodes", np.zeros((n_node, 3)), units="m")
        self.add_input("jacket_elem_N", np.zeros((n_elem, 2)))
        self.add_input("jacket_elem_L", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_D", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_t", np.zeros(n_elem), units="m")
        self.add_input("jacket_mass", 0.0, units="kg")

        self.add_input("tower_cost", val=0.0, units="USD")
        self.add_input("transition_piece_cost", 0.0, units="USD")

        self.add_output("labor_hours", 0.0, units="h")
        self.add_output("jacket_cost", 0.0, units="USD")
        self.add_output("structural_cost", val=0.0, units="USD")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        material_name = mod_opt["WISDEM"]["FixedBottomSE"]["material"]
        eps = 1e-8

        # Unpack inputs and ignore ghost nodes
        c_trans = inputs["transition_piece_cost"]
        leg_nodes = inputs["leg_nodes"]
        connect_mat = np.int_(inputs["jacket_elem_N"][:-n_legs, :])
        L = inputs["jacket_elem_L"][:-n_legs]
        D = inputs["jacket_elem_D"][:-n_legs].copy()
        t = inputs["jacket_elem_t"][:-n_legs]
        xyz = inputs["jacket_nodes"]
        m_total = inputs["jacket_mass"]
        n_edges = L.size
        N0 = connect_mat[:, 0]
        N1 = connect_mat[:, 1]

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        imat = discrete_inputs["material_names"].index(material_name)
        k_m = inputs["unit_cost_mat"][imat]  # 1.1 # USD / kg carbon steel plate
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        k_e = 0.064  # Industrial electricity rate $/kWh https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
        e_f = 15.9  # Electricity usage kWh/kg for steel
        # e_fo = 26.9  # Electricity usage kWh/kg for stainless steel
        theta = 3  # Manufacturing difficulty factor

        # Prep vectors for elements and legs for some vector math
        edge_vec = xyz[N0, :] - xyz[N1, :]
        leg_vec = np.squeeze(leg_nodes[:, -1, :] - leg_nodes[:, 0, :])
        leg_L = np.linalg.norm(leg_vec, axis=1)

        # Get the angle of intersection between all edges (as vectors) and all legs (as vectors)
        vec_vals = np.dot(edge_vec, leg_vec.T) / np.outer(L, leg_L)
        leg_alpha = np.arccos(np.minimum(np.maximum(vec_vals, -1.0), 1.0))
        # If angle of intersection is close to 0 or 180, the edge is part of a leg
        tol = np.deg2rad(5)
        idx_leg = np.any((np.abs(leg_alpha) < tol) | (np.abs(leg_alpha - np.pi) < tol), axis=1)
        D[idx_leg] = 0.0  # No double-counting time for leg elements since single piece

        # Now determine which angle to use based on which leg a given edge node is on
        edge_alpha = 0.5 * np.pi * np.ones(n_edges)
        for k in range(n_legs):
            tol = 1e-2 * leg_L[k]
            sec1 = np.linalg.norm(np.squeeze(leg_nodes[k, -1, :])[np.newaxis, :] - xyz[N0, :], axis=1)
            sec2 = np.linalg.norm(xyz[N0, :] - np.squeeze(leg_nodes[k, 0, :])[np.newaxis, :], axis=1)
            on_leg_k = np.abs(leg_L[k] - sec1 - sec2) < tol
            edge_alpha[on_leg_k] = leg_alpha[on_leg_k, k]
        edge_alpha = np.minimum(edge_alpha, np.pi - edge_alpha) + eps

        # Run manufacturing time estimate functions
        weld_L = 2 * np.pi * D / np.sin(edge_alpha)  # Multiply by 2 for both ends
        n_pieces = n_edges - np.count_nonzero(D)
        t_cut = 2 * manu.steel_tube_cutgrind_time(theta, 0.5 * D, t, edge_alpha)  # Multiply by 2 for both ends
        t_weld = manu.steel_tube_welding_time(theta, n_pieces, m_total, weld_L, t)
        t_manufacture = t_cut + t_weld
        K_f = k_f * t_manufacture
        outputs["labor_hours"] = t_manufacture / 60.0
        
        # Cost step 5) Painting by surface area
        theta_p = 2
        K_p = k_p * theta_p * np.pi * np.sum(D[:-n_legs] * L[:-n_legs])

        # Material cost with waste fraction, but without outfitting,
        K_m = 1.21 * np.sum(k_m * m_total)

        # Electricity usage
        K_e = k_e * (e_f * m_total)  # + e_fo * (coeff - 1.0) * m_total

        # Assemble all costs for now
        tempSum = K_m + K_e + K_p + K_f

        # Capital cost share from BLS MFP by NAICS
        K_c = 0.118 * tempSum / (1.0 - 0.118)

        outputs["jacket_cost"] = K_c + tempSum + c_trans
        outputs["structural_cost"] = outputs["jacket_cost"] + inputs["tower_cost"]


# Assemble the system together in an OpenMDAO Group
class JacketSEProp(om.Group):
    """
    Group to contain all subsystems needed for jacket analysis and design.
    Can be used as a standalone or within the larger WISDEM stack.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem("pre", PreDiscretization(), promotes=["*"])
        self.add_subsystem("nodes", ComputeJacketNodes(modeling_options=modeling_options), promotes=["*"])


class JacketSEPerf(om.Group):
    """
    Group to contain all subsystems needed for jacket analysis and design.
    Can be used as a standalone or within the larger WISDEM stack.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem("frame3dd", ComputeFrame3DD(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem("post", JacketPost(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem("cost", JacketCost(modeling_options=modeling_options), promotes=["*"])


class JacketSE(om.Group):
    """
    Group to contain all subsystems needed for jacket analysis and design.
    Can be used as a standalone or within the larger WISDEM stack.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem("prop", JacketSEProp(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem("perf", JacketSEPerf(modeling_options=modeling_options), promotes=["*"])

