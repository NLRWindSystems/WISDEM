import numpy as np

import wisdem.inputs as sch
from wisdem.commonse.utilities import sectional2nodal


class WindTurbineOntologyPython(object):
    def __init__(self, fname_input_wt, fname_input_modeling, fname_input_analysis):
        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)
        if fname_input_wt is None:
            self.wt_init = None
        else:
            self.wt_init = sch.load_geometry_yaml(fname_input_wt)
        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_opt_flags()

    def get_input_data(self):
        return self.wt_init, self.modeling_options, self.analysis_options

    def set_run_flags(self):
        # Create components flag struct
        flags = self.modeling_options["flags"] = {}

        # Backwards compatibility
        modules = [
            "RotorSE",
            "DriveSE",
            "TowerSE",
            "FixedBottomSE",
            "FloatingSE",
            "Loading",
            "Environment",
            "BOS",
            "LCOE",
        ]
        for m in modules:
            if m in self.modeling_options:
                if m in self.modeling_options["WISDEM"]:
                    self.modeling_options["WISDEM"][m].update(self.modeling_options[m])
                else:
                    self.modeling_options["WISDEM"][m] = self.modeling_options[m]

        for k in [
            "blade",
            "hub",
            "drivetrain",
            "yaw",
            "tower",
            "monopile",
            "jacket",
            "floating_platform",
            "mooring",
            "RNA",
        ]:
            flags[k] = k in self.wt_init["components"]

        for k in ["assembly", "components", "airfoils", "materials", "control"]:
            flags[k] = k in self.wt_init

        # Generator flag
        flags["generator"] = (
            flags["drivetrain"]
            and "generator" in self.wt_init["components"]["drivetrain"]
            and self.wt_init["components"]["drivetrain"]["generator"]["h_s"] > 0.0
        )
        if flags["generator"]:
            self.modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] = self.wt_init["components"]["drivetrain"][
                "generator"
            ]["type"].lower()

        # Offshore flags
        flags["floating"] = self.modeling_options["flags"]["floating_platform"]

        # Even if the block is in the inputs, the user can turn off via modeling options
        # The "flags" here should come in as a string, not Boolean, to allow for the user_elastic option
        self.modeling_options["user_elastic"] = {m: False for m in flags.keys()}

        flag_pairings = [
            ("blade", "RotorSE"),
            ("tower", "TowerSE"),
            ("monopile", "FixedBottomSE"),
            ("jacket", "FixedBottomSE"),
            ("floating", "FloatingSE"),
            ("mooring", "FloatingSE"),
            ("hub", "DriveSE"),
            ("drivetrain", "DriveSE"),
            ("yaw", "DriveSE"),
            ("generator", "DriveSE"),
        ]
        for i, j in flag_pairings:
            if flags[i]:
                if isinstance(self.modeling_options["WISDEM"][j]["flag"], bool):
                    flags[i] = self.modeling_options["WISDEM"][j]["flag"]
            if (
                isinstance(self.modeling_options["WISDEM"][j]["flag"], str)
                and self.modeling_options["WISDEM"][j]["flag"].lower() == "user_elastic"
            ):
                self.modeling_options["user_elastic"][i] = True
                flags[i] = False
        flags["hub"] = flags["drivetrain"] = (
            flags["hub"] or flags["drivetrain"]
        )  # Hub and drivetrain have to go together
        flags["bos"] = self.modeling_options["WISDEM"]["BOS"]["flag"]
        flags["opex"] = self.modeling_options["WISDEM"]["OpEx"]["flag"]
        flags["environment"] = "Environment" in self.modeling_options["WISDEM"]
        flags["costs"] = self.modeling_options["WISDEM"]["LCOE"]["flag"]
        flags["offshore"] = flags["floating"] or flags["monopile"] or flags["jacket"]

        # Blades and airfoils
        if flags["blade"] and not flags["airfoils"]:
            raise ValueError("Blades/rotor analysis is requested but no airfoils are found")
        if flags["airfoils"] and not flags["blade"]:
            print("WARNING: Airfoils provided but no blades/rotor found or RotorSE deactivated")

        # Blades, tower, monopile and environment
        if flags["jacket"] and flags["monopile"]:
            raise ValueError("Cannot specify both monopile and jacket support structures")

        # Floating/monopile
        if flags["floating"] and (flags["monopile"] or flags["jacket"]):
            raise ValueError("Cannot have both floating and fixed-bottom components")

        # Water depth check
        if self.modeling_options["WISDEM"]["Environment"]["water_depth"] <= 0.0 and flags["offshore"]:
            raise ValueError("Water depth must be > 0 to do fixed-bottom or floating analysis")

    def set_openmdao_vectors(self):
        # Class instance to determine all the parameters used to initialize the openmdao arrays, i.e. number of airfoils, number of angles of attack, number of blade spanwise stations, etc
        # ==modeling_options = {}

        # Materials
        self.modeling_options["materials"] = {}
        self.modeling_options["materials"]["n_mat"] = len(self.wt_init["materials"])

        # Airfoils
        if self.modeling_options["flags"]["airfoils"]:
            self.modeling_options["WISDEM"]["RotorSE"]["n_af_database"] = len(self.wt_init["airfoils"])
            self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"]
            if self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0 == int(
                self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0
            ):
                # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
                self.modeling_options["WISDEM"]["RotorSE"]["aoa"] = np.unique(
                    np.hstack(
                        [
                            np.linspace(
                                -180.0, -180.0 / 6.0, int(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0 + 1)
                            ),
                            np.linspace(
                                -180.0 / 6.0,
                                180.0 / 6.0,
                                int(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 2.0),
                            ),
                            np.linspace(
                                180.0 / 6.0, 180.0, int(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0 + 1)
                            ),
                        ]
                    )
                )
            else:
                self.modeling_options["WISDEM"]["RotorSE"]["aoa"] = np.linspace(
                    -180.0, 180.0, self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"]
                )
                print(
                    "WARNING: If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of "
                    + str(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"])
                    + " is not a multiple of 4 and an equally spaced grid is adopted."
                )
            Re_all = []
            self.modeling_options["WISDEM"]["RotorSE"]["AFTabMod"] = 1
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_af_database"]):
                for j in range(len(self.wt_init["airfoils"][i]["polars"])):
                    for k in range(len(self.wt_init["airfoils"][i]["polars"][j]["re_sets"])):
                        Re_all.append(self.wt_init["airfoils"][i]["polars"][j]["re_sets"][k]["re"])
                if len(self.wt_init["airfoils"][i]["polars"]) > 1:
                    self.modeling_options["WISDEM"]["RotorSE"]["AFTabMod"] = 2
            self.modeling_options["WISDEM"]["RotorSE"]["n_Re"] = len(np.unique(Re_all))
            self.modeling_options["WISDEM"]["RotorSE"]["Re"] = np.unique(Re_all)
            self.modeling_options["WISDEM"]["RotorSE"]["n_xy"] = self.modeling_options["WISDEM"]["RotorSE"]["n_xy"]
            n_af_master = len(self.wt_init["components"]["blade"]["outer_shape"]["airfoils"])
            self.modeling_options["WISDEM"]["RotorSE"]["n_af_master"] = n_af_master
            self.modeling_options["WISDEM"]["RotorSE"]["af_master"] = [""] * n_af_master
            for i in range(n_af_master):
                self.modeling_options["WISDEM"]["RotorSE"]["af_master"][i] = self.wt_init["components"]["blade"][
                    "outer_shape"
                ]["airfoils"][i]["name"]

        # Blade
        if self.modeling_options["flags"]["blade"] or self.modeling_options["user_elastic"]["blade"]:
            self.modeling_options["WISDEM"]["RotorSE"]["nd_span"] = np.linspace(
                0.0, 1.0, self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
            )  # Equally spaced non-dimensional spanwise grid

            self.modeling_options["WISDEM"]["RotorSE"][
                "lofted_output"
            ] = False  # Is this always false? It is not in the schema and not changed anywhere else.
            self.modeling_options["WISDEM"]["RotorSE"][
                "n_freq"
            ] = 10  # Number of blade nat frequencies computed, this should be common so moved out of the conditional
            if not self.modeling_options["user_elastic"]["blade"]:
                self.modeling_options["WISDEM"]["RotorSE"]["n_webs"] = len(
                    self.wt_init["components"]["blade"]["structure"]["webs"]
                )
                self.modeling_options["WISDEM"]["RotorSE"]["n_layers"] = len(
                    self.wt_init["components"]["blade"]["structure"]["layers"]
                )

                self.modeling_options["WISDEM"]["RotorSE"]["layer_name"] = self.modeling_options["WISDEM"]["RotorSE"][
                    "n_layers"
                ] * [""]
                self.modeling_options["WISDEM"]["RotorSE"]["layer_mat"] = self.modeling_options["WISDEM"]["RotorSE"][
                    "n_layers"
                ] * [""]
                for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_layers"]):
                    self.modeling_options["WISDEM"]["RotorSE"]["layer_name"][i] = self.wt_init["components"]["blade"][
                        "structure"
                    ]["layers"][i]["name"]
                    self.modeling_options["WISDEM"]["RotorSE"]["layer_mat"][i] = self.wt_init["components"]["blade"][
                        "structure"
                    ]["layers"][i]["material"]
                self.modeling_options["WISDEM"]["RotorSE"]["web_name"] = self.modeling_options["WISDEM"]["RotorSE"][
                    "n_webs"
                ] * [""]
                for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_webs"]):
                    self.modeling_options["WISDEM"]["RotorSE"]["web_name"][i] = self.wt_init["components"]["blade"][
                        "structure"
                    ]["webs"][i]["name"]
                joint_pos = self.wt_init["components"]["blade"]["structure"]["joint"]["position"]
                if joint_pos > 0.0:
                    # Adjust grid to have grid point at join location
                    closest_grid_pt = np.argmin(abs(self.modeling_options["WISDEM"]["RotorSE"]["nd_span"] - joint_pos))
                    self.modeling_options["WISDEM"]["RotorSE"]["nd_span"][closest_grid_pt] = joint_pos
                    self.modeling_options["WISDEM"]["RotorSE"]["id_joint_position"] = closest_grid_pt
                else:
                    self.modeling_options["WISDEM"]["RotorSE"]["id_joint_position"] = 0

        # Drivetrain config
        self.modeling_options["WISDEM"]["DriveSE"]["direct"] = self.wt_init["assembly"]["drivetrain"].lower() in [
            "direct",
            "direct_drive",
            "pm_direct_drive",
        ]

        # Tower
        if self.modeling_options["flags"]["tower"]:
            svec = np.unique(
                np.r_[
                    self.wt_init["components"]["tower"]["outer_shape"]["outer_diameter"]["grid"],
                    self.wt_init["components"]["tower"]["reference_axis"]["x"]["grid"],
                    self.wt_init["components"]["tower"]["reference_axis"]["y"]["grid"],
                    self.wt_init["components"]["tower"]["reference_axis"]["z"]["grid"],
                ]
            )
            self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = self.modeling_options["WISDEM"]["TowerSE"][
                "n_height_tower"
            ] = len(svec)
            self.modeling_options["WISDEM"]["TowerSE"]["n_layers"] = self.modeling_options["WISDEM"]["TowerSE"][
                "n_layers_tower"
            ] = len(self.wt_init["components"]["tower"]["structure"]["layers"])

        # Monopile
        if self.modeling_options["flags"]["monopile"]:
            monopile = self.wt_init["components"]["monopile"]
            svec = np.unique(
                np.r_[
                    monopile["outer_shape"]["outer_diameter"]["grid"],
                    monopile["reference_axis"]["x"]["grid"],
                    monopile["reference_axis"]["y"]["grid"],
                    monopile["reference_axis"]["z"]["grid"],
                ]
            )
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = self.modeling_options["WISDEM"][
                "FixedBottomSE"
            ]["n_height_monopile"] = len(svec)
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = self.modeling_options["WISDEM"][
                "FixedBottomSE"
            ]["n_layers_monopile"] = len(self.wt_init["components"]["monopile"]["structure"]["layers"])

        # Jacket
        if self.modeling_options["flags"]["jacket"]:
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = self.wt_init["components"]["jacket"]["n_legs"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = self.wt_init["components"]["jacket"]["n_bays"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = self.wt_init["components"]["jacket"]["x_mb"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["material"] = self.wt_init["components"]["jacket"][
                "material"
            ]

        # Floating platform
        self.modeling_options["floating"] = {}
        if self.modeling_options["flags"]["floating"]:
            n_joints = len(self.wt_init["components"]["floating_platform"]["joints"])
            self.modeling_options["floating"]["joints"] = {}
            self.modeling_options["floating"]["joints"]["n_joints"] = n_joints
            self.modeling_options["floating"]["joints"]["name"] = [""] * n_joints
            self.modeling_options["floating"]["joints"]["transition"] = [False] * n_joints
            self.modeling_options["floating"]["joints"]["cylindrical"] = [False] * n_joints
            self.modeling_options["floating"]["joints"]["relative"] = ["origin"] * n_joints
            self.modeling_options["floating"]["joints"]["relative_dims"] = [[True, True, True]] * n_joints
            self.modeling_options["floating"]["joints"]["axial_coeffs"] = [{}] * n_joints
            for i in range(n_joints):
                self.modeling_options["floating"]["joints"]["name"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["name"]
                self.modeling_options["floating"]["joints"]["transition"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["transition"]
                self.modeling_options["floating"]["joints"]["cylindrical"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["cylindrical"]
                self.modeling_options["floating"]["joints"]["relative"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["relative"]
                self.modeling_options["floating"]["joints"]["relative_dims"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["relative_dims"]
                # Axial coefficients: pass through for now.  If we want them to be a DV someday, will need to make part openmdao glue code
                self.modeling_options["floating"]["joints"]["axial_coeffs"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["axial_coeffs"]

            # Create name->index dictionary for joint names, will add on axial joints later
            name2idx = dict(zip(self.modeling_options["floating"]["joints"]["name"], range(n_joints)))

            # Check that there is at most one transition joint
            if self.modeling_options["floating"]["joints"]["transition"].count(True) > 1:
                raise ValueError("Can only support one tower on the floating platform for now")
            try:
                itrans = self.modeling_options["floating"]["joints"]["transition"].index(True)
                self.modeling_options["floating"]["transition_joint"] = itrans
            except:
                self.modeling_options["floating"]["transition_joint"] = None

            n_members = len(self.wt_init["components"]["floating_platform"]["members"])
            self.modeling_options["floating"]["members"] = {}
            self.modeling_options["floating"]["members"]["n_members"] = n_members
            self.modeling_options["floating"]["members"]["name"] = [""] * n_members
            self.modeling_options["floating"]["members"]["joint1"] = [""] * n_members
            self.modeling_options["floating"]["members"]["joint2"] = [""] * n_members
            self.modeling_options["floating"]["members"]["outer_shape"] = [""] * n_members
            self.modeling_options["floating"]["members"]["n_height"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_geom"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_layers"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_ballasts"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_bulkheads"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_axial_joints"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["no_intersect"] = [False] * n_members
            ballast_types = []
            for i in range(n_members):
                self.modeling_options["floating"]["members"]["name"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["name"]
                self.modeling_options["floating"]["members"]["joint1"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["joint1"]
                self.modeling_options["floating"]["members"]["joint2"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["joint2"]
                self.modeling_options["floating"]["members"]["outer_shape"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["outer_shape"]["shape"]

                # Master grid for all bulkheads, internal joints, ballasts, geometry changes, etc
                member_shape = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"]["shape"]
                if member_shape == "circular":
                    grid = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"][
                        "outer_diameter"
                    ]["grid"][:]
                elif member_shape == "rectangular":
                    grid = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"][
                        "side_length_a"
                    ]["grid"][:]
                    grid_b = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"][
                        "side_length_b"
                    ]["grid"][:]
                    assert grid == grid_b, "Side length a and b don't have the same grid but they should."

                # Grid for just diameter / thickness design
                geom_grid = grid[:]

                n_layers = len(self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["layers"])
                self.modeling_options["floating"]["members"]["n_layers"][i] = n_layers
                if "ballast" in self.wt_init["components"]["floating_platform"]["members"][i]["structure"]:
                    n_ballasts = len(
                        self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["ballast"]
                    )
                else:
                    n_ballasts = 0
                self.modeling_options["floating"]["members"]["n_ballasts"][i] = n_ballasts

                # Add in bulkheads and enforce at least endcaps for submerged environment
                # Don't add to master grid as they are handled differently in FloatingSE
                if "bulkhead" in self.wt_init["components"]["floating_platform"]["members"][i]["structure"]:
                    bulkgrid = self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                        "thickness"
                    ]["grid"]
                    if not 0.0 in bulkgrid:
                        self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                            "thickness"
                        ]["grid"].append(0.0)
                        self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                            "thickness"
                        ]["values"].append(0.02)
                    if not 1.0 in bulkgrid:
                        self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                            "thickness"
                        ]["grid"].append(1.0)
                        self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                            "thickness"
                        ]["values"].append(0.02)
                    # grid += bulkgrid # Handled differently in the floating code
                else:
                    self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"] = {}
                    self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                        "material"
                    ] = self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["layers"][0][
                        "material"
                    ]
                    self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"][
                        "thickness"
                    ] = {}
                    self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"]["thickness"][
                        "grid"
                    ] = [0.0, 1.0]
                    self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"]["thickness"][
                        "values"
                    ] = [0.02, 0.02]

                n_bulk = len(
                    self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["bulkhead"]["thickness"][
                        "grid"
                    ]
                )
                self.modeling_options["floating"]["members"]["n_bulkheads"][i] = n_bulk

                self.modeling_options["floating"]["members"][
                    "layer_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = [""] * n_layers
                for j in range(n_layers):
                    self.modeling_options["floating"]["members"][
                        "layer_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ][j] = self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["layers"][j][
                        "material"
                    ]
                    grid += self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["layers"][j][
                        "thickness"
                    ]["grid"]
                    geom_grid += self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["layers"][
                        j
                    ]["thickness"]["grid"]

                self.modeling_options["floating"]["members"][
                    "ballast_flag_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = [False] * n_ballasts
                self.modeling_options["floating"]["members"][
                    "ballast_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = [""] * n_ballasts
                for k in range(n_ballasts):
                    self.modeling_options["floating"]["members"][
                        "ballast_flag_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ][k] = self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["ballast"][k][
                        "variable_flag"
                    ]
                    if (
                        self.modeling_options["floating"]["members"][
                            "ballast_flag_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][k]
                        == False
                    ):
                        ballast_types.append(
                            self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["ballast"][k][
                                "material"
                            ]
                        )
                        self.modeling_options["floating"]["members"][
                            "ballast_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][k] = ballast_types[-1]
                    else:
                        ballast_types.append("variable")

                    grid += self.wt_init["components"]["floating_platform"]["members"][i]["structure"]["ballast"][k][
                        "grid"
                    ]

                if "axial_joints" in self.wt_init["components"]["floating_platform"]["members"][i]:
                    n_axial_joints = len(self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"])
                    self.modeling_options["floating"]["members"]["n_axial_joints"][i] = n_axial_joints
                    self.modeling_options["floating"]["members"][
                        "axial_joint_name_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ] = [""] * n_axial_joints
                    for m in range(n_axial_joints):
                        self.modeling_options["floating"]["members"][
                            "axial_joint_name_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][m] = self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["name"]
                        grid.append(
                            self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["grid"]
                        )
                        name2idx[
                            self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["name"]
                        ] = len(name2idx)
                else:
                    self.modeling_options["floating"]["members"]["n_axial_joints"][i] = 0

                self.modeling_options["floating"]["members"]["no_intersect"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["no_intersect"]

                final_grid = np.unique(grid)
                final_geom_grid = np.unique(geom_grid)
                self.modeling_options["floating"]["members"][
                    "grid_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = final_grid
                self.modeling_options["floating"]["members"][
                    "geom_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = final_geom_grid
                self.modeling_options["floating"]["members"]["n_height"][i] = len(final_grid)
                self.modeling_options["floating"]["members"]["n_geom"][i] = len(final_geom_grid)

            self.modeling_options["floating"]["members"]["ballast_types"] = set(ballast_types)

            # Store joint info
            self.modeling_options["floating"]["joints"]["name2idx"] = name2idx

            # Store rigid_bodies info
            self.modeling_options["floating"]["rigid_bodies"] = {}
            self.modeling_options["floating"]["rigid_bodies"]["n_bodies"] = len(
                self.wt_init["components"]["floating_platform"]["rigid_bodies"]
            )
            self.modeling_options["floating"]["rigid_bodies"]["joint1"] = [
                rb["joint1"] for rb in self.wt_init["components"]["floating_platform"]["rigid_bodies"]
            ]

            # Floating tower params
            self.modeling_options["floating"]["tower"] = {}
            self.modeling_options["floating"]["tower"]["n_ballasts"] = [0]
            self.modeling_options["floating"]["tower"]["n_bulkheads"] = [0]
            self.modeling_options["floating"]["tower"]["n_axial_joints"] = [0]
            if self.modeling_options["flags"]["tower"]:
                self.modeling_options["floating"]["tower"]["n_height"] = [
                    self.modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]
                ]
                self.modeling_options["floating"]["tower"]["n_layers"] = [
                    self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"]
                ]
            else:
                self.modeling_options["floating"]["tower"]["n_height"] = [0]
                self.modeling_options["floating"]["tower"]["n_layers"] = [0]

        # Mooring
        self.modeling_options["mooring"] = {}
        if self.modeling_options["flags"]["mooring"]:
            n_nodes = len(self.wt_init["components"]["mooring"]["nodes"])
            n_lines = len(self.wt_init["components"]["mooring"]["lines"])
            n_line_types = len(self.wt_init["components"]["mooring"]["line_types"])
            n_anchor_types = len(self.wt_init["components"]["mooring"]["anchor_types"])
            self.modeling_options["mooring"]["symmetric"] = self.modeling_options["WISDEM"]["FloatingSE"][
                "symmetric_moorings"
            ]
            self.modeling_options["mooring"]["n_nodes"] = n_nodes
            self.modeling_options["mooring"]["n_lines"] = n_lines
            self.modeling_options["mooring"]["n_anchors"] = np.sum(
                np.array([n["node_type"] == "fixed" for n in self.wt_init["components"]["mooring"]["nodes"]])
            )
            self.modeling_options["mooring"]["n_line_types"] = n_line_types
            self.modeling_options["mooring"]["n_anchor_types"] = n_anchor_types
            self.modeling_options["mooring"]["node_type"] = [""] * n_nodes
            self.modeling_options["mooring"]["node_names"] = [""] * n_nodes
            self.modeling_options["mooring"]["anchor_type"] = [""] * n_nodes
            self.modeling_options["mooring"]["fairlead_type"] = [""] * n_nodes
            for i in range(n_nodes):
                self.modeling_options["mooring"]["node_type"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "node_type"
                ]
                self.modeling_options["mooring"]["node_names"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "name"
                ]
                self.modeling_options["mooring"]["anchor_type"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "anchor_type"
                ]
                self.modeling_options["mooring"]["fairlead_type"][i] = self.wt_init["components"]["mooring"]["nodes"][
                    i
                ]["fairlead_type"]
            self.modeling_options["mooring"]["node1"] = [""] * n_lines
            self.modeling_options["mooring"]["node2"] = [""] * n_lines
            self.modeling_options["mooring"]["line_type"] = [""] * n_lines
            self.modeling_options["mooring"]["line_material"] = [""] * n_lines
            self.modeling_options["mooring"]["line_anchor"] = [""] * n_lines
            fairlead_nodes = []
            for i in range(n_lines):
                self.modeling_options["mooring"]["node1"][i] = self.wt_init["components"]["mooring"]["lines"][i][
                    "node1"
                ]
                self.modeling_options["mooring"]["node2"][i] = self.wt_init["components"]["mooring"]["lines"][i][
                    "node2"
                ]
                self.modeling_options["mooring"]["line_type"][i] = self.wt_init["components"]["mooring"]["lines"][i][
                    "line_type"
                ]
                # For the vessel attachments, find the list of fairlead nodes on the structure
                node1id = self.modeling_options["mooring"]["node_names"].index(
                    self.modeling_options["mooring"]["node1"][i]
                )
                node2id = self.modeling_options["mooring"]["node_names"].index(
                    self.modeling_options["mooring"]["node2"][i]
                )
                if self.modeling_options["mooring"]["node_type"][node1id] == "vessel":
                    fairlead_nodes.append(self.wt_init["components"]["mooring"]["nodes"][node1id]["joint"])
                if self.modeling_options["mooring"]["node_type"][node2id] == "vessel":
                    fairlead_nodes.append(self.wt_init["components"]["mooring"]["nodes"][node2id]["joint"])
                # Store the anchor type names to start
                if "fix" in self.modeling_options["mooring"]["node_type"][node1id]:
                    self.modeling_options["mooring"]["line_anchor"][i] = self.modeling_options["mooring"][
                        "anchor_type"
                    ][node1id]
                if "fix" in self.modeling_options["mooring"]["node_type"][node2id]:
                    self.modeling_options["mooring"]["line_anchor"][i] = self.modeling_options["mooring"][
                        "anchor_type"
                    ][node2id]

            self.modeling_options["mooring"]["line_type_name"] = [""] * n_line_types
            self.modeling_options["mooring"]["line_type_type"] = [""] * n_line_types
            for i in range(n_line_types):
                self.modeling_options["mooring"]["line_type_name"][i] = self.wt_init["components"]["mooring"][
                    "line_types"
                ][i]["name"]
                self.modeling_options["mooring"]["line_type_type"][i] = self.wt_init["components"]["mooring"][
                    "line_types"
                ][i]["type"].lower()
                for j in range(n_lines):
                    if (
                        self.modeling_options["mooring"]["line_type"][j]
                        == self.modeling_options["mooring"]["line_type_name"][i]
                    ):
                        self.modeling_options["mooring"]["line_material"][j] = self.modeling_options["mooring"][
                            "line_type_type"
                        ][i]
            self.modeling_options["mooring"]["anchor_type_name"] = [""] * n_anchor_types
            self.modeling_options["mooring"]["anchor_type_type"] = [""] * n_anchor_types
            for i in range(n_anchor_types):
                self.modeling_options["mooring"]["anchor_type_name"][i] = self.wt_init["components"]["mooring"][
                    "anchor_types"
                ][i]["name"]
                self.modeling_options["mooring"]["anchor_type_type"][i] = self.wt_init["components"]["mooring"][
                    "anchor_types"
                ][i]["type"].lower()
                for j in range(n_lines):
                    if (
                        self.modeling_options["mooring"]["line_anchor"][j]
                        == self.modeling_options["mooring"]["anchor_type_name"][i]
                    ):
                        self.modeling_options["mooring"]["line_anchor"][j] = self.modeling_options["mooring"][
                            "anchor_type_type"
                        ][i]
            self.modeling_options["mooring"]["n_attach"] = len(set(fairlead_nodes))

        # Assembly
        self.modeling_options["assembly"] = {}
        self.modeling_options["assembly"]["number_of_blades"] = int(self.wt_init["assembly"]["number_of_blades"])

    def set_opt_flags(self):
        # Recursively look for flags to set global optimization flag
        def recursive_flag(d):
            opt_flag = False
            for k, v in d.items():
                if isinstance(v, dict):
                    opt_flag = opt_flag or recursive_flag(v)
                elif k == "flag":
                    opt_flag = opt_flag or v
            return opt_flag

        # The user can provide `opt_flag` in analysis_options.yaml,
        # but if it's not provided, we check the individual opt flags
        # from analysis_options.yaml and set a global `opt_flag`
        if "flag" in self.analysis_options["driver"]["optimization"]:
            self.analysis_options["opt_flag"] = self.analysis_options["driver"]["optimization"]["flag"]
        else:
            self.analysis_options["opt_flag"] = recursive_flag(self.analysis_options["design_variables"])

        if self.analysis_options["opt_flag"] == False and (
            self.analysis_options["driver"]["step_size_study"]["flag"] == True
            or self.analysis_options["driver"]["design_of_experiments"]["flag"] == True
        ):
            self.analysis_options["opt_flag"] = True

        # Blade design variables
        # If not an optimization DV, then the number of points should be same as the discretization
        blade_opt_options = self.analysis_options["design_variables"]["blade"]
        # Blade aero design variables
        if (
            not blade_opt_options["aero_shape"]["twist"]["flag"]
            and not blade_opt_options["aero_shape"]["twist"]["inverse"]
        ):
            blade_opt_options["aero_shape"]["twist"]["n_opt"] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError("you are attempting to do an analysis using fewer analysis points than control points.")
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] < 4:
            raise ValueError("Cannot optimize twist with less than 4 control points along blade span")
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError(
                """Please set WISDEM->RotorSE->n_span in the modeling options yaml larger
                than aero_shape->twist->n_opt in the analysis options yaml. n_span and twist n_opt are """,
                self.modeling_options["WISDEM"]["RotorSE"]["n_span"],
                blade_opt_options["aero_shape"]["twist"]["n_opt"],
            )

        if not blade_opt_options["aero_shape"]["chord"]["flag"]:
            blade_opt_options["aero_shape"]["chord"]["n_opt"] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError("you are attempting to do an analysis using fewer analysis points than control points.")
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] < 4:
            raise ValueError("Cannot optimize chord with less than 4 control points along blade span")
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError(
                """Please set WISDEM->RotorSE->n_span in the modeling options yaml larger
                than aero_shape->chord->n_opt in the analysis options yaml. n_span and chord n_opt are """,
                self.modeling_options["WISDEM"]["RotorSE"]["n_span"],
                blade_opt_options["aero_shape"]["chord"]["n_opt"],
            )

        # # Blade structural design variables
        if self.modeling_options["flags"]["blade"] and (not self.modeling_options["user_elastic"]["blade"]):
            n_layers = self.modeling_options["WISDEM"]["RotorSE"]["n_layers"]
            layer_name = self.modeling_options["WISDEM"]["RotorSE"]["layer_name"]
            spars_tereinf = np.zeros(4, dtype=int)
            blade_opt_options["n_opt_struct"] = np.ones(n_layers, dtype=int)
            if "structure" in blade_opt_options:
                n_layers_opt = len(blade_opt_options["structure"])
                blade_opt_options["layer_index_opt"] = -np.ones(n_layers_opt, dtype=int)
                for i in range(n_layers):
                    foundit = False
                    for j in range(n_layers_opt):
                        if blade_opt_options["structure"][j]["layer_name"] == layer_name[i]:
                            blade_opt_options["n_opt_struct"][i] = blade_opt_options["structure"][j]["n_opt"]
                            blade_opt_options["layer_index_opt"][j] = i
                            foundit = True
                            break
                    if not foundit:
                        blade_opt_options["n_opt_struct"][i] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
                    if layer_name[i].lower() == self.modeling_options["WISDEM"]["RotorSE"]["spar_cap_ss"].lower():
                        spars_tereinf[0] = i
                    if layer_name[i].lower() == self.modeling_options["WISDEM"]["RotorSE"]["spar_cap_ps"].lower():
                        spars_tereinf[1] = i
                    if layer_name[i].lower() == self.modeling_options["WISDEM"]["RotorSE"]["te_ss"].lower():
                        spars_tereinf[2] = i
                    if layer_name[i].lower() == self.modeling_options["WISDEM"]["RotorSE"]["te_ps"].lower():
                        spars_tereinf[3] = i

                index_not_found = np.where(blade_opt_options["layer_index_opt"] == -1)[0]
                if len(index_not_found) > 0:
                    raise Exception(
                        "WISDEM is set to optimize the thickness of blade composite layer {}, but this layer "
                        "is not found in the input geometry yaml".format(
                            blade_opt_options["structure"][index_not_found[0]]["layer_name"]
                        )
                    )
            else:
                blade_opt_options["structure"] = []
                blade_opt_options["n_opt_struct"] *= self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
            self.modeling_options["WISDEM"]["RotorSE"]["spars_tereinf"] = spars_tereinf

            if any(blade_opt_options["n_opt_struct"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]):
                raise ValueError(
                    "You are attempting to do a blade structural design optimization with more DVs than spanwise stations."
                )
            blade_opt_options["n_opt_struct"] = blade_opt_options["n_opt_struct"].tolist()
            if "layer_index_opt" in blade_opt_options:
                blade_opt_options["layer_index_opt"] = blade_opt_options["layer_index_opt"].tolist()

        # Handle linked joints and members in floating platform
        if self.modeling_options["flags"]["floating"]:
            float_opt_options = self.analysis_options["design_variables"]["floating"]

            # First the joints
            dv_info = []
            for c in ["z", "r"]:
                for idv in float_opt_options["joints"][c + "_coordinate"]:
                    inames = idv["names"]
                    idx = [self.modeling_options["floating"]["joints"]["name"].index(m) for m in inames]

                    idict = {}
                    idict["indices"] = idx
                    idict["dimension"] = 0 if c == "r" else 2
                    dv_info.append(idict)

            # Check for r-coordinate dv and cylindrical consistency
            for idict in dv_info:
                if idict["dimension"] != 0:
                    continue
                for k in idict["indices"]:
                    m = self.modeling_options["floating"]["joints"]["name"][k]
                    if not self.modeling_options["floating"]["joints"]["cylindrical"][k]:
                        raise ValueError(f"Cannot optimize r-coordinate of, {m}, becase it is not a cylindrical joint")

            # Store DV information for needed linking and IVC assignment
            self.modeling_options["floating"]["joints"]["design_variable_data"] = dv_info

            # Now the members
            memgrps = [[m] for m in self.modeling_options["floating"]["members"]["name"]]
            for idv in float_opt_options["members"]["groups"]:
                inames = idv["names"]
                idx = [self.modeling_options["floating"]["members"]["name"].index(m) for m in inames]
                for k in range(1, len(idx)):
                    try:
                        memgrps[idx[k]].remove(inames[k])
                        memgrps[idx[0]].append(inames[k])
                    except ValueError:
                        raise ValueError("Cannot put member," + inames[k] + ", as part of multiple groups")

            # Remove entries for members that are now linked with others
            while [] in memgrps:
                memgrps.remove([])
            self.modeling_options["floating"]["members"]["linked_members"] = memgrps

            # Make a name 2 group index lookup
            name2grp = {}
            for k, kgrp in enumerate(memgrps):
                for kname in kgrp:
                    name2grp[kname] = k
            self.modeling_options["floating"]["members"]["name2idx"] = name2grp

    def update_ontology(self, wt_opt):
        # Update blade
        if self.modeling_options["flags"]["blade"] or self.modeling_options["user_elastic"]["blade"]:
            # Update blade outer shape
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_af_master"]):
                self.wt_init["components"]["blade"]["outer_shape"]["airfoils"][i]["spanwise_position"] = float(
                    wt_opt["blade.opt_var.af_position"][i]
                )
            self.wt_init["components"]["blade"]["outer_shape"]["chord"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["chord"]["values"] = wt_opt[
                "blade.pa.chord_param"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["twist"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["twist"]["values"] = np.rad2deg(
                wt_opt["rotorse.theta"]
            ).tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["section_offset_y"]["grid"] = wt_opt[
                "blade.outer_shape.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["section_offset_y"]["values"] = wt_opt[
                "blade.pa.section_offset_y_param"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["rthick"] = {}
            self.wt_init["components"]["blade"]["outer_shape"]["rthick"]["grid"] = wt_opt[
                "blade.outer_shape.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape"]["rthick"]["values"] = wt_opt[
                "blade.interp_airfoils.rthick_interp"
            ].tolist()
            self.wt_init["components"]["blade"]["reference_axis"]["x"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
            self.wt_init["components"]["blade"]["reference_axis"]["y"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
            self.wt_init["components"]["blade"]["reference_axis"]["z"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
            self.wt_init["components"]["blade"]["reference_axis"]["x"]["values"] = wt_opt[
                "blade.high_level_blade_props.blade_ref_axis"
            ][:, 0].tolist()
            self.wt_init["components"]["blade"]["reference_axis"]["y"]["values"] = wt_opt[
                "blade.high_level_blade_props.blade_ref_axis"
            ][:, 1].tolist()
            self.wt_init["components"]["blade"]["reference_axis"]["z"]["values"] = wt_opt[
                "blade.high_level_blade_props.blade_ref_axis"
            ][:, 2].tolist()

            # Update blade structure
            # Reference axis from blade outer shape
            if not self.modeling_options["user_elastic"]["blade"]:
                # Webs positions TBD
                # Structural layers
                anchors = self.wt_init["components"]["blade"]["structure"]["anchors"]
                n_anchors = len(anchors)
                for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_layers"]):
                    self.wt_init["components"]["blade"]["structure"]["layers"][i]["thickness"]["grid"] = wt_opt[
                        "blade.outer_shape.s"
                    ].tolist()
                    self.wt_init["components"]["blade"]["structure"]["layers"][i]["thickness"]["values"] = wt_opt[
                        "blade.ps.layer_thickness_param"
                    ][i, :].tolist()
                    # Update start_nd_arc for the anchor
                    if "anchor" in self.wt_init["components"]["blade"]["structure"]["layers"][i]["start_nd_arc"]:
                        anchor_name = self.wt_init["components"]["blade"]["structure"]["layers"][i]["start_nd_arc"]["anchor"]["name"]
                        anchor_handle = self.wt_init["components"]["blade"]["structure"]["layers"][i]["start_nd_arc"]["anchor"]["handle"]
                        for j in range(n_anchors):
                            if anchors[j]["name"] == anchor_name:
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["grid"] = wt_opt[
                                    "blade.outer_shape.s"
                                ].tolist()
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["values"] = wt_opt[
                                    "blade.structure.layer_start_nd"
                                ][i, :].tolist()
                                if "width" in self.wt_init["components"]["blade"]["structure"]["anchors"][j]:
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["width"]["grid"] = wt_opt[
                                        "blade.outer_shape.s"
                                    ].tolist()
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["width"]["values"] = wt_opt[
                                        "blade.structure.layer_width_adjusted"
                                    ][i, :].tolist()
                                if "plane_intersection" in self.wt_init["components"]["blade"]["structure"]["anchors"][j]:
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["values"] = wt_opt[
                                        "blade.structure.layer_offset_adjusted"
                                    ][i, :].tolist()
                                break
                    # Update end_nd_arc for the anchor
                    if "anchor" in self.wt_init["components"]["blade"]["structure"]["layers"][i]["end_nd_arc"]:
                        anchor_name = self.wt_init["components"]["blade"]["structure"]["layers"][i]["end_nd_arc"]["anchor"]["name"]
                        anchor_handle = self.wt_init["components"]["blade"]["structure"]["layers"][i]["end_nd_arc"]["anchor"]["handle"]
                        for j in range(n_anchors):
                            if anchors[j]["name"] == anchor_name:
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["grid"] = wt_opt[
                                    "blade.outer_shape.s"
                                ].tolist()
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["values"] = wt_opt[
                                    "blade.structure.layer_end_nd"
                                ][i, :].tolist()
                                if "width" in self.wt_init["components"]["blade"]["structure"]["anchors"][j]:
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["width"]["grid"] = wt_opt[
                                        "blade.outer_shape.s"
                                    ].tolist()
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["width"]["values"] = wt_opt[
                                        "blade.structure.layer_width_adjusted"
                                    ][i, :].tolist()
                                if "plane_intersection" in self.wt_init["components"]["blade"]["structure"]["anchors"][j]:
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["values"] = wt_opt[
                                        "blade.structure.layer_offset_adjusted"
                                    ][i, :].tolist()
                                break
                for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_webs"]):
                    # Update start_nd_arc for the anchor
                    if "anchor" in self.wt_init["components"]["blade"]["structure"]["webs"][i]["start_nd_arc"]:
                        anchor_name = self.wt_init["components"]["blade"]["structure"]["webs"][i]["start_nd_arc"]["anchor"]["name"]
                        anchor_handle = self.wt_init["components"]["blade"]["structure"]["webs"][i]["start_nd_arc"]["anchor"]["handle"]
                        for j in range(n_anchors):
                            if anchors[j]["name"] == anchor_name:
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["grid"] = wt_opt[
                                    "blade.outer_shape.s"
                                ].tolist()
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["values"] = wt_opt[
                                    "blade.structure.web_start_nd"
                                ][i, :].tolist()
                                if "plane_intersection" in self.wt_init["components"]["blade"]["structure"]["anchors"][j]:
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["values"] = wt_opt[
                                        "blade.structure.web_offset_adjusted"
                                    ][i, :].tolist()
                                break
                    # Update end_nd_arc for the anchor
                    if "anchor" in self.wt_init["components"]["blade"]["structure"]["webs"][i]["end_nd_arc"]:
                        anchor_name = self.wt_init["components"]["blade"]["structure"]["webs"][i]["end_nd_arc"]["anchor"]["name"]
                        anchor_handle = self.wt_init["components"]["blade"]["structure"]["webs"][i]["end_nd_arc"]["anchor"]["handle"]
                        for j in range(n_anchors):
                            if anchors[j]["name"] == anchor_name:
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["grid"] = wt_opt[
                                    "blade.outer_shape.s"
                                ].tolist()
                                self.wt_init["components"]["blade"]["structure"]["anchors"][j][anchor_handle]["values"] = wt_opt[
                                    "blade.structure.web_end_nd"
                                ][i, :].tolist()
                                if "plane_intersection" in self.wt_init["components"]["blade"]["structure"]["anchors"][j]:
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["grid"] = wt_opt["blade.outer_shape.s"].tolist()
                                    self.wt_init["components"]["blade"]["structure"]["anchors"][j]["plane_intersection"]["offset"]["values"] = wt_opt[
                                        "blade.structure.web_offset_adjusted"
                                    ][i, :].tolist()
                                break


            if "elastic_properties" not in self.wt_init["components"]["blade"]["structure"]:
                self.wt_init["components"]["blade"]["structure"]["elastic_properties"] = {}
                self.wt_init["components"]["blade"]["structure"]["elastic_properties"]["stiffness_matrix"] = {}
                self.wt_init["components"]["blade"]["structure"]["elastic_properties"]["structural_damping"] = {}
                self.wt_init["components"]["blade"]["structure"]["elastic_properties"]["structural_damping"]["mu"] = (
                    np.zeros(6).tolist()
                )

            self.wt_init["components"]["blade"]["structure"]["elastic_properties"]["stiffness_matrix"]["grid"] = wt_opt[
                "blade.outer_shape.s"
            ].tolist()

            stiff_terms = [11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 33, 34, 35, 36, 44, 45, 46, 55, 56, 66]
            for term in stiff_terms:
                self.wt_init["components"]["blade"]["structure"]["elastic_properties"]["stiffness_matrix"][
                    "K" + str(term)
                ] = np.array(wt_opt["rotorse.re.K"][:, int(term // 10 - 1), int(term % 10 - 1)]).tolist()

            I = {}
            I["grid"] = wt_opt["blade.outer_shape.s"].tolist()
            I["mass"] = wt_opt["rotorse.re.I"][:, 0, 0].tolist()
            I["cm_x"] = wt_opt["rotorse.re.x_cg"].tolist()
            I["cm_y"] = wt_opt["rotorse.re.y_cg"].tolist()
            I["i_edge"] = wt_opt["rotorse.re.I"][:, 3, 3].tolist()
            I["i_flap"] = wt_opt["rotorse.re.I"][:, 4, 4].tolist()
            I["i_plr"] = (np.asarray(I["i_edge"]) + np.asarray(I["i_flap"])).tolist()
            I["i_cp"] = wt_opt["rotorse.re.I"][:, 3, 4].tolist()

            self.wt_init["components"]["blade"]["structure"]["elastic_properties"]["inertia_matrix"] = I

        # Update hub
        if self.modeling_options["flags"]["hub"]:
            # Update hub
            self.wt_init["components"]["hub"]["cone_angle"] = float(wt_opt["hub.cone"][0])
            self.wt_init["components"]["hub"]["flange_t2shell_t"] = float(wt_opt["hub.flange_t2shell_t"][0])
            self.wt_init["components"]["hub"]["flange_OD2hub_D"] = float(wt_opt["hub.flange_OD2hub_D"][0])
            self.wt_init["components"]["hub"]["flange_ID2OD"] = float(wt_opt["hub.flange_ID2flange_OD"][0])
            self.wt_init["components"]["hub"]["hub_blade_spacing_margin"] = float(wt_opt["hub.hub_in2out_circ"][0])
            self.wt_init["components"]["hub"]["hub_stress_concentration"] = float(
                wt_opt["hub.hub_stress_concentration"][0]
            )
            self.wt_init["components"]["hub"]["n_front_brackets"] = int(wt_opt["hub.n_front_brackets"])
            self.wt_init["components"]["hub"]["n_rear_brackets"] = int(wt_opt["hub.n_rear_brackets"])
            self.wt_init["components"]["hub"]["clearance_hub_spinner"] = float(wt_opt["hub.clearance_hub_spinner"][0])
            self.wt_init["components"]["hub"]["spin_hole_incr"] = float(wt_opt["hub.spin_hole_incr"][0])
            self.wt_init["components"]["hub"]["pitch_system_scaling_factor"] = float(
                wt_opt["hub.pitch_system_scaling_factor"][0]
            )
            if "elastic_properties" not in self.wt_init["components"]["hub"]:
                self.wt_init["components"]["hub"]["elastic_properties"] = {}
            self.wt_init["components"]["hub"]["elastic_properties"]["mass"] = float(
                wt_opt["drivese.hub_system_mass"][0]
            )
            self.wt_init["components"]["hub"]["elastic_properties"]["inertia"] = wt_opt["drivese.hub_system_I"].tolist()
            # windIO and OpenFAST center of mass is measured from rotor apex.  WISDEM center of mass is measured from hub flange.
            cm_rotor_apex = 0.5 * wt_opt["drivese.hub_diameter"][0] - wt_opt["drivese.hub_system_cm"][0]
            self.wt_init["components"]["hub"]["elastic_properties"]["location"] = [float(cm_rotor_apex), 0.0, 0.0]

        # Update drivetrain
        if self.modeling_options["flags"]["drivetrain"]:
            # Common direct and geared
            self.wt_init["components"]["drivetrain"]["outer_shape"]["uptilt"] = float(wt_opt["drivetrain.uptilt"][0])
            self.wt_init["components"]["drivetrain"]["outer_shape"]["distance_tt_hub"] = float(
                wt_opt["drivetrain.distance_tt_hub"][0]
            )
            self.wt_init["components"]["drivetrain"]["outer_shape"]["overhang"] = float(
                wt_opt["drivetrain.overhang"][0]
            )
            self.wt_init["components"]["drivetrain"]["outer_shape"]["distance_hub_mb"] = float(
                wt_opt["drivetrain.distance_hub_mb"][0]
            )
            self.wt_init["components"]["drivetrain"]["outer_shape"]["distance_mb_mb"] = float(
                wt_opt["drivetrain.distance_mb_mb"][0]
            )
            self.wt_init["components"]["drivetrain"]["lss"]["diameter"] = wt_opt["drivetrain.lss_diameter"].tolist()
            self.wt_init["components"]["drivetrain"]["lss"]["wall_thickness"] = wt_opt[
                "drivetrain.lss_wall_thickness"
            ].tolist()
            self.wt_init["components"]["drivetrain"]["gearbox"]["gear_ratio"] = float(
                wt_opt["drivetrain.gear_ratio"][0]
            )
            self.wt_init["components"]["drivetrain"]["gearbox"]["efficiency"] = float(
                wt_opt["drivetrain.gearbox_efficiency"][0]
            )
            self.wt_init["components"]["drivetrain"]["other_components"]["mb1Type"] = wt_opt["drivetrain.mb1Type"]
            self.wt_init["components"]["drivetrain"]["other_components"]["mb2Type"] = wt_opt["drivetrain.mb2Type"]
            self.wt_init["components"]["drivetrain"]["other_components"]["uptower"] = wt_opt["drivetrain.uptower"]
            self.wt_init["components"]["drivetrain"]["lss"]["material"] = wt_opt["drivetrain.lss_material"]
            self.wt_init["components"]["drivetrain"]["bedplate"]["material"] = wt_opt["drivetrain.bedplate_material"]
            # WindIO v2
            if "elastic_properties" not in self.wt_init["components"]["drivetrain"]:
                self.wt_init["components"]["drivetrain"]["elastic_properties"] = {}
            self.wt_init["components"]["drivetrain"]["elastic_properties"]["mass"] = float(
                wt_opt["drivese.above_yaw_mass"][0]
            )
            self.wt_init["components"]["drivetrain"]["elastic_properties"]["inertia"] = wt_opt[
                "drivese.above_yaw_I"
            ].tolist()
            self.wt_init["components"]["drivetrain"]["elastic_properties"]["inertia_tt"] = wt_opt[
                "drivese.above_yaw_I_TT"
            ].tolist()
            self.wt_init["components"]["drivetrain"]["elastic_properties"]["location"] = wt_opt[
                "drivese.above_yaw_cm"
            ].tolist()
            self.wt_init["components"]["drivetrain"]["elastic_properties"]["spring_constant"] = float(
                wt_opt["drivese.drivetrain_spring_constant"][0]
            )
            self.wt_init["components"]["drivetrain"]["elastic_properties"]["damping_coefficient"] = float(
                wt_opt["drivese.drivetrain_damping_coefficient"][0]
            )

            if "yaw" not in self.wt_init["components"]:
                self.wt_init["components"]["yaw"] = {}
            if "elastic_properties" not in self.wt_init["components"]["yaw"]:
                self.wt_init["components"]["yaw"]["elastic_properties"] = {}
            self.wt_init["components"]["yaw"]["elastic_properties"]["mass"] = float(wt_opt["drivese.yaw_mass"][0])
            self.wt_init["components"]["yaw"]["elastic_properties"]["inertia"] = np.zeros(3).tolist()
            self.wt_init["components"]["yaw"]["elastic_properties"]["location"] = np.zeros(3).tolist()

            if self.modeling_options["WISDEM"]["DriveSE"]["direct"]:
                # Direct only
                s_bed = np.linspace(0.0, 1.0, len(wt_opt["drivetrain.bedplate_wall_thickness"])).tolist()
                self.wt_init["components"]["drivetrain"]["nose"]["diameter"] = wt_opt[
                    "drivetrain.nose_diameter"
                ].tolist()
                self.wt_init["components"]["drivetrain"]["nose"]["wall_thickness"] = wt_opt[
                    "drivetrain.nose_wall_thickness"
                ].tolist()
                self.wt_init["components"]["drivetrain"]["bedplate"]["wall_thickness"]["grid"] = s_bed
                self.wt_init["components"]["drivetrain"]["bedplate"]["wall_thickness"]["values"] = wt_opt[
                    "drivetrain.bedplate_wall_thickness"
                ].tolist()
            else:
                # Geared only
                self.wt_init["components"]["drivetrain"]["hss"]["length"] = float(wt_opt["drivetrain.hss_length"][0])
                self.wt_init["components"]["drivetrain"]["hss"]["diameter"] = wt_opt["drivetrain.hss_diameter"].tolist()
                self.wt_init["components"]["drivetrain"]["hss"]["wall_thickness"] = wt_opt[
                    "drivetrain.hss_wall_thickness"
                ].tolist()
                self.wt_init["components"]["drivetrain"]["bedplate"]["flange_width"] = float(
                    wt_opt["drivetrain.bedplate_flange_width"][0]
                )
                self.wt_init["components"]["drivetrain"]["bedplate"]["flange_thickness"] = float(
                    wt_opt["drivetrain.bedplate_flange_thickness"][0]
                )
                self.wt_init["components"]["drivetrain"]["bedplate"]["web_thickness"] = float(
                    wt_opt["drivetrain.bedplate_web_thickness"][0]
                )
                self.wt_init["components"]["drivetrain"]["gearbox"]["gear_configuration"] = wt_opt[
                    "drivetrain.gear_configuration"
                ]
                self.wt_init["components"]["drivetrain"]["gearbox"]["planet_numbers"] = wt_opt[
                    "drivetrain.planet_numbers"
                ]
                self.wt_init["components"]["drivetrain"]["hss"]["material"] = wt_opt["drivetrain.hss_material"]

        # Update generator
        if self.modeling_options["flags"]["drivetrain"] and "generator" in self.wt_init["components"]["drivetrain"]:
            self.wt_init["components"]["drivetrain"]["generator"]["length"] = float(wt_opt["generator.L_generator"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["mass"] = float(
                wt_opt["generator.generator_mass_user"][0]
            )
            if not self.modeling_options["flags"]["generator"]:
                self.wt_init["components"]["drivetrain"]["generator"]["radius"] = float(
                    wt_opt["generator.generator_radius_user"][0]
                )
                self.wt_init["components"]["drivetrain"]["generator"]["rpm_efficiency"]["grid"] = wt_opt[
                    "generator.generator_efficiency_user"
                ][:, 0].tolist()
                self.wt_init["components"]["drivetrain"]["generator"]["rpm_efficiency"]["values"] = wt_opt[
                    "generator.generator_efficiency_user"
                ][:, 1].tolist()

            if "elastic_properties" not in self.wt_init["components"]["drivetrain"]["generator"]:
                self.wt_init["components"]["drivetrain"]["generator"]["elastic_properties"] = {}
            self.wt_init["components"]["drivetrain"]["generator"]["elastic_properties"]["mass"] = float(
                wt_opt["drivese.generator_mass"][0]
            )
            self.wt_init["components"]["drivetrain"]["generator"]["elastic_properties"]["inertia"] = wt_opt[
                "drivese.generator_rotor_I"
            ].tolist()
            self.wt_init["components"]["drivetrain"]["generator"]["elastic_properties"]["location"] = wt_opt[
                "drivese.generator_cm"
            ].tolist()

        if self.modeling_options["flags"]["generator"]:
            self.wt_init["components"]["drivetrain"]["generator"]["B_r"] = float(wt_opt["generator.B_r"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["P_Fe0e"] = float(wt_opt["generator.P_Fe0e"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["P_Fe0h"] = float(wt_opt["generator.P_Fe0h"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["S_N"] = float(wt_opt["generator.S_N"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["alpha_p"] = float(wt_opt["generator.alpha_p"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["b_r_tau_r"] = float(wt_opt["generator.b_r_tau_r"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["b_ro"] = float(wt_opt["generator.b_ro"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["b_s_tau_s"] = float(wt_opt["generator.b_s_tau_s"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["b_so"] = float(wt_opt["generator.b_so"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["cofi"] = float(wt_opt["generator.cofi"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["freq"] = float(wt_opt["generator.freq"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["h_i"] = float(wt_opt["generator.h_i"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["h_sy0"] = float(wt_opt["generator.h_sy0"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["h_w"] = float(wt_opt["generator.h_w"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["k_fes"] = float(wt_opt["generator.k_fes"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["k_fillr"] = float(wt_opt["generator.k_fillr"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["k_fills"] = float(wt_opt["generator.k_fills"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["k_s"] = float(wt_opt["generator.k_s"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["m"] = wt_opt["generator.m"]
            self.wt_init["components"]["drivetrain"]["generator"]["mu_0"] = float(wt_opt["generator.mu_0"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["mu_r"] = float(wt_opt["generator.mu_r"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["p"] = float(wt_opt["generator.p"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["phi"] = float(wt_opt["generator.phi"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["q1"] = wt_opt["generator.q1"]
            self.wt_init["components"]["drivetrain"]["generator"]["q2"] = wt_opt["generator.q2"]
            self.wt_init["components"]["drivetrain"]["generator"]["ratio_mw2pp"] = float(
                wt_opt["generator.ratio_mw2pp"][0]
            )
            self.wt_init["components"]["drivetrain"]["generator"]["resist_Cu"] = float(wt_opt["generator.resist_Cu"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["sigma"] = float(wt_opt["generator.sigma"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["y_tau_p"] = float(wt_opt["generator.y_tau_p"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["y_tau_pr"] = float(wt_opt["generator.y_tau_pr"][0])

            self.wt_init["components"]["drivetrain"]["generator"]["I_0"] = float(wt_opt["generator.I_0"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["d_r"] = float(wt_opt["generator.d_r"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["h_m"] = float(wt_opt["generator.h_m"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["h_0"] = float(wt_opt["generator.h_0"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["h_s"] = float(wt_opt["generator.h_s"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["len_s"] = float(wt_opt["generator.len_s"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["n_r"] = float(wt_opt["generator.n_r"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["rad_ag"] = float(wt_opt["generator.rad_ag"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["t_wr"] = float(wt_opt["generator.t_wr"][0])

            self.wt_init["components"]["drivetrain"]["generator"]["n_s"] = float(wt_opt["generator.n_s"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["b_st"] = float(wt_opt["generator.b_st"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["d_s"] = float(wt_opt["generator.d_s"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["t_ws"] = float(wt_opt["generator.t_ws"][0])

            self.wt_init["components"]["drivetrain"]["generator"]["rho_Copper"] = float(
                wt_opt["generator.rho_Copper"][0]
            )
            self.wt_init["components"]["drivetrain"]["generator"]["rho_Fe"] = float(wt_opt["generator.rho_Fe"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["rho_Fes"] = float(wt_opt["generator.rho_Fes"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["rho_PM"] = float(wt_opt["generator.rho_PM"][0])

            self.wt_init["components"]["drivetrain"]["generator"]["C_Cu"] = float(wt_opt["generator.C_Cu"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["C_Fe"] = float(wt_opt["generator.C_Fe"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["C_Fes"] = float(wt_opt["generator.C_Fes"][0])
            self.wt_init["components"]["drivetrain"]["generator"]["C_PM"] = float(wt_opt["generator.C_PM"][0])

            if self.modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] in ["pmsg_outer"]:
                self.wt_init["components"]["drivetrain"]["generator"]["N_c"] = float(wt_opt["generator.N_c"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["b"] = float(wt_opt["generator.b"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["c"] = float(wt_opt["generator.c"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["E_p"] = float(wt_opt["generator.E_p"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["h_yr"] = float(wt_opt["generator.h_yr"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["h_ys"] = float(wt_opt["generator.h_ys"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["h_sr"] = float(wt_opt["generator.h_sr"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["h_ss"] = float(wt_opt["generator.h_ss"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["t_r"] = float(wt_opt["generator.t_r"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["t_s"] = float(wt_opt["generator.t_s"][0])

                self.wt_init["components"]["drivetrain"]["generator"]["u_allow_pcent"] = float(
                    wt_opt["generator.u_allow_pcent"][0]
                )
                self.wt_init["components"]["drivetrain"]["generator"]["y_allow_pcent"] = float(
                    wt_opt["generator.y_allow_pcent"][0]
                )
                self.wt_init["components"]["drivetrain"]["generator"]["z_allow_deg"] = float(
                    wt_opt["generator.z_allow_deg"][0]
                )
                self.wt_init["components"]["drivetrain"]["generator"]["B_tmax"] = float(wt_opt["generator.B_tmax"][0])

            if self.modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                self.wt_init["components"]["drivetrain"]["generator"]["tau_p"] = float(wt_opt["generator.tau_p"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["h_ys"] = float(wt_opt["generator.h_ys"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["h_yr"] = float(wt_opt["generator.h_yr"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["b_arm"] = float(wt_opt["generator.b_arm"][0])

            elif self.modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] in ["scig", "dfig"]:
                self.wt_init["components"]["drivetrain"]["generator"]["B_symax"] = float(wt_opt["generator.B_symax"][0])
                self.wt_init["components"]["drivetrain"]["generator"]["S_Nmax"] = float(wt_opt["generator.S_Nmax"][0])

        # Update tower
        if self.modeling_options["flags"]["tower"]:
            self.wt_init["components"]["tower"]["outer_shape"]["outer_diameter"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["outer_shape"]["outer_diameter"]["values"] = wt_opt[
                "tower.diameter"
            ].tolist()
            self.wt_init["components"]["tower"]["reference_axis"]["x"]["grid"] = wt_opt["tower_grid.s"].tolist()
            self.wt_init["components"]["tower"]["reference_axis"]["y"]["grid"] = wt_opt["tower_grid.s"].tolist()
            self.wt_init["components"]["tower"]["reference_axis"]["z"]["grid"] = wt_opt["tower_grid.s"].tolist()
            self.wt_init["components"]["tower"]["reference_axis"]["x"]["values"] = wt_opt["tower.ref_axis"][
                :, 0
            ].tolist()
            self.wt_init["components"]["tower"]["reference_axis"]["y"]["values"] = wt_opt["tower.ref_axis"][
                :, 1
            ].tolist()
            self.wt_init["components"]["tower"]["reference_axis"]["z"]["values"] = wt_opt["tower.ref_axis"][
                :, 2
            ].tolist()
            self.wt_init["components"]["tower"]["structure"]["outfitting_factor"] = float(
                wt_opt["tower.outfitting_factor"][0]
            )
            for i in range(self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"]):
                self.wt_init["components"]["tower"]["structure"]["layers"][i]["thickness"]["grid"] = wt_opt[
                    "tower_grid.s"
                ].tolist()
                self.wt_init["components"]["tower"]["structure"]["layers"][i]["thickness"]["values"] = wt_opt[
                    "tower.layer_thickness"
                ][i, :].tolist()
            # Update elastic properties of the tower
            if "elastic_properties" not in self.wt_init["components"]["tower"]["structure"]:
                self.wt_init["components"]["tower"]["structure"]["elastic_properties"] = {}
                self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["stiffness_matrix"] = {}
                self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["inertia_matrix"] = {}
                self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["structural_damping"] = {}
                self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["structural_damping"]["mu"] = (
                    np.zeros(6).tolist()
                )
            self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["stiffness_matrix"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["stiffness_matrix"]["K44"] = (
                sectional2nodal(wt_opt["towerse.member.sideside_stff"]).tolist()
            )
            self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["stiffness_matrix"]["K55"] = (
                sectional2nodal(wt_opt["towerse.member.foreaft_stff"]).tolist()
            )
            self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["stiffness_matrix"]["K66"] = (
                sectional2nodal(wt_opt["towerse.member.tor_stff"]).tolist()
            )
            self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["inertia_matrix"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["structure"]["elastic_properties"]["inertia_matrix"]["mass"] = (
                sectional2nodal(wt_opt["towerse.member.mass_den"]).tolist()
            )

        # Update monopile
        if self.modeling_options["flags"]["monopile"]:
            self.wt_init["components"]["monopile"]["outer_shape"]["outer_diameter"]["grid"] = wt_opt[
                "monopile.s"
            ].tolist()
            self.wt_init["components"]["monopile"]["outer_shape"]["outer_diameter"]["values"] = wt_opt[
                "monopile.diameter"
            ].tolist()
            self.wt_init["components"]["monopile"]["reference_axis"]["x"]["grid"] = wt_opt["monopile.s"].tolist()
            self.wt_init["components"]["monopile"]["reference_axis"]["y"]["grid"] = wt_opt["monopile.s"].tolist()
            self.wt_init["components"]["monopile"]["reference_axis"]["z"]["grid"] = wt_opt["monopile.s"].tolist()
            self.wt_init["components"]["monopile"]["reference_axis"]["x"]["values"] = wt_opt["monopile.ref_axis"][
                :, 0
            ].tolist()
            self.wt_init["components"]["monopile"]["reference_axis"]["y"]["values"] = wt_opt["monopile.ref_axis"][
                :, 1
            ].tolist()
            self.wt_init["components"]["monopile"]["reference_axis"]["z"]["values"] = wt_opt["monopile.ref_axis"][
                :, 2
            ].tolist()
            self.wt_init["components"]["monopile"]["structure"]["outfitting_factor"] = float(
                wt_opt["monopile.outfitting_factor"][0]
            )
            for i in range(self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers_monopile"]):
                self.wt_init["components"]["monopile"]["structure"]["layers"][i]["thickness"]["grid"] = wt_opt[
                    "monopile.s"
                ].tolist()
                self.wt_init["components"]["monopile"]["structure"]["layers"][i]["thickness"]["values"] = wt_opt[
                    "monopile.layer_thickness"
                ][i, :].tolist()
                # Update elastic properties of the tower
            if "elastic_properties" not in self.wt_init["components"]["monopile"]["structure"]:
                self.wt_init["components"]["monopile"]["structure"]["elastic_properties"] = {}
                self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["stiffness_matrix"] = {}
                self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["inertia_matrix"] = {}
                self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["structural_damping"] = {}
                self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["structural_damping"][
                    "mu"
                ] = np.zeros(6).tolist()
            self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["stiffness_matrix"]["grid"] = (
                wt_opt["fixedse.member.s"].tolist()
            )
            self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["stiffness_matrix"]["K44"] = (
                sectional2nodal(wt_opt["fixedse.member.sideside_stff"]).tolist()
            )
            self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["stiffness_matrix"]["K55"] = (
                sectional2nodal(wt_opt["fixedse.member.foreaft_stff"]).tolist()
            )
            self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["stiffness_matrix"]["K66"] = (
                sectional2nodal(wt_opt["fixedse.member.tor_stff"]).tolist()
            )
            self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["inertia_matrix"]["grid"] = (
                wt_opt["fixedse.member.s"].tolist()
            )
            self.wt_init["components"]["monopile"]["structure"]["elastic_properties"]["inertia_matrix"]["mass"] = (
                sectional2nodal(wt_opt["fixedse.member.added_mass"]).tolist()
            )

        # Update jacket
        if self.modeling_options["flags"]["jacket"]:
            self.wt_init["components"]["jacket"]["r_head"] = float(wt_opt["jacket.r_head"][0])
            self.wt_init["components"]["jacket"]["r_foot"] = float(
                wt_opt["jacket.r_head"][0] * wt_opt["jacket.foot_head_ratio"][0]
            )
            self.wt_init["components"]["jacket"]["height"] = float(wt_opt["jacket.height"][0])
            self.wt_init["components"]["jacket"]["leg_diameter"] = float(wt_opt["jacket.leg_diameter"][0])
            self.wt_init["components"]["jacket"]["leg_thickness"] = float(wt_opt["jacket.leg_thickness"][0])
            self.wt_init["components"]["jacket"]["brace_diameters"] = wt_opt["jacket.brace_diameters"].tolist()
            self.wt_init["components"]["jacket"]["brace_thicknesses"] = wt_opt["jacket.brace_thicknesses"].tolist()
            self.wt_init["components"]["jacket"]["bay_spacing"] = wt_opt["jacket.bay_spacing"].tolist()

        # Update floating platform and mooring
        if self.modeling_options["flags"]["floating"]:
            yaml_out = self.wt_init["components"]["floating_platform"]
            n_joints = self.modeling_options["floating"]["joints"]["n_joints"]
            for i in range(n_joints):
                yaml_out["joints"][i]["location"] = wt_opt["floating.location"][i, :].tolist()

            n_members = self.modeling_options["floating"]["members"]["n_members"]
            for i in range(n_members):
                name_member = self.modeling_options["floating"]["members"]["name"][i]
                idx = self.modeling_options["floating"]["members"]["name2idx"][name_member]
                member_shape = yaml_out["members"][i]["outer_shape"]["shape"]

                s_in = wt_opt[f"floating.memgrp{idx}.s_in"].tolist()
                if member_shape == "circular":
                    yaml_out["members"][i]["outer_shape"]["outer_diameter"]["grid"] = s_in
                    d_in = wt_opt[f"floating.memgrp{idx}.outer_diameter_in"]
                    if len(d_in) == len(s_in):
                        yaml_out["members"][i]["outer_shape"]["outer_diameter"]["values"] = d_in.tolist()
                    else:
                        d_in2 = d_in * np.ones(len(s_in))
                        yaml_out["members"][i]["outer_shape"]["outer_diameter"]["values"] = d_in2.tolist()
                elif member_shape == "rectangular":
                    yaml_out["members"][i]["outer_shape"]["side_length_a"]["grid"] = s_in
                    yaml_out["members"][i]["outer_shape"]["side_length_b"]["grid"] = s_in
                    length_a_in = wt_opt[f"floating.memgrp{idx}.side_length_a_in"]
                    length_b_in = wt_opt[f"floating.memgrp{idx}.side_length_b_in"]
                    if len(length_a_in) == len(s_in):
                        yaml_out["members"][i]["outer_shape"]["side_length_a"]["values"] = length_a_in.tolist()
                        yaml_out["members"][i]["outer_shape"]["side_length_b"]["values"] = length_b_in.tolist()
                    else:
                        length_a_in2 = length_a_in * np.ones(len(s_in))
                        length_b_in2 = length_b_in * np.ones(len(s_in))
                        yaml_out["members"][i]["outer_shape"]["side_length_a"]["values"] = length_a_in2.tolist()
                        yaml_out["members"][i]["outer_shape"]["side_length_b"]["values"] = length_b_in2.tolist()

                istruct = yaml_out["members"][i]["structure"]

                n_layers = self.modeling_options["floating"]["members"]["n_layers"][i]
                for j in range(n_layers):
                    istruct["layers"][j]["thickness"]["grid"] = wt_opt[f"floating.memgrp{idx}.s_in"].tolist()
                    istruct["layers"][j]["thickness"]["values"] = wt_opt[f"floating.memgrp{idx}.layer_thickness_in"][
                        j, :
                    ].tolist()

                if "ring_stiffeners" in istruct:
                    istruct["ring_stiffeners"]["web_height"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_web_height"][0]
                    )
                    istruct["ring_stiffeners"]["web_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_web_thickness"][0]
                    )
                    istruct["ring_stiffeners"]["flange_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_flange_thickness"][0]
                    )
                    istruct["ring_stiffeners"]["flange_width"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_flange_width"][0]
                    )
                    istruct["ring_stiffeners"]["spacing"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_spacing"][0]
                    )

                if "longitudinal_stiffeners" in istruct:
                    istruct["longitudinal_stiffeners"]["web_height"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_web_height"][0]
                    )
                    istruct["longitudinal_stiffeners"]["web_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_web_thickness"][0]
                    )
                    istruct["longitudinal_stiffeners"]["flange_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_flange_thickness"][0]
                    )
                    istruct["longitudinal_stiffeners"]["flange_width"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_flange_width"][0]
                    )
                    istruct["longitudinal_stiffeners"]["spacing"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_spacing"][0]
                    )

                n_ballasts = self.modeling_options["floating"]["members"]["n_ballasts"][i]
                for j in range(n_ballasts):
                    if self.modeling_options["floating"]["members"]["ballast_flag_member_" + name_member][j] == False:
                        istruct["ballast"][j]["volume"] = float(wt_opt[f"floating.memgrp{idx}.ballast_volume"][j])

                if self.modeling_options["floating"]["members"]["n_axial_joints"][i] > 0:
                    for j in range(self.modeling_options["floating"]["members"]["n_axial_joints"][i]):
                        yaml_out["members"][i]["axial_joints"][j]["grid"] = float(
                            wt_opt[f"floating.memgrp{idx}.grid_axial_joints"][j]
                        )

        if self.modeling_options["flags"]["mooring"]:
            n_lines = self.modeling_options["mooring"]["n_lines"]
            n_line_types = self.modeling_options["mooring"]["n_line_types"]
            line_names = [self.wt_init["components"]["mooring"]["line_types"][i]["name"] for i in range(n_line_types)]
            line_id = [self.wt_init["components"]["mooring"]["lines"][i]["line_type"] for i in range(n_lines)]

            for i in range(n_lines):
                self.wt_init["components"]["mooring"]["lines"][i]["unstretched_length"] = float(
                    wt_opt["mooring.unstretched_length"][i]
                )

            for jj, jname in enumerate(line_id):
                for ii, iname in enumerate(line_names):
                    if jname == iname:
                        self.wt_init["components"]["mooring"]["line_types"][ii]["diameter"] = float(
                            wt_opt["mooring.line_diameter"][jj]
                        )

        # Update rotor nacelle assembly
        if self.modeling_options["flags"]["drivetrain"]:
            self.wt_init["components"]["RNA"] = {}
            self.wt_init["components"]["RNA"]["elastic_properties"] = {}
            self.wt_init["components"]["RNA"]["elastic_properties"]["mass"] = float(wt_opt["drivese.rna_mass"][0])
            self.wt_init["components"]["RNA"]["elastic_properties"]["inertia"] = wt_opt["drivese.rna_I_TT"].tolist()
            self.wt_init["components"]["RNA"]["elastic_properties"]["center_mass"] = wt_opt["drivese.rna_cm"].tolist()

        # Update rotor diameter and hub height
        if self.modeling_options["flags"]["blade"]:
            self.wt_init["assembly"]["rotor_diameter"] = float(wt_opt["blade.high_level_blade_props.rotor_diameter"][0])
        self.wt_init["assembly"]["hub_height"] = float(wt_opt["high_level_tower_props.hub_height"][0])

        # Update controller
        if self.modeling_options["flags"]["control"]:
            self.wt_init["control"]["optimal_tsr"] = float(wt_opt["control.rated_TSR"][0])
            if "ROSCO" not in self.modeling_options:  # If using WEIS, will have ROSCO, and peak_thrust_shaving will be set there
                self.wt_init["control"]["peak_thrust_shaving"] = float(wt_opt["control.peak_thrust_shaving"][0])

        # Update cost coefficients
        if self.modeling_options["flags"]["costs"]:
            if float(wt_opt["tcc.blade_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["blade_mass_cost_coeff"] = float(
                    (wt_opt["tcc.blade_cost"] / wt_opt["tcc.blade_mass"])[0]
                )
            if float(wt_opt["tcc.hub_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["hub_mass_cost_coeff"] = float(
                    (wt_opt["tcc.hub_cost"] / wt_opt["tcc.hub_mass"])[0]
                )
            if float(wt_opt["tcc.pitch_system_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["pitch_system_mass_cost_coeff"] = float(
                    (wt_opt["tcc.pitch_system_cost"] / wt_opt["tcc.pitch_system_mass"])[0]
                )
            if float(wt_opt["tcc.spinner_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["spinner_mass_cost_coeff"] = float(
                    (wt_opt["tcc.spinner_cost"] / wt_opt["tcc.spinner_mass"])[0]
                )
            if float(wt_opt["tcc.lss_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["lss_mass_cost_coeff"] = float(
                    (wt_opt["tcc.lss_cost"] / wt_opt["tcc.lss_mass"])[0]
                )
            if float(wt_opt["tcc.main_bearing_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["bearing_mass_cost_coeff"] = float(
                    (wt_opt["tcc.main_bearing_cost"] / wt_opt["tcc.main_bearing_mass"])[0]
                )
            if self.modeling_options["flags"]["drivetrain"]:
                if float(wt_opt["drivese.gearbox_mass"][0]) > 0.0:
                    self.modeling_options["WISDEM"]["LCOE"]["gearbox_torque_cost"] = float(
                        (wt_opt["tcc.gearbox_cost"] / wt_opt["drivese.rated_torque"])[0] * 1.0e3
                    )
            if float(wt_opt["tcc.hss_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["hss_mass_cost_coeff"] = float(
                    (wt_opt["tcc.hss_cost"] / wt_opt["tcc.hss_mass"])[0]
                )
            if float(wt_opt["tcc.generator_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["generator_mass_cost_coeff"] = float(
                    (wt_opt["tcc.generator_cost"] / wt_opt["tcc.generator_mass"])[0]
                )
            if float(wt_opt["tcc.bedplate_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["bedplate_mass_cost_coeff"] = float(
                    (wt_opt["tcc.bedplate_cost"] / wt_opt["tcc.bedplate_mass"])[0]
                )
            if float(wt_opt["tcc.yaw_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["yaw_mass_cost_coeff"] = float(
                    (wt_opt["tcc.yaw_system_cost"] / wt_opt["tcc.yaw_mass"])[0]
                )
            if float(wt_opt["tcc.converter_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["converter_mass_cost_coeff"] = float(
                    (wt_opt["tcc.converter_cost"] / wt_opt["tcc.converter_mass"])[0]
                )
            if float(wt_opt["tcc.transformer_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["transformer_mass_cost_coeff"] = float(
                    (wt_opt["tcc.transformer_cost"] / wt_opt["tcc.transformer_mass"])[0]
                )
            if float(wt_opt["tcc.hvac_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["hvac_mass_cost_coeff"] = float(
                    (wt_opt["tcc.hvac_cost"] / wt_opt["tcc.hvac_mass"])[0]
                )
            if float(wt_opt["tcc.cover_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["cover_mass_cost_coeff"] = float(
                    (wt_opt["tcc.cover_cost"] / wt_opt["tcc.cover_mass"])[0]
                )
            if float(wt_opt["tcc.tower_mass"][0]) > 0.0:
                self.modeling_options["WISDEM"]["LCOE"]["tower_mass_cost_coeff"] = float(
                    (wt_opt["tcc.tower_cost"] / wt_opt["tcc.tower_mass"])[0]
                )

    def write_outputs(self, fname_output):
        # Write yamls with updated values
        sch.write_geometry_yaml(self.wt_init, fname_output)

        sch.write_modeling_yaml(self.modeling_options, fname_output)
        sch.write_analysis_yaml(self.analysis_options, fname_output)
