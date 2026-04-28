import copy
import logging

import numpy as np
import openmdao.api as om
from moorpy.helpers import getLineProps
from scipy.interpolate import PchipInterpolator

from wisdem.ccblade.Polar import Polar
from wisdem.commonse.utilities import arc_length, arc_length_deriv
from wisdem.rotorse.parametrize_rotor import ComputeReynolds, ParametrizeBladeAero, ParametrizeBladeStruct
from wisdem.rotorse.geometry_tools.geometry import remap2grid, trailing_edge_smoothing

logger = logging.getLogger("wisdem/weis")


class WindTurbineOntologyOpenMDAO(om.Group):
    # Openmdao group with all wind turbine data

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        # Material dictionary inputs
        self.add_subsystem(
            "materials",
            Materials(mat_init_options=modeling_options["materials"], composites=modeling_options["flags"]["blade"]),
        )

        # Environment inputs
        if modeling_options["flags"]["environment"]:
            env_ivc = self.add_subsystem("env", om.IndepVarComp())
            env_ivc.add_output("rho_air", val=1.225, units="kg/m**3", desc="Density of air")
            env_ivc.add_output("mu_air", val=1.81e-5, units="kg/(m*s)", desc="Dynamic viscosity of air")
            env_ivc.add_output("shear_exp", val=0.2, desc="Shear exponent of the wind.")
            env_ivc.add_output("speed_sound_air", val=340.0, units="m/s", desc="Speed of sound in air.")
            env_ivc.add_output(
                "weibull_k", val=2.0, desc="Shape parameter of the Weibull probability density function of the wind."
            )
            env_ivc.add_output("rho_water", val=1025.0, units="kg/m**3", desc="Density of ocean water")
            env_ivc.add_output("mu_water", val=1.3351e-3, units="kg/(m*s)", desc="Dynamic viscosity of ocean water")
            env_ivc.add_output(
                "water_depth", val=0.0, units="m", desc="Water depth for analysis.  Values > 0 mean offshore"
            )
            env_ivc.add_output("Hsig_wave", val=0.0, units="m", desc="Significant wave height")
            env_ivc.add_output("Tsig_wave", val=0.0, units="s", desc="Significant wave period")
            env_ivc.add_output("G_soil", val=140e6, units="N/m**2", desc="Shear stress of soil")
            env_ivc.add_output("nu_soil", val=0.4, desc="Poisson ratio of soil")

        # Airfoil dictionary inputs
        if modeling_options["flags"]["airfoils"]:
            airfoils = om.IndepVarComp()
            rotorse_options = modeling_options["WISDEM"]["RotorSE"]
            n_af_master = rotorse_options["n_af_master"]  # Number of airfoils
            n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
            n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
            n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
            airfoils.add_output(
                "ac",
                val=np.zeros(n_af_master),
                desc="1D array of the aerodynamic centers of each airfoil used along span.",
            )
            airfoils.add_output(
                "rthick_master",
                val=np.zeros(n_af_master),
                desc="1D array of the relative thicknesses of each airfoil used along span.",
            )
            airfoils.add_output(
                "aoa",
                val=np.zeros(n_aoa),
                units="deg",
                desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
            )
            airfoils.add_output(
                "Re",
                val=np.zeros(n_Re),
                desc="1D array of the Reynolds numbers used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
            )
            airfoils.add_output(
                "cl",
                val=np.zeros((n_af_master, n_aoa, n_Re)),
                desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
            )
            airfoils.add_output(
                "cd",
                val=np.zeros((n_af_master, n_aoa, n_Re)),
                desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
            )
            airfoils.add_output(
                "cm",
                val=np.zeros((n_af_master, n_aoa, n_Re)),
                desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
            )
            # Airfoil coordinates
            airfoils.add_output(
                "coord_xy",
                val=np.zeros((n_af_master, n_xy, 2)),
                desc="3D array of the x and y airfoil coordinates of the n_af_master airfoils used along blade span.",
            )
            self.add_subsystem("airfoils", airfoils)

        # Wind turbine configuration inputs
        conf_ivc = self.add_subsystem("configuration", om.IndepVarComp())
        conf_ivc.add_discrete_output(
            "ws_class",
            val="",
            desc="IEC wind turbine class. I - offshore, II coastal, III - land-based, IV - low wind speed site.",
        )
        conf_ivc.add_discrete_output(
            "turb_class",
            val="",
            desc="IEC wind turbine category. A - high turbulence intensity (land-based), B - mid turbulence, C - low turbulence (offshore).",
        )
        conf_ivc.add_discrete_output(
            "gearbox_type", val="geared", desc="Gearbox configuration (geared, direct-drive, etc.)."
        )
        conf_ivc.add_discrete_output(
            "rotor_orientation", val="upwind", desc="Rotor orientation, either upwind or downwind."
        )
        conf_ivc.add_discrete_output(
            "upwind", val=True, desc="Convenient boolean for upwind (True) or downwind (False)."
        )
        conf_ivc.add_discrete_output("n_blades", val=3, desc="Number of blades of the rotor.")
        conf_ivc.add_output("rated_power", val=0.0, units="W", desc="Electrical rated power of the generator.")
        conf_ivc.add_output("lifetime", val=25.0, units="yr", desc="Turbine design lifetime.")

        conf_ivc.add_output(
            "rotor_diameter_user",
            val=0.0,
            units="m",
            desc="Diameter of the wind turbine rotor specified by the user, defined as 2 x (Rhub + blade length along z) * cos(precone).",
        )
        conf_ivc.add_output(
            "hub_height_user",
            val=0.0,
            units="m",
            desc="Height of the hub center over the ground (land-based) or the mean sea level (offshore) specified by the user.",
        )
        # Control inputs
        if modeling_options["flags"]["control"]:
            ctrl_ivc = self.add_subsystem("control", om.IndepVarComp())
            ctrl_ivc.add_output(
                "V_in", val=0.0, units="m/s", desc="Cut in wind speed. This is the wind speed where region II begins."
            )
            ctrl_ivc.add_output(
                "V_out", val=0.0, units="m/s", desc="Cut out wind speed. This is the wind speed where region III ends."
            )
            ctrl_ivc.add_output("minOmega", val=0.0, units="rpm", desc="Minimum allowed rotor speed.")
            ctrl_ivc.add_output("maxOmega", val=0.0, units="rpm", desc="Maximum allowed rotor speed.")
            ctrl_ivc.add_output("max_allowable_blade_tip_speed", val=0.0, units="m/s", desc="Maximum allowed blade tip speed.")
            ctrl_ivc.add_output("rated_TSR", val=0.0, desc="Constant tip speed ratio in region II.")
            ctrl_ivc.add_output("rated_pitch", val=0.0, units="deg", desc="Constant pitch angle in region II.")
            if "ROSCO" not in modeling_options:  # If using WEIS, peak_thrust_shaving will be set there
                ctrl_ivc.add_output(
                    "peak_thrust_shaving",
                    val=1.0,
                    desc="Scalar applied to the max thrust within RotorSE for peak thrust shaving.",
                )

        # Blade inputs and connections from airfoils
        if modeling_options["flags"]["blade"]:
            self.add_subsystem(
                "blade",
                Blade(
                    rotorse_options=modeling_options["WISDEM"]["RotorSE"],
                    opt_options=opt_options,
                    user_elastic=modeling_options["user_elastic"]["blade"],
                ),
            )
            self.connect("airfoils.rthick_master", "blade.interp_airfoils.rthick_master")
            self.connect("airfoils.ac", "blade.interp_airfoils.ac")
            self.connect("airfoils.coord_xy", "blade.interp_airfoils.coord_xy")
            self.connect("airfoils.aoa", "blade.interp_airfoils.aoa")
            self.connect("airfoils.cl", "blade.interp_airfoils.cl")
            self.connect("airfoils.cd", "blade.interp_airfoils.cd")
            self.connect("airfoils.cm", "blade.interp_airfoils.cm")

            self.connect("hub.radius", "blade.high_level_blade_props.hub_radius")
            self.connect("hub.cone", "blade.high_level_blade_props.cone")
            self.connect("configuration.rotor_diameter_user", "blade.high_level_blade_props.rotor_diameter_user")
            self.connect("configuration.n_blades", "blade.high_level_blade_props.n_blades")

        # Hub inputs
        if (
            modeling_options["flags"]["hub"]
            or modeling_options["flags"]["blade"]
            or modeling_options["user_elastic"]["hub"]
            or modeling_options["user_elastic"]["blade"]
        ):
            self.add_subsystem("hub", Hub(flags=modeling_options["flags"]))

        # Drivetrain inputs
        if (
            modeling_options["flags"]["drivetrain"]
            or modeling_options["flags"]["blade"]
            or modeling_options["user_elastic"]["drivetrain"]
            or modeling_options["user_elastic"]["blade"]
        ):
            self.add_subsystem(
                "drivetrain",
                Drivetrain(
                    flags=modeling_options["flags"], direct_drive=modeling_options["WISDEM"]["DriveSE"]["direct"]
                ),
            )

        # Generator inputs
        if modeling_options["flags"]["drivetrain"]:
            self.add_subsystem(
                "generator",
                Generator(
                    flags=modeling_options["flags"],
                    gentype=modeling_options["WISDEM"]["DriveSE"]["generator"]["type"],
                    n_pc=modeling_options["WISDEM"]["RotorSE"]["n_pc"],
                ),
            )

        if modeling_options["user_elastic"]["hub"] or modeling_options["user_elastic"]["drivetrain"]:
            # User wants to bypass all of DrivetrainSE with elastic summary properties
            drivese_ivc = om.IndepVarComp()
            drivese_ivc.add_output("hub_system_mass", val=0, units="kg")
            drivese_ivc.add_output("hub_system_I", val=np.zeros(6), units="kg*m**2")
            drivese_ivc.add_output("hub_system_cm", val=0.0, units="m")
            drivese_ivc.add_output("drivetrain_spring_constant", 0.0, units="N*m/rad")
            drivese_ivc.add_output("drivetrain_damping_coefficient", 0.0, units="N*m*s/rad")
            drivese_ivc.add_output("above_yaw_mass", 0.0, units="kg")
            drivese_ivc.add_output("above_yaw_cm", np.zeros(3), units="m")
            drivese_ivc.add_output("above_yaw_I", np.zeros(6), units="kg*m**2")
            drivese_ivc.add_output("above_yaw_I_TT", np.zeros(6), units="kg*m**2")
            drivese_ivc.add_output("yaw_mass", val=0.0, units="kg")
            drivese_ivc.add_output("rna_mass", 0.0, units="kg")
            drivese_ivc.add_output("rna_cm", np.zeros(3), units="m")
            drivese_ivc.add_output("rna_I_TT", np.zeros(6), units="kg*m**2")
            drivese_ivc.add_output("generator_rotor_I", val=np.zeros(3), units="kg*m**2")
            self.add_subsystem("drivese", drivese_ivc)

        # Tower inputs
        if modeling_options["flags"]["tower"]:
            tower_init_options = modeling_options["WISDEM"]["TowerSE"]
            n_height_tower = tower_init_options["n_height"]
            n_layers_tower = tower_init_options["n_layers"]
            ivc = self.add_subsystem("tower", om.IndepVarComp())
            ivc.add_output(
                "ref_axis",
                val=np.zeros((n_height_tower, 3)),
                units="m",
                desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
            )
            ivc.add_output(
                "diameter",
                val=np.zeros(n_height_tower),
                units="m",
                desc="1D array of the outer diameter values defined along the tower axis.",
            )
            ivc.add_output(
                "cd",
                val=np.zeros(n_height_tower),
                desc="1D array of the drag coefficients defined along the tower height.",
            )
            ivc.add_output(
                "layer_thickness",
                val=np.zeros((n_layers_tower, n_height_tower)),
                units="m",
                desc="2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the tower sections.",
            )
            ivc.add_output(
                "outfitting_factor",
                val=0.0,
                desc="Multiplier that accounts for secondary structure mass inside of tower",
            )
            ivc.add_output(
                "tower_mass_user",
                val=0.0,
                units="kg",
                desc="Override bottom-up calculation of total tower mass with this value",
            )
            ivc.add_discrete_output(
                "layer_name", val=[], desc="1D array of the names of the layers modeled in the tower structure."
            )
            ivc.add_discrete_output(
                "layer_mat",
                val=[],
                desc="1D array of the names of the materials of each layer modeled in the tower structure.",
            )
            ivc.add_output(
                "lumped_mass",
                val=np.zeros(n_height_tower),
                units="kg",
                desc="1D array of the lumped mass values defined along the tower axis.",
            )

        # Monopile inputs
        if modeling_options["flags"]["monopile"]:
            self.add_subsystem("monopile", Monopile(fixedbottomse_options=modeling_options["WISDEM"]["FixedBottomSE"]))

        # Jacket inputs
        if modeling_options["flags"]["jacket"]:
            self.add_subsystem("jacket", Jacket(fixedbottomse_options=modeling_options["WISDEM"]["FixedBottomSE"]))

        # Floating substructure inputs
        if modeling_options["flags"]["floating"]:
            self.add_subsystem(
                "floating",
                Floating(floating_init_options=modeling_options["floating"], opt_options=self.options["opt_options"]),
            )
            self.add_subsystem("mooring", Mooring(options=modeling_options))
            self.connect("floating.joints_xyz", "mooring.joints_xyz")

        # Balance of station inputs
        if modeling_options["flags"]["bos"]:
            bos_ivc = self.add_subsystem("bos", om.IndepVarComp())
            bos_ivc.add_output("plant_turbine_spacing", 7, desc="Distance between turbines in rotor diameters")
            bos_ivc.add_output("plant_row_spacing", 7, desc="Distance between turbine rows in rotor diameters")
            bos_ivc.add_output("commissioning_cost_kW", 44.0, units="USD/kW")
            bos_ivc.add_output("decommissioning_cost_kW", 58.0, units="USD/kW")
            bos_ivc.add_output("distance_to_substation", 50.0, units="km")
            bos_ivc.add_output("distance_to_interconnection", 5.0, units="km")
            if modeling_options["flags"]["offshore"]:
                bos_ivc.add_output("site_distance", 40.0, units="km")
                bos_ivc.add_output("distance_to_landfall", 40.0, units="km")
                bos_ivc.add_output("port_cost_per_month", 2e6, units="USD/mo")
                bos_ivc.add_output("site_auction_price", 100e6, units="USD")
                bos_ivc.add_output("site_assessment_cost", 50e6, units="USD")
                bos_ivc.add_output("boem_review_cost", 0.0, units="USD")
                bos_ivc.add_output("installation_plan_cost", 2.5e5, units="USD")
                bos_ivc.add_output("construction_plan_cost", 1e6, units="USD")
                bos_ivc.add_output("construction_insurance", 44.0, units="USD/kW")
                bos_ivc.add_output("construction_financing", 183.0, units="USD/kW")
                bos_ivc.add_output("contingency", 316.0, units="USD/kW")
            else:
                bos_ivc.add_output("interconnect_voltage", 130.0, units="kV")

        # Operation and maintenance inputs
        if modeling_options["flags"]["opex"]:
            opex_ivc = self.add_subsystem("opex", om.IndepVarComp())
            opex_ivc.add_discrete_output(
                "workday_start", 7, desc="Hour of the day where any work-related activities begin"
            )
            opex_ivc.add_discrete_output(
                "workday_end", 19, desc="Hour of the day where any work-related activities end"
            )
            opex_ivc.add_output(
                "equipment_dispatch_distance",
                50,
                units="km",
                desc="Distance, in km, that servicing equipment must travel daily to reach the wind farm",
            )
            opex_ivc.add_discrete_output(
                "n_ctv", 3, desc="Number of crew transfer vessels that should be made available to the wind farm."
            )
            opex_ivc.add_discrete_output(
                "n_hlv",
                1,
                desc="Number of heavy lift vessels that should be made available to the wind farm (fixed-bottom simulations only)",
            )
            opex_ivc.add_discrete_output(
                "n_tugboat",
                2,
                desc="Number of tugboat groups that should be available to the port to tow floating turbines to port and back",
            )
            opex_ivc.add_discrete_output(
                "port_workday_start",
                6,
                desc="Hour of the day where any work-related activities begin for port-side repairs",
            )
            opex_ivc.add_discrete_output(
                "port_workday_end",
                18,
                desc="Hour of the day where any work-related activities end for port-side repairs",
            )
            opex_ivc.add_discrete_output(
                "n_port_crews",
                2,
                desc="Number of port-side crews available to work on simultaneous repairs for any at-port turbine",
            )
            opex_ivc.add_discrete_output(
                "max_port_operations", 2, desc="Number of turbines that can be at port at once"
            )
            opex_ivc.add_output(
                "repair_port_distance",
                116,
                units="km",
                desc="Distance, in km, that tugboats must travel to reach the wind farm for tow-to-port repairs",
            )
            opex_ivc.add_discrete_output(
                "maintenance_start",
                None,
                desc="Date of first maintenance event to determine regular interval timing. Can be set to prior to the starting year to ensure staggered starts.",
            )
            opex_ivc.add_discrete_output(
                "non_operational_start",
                None,
                desc="Starting date, in MM/DD format, for an annual period where the site is inaccessible",
            )
            opex_ivc.add_discrete_output(
                "non_operational_end",
                None,
                desc="Ending date, in MM/DD format, for an annual period where the site is inaccessible",
            )
            opex_ivc.add_discrete_output(
                "reduced_speed_start",
                None,
                desc="Starting date, in MM/DD format, for an annual period where traveling speed is reduced",
            )
            opex_ivc.add_discrete_output(
                "reduced_speed_end",
                None,
                desc="Ending date, in MM/DD format, for an annual period where traveling speed is reduced",
            )
            opex_ivc.add_output(
                "reduced_speed",
                0,
                units="km/h",
                desc="Reduced speed applied to servicing equipment in the reduced speed period",
            )
            opex_ivc.add_discrete_output("random_seed", 42, desc="Random seed for the internal random generator")

        # Cost analysis inputs
        if modeling_options["flags"]["costs"]:
            costs_ivc = self.add_subsystem("costs", om.IndepVarComp())
            costs_ivc.add_discrete_output("turbine_number", val=0, desc="Number of turbines at plant")
            costs_ivc.add_output("offset_tcc_per_kW", val=0.0, units="USD/kW", desc="Offset to turbine capital cost")
            costs_ivc.add_output("bos_per_kW", val=0.0, units="USD/kW", desc="Balance of station/plant capital cost")
            costs_ivc.add_output(
                "opex_per_kW", val=0.0, units="USD/kW/yr", desc="Average annual operational expenditures of the turbine"
            )
            costs_ivc.add_output("wake_loss_factor", val=0.0, desc="The losses in AEP due to waked conditions")
            costs_ivc.add_output("fixed_charge_rate", val=0.0, desc="Fixed charge rate for coe calculation")
            costs_ivc.add_output("labor_rate", 0.0, units="USD/h")
            costs_ivc.add_output("painting_rate", 0.0, units="USD/m**2")

            costs_ivc.add_output("blade_mass_cost_coeff", units="USD/kg", val=14.6)
            costs_ivc.add_output("hub_mass_cost_coeff", units="USD/kg", val=3.9)
            costs_ivc.add_output("pitch_system_mass_cost_coeff", units="USD/kg", val=22.1)
            costs_ivc.add_output("spinner_mass_cost_coeff", units="USD/kg", val=11.1)
            costs_ivc.add_output("lss_mass_cost_coeff", units="USD/kg", val=11.9)
            costs_ivc.add_output("bearing_mass_cost_coeff", units="USD/kg", val=4.5)
            costs_ivc.add_output("gearbox_torque_cost", units="USD/kN/m", val=50.0)
            costs_ivc.add_output("hss_mass_cost_coeff", units="USD/kg", val=6.8)
            costs_ivc.add_output("generator_mass_cost_coeff", units="USD/kg", val=12.4)
            costs_ivc.add_output("bedplate_mass_cost_coeff", units="USD/kg", val=2.9)
            costs_ivc.add_output("yaw_mass_cost_coeff", units="USD/kg", val=8.3)
            costs_ivc.add_output("converter_mass_cost_coeff", units="USD/kg", val=18.8)
            costs_ivc.add_output("transformer_mass_cost_coeff", units="USD/kg", val=18.8)
            costs_ivc.add_output("hvac_mass_cost_coeff", units="USD/kg", val=124.0)
            costs_ivc.add_output("cover_mass_cost_coeff", units="USD/kg", val=5.7)
            costs_ivc.add_output("elec_connec_machine_rating_cost_coeff", units="USD/kW", val=41.85)
            costs_ivc.add_output("platforms_mass_cost_coeff", units="USD/kg", val=17.1)
            costs_ivc.add_output("tower_mass_cost_coeff", units="USD/kg", val=2.9)
            costs_ivc.add_output("controls_machine_rating_cost_coeff", units="USD/kW", val=21.15)
            costs_ivc.add_output("crane_cost", units="USD", val=12e3)
            costs_ivc.add_output("electricity_price", val=0.04, units="USD/kW/h")
            costs_ivc.add_output("reserve_margin_price", val=120.0, units="USD/kW/yr")
            costs_ivc.add_output("capacity_credit", val=1.0)
            costs_ivc.add_output("benchmark_price", val=0.071, units="USD/kW/h")

        # Assembly setup
        self.add_subsystem("high_level_tower_props", ComputeHighLevelTowerProperties(modeling_options=modeling_options))
        self.connect("configuration.hub_height_user", "high_level_tower_props.hub_height_user")
        if modeling_options["flags"]["blade"]:
            self.connect("blade.high_level_blade_props.rotor_diameter", "high_level_tower_props.rotor_diameter")
            self.add_subsystem("af_3d", Airfoil3DCorrection(rotorse_options=modeling_options["WISDEM"]["RotorSE"]))
            self.connect("airfoils.aoa", "af_3d.aoa")
            self.connect("airfoils.Re", "af_3d.Re")
            self.connect("blade.interp_airfoils.cl_interp", "af_3d.cl")
            self.connect("blade.interp_airfoils.cd_interp", "af_3d.cd")
            self.connect("blade.interp_airfoils.cm_interp", "af_3d.cm")
            self.connect("blade.high_level_blade_props.rotor_diameter", "af_3d.rotor_diameter")
            self.connect("blade.high_level_blade_props.r_blade", "af_3d.r_blade")
            self.connect("blade.interp_airfoils.rthick_interp", "af_3d.rthick")
            self.connect("blade.pa.chord_param", "af_3d.chord")
            self.connect("control.rated_TSR", "af_3d.rated_TSR")
            self.connect("control.maxOmega", "blade.compute_reynolds.maxOmega")
            self.connect("control.V_out", "blade.compute_reynolds.V_out")
        if modeling_options["flags"]["tower"]:
            self.connect("tower.ref_axis", "high_level_tower_props.tower_ref_axis_user")
            self.add_subsystem("tower_grid", Compute_Grid(n_height=n_height_tower))
            self.connect("high_level_tower_props.tower_ref_axis", "tower_grid.ref_axis")
        if modeling_options["flags"]["drivetrain"] or modeling_options["flags"]["blade"]:
            self.connect("drivetrain.distance_tt_hub", "high_level_tower_props.distance_tt_hub")


class Blade(om.Group):
    # Openmdao group with components with the blade data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")
        self.options.declare("user_elastic")

    def setup(self):
        # Options
        rotorse_options = self.options["rotorse_options"]
        opt_options = self.options["opt_options"]
        user_elastic = self.options["user_elastic"]

        # Optimization parameters initialized as indipendent variable component
        opt_var = om.IndepVarComp()
        opt_var.add_output(
            "s_opt_twist", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"])
        )
        opt_var.add_output(
            "s_opt_chord", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"])
        )
        opt_var.add_output(
            "twist_opt",
            val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]),
            units="rad",
        )
        opt_var.add_output(
            "chord_opt",
            units="m",
            val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"]),
        )
        opt_var.add_output("af_position", val=np.ones(rotorse_options["n_af_master"]))

        if not user_elastic:
            for i in range(rotorse_options["n_layers"]):
                opt_var.add_output(
                    "s_opt_layer_%d" % i,
                    val=np.ones(opt_options["design_variables"]["blade"]["n_opt_struct"][i]),
                )
                opt_var.add_output(
                    "layer_%d_opt" % i,
                    units="m",
                    val=np.ones(opt_options["design_variables"]["blade"]["n_opt_struct"][i]),
                )
        else:
            user_KI = om.IndepVarComp()
            n_span = rotorse_options["n_span"]
            user_KI.add_output(
                "K11",
                val=np.zeros(n_span),
                desc="Distribution of the K11 element of the stiffness matrix along blade span. K11 corresponds to the shear stiffness along the x axis (in a blade, x points to the trailing edge)",
                units="N",
            )
            user_KI.add_output(
                "K22",
                val=np.zeros(n_span),
                desc="Distribution of the K22 element of the stiffness matrix along blade span. K22 corresponds to the shear stiffness along the y axis (in a blade, y points to the suction side)",
                units="N",
            )
            user_KI.add_output(
                "K33",
                val=np.zeros(n_span),
                desc="Distribution of the K33 element of the stiffness matrix along blade span. K33 corresponds to the axial stiffness along the z axis (in a blade, z runs along the span and points to the tip)",
                units="N",
            )
            user_KI.add_output(
                "K44",
                val=np.zeros(n_span),
                desc="Distribution of the K44 element of the stiffness matrix along blade span. K44 corresponds to the bending stiffness around the x axis (in a blade, x points to the trailing edge and K44 corresponds to the flapwise stiffness)",
                units="N*m**2",
            )
            user_KI.add_output(
                "K55",
                val=np.zeros(n_span),
                desc="Distribution of the K55 element of the stiffness matrix along blade span. K55 corresponds to the bending stiffness around the y axis (in a blade, y points to the suction side and K55 corresponds to the edgewise stiffness)",
                units="N*m**2",
            )
            user_KI.add_output(
                "K66",
                val=np.zeros(n_span),
                desc="Distribution of K66 element of the stiffness matrix along blade span. K66 corresponds to the torsional stiffness along the z axis (in a blade, z runs along the span and points to the tip)",
                units="N*m**2",
            )
            user_KI.add_output(
                "K12",
                val=np.zeros(n_span),
                desc="Distribution of the K12 element of the stiffness matrix along blade span. K12 is a cross term between shear terms",
                units="N",
            )
            user_KI.add_output(
                "K13",
                val=np.zeros(n_span),
                desc="Distribution of the K13 element of the stiffness matrix along blade span. K13 is a cross term shear - axial",
                units="N",
            )
            user_KI.add_output(
                "K14",
                val=np.zeros(n_span),
                desc="Distribution of the K14 element of the stiffness matrix along blade span. K14 is a cross term shear - bending",
                units="N*m**2",
            )
            user_KI.add_output(
                "K15",
                val=np.zeros(n_span),
                desc="Distribution of the K15 element of the stiffness matrix along blade span. K15 is a cross term shear - bending",
                units="N*m**2",
            )
            user_KI.add_output(
                "K16",
                val=np.zeros(n_span),
                desc="Distribution of the K16 element of the stiffness matrix along blade span. K16 is a cross term shear - torsion",
                units="N*m**2",
            )
            user_KI.add_output(
                "K23",
                val=np.zeros(n_span),
                desc="Distribution of the K23 element of the stiffness matrix along blade span. K23 is a cross term shear - axial",
                units="N*m**2",
            )
            user_KI.add_output(
                "K24",
                val=np.zeros(n_span),
                desc="Distribution of the K24 element of the stiffness matrix along blade span. K24 is a cross term shear - bending",
                units="N/m**2",
            )
            user_KI.add_output(
                "K25",
                val=np.zeros(n_span),
                desc="Distribution of the K25 element of the stiffness matrix along blade span. K25 is a cross term shear - bending",
                units="N*m**2",
            )
            user_KI.add_output(
                "K26",
                val=np.zeros(n_span),
                desc="Distribution of the K26 element of the stiffness matrix along blade span. K26 is a cross term shear - torsion",
                units="N*m**2",
            )
            user_KI.add_output(
                "K34",
                val=np.zeros(n_span),
                desc="Distribution of the K34 element of the stiffness matrix along blade span. K34 is a cross term axial - bending",
                units="N*m**2",
            )
            user_KI.add_output(
                "K35",
                val=np.zeros(n_span),
                desc="Distribution of the K35 element of the stiffness matrix along blade span. K35 is a cross term axial - bending",
                units="N*m**2",
            )
            user_KI.add_output(
                "K36",
                val=np.zeros(n_span),
                desc="Distribution of the K36 element of the stiffness matrix along blade span. K36 is a cross term axial - torsion",
                units="N*m**2",
            )
            user_KI.add_output(
                "K45",
                val=np.zeros(n_span),
                desc="Distribution of the K45 element of the stiffness matrix along blade span. K45 is a cross term flapwise bending - edgewise bending",
                units="N*m**2",
            )
            user_KI.add_output(
                "K46",
                val=np.zeros(n_span),
                desc="Distribution of the K46 element of the stiffness matrix along blade span. K46 is a cross term flapwise bending - torsion",
                units="N*m**2",
            )
            user_KI.add_output(
                "K56",
                val=np.zeros(n_span),
                desc="Distribution of the K56 element of the stiffness matrix along blade span. K56 is a cross term edgewise bending - torsion",
                units="N*m**2",
            )

            # mass matrix inputs
            user_KI.add_output(
                "mass",
                val=np.zeros(n_span),
                desc="Mass per unit length along the beam, expressed in kilogram per meter",
                units="kg/m",
            )
            user_KI.add_output(
                "cm_x",
                val=np.zeros(n_span),
                desc="Distance between the reference axis and the center of mass along the x axis",
                units="m",
            )
            user_KI.add_output(
                "cm_y",
                val=np.zeros(n_span),
                desc="Distance between the reference axis and the center of mass along the y axis",
                units="m",
            )
            user_KI.add_output(
                "i_edge",
                val=np.zeros(n_span),
                desc="Edgewise mass moment of inertia per unit span (around y axis)",
                units="kg*m**2",
            )
            user_KI.add_output(
                "i_flap",
                val=np.zeros(n_span),
                desc="Flapwise mass moment of inertia per unit span (around x axis)",
                units="kg*m**2",
            )
            user_KI.add_output(
                "i_plr",
                val=np.zeros(n_span),
                desc="Polar moment of inertia per unit span (around z axis). Please note that for beam-like structures iplr must be equal to iedge plus iflap.",
                units="kg*m**2",
            )
            user_KI.add_output(
                "i_cp",
                val=np.zeros(n_span),
                desc="Sectional cross-product of inertia per unit span (cross term x y)",
                units="kg*m**2",
            )

            self.add_subsystem("user_KI", user_KI)

        self.add_subsystem("opt_var", opt_var)

        ivc = self.add_subsystem("blade_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "ref_axis",
            val=np.zeros((rotorse_options["n_span"], 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        # Import outer shape BEM
        self.add_subsystem("outer_shape", Blade_Outer_Shape(rotorse_options=rotorse_options))

        # Parametrize blade outer shape
        self.add_subsystem(
            "pa", ParametrizeBladeAero(rotorse_options=rotorse_options, opt_options=opt_options)
        )  # Parameterize aero (chord and twist)
        # Connections to blade aero parametrization
        self.connect("opt_var.s_opt_twist", "pa.s_opt_twist")
        self.connect("opt_var.s_opt_chord", "pa.s_opt_chord")
        self.connect("opt_var.twist_opt", "pa.twist_opt")
        self.connect("opt_var.chord_opt", "pa.chord_opt")
        self.connect("outer_shape.s", "pa.s")
        self.connect("outer_shape.chord", "pa.chord_original")
        self.connect("outer_shape.section_offset_y", "pa.section_offset_y")

        # Interpolate airfoil profiles and coordinates
        self.add_subsystem(
            "interp_airfoils",
            Blade_Interp_Airfoils(rotorse_options=rotorse_options),
        )

        # Connections from oute_shape_bem to interp_airfoils
        self.connect("outer_shape.s", "interp_airfoils.s")
        self.connect("outer_shape.rthick_yaml", "interp_airfoils.rthick_yaml")
        self.connect("pa.chord_param", ["interp_airfoils.chord", "compute_coord_xy_dim.chord"])
        self.connect(
            "pa.section_offset_y_param", ["interp_airfoils.section_offset_y", "compute_coord_xy_dim.section_offset_y"]
        )
        self.connect("opt_var.af_position", "interp_airfoils.af_position")

        self.add_subsystem("high_level_blade_props", ComputeHighLevelBladeProperties(rotorse_options=rotorse_options))
        self.connect("ref_axis", "high_level_blade_props.blade_ref_axis_user")
        self.connect("pa.chord_param", "high_level_blade_props.chord")

        # TODO : Compute Reynolds here
        self.add_subsystem("compute_reynolds", ComputeReynolds(n_span=rotorse_options["n_span"]))
        self.connect("high_level_blade_props.r_blade", "compute_reynolds.r_blade")
        self.connect("high_level_blade_props.rotor_diameter", "compute_reynolds.rotor_diameter")

        self.add_subsystem(
            "compute_coord_xy_dim",
            Compute_Coord_XY_Dim(rotorse_options=rotorse_options),
        )
        self.connect("pa.twist_param", "compute_coord_xy_dim.twist")
        self.connect("high_level_blade_props.blade_ref_axis", "compute_coord_xy_dim.ref_axis")

        self.connect("interp_airfoils.coord_xy_interp", "compute_coord_xy_dim.coord_xy_interp")

        # If the flag is true, generate the 3D x,y,z points of the outer blade shape
        if rotorse_options["lofted_output"] == True:
            self.add_subsystem(
                "blade_lofted",
                Blade_Lofted_Shape(rotorse_options=rotorse_options),
            )
            self.connect("compute_coord_xy_dim.coord_xy_dim_twisted", "blade_lofted.coord_xy_dim_twisted")
            self.connect("high_level_blade_props.blade_ref_axis", "blade_lofted.ref_axis")

        # Import blade internal structure data and remap composites on the outer blade shape
        # when not using the user-defined elastic properties only
        if not user_elastic:
            self.add_subsystem(
                "structure",
                Blade_Structure(rotorse_options=rotorse_options),
            )

            self.add_subsystem(
                "ps", ParametrizeBladeStruct(rotorse_options=rotorse_options, opt_options=opt_options)
            )  # Parameterize struct (spar caps ss and ps)

            # Connections to blade struct parametrization
            for i in range(rotorse_options["n_layers"]):
                self.connect("opt_var.layer_%d_opt" % i, "ps.layer_%d_opt" % i)
                self.connect("opt_var.s_opt_layer_%d" % i, "ps.s_opt_layer_%d" % i)

            self.connect("outer_shape.s", "ps.s")
            self.connect("compute_coord_xy_dim.coord_xy_dim", "structure.coord_xy_dim")
            self.connect("structure.layer_thickness", "ps.layer_thickness_original")

        # Fatigue specific parameters
        fat_var = om.IndepVarComp()
        fat_var.add_output("sparU_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("sparU_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("sparU_wohlerexp", val=1.0)
        fat_var.add_output("sparL_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("sparL_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("sparL_wohlerexp", val=1.0)
        fat_var.add_output("teU_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("teU_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("teU_wohlerexp", val=1.0)
        fat_var.add_output("teL_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("teL_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("teL_wohlerexp", val=1.0)
        self.add_subsystem("fatigue", fat_var)


class Blade_Outer_Shape(om.Group):
    # Openmdao group with the blade outer shape data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        n_af_master = rotorse_options["n_af_master"]
        self.n_span = n_span = rotorse_options["n_span"]

        ivc = self.add_subsystem("blade_outer_shape_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "af_position",
            val=np.zeros(n_af_master),
            desc="1D array of the non dimensional positions of the airfoils af_master defined along blade span.",
        )
        ivc.add_output(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        ivc.add_output(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        ivc.add_output(
            "twist",
            val=np.zeros(n_span),
            units="deg",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        ivc.add_output(
            "section_offset_y",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the distance in meters along the chordline from the reference axis to the leading edge. 0 means that the airfoil is pinned at the leading edge, a positive offset means that the leading edge is upstream of the reference axis in local chordline coordinates, and a negative offset that the leading edge aft of the reference axis.",
        )
        ivc.add_output(
            "section_offset_x",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the chordline normal distance in meters from the reference axis. 0 means that the reference axis lies on the airfoil chordline, a positive offset means that the chordline is shifted in the direction of the suction side relative to the reference axis, and a negative offset that the section is shifted in the direction of the pressure side of the airfoil.",
        )
        ivc.add_output(
            "rthick_yaml",
            val=np.zeros(n_span),
            desc="1D array of the relative thickness values defined along blade span.",
        )


class Blade_Interp_Airfoils(om.ExplicitComponent):
    # Openmdao component to interpolate airfoil coordinates and airfoil polars along the span of the blade for a predefined set of airfoils coming from component Airfoils.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_af_master = n_af_master = rotorse_options["n_af_master"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        self.af_master = rotorse_options["af_master"]  # Names of the airfoils adopted along blade span

        self.add_input(
            "af_position",
            val=np.zeros(n_af_master),
            desc="1D array of the non dimensional positions of the airfoils af_master defined along blade span.",
        )
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "section_offset_y",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the distance in meters along the chordline from the reference axis to the leading edge. 0 means that the airfoil is pinned at the leading edge, a positive offset means that the leading edge is upstream of the reference axis in local chordline coordinates, and a negative offset that the leading edge aft of the reference axis..",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )

        # Airfoil properties
        self.add_input("ac", val=np.zeros(n_af_master), desc="1D array of the aerodynamic centers of each airfoil.")
        self.add_input(
            "rthick_master", val=np.zeros(n_af_master), desc="1D array of the relative thicknesses of each airfoil."
        )
        self.add_input(
            "aoa",
            val=np.zeros(n_aoa),
            units="deg",
            desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "cl",
            val=np.zeros((n_af_master, n_aoa, n_Re)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_input(
            "cd",
            val=np.zeros((n_af_master, n_aoa, n_Re)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_input(
            "cm",
            val=np.zeros((n_af_master, n_aoa, n_Re)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )

        # Airfoil coordinates
        self.add_input(
            "coord_xy",
            val=np.zeros((n_af_master, n_xy, 2)),
            desc="3D array of the x and y airfoil coordinates of the n_af_master airfoils used along span.",
        )
        self.add_input(
            "rthick_yaml",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )

        # Polars and coordinates interpolated along span
        self.add_output(
            "rthick_interp",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )
        self.add_output(
            "ac_interp",
            val=np.zeros(n_span),
            desc="1D array of the aerodynamic center of the blade defined along span.",
        )
        self.add_output(
            "cl_interp",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_output(
            "cd_interp",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_output(
            "cm_interp",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_output(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.",
        )

    def compute(self, inputs, outputs):

        # Pchip does have an associated derivative method built-in:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.derivative.html#scipy.interpolate.PchipInterpolator.derivative
        spline = PchipInterpolator
        if max(inputs["rthick_yaml"]) < 1.0e-6:
            rthick_spline = spline(inputs["af_position"], inputs["rthick_master"])
            outputs["rthick_interp"] = rthick_spline(inputs["s"])
        else:
            outputs["rthick_interp"] = inputs["rthick_yaml"]

        ac_spline = spline(inputs["af_position"], inputs["ac"])
        outputs["ac_interp"] = ac_spline(inputs["s"])

        # Spanwise interpolation of the profile coordinates with a pchip
        # Is this unique an issue? Does it assume no two airfoils have the same relative thickness?
        rthick_unique, indices = np.unique(inputs["rthick_master"], return_index=True)
        profile_spline = spline(rthick_unique, inputs["coord_xy"][indices, :, :])
        coord_xy_interp = np.flip(profile_spline(np.flip(outputs["rthick_interp"])), axis=0)

        for i in range(self.n_span):
            # Correction to move the leading edge (min x point) to (0,0)
            af_le = coord_xy_interp[i, np.argmin(coord_xy_interp[i, :, 0]), :]
            coord_xy_interp[i, :, 0] -= af_le[0]
            coord_xy_interp[i, :, 1] -= af_le[1]
            c = max(coord_xy_interp[i, :, 0]) - min(coord_xy_interp[i, :, 0])
            coord_xy_interp[i, :, :] /= c
            # If the rel thickness is smaller than 0.4 apply a trailing ege smoothing step
            if outputs["rthick_interp"][i] < 0.4:
                coord_xy_interp[i, :, :] = trailing_edge_smoothing(coord_xy_interp[i, :, :])

        # Spanwise interpolation of the airfoil polars with a pchip
        cl_spline = spline(rthick_unique, inputs["cl"][indices, :, :])
        cl_interp = np.flip(cl_spline(np.flip(outputs["rthick_interp"])), axis=0)
        cd_spline = spline(rthick_unique, inputs["cd"][indices, :, :])
        cd_interp = np.flip(cd_spline(np.flip(outputs["rthick_interp"])), axis=0)
        cm_spline = spline(rthick_unique, inputs["cm"][indices, :, :])
        cm_interp = np.flip(cm_spline(np.flip(outputs["rthick_interp"])), axis=0)

        outputs["coord_xy_interp"] = coord_xy_interp
        outputs["cl_interp"] = cl_interp
        outputs["cd_interp"] = cd_interp
        outputs["cm_interp"] = cm_interp


class Compute_Coord_XY_Dim(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_input(
            "section_offset_y",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the distance in meters along the chordline from the reference axis to the leading edge. 0 means that the airfoil is pinned at the leading edge, a positive offset means that the leading edge is upstream of the reference axis in local chordline coordinates, and a negative offset that the leading edge aft of the reference axis.",
        )
        self.add_input(
            "twist",
            val=np.zeros(n_span),
            units="deg",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        self.add_input(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.",
        )
        self.add_input(
            "ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

        self.add_output(
            "coord_xy_dim",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_output(
            "coord_xy_dim_twisted",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_output("wetted_area", val=0.0, units="m**2", desc="The wetted (painted) surface area of the blade")
        self.add_output("projected_area", val=0.0, units="m**2", desc="The projected surface area of the blade")

    def compute(self, inputs, outputs):
        section_offset_y = inputs["section_offset_y"]
        chord = inputs["chord"]
        twist = inputs["twist"]
        coord_xy_interp = inputs["coord_xy_interp"]

        coord_xy_dim = copy.copy(coord_xy_interp)
        coord_xy_dim[:, :, 0] -= section_offset_y[:, np.newaxis] / chord[:, np.newaxis]
        coord_xy_dim = coord_xy_dim * chord[:, np.newaxis, np.newaxis]

        outputs["coord_xy_dim"] = coord_xy_dim

        coord_xy_twist = copy.copy(coord_xy_interp)
        x = coord_xy_dim[:, :, 0]
        y = coord_xy_dim[:, :, 1]
        coord_xy_twist[:, :, 0] = x * np.cos(np.deg2rad(twist[:, np.newaxis])) - y * np.sin(
            np.deg2rad(twist[:, np.newaxis])
        )
        coord_xy_twist[:, :, 1] = y * np.cos(np.deg2rad(twist[:, np.newaxis])) + x * np.sin(
            np.deg2rad(twist[:, np.newaxis])
        )
        outputs["coord_xy_dim_twisted"] = coord_xy_twist

        # Integrate along span for surface area
        wetted_chord = coord_xy_dim[:, :, 1].max(axis=1) - coord_xy_dim[:, :, 1].min(axis=1)
        projected_chord = coord_xy_twist[:, :, 1].max(axis=1) - coord_xy_twist[:, :, 1].min(axis=1)
        try:
            # Numpy v1/2 clash
            outputs["wetted_area"] = np.trapezoid(wetted_chord, inputs["ref_axis"][:, 2])
            outputs["projected_area"] = np.trapezoid(projected_chord, inputs["ref_axis"][:, 2])
        except AttributeError:
            outputs["wetted_area"] = np.trapz(wetted_chord, inputs["ref_axis"][:, 2])
            outputs["projected_area"] = np.trapz(projected_chord, inputs["ref_axis"][:, 2])


class Blade_Lofted_Shape(om.ExplicitComponent):
    # Openmdao component to generate the x, y, z coordinates of the points describing the blade outer shape.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        self.add_input(
            "ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

        self.add_input(
            "coord_xy_dim_twisted",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_output(
            "3D_shape",
            val=np.zeros((n_span * n_xy, 4)),
            units="m",
            desc="4D array of the s, and x, y, and z coordinates of the points describing the outer shape of the blade. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

    def compute(self, inputs, outputs):
        k = 0
        for i in range(self.n_span):
            for j in range(self.n_xy):
                outputs["3D_shape"][k, :] = np.array(
                    [k, inputs["coord_xy_dim_twisted"][i, j, 1], inputs["coord_xy_dim_twisted"][i, j, 0], 0.0]
                ) + np.hstack([0, inputs["ref_axis"][i, :]])
                k = k + 1

            # Debug output
            np.savetxt(
                "3d_xyz_blade_lofted.dat",
                outputs["3D_shape"],
                header="\t point number [-]\t\t\t\t x [m] \t\t\t\t\t y [m]  \t\t\t\t z [m] \t\t\t\t The coordinate system follows the BeamDyn one.",
            )


class Blade_Structure(om.Group):
    # Openmdao group with the blade internal structure data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_webs = n_webs = rotorse_options["n_webs"]
        self.n_layers = n_layers = rotorse_options["n_layers"]

        ivc = self.add_subsystem("blade_struct_indep_vars", om.IndepVarComp(), promotes=["*"])

        ivc.add_output(
            "web_start_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "web_offset",
            val=np.zeros((n_webs, n_span)),
            units="m",
            desc="2D array of the dimensional offset of a web with respect to the reference axis. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        ivc.add_discrete_output(
            "build_web",
            val=[False] * n_webs,
            desc="1D array of boolean values indicating whether to build a web from offset and rotation.",
        )
        ivc.add_output(
            "web_rotation",
            val=np.zeros(n_webs),
            units="deg",
            desc="1D array of the dimensional rotation of a web with respect to the reference axis. The dimension represents each web.",
        )
        ivc.add_output(
            "web_end_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )

        ivc.add_output(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents span.",
        )
        ivc.add_output(
            "layer_start_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the start_nd_arc of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        ivc.add_output(
            "layer_end_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the end_nd_arc of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        ivc.add_discrete_output(
            "build_layer",
            val=np.zeros(n_layers),
            desc="1D array of boolean values indicating how to build a layer.",
        )
        ivc.add_discrete_output(
            "index_layer_start", val=np.zeros(n_layers), desc="Index used to fix a layer to another"
        )
        ivc.add_discrete_output("index_layer_end", val=np.zeros(n_layers), desc="Index used to fix a layer to another")
        ivc.add_output(
            "layer_width",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the width of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        ivc.add_output(
            "layer_offset",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the dimensional offset of a layer with respect to the reference axis. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_rotation",
            val=np.zeros(n_layers),
            units="deg",
            desc="1D array of the dimensional rotation of a layer with respect to the reference axis. The dimension represents each layer.",
        )
        ivc.add_output(
            "layer_fiber_orientation",
            val=np.zeros((n_layers, n_span)),
            units="deg",
            desc="2D array of the orientation of the layers of the blade structure. The first dimension represents each layer, the second dimension represents span.",
        )
        ivc.add_output(
            "joint_position",
            val=0.0,
            desc="Spanwise position of a blade segmentation joint.",
        )
        ivc.add_output("joint_mass", val=0.0, units="kg", desc="Mass of the blade spanwise joint.")
        ivc.add_output("joint_cost", val=0.0, units="USD", desc="Cost of the joint.")
        ivc.add_output("d_f", val=0.0, units="m", desc="Diameter of the blade root fastener.")
        ivc.add_output("sigma_max", val=0.0, units="Pa", desc="Max stress on each blade root bolt.")

        self.add_subsystem(
            "compute_structure",
            Compute_Blade_Structure(rotorse_options=rotorse_options),
            promotes=["*"],
        )


class Compute_Blade_Structure(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_webs = n_webs = rotorse_options["n_webs"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        self.add_input(
            "web_start_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "web_end_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "web_offset",
            val=np.zeros((n_webs, n_span)),
            units="m",
            desc="2D array of the dimensional offset of a web with respect to the reference axis. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "web_rotation",
            val=np.zeros(n_webs),
            units="deg",
            desc="1D array of the dimensional rotation of a web with respect to the reference axis. The dimension represents each web.",
        )
        self.add_discrete_input(
            "build_web",
            val=[False] * n_webs,
            desc="1D array of boolean values indicating whether to build a web from offset and rotation.",
        )
        self.add_input(
            "layer_start_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the start_nd_arc of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        self.add_input(
            "layer_end_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the end_nd_arc of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        self.add_input(
            "layer_width",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the width of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        self.add_input(
            "layer_offset",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the dimensional offset of a layer with respect to the reference axis. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_discrete_input(
            "build_layer",
            val=np.zeros(n_layers),
            desc="1D array of boolean values indicating how to build a layer. 0 - start and end are set constant, 1 - from offset and rotation suction side, 2 - from offset and rotation pressure side, 3 - LE and width, 4 - TE SS width, 5 - TE PS width, 6 - locked to another layer. Negative values place the layer on webs (-1 first web, -2 second web, etc.).",
        )
        self.add_discrete_input(
            "index_layer_start", val=np.zeros(n_layers), desc="Index used to fix a layer to another"
        )
        self.add_discrete_input("index_layer_end", val=np.zeros(n_layers), desc="Index used to fix a layer to another")
        self.add_input(
            "layer_rotation",
            val=np.zeros(n_layers),
            units="deg",
            desc="1D array of the dimensional rotation of a layer with respect to the reference axis. The dimension represents each layer.",
        )
        self.add_input(
            "coord_xy_dim",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )

        # Outputs
        self.add_output(
            "web_start_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "web_end_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "layer_start_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the start_nd_arc of the layers. The first dimension represents each layer, the second dimension represents span.",
        )
        self.add_output(
            "layer_end_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the end_nd_arc of the layers. The first dimension represents each layer, the second dimension represents span.",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Initialize arrays
        web_start_nd = np.zeros((self.n_webs, self.n_span))
        web_end_nd = np.zeros((self.n_webs, self.n_span))
        layer_start_nd = np.zeros((self.n_layers, self.n_span))
        layer_end_nd = np.zeros((self.n_layers, self.n_span))
        import matplotlib.pyplot as plt

        # Compute the start and end points of the webs
        for j in range(self.n_webs):
            for i in range(self.n_span):

                xy_coord_i = inputs["coord_xy_dim"][i, :, :]
                idx_le = np.argmin(xy_coord_i[:, 0])
                theta = np.deg2rad(inputs["web_rotation"][j])
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                xy_coord_rotated = xy_coord_i @ rotation_matrix.T
                web_offset = inputs["web_offset"][j, i]
                idx_web_ss = np.argmin(abs(xy_coord_rotated[:idx_le, 0] - web_offset))
                idx_web_ps = np.argmin(abs(xy_coord_rotated[idx_le:, 0] - web_offset)) + idx_le
                xy_arc_i = arc_length(xy_coord_i)
                web_start_nd[j, i] = xy_arc_i[idx_web_ss] / xy_arc_i[-1]
                web_end_nd[j, i] = xy_arc_i[idx_web_ps] / xy_arc_i[-1]

        if np.any(web_start_nd < 0):
            logger.debug("Web start points must be larger than 0. Setting the value to 0.")
            web_start_nd[web_start_nd < 0] = 0
        if np.any(web_start_nd > 1):
            logger.debug("Web start points must be smaller than 1. Setting the value to 1.")
            web_start_nd[web_start_nd > 1] = 1
        if np.any(web_end_nd < 0):
            logger.debug("Web end points must be larger than 0. Setting the value to 0.")
            web_end_nd[web_end_nd < 0] = 0
        if np.any(web_end_nd > 1):
            logger.debug("Web end points must be smaller than 1. Setting the value to 1.")
            web_end_nd[web_end_nd > 1] = 1

        outputs["web_start_nd"] = web_start_nd
        outputs["web_end_nd"] = web_end_nd

        # Compute the start and end points of the layers
        for j in range(self.n_layers):
            if discrete_inputs["build_layer"][j] == 0:
                layer_start_nd[j, :] = inputs["layer_start_nd_yaml"][j, :]
                layer_end_nd[j, :] = inputs["layer_end_nd_yaml"][j, :]

            elif discrete_inputs["build_layer"][j] == 1 or discrete_inputs["build_layer"][j] == 2:
                for i in range(self.n_span):

                    xy_coord_i = inputs["coord_xy_dim"][i, :, :]
                    idx_le = np.argmin(xy_coord_i[:, 0])
                    theta = np.deg2rad(inputs["layer_rotation"][j])
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    xy_coord_rotated = xy_coord_i @ rotation_matrix.T
                    layer_offset = inputs["layer_offset"][j, i]
                    if discrete_inputs["build_layer"][j] == 1:  # suction side
                        idx_layer = np.argmin(abs(xy_coord_rotated[:idx_le, 0] - layer_offset))
                    else:  # pressure side
                        idx_layer = np.argmin(abs(xy_coord_rotated[idx_le:, 0] - layer_offset)) + idx_le
                    xy_arc_i = arc_length(xy_coord_i)
                    arc_L_i = xy_arc_i[-1]
                    width_i = inputs["layer_width"][j, i]

                    layer_start_nd[j, i] = (xy_arc_i[idx_layer] - 0.5 * width_i) / arc_L_i
                    layer_end_nd[j, i] = (xy_arc_i[idx_layer] + 0.5 * width_i) / arc_L_i

            elif discrete_inputs["build_layer"][j] == 3:
                for i in range(self.n_span):
                    xy_coord_i = inputs["coord_xy_dim"][i, :, :]
                    xy_arc_i = arc_length(xy_coord_i)
                    arc_L_i = xy_arc_i[-1]
                    idx_le = np.argmin(xy_coord_i[:, 0])
                    LE_loc_i = xy_arc_i[idx_le]
                    width_i = inputs["layer_width"][j, i]

                    layer_start_nd[j, i] = (LE_loc_i - 0.5 * width_i) / arc_L_i
                    layer_end_nd[j, i] = (LE_loc_i + 0.5 * width_i) / arc_L_i

            elif discrete_inputs["build_layer"][j] == 4:
                for i in range(self.n_span):
                    xy_coord_i = inputs["coord_xy_dim"][i, :, :]
                    xy_arc_i = arc_length(xy_coord_i)
                    arc_L_i = xy_arc_i[-1]
                    width_i = inputs["layer_width"][j, i]

                    layer_start_nd[j, i] = 0.0
                    layer_end_nd[j, i] = width_i / arc_L_i

            elif discrete_inputs["build_layer"][j] == 5:
                for i in range(self.n_span):
                    xy_coord_i = inputs["coord_xy_dim"][i, :, :]
                    xy_arc_i = arc_length(xy_coord_i)
                    arc_L_i = xy_arc_i[-1]
                    width_i = inputs["layer_width"][j, i]

                    layer_start_nd[j, i] = 1.0 - width_i / arc_L_i
                    layer_end_nd[j, i] = 1.0

            elif discrete_inputs["build_layer"][j] == 6:
                # start a layer from the end of another layer, and end where the other starts
                layer_start_nd[j, :] = layer_end_nd[int(discrete_inputs["index_layer_start"][j]), :]
                layer_end_nd[j, :] = layer_start_nd[int(discrete_inputs["index_layer_end"][j]), :]

        if np.any(layer_start_nd < 0):
            logger.debug("Layer start points must be larger than 0. Setting the value to 0.")
            layer_start_nd[layer_start_nd < 0] = 0
        if np.any(layer_start_nd > 1):
            logger.debug("Layer start points must be smaller than 1. Setting the value to 1.")
            layer_start_nd[layer_start_nd > 1] = 1
        if np.any(layer_end_nd < 0):
            logger.debug("Layer end points must be larger than 0. Setting the value to 0.")
            layer_end_nd[layer_end_nd < 0] = 0
        if np.any(layer_end_nd > 1):
            logger.debug("Layer end points must be smaller than 1. Setting the value to 1.")
            layer_end_nd[layer_end_nd > 1] = 1

        outputs["layer_start_nd"] = layer_start_nd
        outputs["layer_end_nd"] = layer_end_nd


class Hub(om.Group):
    # Openmdao group with the hub data coming from the input yaml file.
    def initialize(self):
        self.options.declare("flags")

    def setup(self):
        ivc = self.add_subsystem("hub_indep_vars", om.IndepVarComp(), promotes=["*"])

        ivc.add_output(
            "cone",
            val=0.0,
            units="deg",
            desc="Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.",
        )
        # ivc.add_output('drag_coeff',   val=0.0,                desc='Drag coefficient to estimate the aerodynamic forces generated by the hub.') # GB: this doesn't connect to anything
        ivc.add_output("diameter", val=0.0, units="m")

        exec_comp = om.ExecComp(
            "radius = 0.5 * diameter",
            units="m",
            radius={
                "desc": "Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line."
            },
        )
        self.add_subsystem("compute_radius", exec_comp, promotes=["*"])

        if self.options["flags"]["hub"]:
            ivc.add_output("flange_t2shell_t", val=0.0)
            ivc.add_output("flange_OD2hub_D", val=0.0)
            ivc.add_output("flange_ID2flange_OD", val=0.0)
            ivc.add_output("hub_stress_concentration", val=0.0)
            ivc.add_discrete_output("n_front_brackets", val=0)
            ivc.add_discrete_output("n_rear_brackets", val=0)
            ivc.add_output("clearance_hub_spinner", val=0.0, units="m")
            ivc.add_output("spin_hole_incr", val=0.0)
            ivc.add_output("pitch_system_scaling_factor", val=0.54)
            ivc.add_output("hub_in2out_circ", val=1.2)
            ivc.add_discrete_output("hub_material", val="steel")
            ivc.add_discrete_output("spinner_material", val="carbon")
            ivc.add_output("hub_shell_mass_user", val=0.0, units="kg")
            ivc.add_output("spinner_mass_user", val=0.0, units="kg")
            ivc.add_output("pitch_system_mass_user", val=0.0, units="kg")
            ivc.add_output("hub_system_mass_user", val=0, units="kg")
            ivc.add_output("hub_system_I_user", val=np.zeros(6), units="kg*m**2")
            ivc.add_output("hub_system_cm_user", val=0.0, units="m")


class Drivetrain(om.Group):
    # Openmdao group with the hub data coming from the input yaml file.
    def initialize(self):
        self.options.declare("flags")
        self.options.declare("direct_drive")

    def setup(self):
        ivc = self.add_subsystem("nac_indep_vars", om.IndepVarComp(), promotes=["*"])

        # Common direct and geared
        ivc.add_output(
            "uptilt", val=0.0, units="deg", desc="Shaft uptilt angle. A standard machine has positive values."
        )
        ivc.add_output(
            "distance_tt_hub", val=0.0, units="m", desc="Vertical distance from tower top plane to hub flange"
        )
        ivc.add_output("overhang", val=0.0, units="m", desc="Horizontal distance from tower top edge to hub flange")
        ivc.add_output("gearbox_efficiency", val=1.0, desc="Efficiency of the gearbox. Set to 1.0 for direct-drive")
        ivc.add_output("gearbox_mass_user", val=0.0, units="kg", desc="User override of gearbox mass.")
        ivc.add_output(
            "gearbox_radius_user",
            val=0.0,
            units="m",
            desc="User override of gearbox radius (only used if gearbox_mass_user is > 0).",
        )
        ivc.add_output(
            "gearbox_length_user",
            val=0.0,
            units="m",
            desc="User override of gearbox length (only used if gearbox_mass_user is > 0).",
        )
        ivc.add_output("gear_ratio", val=1.0, desc="Total gear ratio of drivetrain (use 1.0 for direct)")

        if self.options["flags"]["drivetrain"]:
            ivc.add_output(
                "distance_hub_mb", val=0.0, units="m", desc="Distance from hub flange to first main bearing along shaft"
            )
            ivc.add_output(
                "distance_mb_mb", val=0.0, units="m", desc="Distance from first to second main bearing along shaft"
            )
            ivc.add_output("lss_diameter", val=np.zeros(2), units="m", desc="Diameter of low speed shaft")
            ivc.add_output("lss_wall_thickness", val=np.zeros(2), units="m", desc="Thickness of low speed shaft")
            ivc.add_output("lss_mass_user", val=0.0, units="kg", desc="User override of low speed shaft mass.")
            ivc.add_output("damping_ratio", val=0.0, desc="Damping ratio for the drivetrain system")
            ivc.add_output(
                "brake_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of brake mass with this value",
            )
            ivc.add_output(
                "hvac_mass_coeff",
                val=0.025,
                units="kg/kW/m",
                desc="Regression-based scaling coefficient on machine rating to get HVAC system mass",
            )
            ivc.add_output(
                "converter_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of converter mass with this value",
            )
            ivc.add_output(
                "transformer_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of transformer mass with this value",
            )
            ivc.add_output(
                "platform_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of platform mass with this value",
            )
            ivc.add_output(
                "cover_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of cover mass with this value",
            )
            ivc.add_output(
                "mb1_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of first main bearing mass with this value",
            )
            ivc.add_output(
                "mb2_mass_user",
                val=0.0,
                units="kg",
                desc="Override regular regression-based calculation of second main bearing mass with this value",
            )
            ivc.add_discrete_output("mb1Type", val="CARB", desc="Type of main bearing: CARB / CRB / SRB / TRB")
            ivc.add_discrete_output("mb2Type", val="SRB", desc="Type of main bearing: CARB / CRB / SRB / TRB")
            ivc.add_discrete_output(
                "uptower", val=True, desc="If power electronics are located uptower (True) or at tower base (False)"
            )
            ivc.add_discrete_output(
                "lss_material", val="steel", desc="Material name identifier for the low speed shaft"
            )
            ivc.add_discrete_output(
                "hss_material", val="steel", desc="Material name identifier for the high speed shaft"
            )
            ivc.add_discrete_output("bedplate_material", val="steel", desc="Material name identifier for the bedplate")
            ivc.add_output(
                "bedplate_mass_user",
                val=0.0,
                units="kg",
                desc="Override bottom-up calculation of bedplate mass with this value",
            )

            if self.options["direct_drive"]:
                # Direct only
                ivc.add_output(
                    "nose_diameter",
                    val=np.zeros(2),
                    units="m",
                    desc="Diameter of nose (also called turret or spindle)",
                )
                ivc.add_output(
                    "nose_wall_thickness",
                    val=np.zeros(2),
                    units="m",
                    desc="Thickness of nose (also called turret or spindle)",
                )
                ivc.add_output(
                    "bedplate_wall_thickness",
                    val=np.zeros(4),
                    units="m",
                    desc="Thickness of hollow elliptical bedplate",
                )
            else:
                # Geared only
                ivc.add_output("hss_length", val=0.0, units="m", desc="Length of high speed shaft")
                ivc.add_output("hss_diameter", val=np.zeros(2), units="m", desc="Diameter of high speed shaft")
                ivc.add_output(
                    "hss_wall_thickness", val=np.zeros(2), units="m", desc="Wall thickness of high speed shaft"
                )
                ivc.add_output("hss_mass_user", val=0.0, units="kg", desc="User override of high speed shaft mass.")
                ivc.add_output("bedplate_flange_width", val=0.0, units="m", desc="Bedplate I-beam flange width")
                ivc.add_output("bedplate_flange_thickness", val=0.0, units="m", desc="Bedplate I-beam flange thickness")
                ivc.add_output("bedplate_web_thickness", val=0.0, units="m", desc="Bedplate I-beam web thickness")
                ivc.add_discrete_output(
                    "gear_configuration",
                    val="eep",
                    desc="3-letter string of Es or Ps to denote epicyclic or parallel gear configuration",
                )
                ivc.add_discrete_output(
                    "planet_numbers", val=[3, 3, 0], desc="Number of planets for epicyclic stages (use 0 for parallel)"
                )

            ivc.add_output("yaw_system_mass_user", 0.0, units="kg")
            ivc.add_output("above_yaw_mass_user", 0.0, units="kg")
            ivc.add_output("above_yaw_cm_user", np.zeros(3), units="m")
            ivc.add_output("above_yaw_I_user", np.zeros(6), units="kg*m**2")
            # ivc.add_output("above_yaw_I_TT_user", np.zeros(6), units="kg*m**2")
            ivc.add_output("drivetrain_spring_constant_user", val=0, units="N*m/rad")
            ivc.add_output("drivetrain_damping_coefficient_user", val=0, units="N*m*s/rad")


class Generator(om.Group):
    # Openmdao group with the hub data coming from the input yaml file.
    def initialize(self):
        self.options.declare("flags")
        self.options.declare("gentype")
        self.options.declare("n_pc")

    def setup(self):
        ivc = self.add_subsystem("gen_indep_vars", om.IndepVarComp(), promotes=["*"])

        # Generator inputs
        ivc.add_output("L_generator", val=0.0, units="m", desc="Generator length along shaft")
        ivc.add_output("generator_mass_user", val=0.0, units="kg")
        ivc.add_output("generator_rotor_I_user", val=np.zeros(3), units="kg*m**2")

        if not self.options["flags"]["generator"]:
            # If using simple (regression) generator scaling, this is an optional input to override default values
            n_pc = self.options["n_pc"]
            ivc.add_output("generator_radius_user", val=0.0, units="m")
            ivc.add_output("generator_efficiency_user", val=np.zeros((n_pc, 2)))
        else:
            ivc.add_output("B_r", val=1.2, units="T")
            ivc.add_output("P_Fe0e", val=1.0, units="W/kg")
            ivc.add_output("P_Fe0h", val=4.0, units="W/kg")
            ivc.add_output("S_N", val=-0.002)
            ivc.add_output("alpha_p", val=0.5 * np.pi * 0.7)
            ivc.add_output("b_r_tau_r", val=0.45)
            ivc.add_output("b_ro", val=0.004, units="m")
            ivc.add_output("b_s_tau_s", val=0.45)
            ivc.add_output("b_so", val=0.004, units="m")
            ivc.add_output("cofi", val=0.85)
            ivc.add_output("freq", val=60, units="Hz")
            ivc.add_output("h_i", val=0.001, units="m")
            ivc.add_output("h_sy0", val=0.0)
            ivc.add_output("h_w", val=0.005, units="m")
            ivc.add_output("k_fes", val=0.9)
            ivc.add_output("k_fillr", val=0.7)
            ivc.add_output("k_fills", val=0.65)
            ivc.add_output("k_s", val=0.2)
            ivc.add_discrete_output("m", val=3)
            ivc.add_output("mu_0", val=np.pi * 4e-7, units="m*kg/s**2/A**2")
            ivc.add_output("mu_r", val=1.06, units="m*kg/s**2/A**2")
            ivc.add_output("p", val=3.0)
            ivc.add_output("phi", val=90, units="deg")
            ivc.add_discrete_output("q1", val=6)
            ivc.add_discrete_output("q2", val=4)
            ivc.add_output("ratio_mw2pp", val=0.7)
            ivc.add_output("resist_Cu", val=1.8e-8 * 1.4, units="ohm/m")
            ivc.add_output("sigma", val=40e3, units="Pa")
            ivc.add_output("y_tau_p", val=1.0)
            ivc.add_output("y_tau_pr", val=10.0 / 12)

            ivc.add_output("I_0", val=0.0, units="A")
            ivc.add_output("d_r", val=0.0, units="m")
            ivc.add_output("h_m", val=0.0, units="m")
            ivc.add_output("h_0", val=0.0, units="m")
            ivc.add_output("h_s", val=0.0, units="m")
            ivc.add_output("len_s", val=0.0, units="m")
            ivc.add_output("n_r", val=0.0)
            ivc.add_output("rad_ag", val=0.0, units="m")
            ivc.add_output("t_wr", val=0.0, units="m")

            ivc.add_output("n_s", val=0.0)
            ivc.add_output("b_st", val=0.0, units="m")
            ivc.add_output("d_s", val=0.0, units="m")
            ivc.add_output("t_ws", val=0.0, units="m")

            ivc.add_output("rho_Copper", val=0.0, units="kg*m**-3")
            ivc.add_output("rho_Fe", val=0.0, units="kg*m**-3")
            ivc.add_output("rho_Fes", val=0.0, units="kg*m**-3")
            ivc.add_output("rho_PM", val=0.0, units="kg*m**-3")

            ivc.add_output("C_Cu", val=0.0, units="USD/kg")
            ivc.add_output("C_Fe", val=0.0, units="USD/kg")
            ivc.add_output("C_Fes", val=0.0, units="USD/kg")
            ivc.add_output("C_PM", val=0.0, units="USD/kg")

            if self.options["gentype"] in ["pmsg_outer"]:
                ivc.add_output("N_c", 0.0)
                ivc.add_output("b", 0.0)
                ivc.add_output("c", 0.0)
                ivc.add_output("E_p", 0.0, units="V")
                ivc.add_output("h_yr", val=0.0, units="m")
                ivc.add_output("h_ys", val=0.0, units="m")
                ivc.add_output("h_sr", 0.0, units="m", desc="Structural Mass")
                ivc.add_output("h_ss", 0.0, units="m")
                ivc.add_output("t_r", 0.0, units="m")
                ivc.add_output("t_s", 0.0, units="m")

                ivc.add_output("u_allow_pcent", 0.0)
                ivc.add_output("y_allow_pcent", 0.0)
                ivc.add_output("z_allow_deg", 0.0, units="deg")
                ivc.add_output("B_tmax", 0.0, units="T")

            if self.options["gentype"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                ivc.add_output("tau_p", val=0.0, units="m")
                ivc.add_output("h_ys", val=0.0, units="m")
                ivc.add_output("h_yr", val=0.0, units="m")
                ivc.add_output("b_arm", val=0.0, units="m")

            elif self.options["gentype"] in ["scig", "dfig"]:
                ivc.add_output("B_symax", val=0.0, units="T")
                ivc.add_output("S_Nmax", val=-0.2)


class Compute_Grid(om.ExplicitComponent):
    """
    Compute the non-dimensional grid for a tower or monopile.

    Using the dimensional `ref_axis` array, this component computes the
    non-dimensional grid, height (vertical distance) and length (curve distance)
    of a tower or monopile.
    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]

        self.add_input(
            "ref_axis",
            val=np.zeros((n_height, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
        )

        self.add_output(
            "s",
            val=np.zeros(n_height),
            desc="1D array of the non-dimensional grid defined along the tower axis (0-tower base, 1-tower top)",
        )
        self.add_output("height", val=0.0, units="m", desc="Scalar of the tower height computed along the z axis.")
        self.add_output(
            "length",
            val=0.0,
            units="m",
            desc="Scalar of the tower length computed along its curved axis. A standard straight tower will be as high as long.",
        )
        self.add_output(
            "foundation_height",
            val=0.0,
            units="m",
            desc="Foundation height in respect to the ground level.",
        )

        # Declare all partial derivatives.
        self.declare_partials("height", "ref_axis")
        self.declare_partials("length", "ref_axis")
        self.declare_partials("s", "ref_axis")
        self.declare_partials("foundation_height", "ref_axis")

    def compute(self, inputs, outputs):
        # Compute tower height and tower length (a straight tower will be high as long)
        outputs["foundation_height"] = inputs["ref_axis"][0, 2]
        outputs["height"] = inputs["ref_axis"][-1, 2] - inputs["ref_axis"][0, 2]
        myarc = arc_length(inputs["ref_axis"])
        outputs["length"] = myarc[-1]

        if myarc[-1] > 0.0:
            outputs["s"] = myarc / myarc[-1]

    def compute_partials(self, inputs, partials):
        n_height = self.options["n_height"]
        partials["height", "ref_axis"] = np.zeros((1, n_height * 3))
        partials["height", "ref_axis"][0, -1] = 1.0
        partials["height", "ref_axis"][0, 2] = -1.0
        partials["foundation_height", "ref_axis"] = np.zeros((1, n_height * 3))
        partials["foundation_height", "ref_axis"][0, 2] = 1.0
        arc_distances, d_arc_distances_d_points = arc_length_deriv(inputs["ref_axis"])

        # The length is based on only the final point in the arc,
        # but that final point has sensitivity to all ref_axis points
        partials["length", "ref_axis"] = d_arc_distances_d_points[-1, :]

        # Do quotient rule to get the non-dimensional grid derivatives
        low_d_high = arc_distances[-1] * d_arc_distances_d_points
        high_d_low = np.outer(arc_distances, d_arc_distances_d_points[-1, :])
        partials["s", "ref_axis"] = (low_d_high - high_d_low) / arc_distances[-1] ** 2


class Monopile(om.Group):
    def initialize(self):
        self.options.declare("fixedbottomse_options")

    def setup(self):
        fixedbottomse_options = self.options["fixedbottomse_options"]
        n_height = fixedbottomse_options["n_height"]
        n_layers = fixedbottomse_options["n_layers"]

        ivc = self.add_subsystem("monopile_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "diameter",
            val=np.zeros(n_height),
            units="m",
            desc="1D array of the outer diameter values defined along the tower axis.",
        )
        ivc.add_discrete_output(
            "layer_name",
            val=n_layers * [""],
            desc="1D array of the names of the layers modeled in the tower structure.",
        )
        ivc.add_discrete_output(
            "layer_mat",
            val=n_layers * [""],
            desc="1D array of the names of the materials of each layer modeled in the tower structure.",
        )
        ivc.add_output(
            "layer_thickness",
            val=np.zeros((n_layers, n_height)),
            units="m",
            desc="2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the tower sections.",
        )
        ivc.add_output(
            "outfitting_factor", val=0.0, desc="Multiplier that accounts for secondary structure mass inside of tower"
        )
        ivc.add_output("transition_piece_mass", val=0.0, units="kg", desc="point mass of transition piece")
        ivc.add_output("transition_piece_cost", val=0.0, units="USD", desc="cost of transition piece")
        ivc.add_output("gravity_foundation_mass", val=0.0, units="kg", desc="extra mass of gravity foundation")
        ivc.add_output(
            "monopile_mass_user",
            val=0.0,
            units="kg",
            desc="Override bottom-up calculation of total monopile mass with this value",
        )

        self.add_subsystem("compute_monopile_grid", Compute_Grid(n_height=n_height), promotes=["*"])


class Jacket(om.Group):
    def initialize(self):
        self.options.declare("fixedbottomse_options")

    def setup(self):
        fixedbottomse_options = self.options["fixedbottomse_options"]
        n_bays = fixedbottomse_options["n_bays"]
        n_legs = fixedbottomse_options["n_legs"]

        ivc = self.add_subsystem("jacket_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "foot_head_ratio",
            val=1.5,
            desc="Ratio of radius of foot (bottom) of jacket to head.",
        )
        ivc.add_output(
            "r_head",
            val=0.0,
            units="m",
            desc="Radius of head (top) of jacket, in meters.",
        )
        ivc.add_output(
            "height",
            val=0.0,
            units="m",
            desc="Overall jacket height, meters.",
        )
        ivc.add_output(
            "leg_diameter",
            val=0.0,
            units="m",
            desc="Leg diameter, meters. Constant throughout each leg.",
        )
        ivc.add_output(
            "leg_thickness",
            val=0.0,
            units="m",
            desc="Leg thickness, meters. Constant throughout each leg.",
        )
        ivc.add_output(
            "brace_diameters",
            val=np.zeros((n_bays)),
            units="m",
            desc="Brace diameter, meters. Array starts at the bottom of the jacket.",
        )
        ivc.add_output(
            "brace_thicknesses",
            val=np.zeros((n_bays)),
            units="m",
            desc="Brace thickness, meters. Array starts at the bottom of the jacket.",
        )
        ivc.add_output(
            "bay_spacing",
            val=np.zeros((n_bays + 1)),
            desc="Bay nodal spacing. Array starts at the bottom of the jacket.",
        )
        ivc.add_output(
            "outfitting_factor", val=0.0, desc="Multiplier that accounts for secondary structure mass inside of jacket"
        )
        ivc.add_output("transition_piece_mass", val=0.0, units="kg", desc="point mass of transition piece")
        ivc.add_output("transition_piece_cost", val=0.0, units="USD", desc="cost of transition piece")
        ivc.add_output("gravity_foundation_mass", val=0.0, units="kg", desc="extra mass of gravity foundation")
        ivc.add_output(
            "jacket_mass_user",
            val=0.0,
            units="kg",
            desc="Override bottom-up calculation of total jacket mass with this value",
        )


class Floating(om.Group):
    def initialize(self):
        self.options.declare("floating_init_options")
        self.options.declare("opt_options")

    def setup(self):
        floating_init_options = self.options["floating_init_options"]
        float_opt = self.options["opt_options"]["design_variables"]["floating"]
        n_joints = floating_init_options["joints"]["n_joints"]
        n_members = floating_init_options["members"]["n_members"]

        jivc = self.add_subsystem("joints", om.IndepVarComp(), promotes=["*"])
        jivc.add_output("location_in", val=np.zeros((n_joints, 3)), units="m")
        jivc.add_output("transition_node", val=np.zeros(3), units="m")
        jivc.add_output("transition_piece_mass", val=0.0, units="kg", desc="point mass of transition piece")
        jivc.add_output("transition_piece_cost", val=0.0, units="USD", desc="cost of transition piece")

        # Rigid body IVCs
        if floating_init_options["rigid_bodies"]["n_bodies"] > 0:
            rb_ivc = self.add_subsystem("rigid_bodies", om.IndepVarComp(), promotes=["*"])
        for k in range(floating_init_options["rigid_bodies"]["n_bodies"]):
            rb_ivc.add_output(f"rigid_body_{k}_node", val=np.zeros(3), units="m", desc="location of rigid body")
            rb_ivc.add_output(f"rigid_body_{k}_mass", val=0.0, units="kg", desc="point mass of rigid body")
            rb_ivc.add_output(f"rigid_body_{k}_inertia", val=np.zeros(3), units="kg*m**2", desc="inertia of rigid body")

        # Additions for optimizing individual nodes or multiple nodes concurrently
        self.add_subsystem("nodedv", NodeDVs(options=floating_init_options["joints"]), promotes=["*"])
        for k in range(len(floating_init_options["joints"]["design_variable_data"])):
            jivc.add_output(f"jointdv_{k}", val=0.0, units="m")

        # Members added in groups to allow for symmetry
        member_link_data = floating_init_options["members"]["linked_members"]
        for k in range(len(member_link_data)):
            name_member = member_link_data[k][0]
            memidx = floating_init_options["members"]["name"].index(name_member)
            n_geom = floating_init_options["members"]["n_geom"][memidx]
            n_height = floating_init_options["members"]["n_height"][memidx]
            n_layers = floating_init_options["members"]["n_layers"][memidx]
            n_ballasts = floating_init_options["members"]["n_ballasts"][memidx]
            n_bulkheads = floating_init_options["members"]["n_bulkheads"][memidx]
            n_axial_joints = floating_init_options["members"]["n_axial_joints"][memidx]

            ivc = self.add_subsystem(f"memgrp{k}", om.IndepVarComp())
            ivc.add_output("s_in", val=np.zeros(n_geom))
            ivc.add_output("s", val=np.zeros(n_height))

            member_shape_assigned = False
            for i, kgrp in enumerate(float_opt["members"]["groups"]):
                memname = kgrp["names"][0]
                idx = floating_init_options["members"]["name2idx"][memname]
                if idx == k:
                    if "diameter" in float_opt["members"]["groups"][i]:
                        if float_opt["members"]["groups"][i]["diameter"]["constant"]:
                            ivc.add_output("outer_diameter_in", val=0.0, units="m")
                        else:
                            ivc.add_output("outer_diameter_in", val=np.zeros(n_geom), units="m")
                        ivc.add_output("ca_usr_geom", val=-1.0 * np.ones(n_geom))
                        ivc.add_output("cd_usr_geom", val=-1.0 * np.ones(n_geom))
                        member_shape_assigned = True
                    if "side_length_a" in float_opt["members"]["groups"][i]:
                        if float_opt["members"]["groups"][i]["side_length_a"]["constant"]:
                            ivc.add_output("side_length_a_in", val=0.0, units="m")
                        else:
                            ivc.add_output("side_length_a_in", val=np.zeros(n_geom), units="m")
                        member_shape_assigned = True
                        ivc.add_output("ca_usr_geom", val=-1.0 * np.ones(n_geom))
                        ivc.add_output("cd_usr_geom", val=-1.0 * np.ones(n_geom))
                    if "side_length_b" in float_opt["members"]["groups"][i]:
                        if float_opt["members"]["groups"][i]["side_length_b"]["constant"]:
                            ivc.add_output("side_length_b_in", val=0.0, units="m")
                        else:
                            ivc.add_output("side_length_b_in", val=np.zeros(n_geom), units="m")
                        ivc.add_output("cay_usr_geom", val=-1.0 * np.ones(n_geom))
                        ivc.add_output("cdy_usr_geom", val=-1.0 * np.ones(n_geom))
                        member_shape_assigned = True

            if not member_shape_assigned:
                # Use the memidx to query the correct member_shape
                if floating_init_options["members"]["outer_shape"][memidx] == "circular":
                    ivc.add_output("outer_diameter_in", val=np.zeros(n_geom), units="m")
                    ivc.add_output("ca_usr_geom", val=-1.0 * np.ones(n_geom))
                    ivc.add_output("cd_usr_geom", val=-1.0 * np.ones(n_geom))
                elif floating_init_options["members"]["outer_shape"][memidx] == "rectangular":
                    ivc.add_output("side_length_a_in", val=np.zeros(n_geom), units="m")
                    ivc.add_output("side_length_b_in", val=np.zeros(n_geom), units="m")
                    ivc.add_output("ca_usr_geom", val=-1.0 * np.ones(n_geom))
                    ivc.add_output("cd_usr_geom", val=-1.0 * np.ones(n_geom))
                    ivc.add_output("cay_usr_geom", val=-1.0 * np.ones(n_geom))
                    ivc.add_output("cdy_usr_geom", val=-1.0 * np.ones(n_geom))

            ivc.add_discrete_output("layer_materials", val=[""] * n_layers)
            ivc.add_output("layer_thickness_in", val=np.zeros((n_layers, n_geom)), units="m")
            ivc.add_output("bulkhead_grid", val=np.zeros(n_bulkheads))
            ivc.add_output("bulkhead_thickness", val=np.zeros(n_bulkheads), units="m")
            ivc.add_output("ballast_grid", val=np.zeros((n_ballasts, 2)))
            ivc.add_output("ballast_volume", val=np.zeros(n_ballasts), units="m**3")
            ivc.add_discrete_output("ballast_materials", val=[""] * n_ballasts)
            ivc.add_output("grid_axial_joints", val=np.zeros(n_axial_joints))
            ivc.add_output("outfitting_factor", 0.0)
            ivc.add_output("ring_stiffener_web_height", 0.0, units="m")
            ivc.add_output("ring_stiffener_web_thickness", 0.0, units="m")
            ivc.add_output("ring_stiffener_flange_width", 0.0, units="m")
            ivc.add_output("ring_stiffener_flange_thickness", 0.0, units="m")
            ivc.add_output("ring_stiffener_spacing", 0.0)
            ivc.add_output("axial_stiffener_web_height", 0.0, units="m")
            ivc.add_output("axial_stiffener_web_thickness", 0.0, units="m")
            ivc.add_output("axial_stiffener_flange_width", 0.0, units="m")
            ivc.add_output("axial_stiffener_flange_thickness", 0.0, units="m")
            ivc.add_output("axial_stiffener_spacing", 0.0, units="deg")
            ivc.add_output(
                "member_mass_user",
                0.0,
                units="kg",
                desc="Override bottom-up calculation of total member mass with this value",
            )

            # Use the memidx to query the correct member_shape
            self.add_subsystem(
                f"memgrid{k}",
                MemberGrid(
                    n_height=n_height,
                    n_geom=n_geom,
                    n_layers=n_layers,
                    member_shape=floating_init_options["members"]["outer_shape"][memidx],
                ),
            )
            self.connect(f"memgrp{k}.s_in", f"memgrid{k}.s_in")
            self.connect(f"memgrp{k}.s", f"memgrid{k}.s_grid")
            # Here looping all dv member groups
            if floating_init_options["members"]["outer_shape"][memidx] == "circular":
                self.connect(f"memgrp{k}.outer_diameter_in", f"memgrid{k}.outer_diameter_in")
                self.connect(f"memgrp{k}.ca_usr_geom", f"memgrid{k}.ca_usr_geom")
                self.connect(f"memgrp{k}.cd_usr_geom", f"memgrid{k}.cd_usr_geom")
            elif floating_init_options["members"]["outer_shape"][memidx] == "rectangular":
                self.connect(f"memgrp{k}.side_length_a_in", f"memgrid{k}.side_length_a_in")
                self.connect(f"memgrp{k}.side_length_b_in", f"memgrid{k}.side_length_b_in")
                self.connect(f"memgrp{k}.ca_usr_geom", f"memgrid{k}.ca_usr_geom")
                self.connect(f"memgrp{k}.cd_usr_geom", f"memgrid{k}.cd_usr_geom")
                self.connect(f"memgrp{k}.cay_usr_geom", f"memgrid{k}.cay_usr_geom")
                self.connect(f"memgrp{k}.cdy_usr_geom", f"memgrid{k}.cdy_usr_geom")

            self.connect(f"memgrp{k}.layer_thickness_in", f"memgrid{k}.layer_thickness_in")

        self.add_subsystem("alljoints", AggregateJoints(floating_init_options=floating_init_options), promotes=["*"])

        for i in range(n_members):
            name_member = floating_init_options["members"]["name"][i]
            idx = floating_init_options["members"]["name2idx"][name_member]
            self.connect(f"memgrp{idx}.grid_axial_joints", f"member{i}_{name_member}:grid_axial_joints")
            if floating_init_options["members"]["outer_shape"][i] == "circular":
                self.connect(f"memgrid{idx}.outer_diameter", f"member{i}_{name_member}:outer_diameter")
            elif floating_init_options["members"]["outer_shape"][i] == "rectangular":
                # TODO: AggregatedJoints hasn't included rectangular yet, so no connection now
                print("WARNING: AggregatedJoints hasn't included rectangular yet")
                # self.connect(f"memgrid{idx}.side_length_a", f"member{i}_{name_member}:side_length_a")
                # self.connect(f"memgrid{idx}.side_length_b", f"member{i}_{name_member}:side_length_b")
            self.connect(f"memgrp{idx}.s", f"member{i}_{name_member}:s")


# Component that links certain nodes together in a specific dimension for optimization
class NodeDVs(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_joints = opt["n_joints"]
        self.add_input("location_in", val=np.zeros((n_joints, 3)), units="m")

        for k in range(len(opt["design_variable_data"])):
            self.add_input(f"jointdv_{k}", val=0.0, units="m")

        self.add_output("location", val=np.zeros((n_joints, 3)), units="m")

    def compute(self, inputs, outputs):
        opt = self.options["options"]

        xyz = inputs["location_in"]
        for i, linked_node_dict in enumerate(opt["design_variable_data"]):
            idx = linked_node_dict["indices"]
            dim = linked_node_dict["dimension"]
            xyz[idx, dim] = inputs[f"jointdv_{i}"]

        outputs["location"] = xyz


# Component that interpolates the diameter/thickness nodes to all of the other points needed in the member discretization
# TODO: This can be cleaned by generalizing the variables, set variables as options and loop all required variables to interpolate
class MemberGrid(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_layers")
        self.options.declare("n_height")
        self.options.declare("n_geom")
        self.options.declare("member_shape")

    def setup(self):
        n_layers = self.options["n_layers"]
        n_height = self.options["n_height"]
        n_geom = self.options["n_geom"]
        member_shape = self.options["member_shape"]

        self.add_input("s_in", val=np.zeros(n_geom))
        self.add_input("s_grid", val=np.zeros(n_height))
        if member_shape == "circular":
            self.add_input("outer_diameter_in", shape_by_conn=True, units="m")
            self.add_input("ca_usr_geom", shape_by_conn=True)
            self.add_input("cd_usr_geom", shape_by_conn=True)
        elif member_shape == "rectangular":
            self.add_input("side_length_a_in", shape_by_conn=True, units="m")
            self.add_input("side_length_b_in", shape_by_conn=True, units="m")
            self.add_input("ca_usr_geom", shape_by_conn=True)
            self.add_input("cd_usr_geom", shape_by_conn=True)
            self.add_input("cay_usr_geom", shape_by_conn=True)
            self.add_input("cdy_usr_geom", shape_by_conn=True)

        self.add_input("layer_thickness_in", val=np.zeros((n_layers, n_geom)), units="m")

        if member_shape == "circular":
            self.add_output("outer_diameter", val=np.zeros(n_height), units="m")
            self.add_output("ca_usr_grid", val=-1.0 * np.ones(n_height))
            self.add_output("cd_usr_grid", val=-1.0 * np.ones(n_height))
        elif member_shape == "rectangular":
            self.add_output("side_length_a", val=np.zeros(n_height), units="m")
            self.add_output("side_length_b", val=np.zeros(n_height), units="m")
            self.add_output("ca_usr_grid", val=-1.0 * np.ones(n_height))
            self.add_output("cd_usr_grid", val=-1.0 * np.ones(n_height))
            self.add_output("cay_usr_grid", val=-1.0 * np.ones(n_height))
            self.add_output("cdy_usr_grid", val=-1.0 * np.ones(n_height))

        self.add_output("layer_thickness", val=np.zeros((n_layers, n_height)), units="m")

    def compute(self, inputs, outputs):
        n_layers = self.options["n_layers"]
        member_shape = self.options["member_shape"]

        s_in = inputs["s_in"]
        s_grid = inputs["s_grid"]

        if member_shape == "circular":
            if len(inputs["outer_diameter_in"]) > 1:
                outputs["outer_diameter"] = PchipInterpolator(s_in, inputs["outer_diameter_in"])(s_grid)
            else:
                outputs["outer_diameter"][:] = inputs["outer_diameter_in"]
            outputs["ca_usr_grid"] = PchipInterpolator(s_in, inputs["ca_usr_geom"])(s_grid)
            outputs["cd_usr_grid"] = PchipInterpolator(s_in, inputs["cd_usr_geom"])(s_grid)
        elif member_shape == "rectangular":
            if len(inputs["side_length_a_in"]) > 1:
                outputs["side_length_a"] = PchipInterpolator(s_in, inputs["side_length_a_in"])(s_grid)
                outputs["side_length_b"] = PchipInterpolator(s_in, inputs["side_length_b_in"])(s_grid)
            else:
                outputs["side_length_a"][:] = inputs["side_length_a_in"]
                outputs["side_length_b"][:] = inputs["side_length_b_in"]
            outputs["ca_usr_grid"] = PchipInterpolator(s_in, inputs["ca_usr_geom"])(s_grid)
            outputs["cd_usr_grid"] = PchipInterpolator(s_in, inputs["cd_usr_geom"])(s_grid)
            outputs["cay_usr_grid"] = PchipInterpolator(s_in, inputs["cay_usr_geom"])(s_grid)
            outputs["cdy_usr_grid"] = PchipInterpolator(s_in, inputs["cdy_usr_geom"])(s_grid)

        for k in range(n_layers):
            outputs["layer_thickness"][k, :] = PchipInterpolator(s_in, inputs["layer_thickness_in"][k, :])(s_grid)


class AggregateJoints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("floating_init_options")

    def setup(self):
        floating_init_options = self.options["floating_init_options"]
        n_joints = floating_init_options["joints"]["n_joints"]
        n_joints_tot = len(floating_init_options["joints"]["name2idx"])
        n_members = floating_init_options["members"]["n_members"]

        self.add_input("location", val=np.zeros((n_joints, 3)), units="m")

        for i in range(n_members):
            iname = floating_init_options["members"]["name"][i]
            i_axial_joints = floating_init_options["members"]["n_axial_joints"][i]
            i_grid = len(floating_init_options["members"]["grid_member_" + iname])

            self.add_input(f"member{i}_{iname}:s", val=np.zeros(i_grid))
            self.add_input(f"member{i}_{iname}:outer_diameter", val=np.zeros(i_grid), units="m")
            self.add_input(f"member{i}_{iname}:grid_axial_joints", val=np.zeros(i_axial_joints))

            self.add_output(f"member{i}_{iname}:joint1", val=np.zeros(3), units="m")
            self.add_output(f"member{i}_{iname}:joint2", val=np.zeros(3), units="m")
            self.add_output(f"member{i}_{iname}:height", val=0.0, units="m")
            self.add_output(f"member{i}_{iname}:s_ghost1", val=0.0)
            self.add_output(f"member{i}_{iname}:s_ghost2", val=1.0)

        self.add_output("joints_xyz", val=np.zeros((n_joints_tot, 3)), units="m")

    def compute(self, inputs, outputs):
        # Unpack options
        floating_init_options = self.options["floating_init_options"]
        memopt = floating_init_options["members"]
        n_joints = floating_init_options["joints"]["n_joints"]
        n_members = memopt["n_members"]
        name2idx = floating_init_options["joints"]["name2idx"]
        n_joint_tot = len(name2idx)
        NULL = -9999.0

        # Unpack inputs
        locations = inputs["location"]
        joints_xyz = NULL * np.ones(outputs["joints_xyz"].shape)

        # Handle relative joints
        joint_names = floating_init_options["joints"]["name"]
        for i_joint in range(floating_init_options["joints"]["n_joints"]):
            rel_joint = floating_init_options["joints"]["relative"][i_joint]  # name of joint relative to this joint
            if rel_joint != "origin":  # is a relative joint
                if rel_joint not in joint_names:
                    raise Exception(
                        f"The relative joint {joint_names[i_joint]} is not relative to an existing joint.  Relative joint provided: {rel_joint}"
                    )

                rel_joint_location = locations[name2idx[rel_joint]]
                relative_dimensions = np.array(
                    floating_init_options["joints"]["relative_dims"][i_joint]
                )  # These joints are relative
                locations[i_joint][relative_dimensions] += rel_joint_location[relative_dimensions]
        
        # Handle cylindrical coordinate joints
        icyl = floating_init_options["joints"]["cylindrical"]
        locations_xyz = locations.copy()
        locations_xyz[icyl, 0] = locations[icyl, 0] * np.cos(np.deg2rad(locations[icyl, 1]))
        locations_xyz[icyl, 1] = locations[icyl, 0] * np.sin(np.deg2rad(locations[icyl, 1]))


        joints_xyz[:n_joints, :] = locations_xyz.copy()

        # Initial biggest radius at each node
        node_r = np.zeros(n_joint_tot)
        intersects = np.zeros(n_joint_tot)

        if n_joints + sum(memopt["n_axial_joints"]) > n_joint_tot:
            raise Exception(
                f'WISDEM has detected {n_joints + sum(memopt["n_axial_joints"])}, but only {n_joint_tot} have been defined in the yaml'
            )

        # Now add axial joints
        member_list = list(range(n_members))
        count = n_joints
        while count < n_joint_tot:
            for k in member_list[:]:
                # Get the end joint locations for members and then compute the axial joint loc
                joint1xyz = joints_xyz[name2idx[memopt["joint1"][k]], :]
                joint2xyz = joints_xyz[name2idx[memopt["joint2"][k]], :]

                # Check if we are ready to compute xyz position of axial joints in this member
                if np.all(joint1xyz != NULL) and np.all(joint2xyz != NULL):
                    member_list.remove(k)
                else:
                    continue

                i_axial_joints = memopt["n_axial_joints"][k]
                if i_axial_joints == 0:
                    continue

                iname = memopt["name"][k]
                s = 0.5 * inputs[f"member{k}_{iname}:s"]
                Rk = 0.5 * inputs[f"member{k}_{iname}:outer_diameter"]
                dxyz = joint2xyz - joint1xyz

                for a in range(i_axial_joints):
                    s_axial = inputs[f"member{k}_{iname}:grid_axial_joints"][a]
                    joints_xyz[count, :] = joint1xyz + s_axial * dxyz

                    Ra = PchipInterpolator(s, Rk)(s_axial)
                    node_r[count] = max(node_r[count], Ra)
                    intersects[count] += 1

                    count += 1

        # Record starting and ending location for each member now.
        # Also log biggest radius at each node intersection to compute ghost nodes
        for k in range(n_members):
            iname = memopt["name"][k]
            joint1id = name2idx[memopt["joint1"][k]]
            joint2id = name2idx[memopt["joint2"][k]]
            joint1xyz = joints_xyz[joint1id, :]
            joint2xyz = joints_xyz[joint2id, :]
            hk = np.sqrt(np.sum((joint2xyz - joint1xyz) ** 2))
            outputs[f"member{k}_{iname}:joint1"] = joint1xyz
            outputs[f"member{k}_{iname}:joint2"] = joint2xyz
            outputs[f"member{k}_{iname}:height"] = hk

            # Largest radius at connection points for this member,
            # Don't check radius and add an intersection if the member is parallel to the one it's connecting to
            # The ghost node calculations pre-suppose that joints join orthogonal members, but if the member is parallel to another, the
            # no_intersect flag should be used.  no_intersect should be used for modeling heave plates
            if not floating_init_options["members"]["no_intersect"][k]:
                Rk = 0.5 * inputs[f"member{k}_{iname}:outer_diameter"]
                node_r[joint1id] = max(node_r[joint1id], Rk[0])
                node_r[joint2id] = max(node_r[joint2id], Rk[-1])
                intersects[joint1id] += 1
                intersects[joint2id] += 1

        # Store the ghost node non-dimensional locations
        for k in range(n_members):
            iname = memopt["name"][k]
            joint1id = name2idx[memopt["joint1"][k]]
            joint2id = name2idx[memopt["joint2"][k]]
            hk = outputs[f"member{k}_{iname}:height"]
            Rk = 0.5 * inputs[f"member{k}_{iname}:outer_diameter"]
            s_ghost1 = 0.0
            s_ghost2 = 1.0
            if intersects[joint1id] > 1 and node_r[joint1id] > Rk[0]:
                s_ghost1 = node_r[joint1id] / hk
            if intersects[joint2id] > 1 and node_r[joint2id] > Rk[-1]:
                s_ghost2 = 1.0 - node_r[joint2id] / hk
            outputs[f"member{k}_{iname}:s_ghost1"] = s_ghost1
            outputs[f"member{k}_{iname}:s_ghost2"] = s_ghost2

        # Store outputs
        outputs["joints_xyz"] = joints_xyz


class Mooring(om.Group):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        mooring_init_options = self.options["options"]["mooring"]

        n_nodes = mooring_init_options["n_nodes"]
        n_lines = mooring_init_options["n_lines"]

        n_design = 1 if mooring_init_options["symmetric"] else n_lines

        ivc = self.add_subsystem("mooring", om.IndepVarComp(), promotes=["*"])

        ivc.add_discrete_output("node_names", val=[""] * n_nodes)
        ivc.add_discrete_output("n_lines", val=0)  # Needed for ORBIT
        ivc.add_output("nodes_location", val=np.zeros((n_nodes, 3)), units="m")
        ivc.add_output("nodes_mass", val=np.zeros(n_nodes), units="kg")
        ivc.add_output("nodes_volume", val=np.zeros(n_nodes), units="m**3")
        ivc.add_output("nodes_added_mass", val=np.zeros(n_nodes))
        ivc.add_output("nodes_drag_area", val=np.zeros(n_nodes), units="m**2")
        ivc.add_discrete_output("nodes_joint_name", val=[""] * n_nodes)
        ivc.add_output("unstretched_length_in", val=np.zeros(n_design), units="m")
        ivc.add_discrete_output("line_id", val=[""] * n_lines)
        ivc.add_output("line_diameter_in", val=np.zeros(n_design), units="m")
        ivc.add_output("line_mass_density_coeff", val=np.zeros(n_lines), units="kg/m**3")
        ivc.add_output("line_stiffness_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("line_breaking_load_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("line_cost_rate_coeff", val=np.zeros(n_lines), units="USD/m**3")
        ivc.add_output("line_transverse_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        ivc.add_output("line_tangential_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        ivc.add_output("line_transverse_drag_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("line_tangential_drag_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("anchor_mass", val=np.zeros(n_lines), units="kg")
        ivc.add_output("anchor_cost", val=np.zeros(n_lines), units="USD")
        ivc.add_output("anchor_max_vertical_load", val=1e30 * np.ones(n_lines), units="N")
        ivc.add_output("anchor_max_lateral_load", val=1e30 * np.ones(n_lines), units="N")

        self.add_subsystem("moorprop", MooringProperties(mooring_init_options=mooring_init_options), promotes=["*"])
        self.add_subsystem("moorjoint", MooringJoints(options=self.options["options"]), promotes=["*"])


class MooringProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("mooring_init_options")

    def setup(self):
        mooring_init_options = self.options["mooring_init_options"]
        n_lines = mooring_init_options["n_lines"]
        n_design = 1 if mooring_init_options["symmetric"] else n_lines

        self.add_input("unstretched_length_in", val=np.zeros(n_design), units="m")
        self.add_input("line_diameter_in", val=np.zeros(n_design), units="m")
        self.add_input("line_mass_density_coeff", val=np.zeros(n_lines), units="kg/m**3")
        self.add_input("line_stiffness_coeff", val=np.zeros(n_lines), units="N/m**2")
        self.add_input("line_breaking_load_coeff", val=np.zeros(n_lines), units="N/m**2")
        self.add_input("line_cost_rate_coeff", val=np.zeros(n_lines), units="USD/m**3")
        self.add_input("line_transverse_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        self.add_input("line_tangential_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        self.add_input("line_transverse_drag_coeff", val=np.zeros(n_lines), units="N/m**2")
        self.add_input("line_tangential_drag_coeff", val=np.zeros(n_lines), units="N/m**2")

        self.add_output("unstretched_length", val=np.zeros(n_lines), units="m")
        self.add_output("line_diameter", val=np.zeros(n_lines), units="m")
        self.add_output("line_mass_density", val=np.zeros(n_lines), units="kg/m")
        self.add_output("line_stiffness", val=np.zeros(n_lines), units="N")
        self.add_output("line_breaking_load", val=np.zeros(n_lines), units="N")
        self.add_output("line_cost_rate", val=np.zeros(n_lines), units="USD/m")
        self.add_output("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_output("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_output("line_transverse_drag", val=np.zeros(n_lines))
        self.add_output("line_tangential_drag", val=np.zeros(n_lines))

    def compute(self, inputs, outputs):
        n_lines = self.options["mooring_init_options"]["n_lines"]
        line_mat = self.options["mooring_init_options"]["line_material"]
        outputs["line_diameter"] = d = inputs["line_diameter_in"] * np.ones(n_lines)
        outputs["unstretched_length"] = inputs["unstretched_length_in"] * np.ones(n_lines)
        d2 = d * d

        for i_line, lm in enumerate(line_mat):
            if lm == "custom":
                varlist = [
                    "line_mass_density",
                    "line_stiffness",
                    "line_breaking_load",
                    "line_cost_rate",
                    "line_transverse_added_mass",
                    "line_tangential_added_mass",
                    "line_transverse_drag",
                    "line_tangential_drag",
                ]
                for var in varlist:
                    outputs[var] = d2 * inputs[var + "_coeff"]
                line_props = None

            elif lm == "chain_stud":
                line_props = getLineProps(1e3 * d[i_line] / 1.89, material="chain_studlink", source="default")
            else:
                line_props = getLineProps(1e3 * d[i_line] / 1.8, material="chain", source="default")
            if line_props is not None:
                outputs["line_mass_density"][i_line] = line_props["m"]
                outputs["line_stiffness"][i_line] = line_props["EA"]
                outputs["line_breaking_load"][i_line] = line_props["MBL"]
                outputs["line_cost_rate"][i_line] = line_props["cost"]

                outputs["line_transverse_added_mass"][i_line] = line_props["Ca"]
                outputs["line_tangential_added_mass"][i_line] = line_props["CaAx"]
                outputs["line_transverse_drag"][i_line] = line_props["Cd"]
                outputs["line_tangential_drag"][i_line] = line_props["CdAx"]


class MooringJoints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        mooring_init_options = self.options["options"]["mooring"]
        n_nodes = mooring_init_options["n_nodes"]
        n_attach = mooring_init_options["n_attach"]
        n_lines = mooring_init_options["n_lines"]
        n_anchors = mooring_init_options["n_anchors"]

        self.add_discrete_input("nodes_joint_name", val=[""] * n_nodes)
        self.add_input("nodes_location", val=np.zeros((n_nodes, 3)), units="m")
        self.add_input("joints_xyz", shape_by_conn=True, units="m")

        self.add_output("mooring_nodes", val=np.zeros((n_nodes, 3)), units="m")
        self.add_output("fairlead_nodes", val=np.zeros((n_attach, 3)), units="m")
        self.add_output("fairlead", val=np.zeros(n_lines), units="m")
        self.add_output("fairlead_radius", val=np.zeros(n_attach), units="m")
        self.add_output("anchor_nodes", val=np.zeros((n_anchors, 3)), units="m")
        self.add_output("anchor_radius", val=np.zeros(n_anchors), units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        mooring_init_options = self.options["options"]["mooring"]
        n_nodes = mooring_init_options["n_nodes"]

        node_joints = discrete_inputs["nodes_joint_name"]
        node_loc = inputs["nodes_location"]
        joints_loc = inputs["joints_xyz"]
        idx_map = self.options["options"]["floating"]["joints"]["name2idx"]

        # Find mooring nodes
        # Use mooring node name that correpsonds to floating joint location
        for k in range(n_nodes):
            if node_joints[k] == "":
                continue
            idx = idx_map[node_joints[k]]
            node_loc[k, :] = joints_loc[idx, :]
        outputs["mooring_nodes"] = node_loc

        # node_loc = np.unique(node_loc, axis=0)      # this step re-orders!  I'm not sure how there would be duplicates, unless there were duplicate mooring nodes
        depth = np.abs(node_loc[:, 2].min())

        ifair = np.where(np.array(mooring_init_options["node_type"]) == "vessel")[0]
        ianch = np.where(np.array(mooring_init_options["node_type"]) == "fixed")[0]

        z_fair = node_loc[ifair, 2].mean()
        z_anch = node_loc[ianch, 2].mean()

        node_fair = node_loc[ifair, :]
        node_anch = node_loc[ianch, :]
        ang_fair = np.arctan2(node_fair[:, 1], node_fair[:, 0])
        ang_anch = np.arctan2(node_anch[:, 1], node_anch[:, 0])
        node_fair = np.unique(node_fair[np.argsort(ang_fair), :], axis=0)
        node_anch = np.unique(node_anch[np.argsort(ang_anch), :], axis=0)

        outputs["fairlead_nodes"] = node_fair
        outputs["anchor_nodes"] = node_anch
        outputs["fairlead"] = -z_fair  # Positive is defined below the waterline here
        outputs["fairlead_radius"] = np.sqrt(np.sum(node_fair[:, :2] ** 2, axis=1))
        outputs["anchor_radius"] = np.sqrt(np.sum(node_anch[:, :2] ** 2, axis=1))


class ComputeMaterialsProperties(om.ExplicitComponent):
    # Openmdao component with the wind turbine materials coming from the input yaml file. The inputs and outputs are arrays where each entry represents a material

    def initialize(self):
        self.options.declare("mat_init_options")
        self.options.declare("composites", default=True)

    def setup(self):
        mat_init_options = self.options["mat_init_options"]
        self.n_mat = n_mat = mat_init_options["n_mat"]

        self.add_discrete_input("name", val=n_mat * [""], desc="1D array of names of materials.")
        self.add_discrete_input(
            "orth",
            val=np.zeros(n_mat),
            desc="1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.",
        )
        self.add_input(
            "rho_fiber",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the fibers of the materials.",
        )
        self.add_input(
            "rho",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the materials. For composites, this is the density of the laminate.",
        )
        self.add_input(
            "rho_area_dry",
            val=np.zeros(n_mat),
            units="kg/m**2",
            desc="1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "ply_t_from_yaml",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "fvf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "fwf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )

        self.add_output(
            "ply_t",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        self.add_output(
            "fvf",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        self.add_output(
            "fwf",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        density_resin = 0.0
        for i in range(self.n_mat):
            if discrete_inputs["name"][i] == "resin":
                density_resin = inputs["rho"][i]
                id_resin = i
        if self.options["composites"] and density_resin == 0.0:
            raise Exception(
                "Warning: a material named resin is not defined in the input yaml.  This is required for blade composite analysis"
            )

        fvf = np.zeros(self.n_mat)
        fwf = np.zeros(self.n_mat)
        ply_t = np.zeros(self.n_mat)

        for i in range(self.n_mat):
            if discrete_inputs["orth"][i] == 1:  # It's a composite
                # Formula to estimate the fiber volume fraction fvf from the laminate and the fiber densities
                fvf[i] = (inputs["rho"][i] - density_resin) / (inputs["rho_fiber"][i] - density_resin)
                if inputs["fvf_from_yaml"][i] > 0.0:
                    if abs(fvf[i] - inputs["fvf_from_yaml"][i]) > 1e-3:
                        raise ValueError(
                            "Error: the fvf of composite "
                            + discrete_inputs["name"][i]
                            + " specified in the yaml is equal to "
                            + str(inputs["fvf_from_yaml"][i] * 100)
                            + "%, but this value is not compatible to the other values provided. Given the fiber, laminate and resin densities, it should instead be equal to "
                            + str(fvf[i] * 100.0)
                            + "%."
                        )
                    else:
                        outputs["fvf"][i] = inputs["fvf_from_yaml"][i]
                else:
                    outputs["fvf"][i] = fvf[i]

                # Formula to estimate the fiber weight fraction fwf from the fiber volume fraction and the fiber densities
                fwf[i] = (
                    inputs["rho_fiber"][i]
                    * outputs["fvf"][i]
                    / (density_resin + ((inputs["rho_fiber"][i] - density_resin) * outputs["fvf"][i]))
                )
                if inputs["fwf_from_yaml"][i] > 0.0:
                    if abs(fwf[i] - inputs["fwf_from_yaml"][i]) > 1e-3:
                        raise ValueError(
                            "Error: the fwf of composite "
                            + discrete_inputs["name"][i]
                            + " specified in the yaml is equal to "
                            + str(inputs["fwf_from_yaml"][i] * 100)
                            + "%, but this value is not compatible to the other values provided. It should instead be equal to "
                            + str(fwf[i] * 100.0)
                            + "%"
                        )
                    else:
                        outputs["fwf"][i] = inputs["fwf_from_yaml"][i]
                else:
                    outputs["fwf"][i] = fwf[i]

                # Formula to estimate the plyt thickness ply_t of a laminate from the aerial density, the laminate density and the fiber weight fraction
                ply_t[i] = inputs["rho_area_dry"][i] / inputs["rho"][i] / outputs["fwf"][i]
                if inputs["ply_t_from_yaml"][i] > 0.0:
                    if abs(ply_t[i] - inputs["ply_t_from_yaml"][i]) > 1.0e-4:
                        raise ValueError(
                            "Error: the ply_t of composite "
                            + discrete_inputs["name"][i]
                            + " specified in the yaml is equal to "
                            + str(inputs["ply_t_from_yaml"][i])
                            + "m, but this value is not compatible to the other values provided. It should instead be equal to "
                            + str(ply_t[i])
                            + "m. Alternatively, adjust the aerial density to "
                            + str(outputs["ply_t"][i] * inputs["rho"][i] * outputs["fwf"][i])
                            + " kg/m2."
                        )
                    else:
                        outputs["ply_t"][i] = inputs["ply_t_from_yaml"][i]
                else:
                    outputs["ply_t"][i] = ply_t[i]


class Materials(om.Group):
    # Openmdao group with the wind turbine materials coming from the input yaml file.
    # The inputs and outputs are arrays where each entry represents a material

    def initialize(self):
        self.options.declare("mat_init_options")
        self.options.declare("composites", default=True)

    def setup(self):
        mat_init_options = self.options["mat_init_options"]
        self.n_mat = n_mat = mat_init_options["n_mat"]

        ivc = self.add_subsystem("materials_indep_vars", om.IndepVarComp(), promotes=["*"])

        ivc.add_discrete_output(
            "orth",
            val=np.zeros(n_mat),
            desc="1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.",
        )
        ivc.add_output(
            "E",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.",
        )
        ivc.add_output(
            "G",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.",
        )
        ivc.add_output(
            "nu",
            val=np.zeros([n_mat, 3]),
            desc="2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.",
        )
        ivc.add_output(
            "Xt",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.",
        )
        ivc.add_output(
            "Xc",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.",
        )
        ivc.add_output(
            "S",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Ultimate Shear Strength (USS) of the materials. Each row represents a material, the three columns represent S12, S13 and S23.",
        )
        ivc.add_output(
            "sigma_y",
            val=np.zeros(n_mat),
            units="Pa",
            desc="Yield stress of the material (in the principle direction for composites).",
        )
        ivc.add_output(
            "wohler_exp",
            val=np.zeros(n_mat),
            desc="Exponent of S-N Wohler fatigue curve in the form of S = A*N^-(1/m).",
        )
        ivc.add_output(
            "wohler_intercept",
            val=np.zeros(n_mat),
            desc="Stress-intercept (A) of S-N Wohler fatigue curve in the form of S = A*N^-(1/m), taken as ultimate stress unless otherwise specified.",
        )
        ivc.add_output(
            "unit_cost", val=np.zeros(n_mat), units="USD/kg", desc="1D array of the unit costs of the materials."
        )
        ivc.add_output(
            "waste", val=np.zeros(n_mat), desc="1D array of the non-dimensional waste fraction of the materials."
        )
        ivc.add_output(
            "roll_mass",
            val=np.zeros(n_mat),
            units="kg",
            desc="1D array of the roll mass of the composite fabrics. Non-composite materials are kept at 0.",
        )

        ivc.add_discrete_output("name", val=n_mat * [""], desc="1D array of names of materials.")
        ivc.add_output(
            "rho_fiber",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the fibers of the materials.",
        )
        ivc.add_output(
            "rho",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the materials. For composites, this is the density of the laminate.",
        )
        ivc.add_output(
            "rho_area_dry",
            val=np.zeros(n_mat),
            units="kg/m**2",
            desc="1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.",
        )
        ivc.add_output(
            "ply_t_from_yaml",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        ivc.add_output(
            "fvf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        ivc.add_output(
            "fwf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )

        self.add_subsystem(
            "compute_materials_properties",
            ComputeMaterialsProperties(mat_init_options=mat_init_options, composites=self.options["composites"]),
            promotes=["*"],
        )


class ComputeHighLevelBladeProperties(om.ExplicitComponent):
    # Openmdao component that computes rotor quantities, such as the rotor coordinate of the blade stations
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]

        n_span = rotorse_options["n_span"]

        self.add_input(
            "blade_ref_axis_user",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        self.add_input(
            "rotor_diameter_user",
            val=0.0,
            units="m",
            desc="Diameter of the rotor specified by the user, defined as 2 x (Rhub + blade length along z) * cos(precone).",
        )
        self.add_input(
            "hub_radius",
            val=0.0,
            units="m",
            desc="Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line.",
        )
        self.add_input(
            "cone",
            val=0.0,
            units="deg",
            desc="Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_discrete_input("n_blades", val=3, desc="Number of blades of the rotor.")

        self.add_output(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Scalar of the rotor diameter, defined as 2 x (Rhub + blade length along z) * cos(precone).",
        )
        self.add_output(
            "r_blade",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the dimensional spanwise grid defined along the rotor (hub radius to blade tip projected on the plane)",
        )
        self.add_output(
            "Rtip",
            val=0.0,
            units="m",
            desc="Distance between rotor center and blade tip along z axis of the blade root c.s.",
        )
        self.add_output(
            "blade_ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis scaled based on rotor diameter, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        self.add_output("prebend", val=np.zeros(n_span), units="m", desc="Blade prebend at each section")
        self.add_output("prebendTip", val=0.0, units="m", desc="Blade prebend at tip")
        self.add_output("presweep", val=np.zeros(n_span), units="m", desc="Blade presweep at each section")
        self.add_output("presweepTip", val=0.0, units="m", desc="Blade presweep at tip")
        self.add_output(
            "blade_length",
            val=0.0,
            units="m",
            desc="Scalar of the 3D blade length computed along its axis, scaled based on the user defined rotor diameter.",
        )
        self.add_output("blade_solidity", val=0.0, desc="Blade solidity")
        self.add_output("rotor_solidity", val=0.0, desc="Rotor solidity")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs["blade_ref_axis"][:, 0] = inputs["blade_ref_axis_user"][:, 0]
        outputs["blade_ref_axis"][:, 1] = inputs["blade_ref_axis_user"][:, 1]
        # Scale z if the blade length provided by the user does not match the rotor diameter. D = (blade length + hub radius) * 2
        if inputs["rotor_diameter_user"] != 0.0:
            outputs["rotor_diameter"] = inputs["rotor_diameter_user"]
            outputs["blade_ref_axis"][:, 2] = (
                inputs["blade_ref_axis_user"][:, 2]
                * inputs["rotor_diameter_user"]
                / (
                    (inputs["blade_ref_axis_user"][-1, 2] + inputs["hub_radius"])
                    * 2.0
                    * np.cos(np.deg2rad(inputs["cone"][0]))
                )
            )
        # If the user does not provide a rotor diameter, this is computed from the hub diameter and the blade length
        else:
            outputs["rotor_diameter"] = (
                (inputs["blade_ref_axis_user"][-1, 2] + inputs["hub_radius"])
                * 2.0
                * np.cos(np.deg2rad(inputs["cone"][0]))
            )
            outputs["blade_ref_axis"][:, 2] = inputs["blade_ref_axis_user"][:, 2]
        outputs["r_blade"] = outputs["blade_ref_axis"][:, 2] + inputs["hub_radius"]
        outputs["Rtip"] = outputs["r_blade"][-1]
        outputs["blade_length"] = arc_length(outputs["blade_ref_axis"])[-1]
        outputs["prebend"] = outputs["blade_ref_axis"][:, 0]
        outputs["prebendTip"] = outputs["blade_ref_axis"][-1, 0]
        outputs["presweep"] = outputs["blade_ref_axis"][:, 1]
        outputs["presweepTip"] = outputs["blade_ref_axis"][-1, 1]
        try:
            # Numpy v1/2 clash
            outputs["blade_solidity"] = np.trapezoid(inputs["chord"], outputs["r_blade"]) / (
                np.pi * outputs["rotor_diameter"] ** 2.0 / 4.0
            )
        except AttributeError:
            outputs["blade_solidity"] = np.trapz(inputs["chord"], outputs["r_blade"]) / (
                np.pi * outputs["rotor_diameter"] ** 2.0 / 4.0
            )
        outputs["rotor_solidity"] = outputs["blade_solidity"] * discrete_inputs["n_blades"]


class ComputeHighLevelTowerProperties(om.ExplicitComponent):
    # Openmdao component that computes tower quantities, such as the hub height, and the blade-tower clearance
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        if modeling_options["flags"]["tower"]:
            n_height_tower = modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]
        else:
            n_height_tower = 0

        self.add_input(
            "tower_ref_axis_user",
            val=np.zeros((n_height_tower, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
        )
        self.add_input("distance_tt_hub", val=0.0, units="m", desc="Vertical distance from tower top to hub center.")
        self.add_input("hub_height_user", val=0.0, units="m", desc="Height of the hub specified by the user.")
        self.add_input(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Scalar of the rotor diameter, defined as 2 x (Rhub + blade length along z) * cos(precone).",
        )

        self.add_output(
            "tower_ref_axis",
            val=np.zeros((n_height_tower, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
        )
        self.add_output(
            "hub_height",
            val=0.0,
            units="m",
            desc="Height of the hub in the global reference system, i.e. distance rotor center to ground.",
        )

    def compute(self, inputs, outputs):
        modeling_options = self.options["modeling_options"]
        if inputs["hub_height_user"][0] != 0.0:
            outputs["hub_height"] = inputs["hub_height_user"]

        if modeling_options["flags"]["tower"]:
            if inputs["hub_height_user"][0] != 0.0:
                z_base = inputs["tower_ref_axis_user"][0, 2]
                z_current = inputs["tower_ref_axis_user"][:, 2] - z_base
                h_needed = inputs["hub_height_user"] - inputs["distance_tt_hub"] - z_base
                z_new = z_current * h_needed / z_current[-1]
                outputs["tower_ref_axis"][:, 2] = z_new + z_base
            else:
                outputs["hub_height"] = inputs["tower_ref_axis_user"][-1, 2] + inputs["distance_tt_hub"]
                outputs["tower_ref_axis"] = inputs["tower_ref_axis_user"]

        if outputs["hub_height"][0] == 0.0:
            raise Exception(
                "The hub height cannot be set.  Please set it in the top level 'assembly' section in the yaml file and/or define the tower reference axis"
            )

        if modeling_options["flags"]["blade"] and inputs["rotor_diameter"] > 2.0 * outputs["hub_height"]:
            raise Exception(
                "The rotor blade extends past the ground or water line. Please adjust hub height and/or rotor diameter."
            )


class Airfoil3DCorrection(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.af_correction = rotorse_options["3d_af_correction"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.add_input(
            "aoa",
            val=np.zeros(n_aoa),
            units="deg",
            desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "Re",
            val=np.zeros(n_Re),
            desc="1D array of the Reynolds numbers used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "cl",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_input(
            "cd",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_input(
            "cm",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number.",
        )
        self.add_input("rated_TSR", val=0.0, desc="Constant tip speed ratio in region II.")
        self.add_input(
            "r_blade",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the dimensional spanwise grid defined along the rotor (hub radius to blade tip projected on the plane)",
        )
        self.add_input(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Diameter of the wind turbine rotor specified by the user, defined as 2 x (Rhub + blade length along z) * cos(precone).",
        )
        self.add_input(
            "rthick",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        # Outputs
        self.add_output(
            "cl_corrected",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="Lift coefficient corrected with CCBlade.Polar.",
        )
        self.add_output(
            "cd_corrected",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="Drag coefficient corrected with CCBlade.Polar.",
        )
        self.add_output(
            "cm_corrected",
            val=np.zeros((n_span, n_aoa, n_Re)),
            desc="Moment coefficient corrected with CCblade.Polar.",
        )

    def compute(self, inputs, outputs):
        cl_corrected = np.zeros((self.n_span, self.n_aoa, self.n_Re))
        cd_corrected = np.zeros((self.n_span, self.n_aoa, self.n_Re))
        cm_corrected = np.zeros((self.n_span, self.n_aoa, self.n_Re))
        for i in range(self.n_span):
            if (
                inputs["rthick"][i] < 0.7 and self.af_correction
            ):  # Only apply 3D correction to airfoils thinner than 70% to avoid numerical problems at blade root
                logger.info("3D correction applied to airfoil polars for section " + str(i))
                for j in range(self.n_Re):
                    polar = Polar(
                        Re=inputs["Re"][j],
                        alpha=inputs["aoa"],
                        cl=inputs["cl"][i, :, j],
                        cd=inputs["cd"][i, :, j],
                        cm=inputs["cm"][i, :, j],
                    )
                    polar3d = polar.correction3D(
                        inputs["r_blade"][i] / (inputs["rotor_diameter"][0] / 2),
                        inputs["chord"][i] / inputs["r_blade"][i],
                        inputs["rated_TSR"],
                    )
                    cl_corrected[i, :, j] = PchipInterpolator(polar3d.alpha, polar3d.cl)(inputs["aoa"])
                    cd_corrected[i, :, j] = PchipInterpolator(polar3d.alpha, polar3d.cd)(inputs["aoa"])
                    cm_corrected[i, :, j] = PchipInterpolator(polar3d.alpha, polar3d.cm)(inputs["aoa"])
            else:
                cl_corrected[i, :, :] = inputs["cl"][i, :, :]
                cd_corrected[i, :, :] = inputs["cd"][i, :, :]
                cm_corrected[i, :, :] = inputs["cm"][i, :, :]
        outputs["cl_corrected"] = cl_corrected
        outputs["cd_corrected"] = cd_corrected
        outputs["cm_corrected"] = cm_corrected
