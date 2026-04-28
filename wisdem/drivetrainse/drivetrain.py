import numpy as np
import openmdao.api as om

import wisdem.drivetrainse.layout as lay
import wisdem.drivetrainse.drive_structure as ds
import wisdem.drivetrainse.drive_components as dc
from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox
from wisdem.drivetrainse.generator import Generator
from wisdem.drivetrainse.converter import Converter

class DriveMaterials(om.ExplicitComponent):
    """
    This component sifts through the material database and sets the material property data structures needed in this module
    """

    def initialize(self):
        self.options.declare("n_mat")
        self.options.declare("direct", default=False)

    def setup(self):
        n_mat = self.options["n_mat"]

        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("Xt_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("Xy_mat", val=np.zeros(n_mat), units="Pa")
        self.add_input("wohler_exp_mat", val=np.zeros(n_mat))
        self.add_input("wohler_A_mat", val=np.zeros(n_mat))
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("unit_cost_mat", val=np.zeros(n_mat), units="USD/kg")
        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_discrete_input("lss_material", "steel")
        self.add_discrete_input("hss_material", "steel")
        self.add_discrete_input("hub_material", "iron")
        self.add_discrete_input("spinner_material", "carbon")
        self.add_discrete_input("bedplate_material", "steel")

        self.add_output("hub_E", val=0.0, units="Pa")
        self.add_output("hub_G", val=0.0, units="Pa")
        self.add_output("hub_rho", val=0.0, units="kg/m**3")
        self.add_output("hub_Xy", val=0.0, units="Pa")
        self.add_output("hub_wohler_exp", val=0.0)
        self.add_output("hub_wohler_A", val=0.0)
        self.add_output("hub_mat_cost", val=0.0, units="USD/kg")
        self.add_output("spinner_rho", val=0.0, units="kg/m**3")
        self.add_output("spinner_Xt", val=0.0, units="Pa")
        self.add_output("spinner_mat_cost", val=0.0, units="USD/kg")
        self.add_output("lss_E", val=0.0, units="Pa")
        self.add_output("lss_G", val=0.0, units="Pa")
        self.add_output("lss_rho", val=0.0, units="kg/m**3")
        self.add_output("lss_Xy", val=0.0, units="Pa")
        self.add_output("lss_Xt", val=0.0, units="Pa")
        self.add_output("lss_wohler_exp", val=0.0)
        self.add_output("lss_wohler_A", val=0.0)
        self.add_output("lss_cost", val=0.0, units="USD/kg")
        self.add_output("hss_E", val=0.0, units="Pa")
        self.add_output("hss_G", val=0.0, units="Pa")
        self.add_output("hss_rho", val=0.0, units="kg/m**3")
        self.add_output("hss_Xy", val=0.0, units="Pa")
        self.add_output("hss_Xt", val=0.0, units="Pa")
        self.add_output("hss_wohler_exp", val=0.0)
        self.add_output("hss_wohler_A", val=0.0)
        self.add_output("hss_cost", val=0.0, units="USD/kg")
        self.add_output("bedplate_E", val=0.0, units="Pa")
        self.add_output("bedplate_G", val=0.0, units="Pa")
        self.add_output("bedplate_rho", val=0.0, units="kg/m**3")
        self.add_output("bedplate_Xy", val=0.0, units="Pa")
        self.add_output("bedplate_mat_cost", val=0.0, units="USD/kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Convert to isotropic material
        E = np.mean(inputs["E_mat"], axis=1)
        G = np.mean(inputs["G_mat"], axis=1)
        # Take the minimum Xt in longitudinal and transversal diretion, neglect direction 3 (through the fibers)
        Xt = inputs["Xt_mat"][:, [0, 1]].min(axis=1)
        sigy = inputs["Xy_mat"]
        m = inputs["wohler_exp_mat"]
        A = inputs["wohler_A_mat"]
        rho = inputs["rho_mat"]
        cost = inputs["unit_cost_mat"]

        hub_name = discrete_inputs["hub_material"]
        spin_name = discrete_inputs["spinner_material"]
        lss_name = discrete_inputs["lss_material"]
        bed_name = discrete_inputs["bedplate_material"]
        mat_names = discrete_inputs["material_names"]

        # Get the index into the material list
        spin_imat = mat_names.index(spin_name)
        hub_imat = mat_names.index(hub_name)
        lss_imat = mat_names.index(lss_name)
        bed_imat = mat_names.index(bed_name)

        outputs["hub_E"] = E[hub_imat]
        outputs["hub_G"] = G[hub_imat]
        outputs["hub_rho"] = rho[hub_imat]
        outputs["hub_Xy"] = sigy[hub_imat]
        outputs["hub_wohler_exp"] = m[hub_imat]
        outputs["hub_wohler_A"] = A[hub_imat]
        outputs["hub_mat_cost"] = cost[hub_imat]

        outputs["spinner_rho"] = rho[spin_imat]
        outputs["spinner_Xt"] = Xt[spin_imat]
        if Xt[spin_imat] == 0.0:
            raise Exception(
                "The tensile strength of the composite used in the rotor hub spinner is zero. Please check your input file."
            )
        outputs["spinner_mat_cost"] = cost[spin_imat]

        outputs["lss_E"] = E[lss_imat]
        outputs["lss_G"] = G[lss_imat]
        outputs["lss_rho"] = rho[lss_imat]
        outputs["lss_Xy"] = sigy[lss_imat]
        outputs["lss_Xt"] = Xt[lss_imat]
        if Xt[lss_imat] == 0.0:
            raise Exception(
                "The tensile strength of the material used in the low speed shaft is zero. Please check your input file."
            )
        outputs["lss_wohler_exp"] = m[lss_imat]
        outputs["lss_wohler_A"] = A[lss_imat]
        outputs["lss_cost"] = cost[lss_imat]

        outputs["bedplate_E"] = E[bed_imat]
        outputs["bedplate_G"] = G[bed_imat]
        outputs["bedplate_rho"] = rho[bed_imat]
        outputs["bedplate_Xy"] = sigy[bed_imat]
        outputs["bedplate_mat_cost"] = cost[bed_imat]

        if not self.options["direct"]:
            hss_name = discrete_inputs["hss_material"]
            hss_imat = mat_names.index(hss_name)
            outputs["hss_E"] = E[hss_imat]
            outputs["hss_G"] = G[hss_imat]
            outputs["hss_rho"] = rho[hss_imat]
            outputs["hss_Xy"] = sigy[hss_imat]
            outputs["hss_Xt"] = Xt[hss_imat]
            if Xt[hss_imat] == 0.0:
                raise Exception(
                    "The tensile strength of the material used in the high speed shaft is zero. Please check your input file."
                )
            outputs["hss_wohler_exp"] = m[hss_imat]
            outputs["hss_wohler_A"] = A[hss_imat]
            outputs["hss_cost"] = cost[hss_imat]


# ----------------------------------------------------------------------------------------------
class DrivetrainSE(om.Group):
    """
    DirectDriveSE defines an OpenMDAO group that represents a wind turbine drivetrain without a gearbox and two main bearings.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]["WISDEM"]["DriveSE"]
        n_dlcs = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        direct = opt["direct"]
        dogen = self.options["modeling_options"]["flags"]["generator"]
        n_pc = self.options["modeling_options"]["WISDEM"]["RotorSE"]["n_pc"]

        self.set_input_defaults("machine_rating", units="kW")
        #self.set_input_defaults("planet_numbers", [3, 3, 0])
        #self.set_input_defaults("gear_configuration", "eep")
        self.set_input_defaults("hvac_mass_coeff", 0.025, units="kg/kW/m")
        # self.set_input_defaults('mb1Type', 'CARB')
        # self.set_input_defaults('mb2Type', 'SRB')
        #self.set_input_defaults("uptower", True)
        #self.set_input_defaults("upwind", True)
        #self.set_input_defaults("n_blades", 3)
       
        # Materials prep
        self.add_subsystem(
            "mat",
            DriveMaterials(direct=direct, n_mat=self.options["modeling_options"]["materials"]["n_mat"]),
            promotes=["*"],
        )

        # Need to do these first, before the layout
        self.add_subsystem("hub", Hub_System(modeling_options=opt["hub"]), promotes=["*"])
        self.add_subsystem("gear", Gearbox(direct_drive=direct, gearbox_torque_density = opt["gearbox_torque_density"]), promotes=["*"])

        # Layout and mass for the big items
        if direct:
            self.add_subsystem("layout", lay.DirectLayout(), promotes=["*"])
        else:
            self.add_subsystem("layout", lay.GearedLayout(), promotes=["*"])

        # All the smaller items
        self.add_subsystem("bear1", dc.MainBearing())
        self.add_subsystem("bear2", dc.MainBearing())
        self.add_subsystem("brake", dc.Brake(direct_drive=direct), promotes=["*"])
        self.add_subsystem("elec", dc.Electronics(), promotes=["*"])
        self.add_subsystem("yaw", dc.YawSystem(), promotes=["yaw_mass", "yaw_system_mass_user", "yaw_I", "yaw_cm", "rotor_diameter", "D_top"])

        # Generator
        self.add_subsystem("rpm", dc.RPM_Input(n_pc=n_pc), promotes=["*"])
        if dogen:
            gentype = self.options["modeling_options"]["WISDEM"]["DriveSE"]["generator"]["type"]
            self.add_subsystem(
                "generator",
                Generator(design=gentype, n_pc=n_pc),
                promotes=[
                    "generator_mass_user",
                    "generator_mass",
                    "generator_cost",
                    "generator_I",
                    "machine_rating",
                    "generator_efficiency",
                    "rated_torque",
                    ("rotor_mass", "generator_rotor_mass"),
                    ("rotor_I", "generator_rotor_I"),
                    ("stator_mass", "generator_stator_mass"),
                    ("stator_I", "generator_stator_I"),
                ],
            )
        else:
            self.add_subsystem("gensimp", dc.GeneratorSimple(direct_drive=direct, n_pc=n_pc), promotes=["*"])

        # Final tallying
        self.add_subsystem("misc", dc.MiscNacelleComponents(direct_drive=direct), promotes=["*"])
        self.add_subsystem("nac", dc.NacelleSystemAdder(direct_drive=direct), promotes=["*"])
        self.add_subsystem("rna", dc.RNA_Adder(), promotes=["*"])

        # Structural analysis
        self.add_subsystem(
            "lss", ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt, direct_drive=direct), promotes=["*"]
        )
        if direct:
            self.add_subsystem(
                "nose", ds.Nose_Stator_Bedplate_Frame(modeling_options=opt, n_dlcs=n_dlcs), promotes=["*"]
            )
        else:
            self.add_subsystem("hss", ds.HSS_Frame(modeling_options=opt, n_dlcs=n_dlcs), promotes=["*"])
            self.add_subsystem("bed", ds.Bedplate_IBeam_Frame(modeling_options=opt, n_dlcs=n_dlcs), promotes=["*"])

        # Dynamics
        self.add_subsystem("dyn", dc.DriveDynamics(), promotes=["*"])

        # Converter costs and efficiency
        self.add_subsystem("converter", Converter(), promotes=["machine_rating"])

        # Output-to-input connections
        self.connect("bedplate_rho", ["pitch_system.rho", "spinner.metal_rho"])
        self.connect("bedplate_Xy", ["pitch_system.Xy", "spinner.Xy"])
        self.connect("bedplate_mat_cost", "spinner.metal_cost")
        self.connect("hub_rho", ["hub_shell.rho", "rho_castiron"])
        self.connect("hub_Xy", "hub_shell.Xy")
        self.connect("hub_mat_cost", "hub_shell.metal_cost")
        self.connect("spinner_rho", ["spinner.composite_rho", "rho_fiberglass"])
        self.connect("spinner_Xt", "spinner.composite_Xt")
        self.connect("spinner_mat_cost", "spinner.composite_cost")

        if direct:
            self.connect("D_bearing1", "bear1.D_bearing")
            self.connect("D_bearing2", "bear2.D_bearing")
        self.connect("D_shaft_mb1", "bear1.D_shaft")
        self.connect("D_shaft_mb2", "bear2.D_shaft")

        self.connect("bear1.mb_mass", "mb1_mass")
        self.connect("bear1.mb_I", "mb1_I")
        self.connect("bear1.mb_max_defl_ang", "mb1_max_defl_ang")
        self.connect("s_mb1", "mb1_cm")
        self.connect("bear2.mb_mass", "mb2_mass")
        self.connect("bear2.mb_I", "mb2_I")
        self.connect("bear2.mb_max_defl_ang", "mb2_max_defl_ang")
        self.connect("s_mb2", "mb2_cm")
        self.connect("bedplate_rho", "yaw.rho")
        self.connect("s_gearbox", "gearbox_cm")
        self.connect("s_generator", "generator_cm")

        if dogen:
            self.connect("generator.R_out", "R_generator")
            self.connect("bedplate_E", "generator.E")
            self.connect("bedplate_G", "generator.G")

            if direct:
                self.connect("lss_rpm", "generator.shaft_rpm")
                self.connect("torq_deflection", "generator.y_sh")
                self.connect("torq_angle", "generator.theta_sh")
                self.connect("stator_deflection", "generator.y_bd")
                self.connect("stator_angle", "generator.theta_bd")

                self.linear_solver = lbgs = om.LinearBlockGS()
                self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
                nlbgs.options["maxiter"] = 3
                nlbgs.options["atol"] = nlbgs.options["atol"] = 1e-2
                nlbgs.options["iprint"] = 0
            else:
                self.connect("hss_rpm", "generator.shaft_rpm")
