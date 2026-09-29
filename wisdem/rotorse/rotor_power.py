"""
Script that computes the regulation trajectory of the rotor and the annual energy production

Nikhar J. Abbas, Pietro Bortolotti
January 2020
"""

import logging

import numpy as np
from openmdao.api import Group, ExplicitComponent
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.interpolate import PchipInterpolator

from wisdem.ccblade.Polar import Polar
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
from wisdem.commonse.utilities import smooth_abs, smooth_min, linspace_with_deriv
from wisdem.commonse.distribution import WeibullWithMeanCDF

logger = logging.getLogger("wisdem/weis")
TOL = 1e-3


class RotorPower(Group):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem(
            "powercurve",
            RegulatedPowerCurve(modeling_options=modeling_options),
            promotes=[
                "v_min",
                "v_max",
                "rated_power",
                "omega_min",
                "omega_max",
                "max_allowable_blade_tip_speed",
                "tsr_operational",
                "control_pitch",
                "drivetrainType",
                "r",
                "chord",
                "theta",
                "Rhub",
                "Rtip",
                "hub_height",
                "precone",
                "tilt",
                "yaw",
                "precurve",
                "precurveTip",
                "presweep",
                "presweepTip",
                "airfoils_aoa",
                "airfoils_Re",
                "airfoils_cl",
                "airfoils_cd",
                "airfoils_cm",
                "nBlades",
                "rho",
                "mu",
                "shearExp",
                "hubloss",
                "tiploss",
                "wakerotation",
                "usecd",
                "nSector",
            ],
        )
        self.add_subsystem("gust", GustETM(std=modeling_options["WISDEM"]["RotorSE"]["gust_std"]))
        self.add_subsystem("cdf", WeibullWithMeanCDF(nspline=modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"]))
        self.add_subsystem("aep", AEP(nspline=modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"]), promotes=["AEP"])

        # Connections to the gust calculation
        self.connect("powercurve.rated_V", "gust.V_hub")

        # Connections to the Weibull CDF
        self.connect("powercurve.V_spline", "cdf.x")

        # Connections to the aep computation component
        self.connect("cdf.F", "aep.CDF_V")
        self.connect("powercurve.P_spline", "aep.P")


class GustETM(ExplicitComponent):
    # OpenMDAO component that generates an "equivalent gust" wind speed by summing an user-defined wind speed at hub height with 3 times sigma. sigma is the turbulent wind speed standard deviation for the extreme turbulence model, see IEC-61400-1 Eq. 19 paragraph 6.3.2.3

    def initialize(self):
        # number of standard deviations for strength of gust
        self.options.declare("std", default=3.0)

    def setup(self):
        # Inputs
        self.add_input("V_mean", val=0.0, units="m/s", desc="IEC average wind speed for turbine class")
        self.add_input("V_hub", val=0.0, units="m/s", desc="hub height wind speed")
        self.add_discrete_input("turbulence_class", val="A", desc="IEC turbulence class")

        # Output
        self.add_output("V_gust", val=0.0, units="m/s", desc="gust wind speed")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        V_mean = float(inputs["V_mean"][0])
        V_hub = float(inputs["V_hub"][0])
        std = self.options["std"]
        turbulence_class = discrete_inputs["turbulence_class"]

        if turbulence_class.upper() == "A":
            Iref = 0.16
        elif turbulence_class.upper() == "B":
            Iref = 0.14
        elif turbulence_class.upper() == "C":
            Iref = 0.12
        else:
            raise ValueError("Unknown Turbulence Class: " + str(turbulence_class) + " . Permitted values are A / B / C")

        c = 2.0
        sigma = c * Iref * (0.072 * (V_mean / c + 3) * (V_hub / c - 4) + 10)
        V_gust = V_hub + std * sigma
        outputs["V_gust"] = V_gust


class RegulatedPowerCurve(Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem("compute_power_curve", ComputePowerCurve(modeling_options=modeling_options), promotes=["*"])

        self.add_subsystem("compute_splines", ComputeSplines(modeling_options=modeling_options), promotes=["*"])


class ComputePowerCurve(ExplicitComponent):
    """
    Iteratively call CCBlade to compute the power curve.
    """

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("debug", default=False)

    def setup(self):
        modeling_options = self.options["modeling_options"]
        self.n_span = n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        self.n_aoa = n_aoa = modeling_options["WISDEM"]["RotorSE"]["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = modeling_options["WISDEM"]["RotorSE"]["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.regulation_reg_III = modeling_options["WISDEM"]["RotorSE"]["regulation_reg_III"]
        self.fix_pitch_regI12 = modeling_options["WISDEM"]["RotorSE"]["fix_pitch_regI12"]
        self.n_pc = modeling_options["WISDEM"]["RotorSE"]["n_pc"]
        self.n_pc_spline = modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"]

        # parameters
        self.add_input("v_min", val=0.0, units="m/s", desc="cut-in wind speed")
        self.add_input("v_max", val=0.0, units="m/s", desc="cut-out wind speed")
        self.add_input("rated_power", val=0.0, units="W", desc="electrical rated power")
        self.add_input("omega_min", val=0.0, units="rpm", desc="minimum allowed rotor rotation speed")
        self.add_input("omega_max", val=0.0, units="rpm", desc="maximum allowed rotor rotation speed")
        self.add_input("max_allowable_blade_tip_speed", val=0.0, units="m/s", desc="maximum allowed blade tip speed")
        self.add_input("tsr_operational", val=0.0, desc="tip-speed ratio in Region 2 (should be optimized externally)")
        self.add_input(
            "control_pitch",
            val=0.0,
            units="deg",
            desc="pitch angle in region 2 (and region 3 for fixed pitch machines)",
        )
        self.add_input("peak_thrust_shaving", val=1.0, desc="Scalar applied to the max torque within RotorSE for peak thrust shaving. Only used if `peak_thrust_shaving` is True.")

        self.add_discrete_input("drivetrainType", val="GEARED")
        self.add_input("gearbox_efficiency", val=1.0)
        self.add_input(
            "generator_efficiency",
            val=np.ones(self.n_pc),
            desc="Generator efficiency at various rpm values to support table lookup",
        )
        self.add_input(
            "lss_rpm",
            val=np.zeros(self.n_pc),
            units="rpm",
            desc="Low speed shaft RPM values at which the generator efficiency values are given",
        )

        self.add_input(
            "r",
            val=np.zeros(n_span),
            units="m",
            desc="radial locations where blade is defined (should be increasing and not go all the way to hub or tip)",
        )
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input(
            "theta",
            val=np.zeros(n_span),
            units="deg",
            desc="twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input("Rhub", val=0.0, units="m", desc="hub radius")
        self.add_input("Rtip", val=0.0, units="m", desc="tip radius")
        self.add_input("hub_height", val=0.0, units="m", desc="hub height")
        self.add_input("precone", val=0.0, units="deg", desc="precone angle")
        self.add_input("tilt", val=0.0, units="deg", desc="shaft tilt")
        self.add_input("yaw", val=0.0, units="deg", desc="yaw error")
        self.add_input("precurve", val=np.zeros(n_span), units="m", desc="precurve at each section")
        self.add_input("precurveTip", val=0.0, units="m", desc="precurve at tip")
        self.add_input("presweep", val=np.zeros(n_span), units="m", desc="presweep at each section")
        self.add_input("presweepTip", val=0.0, units="m", desc="presweep at tip")

        # self.add_discrete_input('airfoils',  val=[0]*n_span,                      desc='CCAirfoil instances')
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re)), desc="lift coefficients, spanwise")
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re)), desc="drag coefficients, spanwise")
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re)), desc="moment coefficients, spanwise")
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg", desc="angle of attack grid for polars")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)), desc="Reynolds numbers of polars")
        self.add_discrete_input("nBlades", val=0, desc="number of blades")
        self.add_input("rho", val=1.225, units="kg/m**3", desc="density of air")
        self.add_input("mu", val=1.81e-5, units="kg/(m*s)", desc="dynamic viscosity of air")
        self.add_input("shearExp", val=0.0, desc="shear exponent")
        self.add_discrete_input(
            "nSector", val=4, desc="number of sectors to divide rotor face into in computing thrust and power"
        )
        self.add_discrete_input("tiploss", val=True, desc="include Prandtl tip loss model")
        self.add_discrete_input("hubloss", val=True, desc="include Prandtl hub loss model")
        self.add_discrete_input(
            "wakerotation",
            val=True,
            desc="include effect of wake rotation (i.e., tangential induction factor is nonzero)",
        )
        self.add_discrete_input("usecd", val=True, desc="use drag coefficient in computing induction factors")

        # outputs
        self.add_output("V", val=np.zeros(self.n_pc), units="m/s", desc="wind vector")
        self.add_output("Omega", val=np.zeros(self.n_pc), units="rpm", desc="rotor rotational speed")
        self.add_output("pitch", val=np.zeros(self.n_pc), units="deg", desc="rotor pitch schedule")
        self.add_output("P", val=np.zeros(self.n_pc), units="W", desc="rotor electrical power")
        self.add_output("P_aero", val=np.zeros(self.n_pc), units="W", desc="rotor mechanical power")
        self.add_output("T", val=np.zeros(self.n_pc), units="N", desc="rotor aerodynamic thrust")
        self.add_output("Q", val=np.zeros(self.n_pc), units="N*m", desc="rotor aerodynamic torque")
        self.add_output("M", val=np.zeros(self.n_pc), units="N*m", desc="blade root moment")
        self.add_output("Cp", val=np.zeros(self.n_pc), desc="rotor electrical power coefficient")
        self.add_output("Cp_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic power coefficient")
        self.add_output("Ct_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic thrust coefficient")
        self.add_output("Cq_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic torque coefficient")
        self.add_output("Cm_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic moment coefficient")
        self.add_output("ax_induct_rotor", val=np.zeros(self.n_pc), desc="rotor aerodynamic induction")

        self.add_output("V_R25", val=0.0, units="m/s", desc="region 2.5 transition wind speed")
        self.add_output("rated_V", val=0.0, units="m/s", desc="rated wind speed")
        self.add_output("rated_Omega", val=0.0, units="rpm", desc="rotor rotation speed at rated")
        self.add_output("rated_pitch", val=0.0, units="deg", desc="pitch setting at rated")
        self.add_output("rated_T", val=0.0, units="N", desc="rotor aerodynamic thrust at rated")
        self.add_output("rated_Q", val=0.0, units="N*m", desc="rotor aerodynamic torque at rated")
        self.add_output("rated_mech", val=0.0, units="W", desc="Mechanical shaft power at rated")
        self.add_output(
            "ax_induct_regII", val=np.zeros(n_span), desc="rotor axial induction at cut-in wind speed along blade span"
        )
        self.add_output(
            "tang_induct_regII",
            val=np.zeros(n_span),
            desc="rotor tangential induction at cut-in wind speed along blade span",
        )
        self.add_output(
            "aoa_regII",
            val=np.zeros(n_span),
            units="deg",
            desc="angle of attack distribution along blade span at cut-in wind speed",
        )
        self.add_output(
            "L_D",
            val=np.zeros(n_span),
            desc="Lift over drag distribution along blade span at cut-in wind speed",
        )
        self.add_output("Cp_regII", val=0.0, desc="power coefficient at cut-in wind speed")
        self.add_output("Ct_regII", val=0.0, desc="thrust coefficient at cut-in wind speed")
        self.add_output(
            "cl_regII", val=np.zeros(n_span), desc="lift coefficient distribution along blade span at cut-in wind speed"
        )
        self.add_output(
            "cd_regII", val=np.zeros(n_span), desc="drag coefficient distribution along blade span at cut-in wind speed"
        )
        self.add_output("rated_efficiency", val=1.0, desc="Efficiency at rated conditions")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Saving out inputs for easy debugging of troublesome cases
        if self.options["debug"]:
            np.savez(
                "debug.npz",
                v_min=inputs["v_min"],
                v_max=inputs["v_max"],
                rated_power=inputs["rated_power"],
                omega_min=inputs["omega_min"],
                omega_max=inputs["omega_max"],
                max_allowable_blade_tip_speed=inputs["max_allowable_blade_tip_speed"],
                peak_thrust_shaving=inputs["peak_thrust_shaving"],
                tsr_operational=inputs["tsr_operational"],
                control_pitch=inputs["control_pitch"],
                gearbox_efficiency=inputs["gearbox_efficiency"],
                generator_efficiency=inputs["generator_efficiency"],
                lss_rpm=inputs["lss_rpm"],
                r=inputs["r"],
                chord=inputs["chord"],
                theta=inputs["theta"],
                Rhub=inputs["Rhub"],
                Rtip=inputs["Rtip"],
                hub_height=inputs["hub_height"],
                precone=inputs["precone"],
                tilt=inputs["tilt"],
                yaw=inputs["yaw"],
                precurve=inputs["precurve"],
                precurveTip=inputs["precurveTip"],
                presweep=inputs["presweep"],
                presweepTip=inputs["presweepTip"],
                airfoils_cl=inputs["airfoils_cl"],
                airfoils_cd=inputs["airfoils_cd"],
                airfoils_cm=inputs["airfoils_cm"],
                airfoils_aoa=inputs["airfoils_aoa"],
                airfoils_Re=inputs["airfoils_Re"],
                rho=inputs["rho"],
                mu=inputs["mu"],
                shearExp=inputs["shearExp"],
                nBlades=discrete_inputs["nBlades"],
            )
        # Create Airfoil class instances
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :],
                inputs["airfoils_cd"][i, :, :],
                inputs["airfoils_cm"][i, :, :],
            )

        self.ccblade = CCBlade(
            inputs["r"],
            inputs["chord"],
            inputs["theta"],
            af,
            inputs["Rhub"][0],
            inputs["Rtip"][0],
            discrete_inputs["nBlades"],
            inputs["rho"][0],
            inputs["mu"][0],
            inputs["precone"][0],
            inputs["tilt"][0],
            inputs["yaw"][0],
            inputs["shearExp"][0],
            inputs["hub_height"][0],
            discrete_inputs["nSector"],
            inputs["precurve"],
            inputs["precurveTip"][0],
            inputs["presweep"],
            inputs["presweepTip"][0],
            discrete_inputs["tiploss"],
            discrete_inputs["hubloss"],
            discrete_inputs["wakerotation"],
            discrete_inputs["usecd"],
        )

        # JPJ: what is this grid for? Seems to be a special distribution of velocities
        # for the hub
        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi / 4.0, np.pi / 2.0, self.n_pc)))))
        grid1 = (grid0 - grid0[0]) / (grid0[-1] - grid0[0])
        Uhub = grid1 * (inputs["v_max"][0] - inputs["v_min"][0]) + inputs["v_min"][0]

        P_aero = np.zeros(Uhub.shape)
        Cp_aero = np.zeros(Uhub.shape)
        Ct_aero = np.zeros(Uhub.shape)
        Cq_aero = np.zeros(Uhub.shape)
        Cm_aero = np.zeros(Uhub.shape)
        P = np.zeros(Uhub.shape)
        Cp = np.zeros(Uhub.shape)
        T = np.zeros(Uhub.shape)
        Q = np.zeros(Uhub.shape)
        M = np.zeros(Uhub.shape)
        min_pitch = inputs["control_pitch"][0]
        pitch = np.zeros(Uhub.shape) + min_pitch

        # Unpack variables
        P_rated = float(inputs["rated_power"][0])
        # Rtip is passed without cone to CCBlade, which multiplies it by the cos(cone) internally, 
        # but it is also multiplied by cos(cone) to be used correctly in this script
        Rtip_cone = float(inputs["Rtip"][0]) * np.cos(np.deg2rad(float(inputs["precone"][0])))
        tsr = float(inputs["tsr_operational"][0])
        driveType = discrete_inputs["drivetrainType"]

        ## POWERCURVE PRELIMS ##
        # Set rotor speed based on TSR
        Omega_tsr = Uhub * tsr / Rtip_cone

        # Determine maximum rotor speed (rad/s)- either by TS or by control input
        if inputs["max_allowable_blade_tip_speed"][0] > 0.0 and inputs["omega_max"][0] > 0.0:
            Omega_max = min([inputs["max_allowable_blade_tip_speed"][0] / Rtip_cone,
                         float(inputs["omega_max"][0]) * np.pi / 30.0])
        elif inputs["omega_max"][0] > 0.0:
            Omega_max = float(inputs["omega_max"][0]) * np.pi / 30.0
        elif inputs["max_allowable_blade_tip_speed"][0] > 0.0:
            Omega_max = inputs["max_allowable_blade_tip_speed"][0] / Rtip_cone
        else:
            Omega_max = np.inf

        # Apply maximum and minimum rotor speed limits
        Omega_min = float(inputs["omega_min"][0]) * np.pi / 30.0
        Omega = np.maximum(np.minimum(Omega_tsr, Omega_max), Omega_min)
        Omega_rpm = Omega * 30.0 / np.pi

        # Create table lookup of total drivetrain efficiency, where rpm is first column and second column is gearbox*generator
        lss_rpm = inputs["lss_rpm"]
        gen_eff = inputs["generator_efficiency"]
        if not np.any(lss_rpm):
            lss_rpm = np.linspace(np.maximum(0.1, Omega_rpm[0]), Omega_rpm[-1], self.n_pc - 1)
            _, gen_eff = compute_P_and_eff(
                P_rated * lss_rpm / lss_rpm[-1],
                P_rated,
                np.zeros(self.n_pc - 1),
                driveType,
                np.zeros((self.n_pc - 1, 2)),
            )

        # driveEta  = np.c_[lss_rpm, float(inputs['gearbox_efficiency'])*gen_eff]
        driveEta = float(inputs["gearbox_efficiency"][0]) * gen_eff

        # Set baseline power production
        myout, derivs = self.ccblade.evaluate(Uhub, Omega_tsr * 30.0 / np.pi, pitch, coefficients=True)
        P_aero, T, Q, M, Cp_aero, Ct_aero, Cq_aero, Cm_aero = [
            myout[key] for key in ["P", "T", "Q", "Mb", "CP", "CT", "CQ", "CMb"]
        ]
        # P, eff  = compute_P_and_eff(P_aero, P_rated, Omega_rpm, driveType, driveEta)
        eff = np.interp(Omega_rpm, lss_rpm, driveEta)
        P = P_aero * eff
        Cp = Cp_aero * eff

        # Find rated index and guess at rated speed
        if P_aero[-1] > P_rated:
            U_rated = np.interp(P_rated, P, Uhub)
            i_rated = np.nonzero(U_rated <= Uhub)[0][0]
        else:
            U_rated = Uhub[-1] + 1e-6

        # Find Region 3 index
        found_rated = P_aero[-1] > P_rated
        region3 = len(np.nonzero(P >= P_rated)[0]) > 0

        # Guess at Region 2.5, but we will do a more rigorous search below
        if Omega_max < Omega_tsr[-1]:
            U_2p5 = np.interp(Omega[-1], Omega_tsr, Uhub)
            outputs["V_R25"] = U_2p5
        else:
            U_2p5 = U_rated
        region2p5 = U_2p5 < U_rated

        # Initialize peak shaving thrust value, will be updated later
        if inputs["peak_thrust_shaving"][0] < 1.:
            peak_thrust_shaving = True
            pts = inputs["peak_thrust_shaving"][0]
        else:
            peak_thrust_shaving = False
        max_T = pts * T.max() if peak_thrust_shaving and found_rated else 1e16

        ## REGION II.5 and RATED ##
        # Solve for rated velocity
        if found_rated:
            i = i_rated

            def const_Urated(x):
                pitch_i = x[0]
                Uhub_i = x[1]
                Omega_i = min([Uhub_i * tsr / Rtip_cone, Omega_max])
                Omega_i_rpm = Omega_i * 30.0 / np.pi
                myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_i_rpm], [pitch_i], coefficients=False)
                P_aero_i = float(myout["P"][0])
                # P_i,_  = compute_P_and_eff(P_aero_i.flatten(), P_rated, Omega_i_rpm, driveType, driveEta)
                eff_i = np.interp(Omega_i_rpm, lss_rpm, driveEta)
                if isinstance(eff_i, type(np.array([]))): eff_i = eff_i[0]
                P_i = float(P_aero_i * eff_i)
                return 1e-4 * (P_i - P_rated)

            if region2p5:
                # Have to search over both pitch and speed
                x0 = [0.0, U_rated]
                imin = max(i - 3, 0)
                imax = min(i + 2, len(Uhub) - 1)
                bnds = [[min_pitch, 15.0], [Uhub[imin] + TOL, Uhub[imax] - TOL]]
                const = {}
                const["type"] = "eq"
                const["fun"] = const_Urated
                params_rated = minimize(
                    lambda x: x[1], x0, method="slsqp", bounds=bnds, constraints=const, tol=TOL, options={"maxiter": 20, "disp": False}
                )

                if params_rated.success and not np.isnan(params_rated.x[1]):
                    U_rated = params_rated.x[1]
                    pitch_rated = params_rated.x[0]
                else:
                    U_rated = U_rated  # Use guessed value earlier
                    pitch_rated = min_pitch
            else:
                # Just search over speed
                pitch_rated = min_pitch
                i_Uhub = np.max([0, i - 2])
                try:
                    U_rated = brentq(
                        lambda x: const_Urated([0.0, x]),
                        Uhub[i_Uhub],
                        Uhub[i + 2],
                        xtol=1e-1 * TOL,
                        rtol=1e-2 * TOL,
                        maxiter=40,
                        disp=False,
                    )
                except ValueError:
                    U_rated = minimize_scalar(
                        lambda x: np.abs(const_Urated([0.0, x])),
                        bounds=[Uhub[i_Uhub], Uhub[i + 2]],
                        method="bounded",
                        options={"disp": False, "xatol": TOL, "maxiter": 40},
                    )["x"]

            Omega_tsr_rated = U_rated * tsr / Rtip_cone
            Omega_rated = np.minimum(Omega_tsr_rated, Omega_max)
            Omega_rpm_rated = Omega_rated * 30.0 / np.pi
            myout, _ = self.ccblade.evaluate([U_rated], [Omega_rpm_rated], [pitch_rated], coefficients=True)
            (
                P_aero_rated,
                T_rated,
                Q_rated,
                M_rated,
                Cp_aero_rated,
                Ct_aero_rated,
                Cq_aero_rated,
                Cm_aero_rated,
            ) = [float(myout[key][0]) for key in ["P", "T", "Q", "Mb", "CP", "CT", "CQ", "CMb"]]
            eff_rated = np.interp(Omega_rpm_rated, lss_rpm, driveEta)
            Cp_rated = Cp_aero_rated * eff_rated
            P_rated = P_rated

            ## REGION II.5 and RATED with peak shaving##
            if peak_thrust_shaving:
                max_T = pts * T_rated

                def const_Urated_Tpeak(x):
                    pitch_i = x[0]
                    Uhub_i = x[1]
                    Omega_i = min([Uhub_i * tsr / Rtip_cone, Omega_max])
                    Omega_i_rpm = Omega_i * 30.0 / np.pi
                    myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_i_rpm], [pitch_i], coefficients=False)
                    P_aero_i = float(myout["P"][0])
                    # P_i,_  = compute_P_and_eff(P_aero_i.flatten(), P_rated, Omega_i_rpm, driveType, driveEta)
                    eff_i = np.interp(Omega_i_rpm, lss_rpm, driveEta)
                    if isinstance(eff_i, type(np.array([]))): eff_i = eff_i[0]
                    P_i = float(P_aero_i * eff_i)
                    T_i = float(myout["T"][0])
                    return 1e-4 * (P_i - P_rated), 1e-4 * (T_i - max_T)

                # Have to search over both pitch and speed
                x0 = [0.0, U_rated]
                i_Uhub = np.max([0, i - 2])
                bnds = [[min_pitch, 15.0], [Uhub[i_Uhub] + TOL, Uhub[-1] - TOL]]
                const = {}
                const["type"] = "eq"
                const["fun"] = const_Urated_Tpeak
                params_rated = minimize(
                    lambda x: x[1], x0, method="slsqp", bounds=bnds, constraints=const, tol=TOL, options={"maxiter": 20, "disp": False}
                )

                if params_rated.success and not np.isnan(params_rated.x[1]):
                    U_rated = params_rated.x[1]
                    pitch_rated = params_rated.x[0]
                else:
                    U_rated = U_rated  # Use guessed value earlier
                    pitch_rated = min_pitch

                Omega_tsr_rated = U_rated * tsr / Rtip_cone
                Omega_rated = np.minimum(Omega_tsr_rated, Omega_max)
                Omega_rpm_rated = Omega_rated * 30.0 / np.pi
                myout, _ = self.ccblade.evaluate([U_rated], [Omega_rpm_rated], [pitch_rated], coefficients=True)
                (
                    P_aero_rated,
                    T_rated,
                    Q_rated,
                    M_rated,
                    Cp_aero_rated,
                    Ct_aero_rated,
                    Cq_aero_rated,
                    Cm_aero_rated,
                ) = [float(myout[key][0]) for key in ["P", "T", "Q", "Mb", "CP", "CT", "CQ", "CMb"]]
                eff_rated = np.interp(Omega_rpm_rated, lss_rpm, driveEta)
                Cp_rated = Cp_aero_rated * eff_rated
                P_rated = P_rated

        else:
            # No rated conditions, so just assume last values
            U_rated = Uhub[-1] + 1e-6
            Omega_tsr_rated = Omega_tsr[-1]
            Omega_rated = Omega[-1]
            Omega_rpm_rated = Omega_rpm[-1]
            pitch_rated = pitch[-1]
            P_aero_rated = P_aero[-1]
            P_rated = P[-1]
            T_rated = T[-1]
            Q_rated = Q[-1]
            M_rated = M[-1]
            Cp_rated = Cp[-1]
            Cp_aero_rated = Cp_aero[-1]
            Ct_aero_rated = Ct_aero[-1]
            Cq_aero_rated = Cq_aero[-1]
            Cm_aero_rated = Cm_aero[-1]
            eff_rated = eff[-1]

        # Store rated speed in array
        Uhub = np.r_[Uhub, U_rated]
        isort = np.argsort(Uhub)
        Uhub = Uhub[isort]

        Omega_tsr = np.r_[Omega_tsr, Omega_tsr_rated][isort]
        Omega = np.r_[Omega, Omega_rated][isort]
        Omega_rpm = np.r_[Omega_rpm, Omega_rpm_rated][isort]
        pitch = np.r_[pitch, pitch_rated][isort]
        P_aero = np.r_[P_aero, P_aero_rated][isort]
        P = np.r_[P, P_rated][isort]
        T = np.r_[T, T_rated][isort]
        Q = np.r_[Q, Q_rated][isort]
        M = np.r_[M, M_rated][isort]
        Cp = np.r_[Cp, Cp_rated][isort]
        Cp_aero = np.r_[Cp_aero, Cp_aero_rated][isort]
        Ct_aero = np.r_[Ct_aero, Ct_aero_rated][isort]
        Cq_aero = np.r_[Cq_aero, Cq_aero_rated][isort]
        Cm_aero = np.r_[Cm_aero, Cm_aero_rated][isort]
        eff = np.r_[eff, eff_rated][isort]

        i_rated = np.where(Uhub == U_rated)[0][0]
        i_3 = np.minimum(i_rated + 1, self.n_pc)

        # Set rated conditions for rest of powercurve
        Omega[i_rated:] = Omega_rated  # Stay at this speed if hit rated too early
        Omega_rpm = Omega * 30.0 / np.pi

        ## REGION II ##
        # Functions to be used inside of power maximization until Region 3
        def maximizePower(pitch_i, Uhub_i, Omega_rpm_i, scaling_power):
            myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_rpm_i], [pitch_i], coefficients=False)
            return -myout["P"] / scaling_power

        def constr_Tmax(pitch_i, Uhub_i, Omega_rpm_i, scaling_thrust):
            myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_rpm_i], [pitch_i], coefficients=False)
            return (max_T - float(myout["T"][0])) / scaling_thrust

        # Maximize power until rated
        for i in range(i_3):
            # No need to optimize if already doing well or if flag
            # fix_pitch_regI12, which locks pitch in region I 1/2, is on
            # For the 1.04 value on thrust shaving, that is about the level of compliance we see for the Region 2.5
            # points coming out of the optimization routines in the next block of code. This way, a Region 2 point that is
            # at optimal TSR-rpm and 0-deg pitch but 1% over the thrust target can be allowed to stand.
            if (
                ((Omega[i] == Omega_tsr[i]) and not peak_thrust_shaving)
                or ((Omega[i] == Omega_tsr[i]) and peak_thrust_shaving and (T[i]/max_T <= 1.04))
                or ((Omega[i] == Omega_min) and self.fix_pitch_regI12)
                or (found_rated and (i == i_rated))
            ):
                continue

            # Find pitch value that gives highest power rating
            pitch0 = pitch[i] if i == 0 else pitch[i - 1]
            bnds = [max([min_pitch, pitch0 - 10.0]), max([min_pitch, pitch0 + 10.0])]
            # For a successfull minimization, find the initial power value to nondimensionalize power and bring the figure of merit close to 1
            myout, _ = self.ccblade.evaluate(Uhub[i], Omega_rpm[i], pitch0, coefficients=False)
            # For better conditioning near cut-in, use rated values
            scaling_power = 0.1*P_rated #myout["P"]
            scaling_thrust = 0.1*T_rated #myout["T"]
            if peak_thrust_shaving and found_rated and (Omega[i] <= Omega_tsr[i]):
                # Have to constrain thrust in Region II.5
                const = {}
                const["type"] = "ineq"
                const["fun"] = lambda x: constr_Tmax(x, Uhub[i], Omega_rpm[i], scaling_thrust)
                params = minimize(
                    lambda x: maximizePower(x, Uhub[i], Omega_rpm[i], scaling_power),
                    pitch0,
                    method="slsqp", #"cobyla",
                    bounds=[bnds],
                    constraints=const,
                    tol=TOL,
                    options={"maxiter": 20, "disp": False},  #'catol':0.01*max_T},
                )
                pitch[i] = params.x[0]
            else:
                # Only adjust pitch- should be mostly region I.5 for peak shaving and also II.5 for non-peak shaving
                pitch[i] = minimize_scalar(
                    lambda x: maximizePower(x, Uhub[i], Omega_rpm[i], scaling_power),
                    bounds=bnds,
                    method="bounded",
                    options={"disp": False, "xatol": TOL, "maxiter": 40},
                )["x"][0]
                
            # Find associated power
            myout, _ = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [
                myout[key][0] for key in ["P", "T", "Q", "Mb", "CP", "CT", "CQ", "CMb"]
            ]
            # P[i], eff[i] = compute_P_and_eff(P_aero[i], P_rated, Omega_rpm[i], driveType, driveEta)
            eff[i] = np.interp(Omega_rpm[i], lss_rpm, driveEta)
            P[i] = P_aero[i] * eff[i]
            Cp[i] = Cp_aero[i] * eff[i]

        ## REGION III ##
        # JPJ: this part can be converted into a BalanceComp with a solver.
        # This will be less expensive and allow us to get derivatives through the process.
        if region3:
            # Function to be used to stay at rated power in Region 3
            def rated_power_dist(pitch_i, Uhub_i, Omega_rpm_i):
                myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_rpm_i], [pitch_i], coefficients=False)
                P_aero_i = myout["P"]
                eff_i = np.interp(Omega_rpm_i, lss_rpm, driveEta)
                P_i = P_aero_i * eff_i
                return 1e-4 * (P_i[0] - P_rated)

            # Solve for Region 3 pitch
            if self.regulation_reg_III:
                for i in range(i_3, self.n_pc):
                    pitch0 = pitch[i - 1]
                    bnds = ([pitch0, pitch0 + 15.0],)
                    try:
                        pitch[i] = brentq(
                            lambda x: rated_power_dist(x, Uhub[i], Omega_rpm[i]),
                            bnds[0][0],
                            bnds[0][1],
                            xtol=1e-1 * TOL,
                            rtol=1e-2 * TOL,
                            maxiter=40,
                            disp=False,
                        )
                    except ValueError:
                        pitch[i] = minimize_scalar(
                            lambda x: np.abs(rated_power_dist(x, Uhub[i], Omega_rpm[i])),
                            bounds=bnds[0],
                            method="bounded",
                            options={"disp": False, "xatol": TOL, "maxiter": 40},
                        )["x"]

                    myout, _ = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
                    P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [
                        myout[key][0] for key in ["P", "T", "Q", "Mb", "CP", "CT", "CQ", "CMb"]
                    ]
                    eff[i] = np.interp(Omega_rpm[i], lss_rpm, driveEta)
                    P[i] = P_aero[i] * eff[i]
                    Cp[i] = Cp_aero[i] * eff[i]
                    # P[i]        = P_rated

                    # If we are thrust shaving, then check if this is a point that must be modified
                    if peak_thrust_shaving and T[i] >= max_T:
                        myout, _ = self.ccblade.evaluate(Uhub[i], Omega_rpm[i], pitch0, coefficients=False)
                        scaling_thrust = 0.1*float(myout["T"][0])
                        const = {}
                        const["type"] = "ineq"
                        const["fun"] = lambda x: constr_Tmax(x, Uhub[i], Omega_rpm[i], scaling_thrust)
                        params = minimize(
                            lambda x: np.abs(rated_power_dist(x, Uhub[i], Omega_rpm[i])),
                            pitch0,
                            method="slsqp",
                            bounds=bnds,
                            constraints=const,
                            tol=TOL,
                            options={"maxiter": 20, "disp": False},
                        )
                        if params.success and not np.isnan(params.x[0]):
                            pitch[i] = params.x[0]

                        myout, _ = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
                        P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [
                            myout[key][0] for key in ["P", "T", "Q", "Mb", "CP", "CT", "CQ", "CMb"]
                        ]
                        eff[i] = np.interp(Omega_rpm[i], lss_rpm, driveEta)
                        P[i] = P_aero[i] * eff[i]
                        Cp[i] = Cp_aero[i] * eff[i]
                        # P[i]        = P_rated

            else:
                P[i_3:] = P_rated
                P_aero[i_3:] = P_aero[i_3 - 1]
                T[i_3:] = 0
                Q[i_3:] = P[i_3:] / Omega[i_3:]
                M[i_3:] = 0
                pitch[i_3:] = 0
                Cp[i_3:] = P[i_3:] / (0.5 * inputs["rho"][0]* np.pi * Rtip_cone**2 * Uhub[i_3:] ** 3)
                Cp_aero[i_3:] = P_aero[i_3:] / (0.5 * inputs["rho"][0] * np.pi * Rtip_cone**2 * Uhub[i_3:] ** 3)
                Ct_aero[i_3:] = 0
                Cq_aero[i_3:] = 0
                Cm_aero[i_3:] = 0

        ## END POWERCURVE ##

        # Store outputs
        outputs["T"] = T
        outputs["Q"] = Q
        outputs["Omega"] = Omega_rpm

        outputs["P"] = P
        outputs["Cp"] = Cp
        outputs["P_aero"] = P_aero
        outputs["Cp_aero"] = Cp_aero
        outputs["Ct_aero"] = Ct_aero
        outputs["Cq_aero"] = Cq_aero
        outputs["Cm_aero"] = Cm_aero
        outputs["V"] = Uhub
        outputs["M"] = M
        outputs["pitch"] = pitch

        outputs["rated_V"] = np.float64(U_rated)
        outputs["rated_Omega"] = Omega_rpm_rated
        outputs["rated_pitch"] = pitch_rated
        outputs["rated_T"] = T_rated
        outputs["rated_Q"] = Q_rated
        outputs["rated_mech"] = P_aero_rated
        outputs["rated_efficiency"] = eff_rated

        self.ccblade.induction_inflow = True
        tsr_vec = Omega_rpm / 30.0 * np.pi * Rtip_cone / Uhub
        id_regII = np.argmin(abs(tsr_vec - inputs["tsr_operational"][0]))
        ax_induct_rotor = np.zeros_like(Uhub)
        for i in range(len(Uhub)):
            loads, _ = self.ccblade.distributedAeroLoads(Uhub[i], Omega_rpm[i], pitch[i], 0.0)
            try:
                # Numpy v1/2 clash
                ax_induct_rotor[i] = 2. / inputs["r"][-1]**2. * np.trapezoid(loads['a'] * inputs["r"], inputs["r"])
            except AttributeError:
                ax_induct_rotor[i] = 2. / inputs["r"][-1]**2. * np.trapz(loads['a'] * inputs["r"], inputs["r"])
            if i == id_regII:
                # outputs
                outputs["ax_induct_regII"] = loads["a"]
                outputs["tang_induct_regII"] = loads["ap"]
                outputs["aoa_regII"] = loads["alpha"]
                outputs["cl_regII"] = loads["Cl"]
                outputs["cd_regII"] = loads["Cd"]
                outputs["L_D"] = loads["Cl"] / loads["Cd"]
                outputs["Cp_regII"] = Cp_aero[i]
                outputs["Ct_regII"] = Ct_aero[i]
        outputs["ax_induct_rotor"] = ax_induct_rotor

class ComputeSplines(ExplicitComponent):
    """
    Compute splined quantities for V, P, and Omega.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        self.n_pc = modeling_options["WISDEM"]["RotorSE"]["n_pc"]
        self.n_pc_spline = modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"]

        self.add_input("v_min", val=0.0, units="m/s", desc="cut-in wind speed")
        self.add_input("v_max", val=0.0, units="m/s", desc="cut-out wind speed")
        self.add_input("V", val=np.zeros(self.n_pc), units="m/s", desc="wind vector")
        self.add_input("Omega", val=np.zeros(self.n_pc), units="rpm", desc="rotor rotational speed")
        self.add_input("P", val=np.zeros(self.n_pc), units="W", desc="rotor electrical power")

        self.add_output("V_spline", val=np.zeros(self.n_pc_spline), units="m/s", desc="wind vector")
        self.add_output("P_spline", val=np.zeros(self.n_pc_spline), units="W", desc="rotor electrical power")
        self.add_output("Omega_spline", val=np.zeros(self.n_pc_spline), units="rpm", desc="omega")

        self.declare_partials(of="V_spline", wrt="v_min")
        self.declare_partials(of="V_spline", wrt="v_max")

        self.declare_partials(of="P_spline", wrt="v_min", method="fd")
        self.declare_partials(of="P_spline", wrt="v_max", method="fd")
        self.declare_partials(of="P_spline", wrt="V", method="fd")
        self.declare_partials(of="P_spline", wrt="P", method="fd")

        self.declare_partials(of="Omega_spline", wrt="v_min", method="fd")
        self.declare_partials(of="Omega_spline", wrt="v_max", method="fd")
        self.declare_partials(of="Omega_spline", wrt="V", method="fd")
        self.declare_partials(of="Omega_spline", wrt="Omega", method="fd")

    def compute(self, inputs, outputs):
        # Fit spline to powercurve for higher grid density
        V_spline = np.linspace(inputs["v_min"][0], inputs["v_max"][0], self.n_pc_spline)
        spline = PchipInterpolator(inputs["V"], inputs["P"])
        P_spline = spline(V_spline)
        spline = PchipInterpolator(inputs["V"], inputs["Omega"])
        Omega_spline = spline(V_spline)

        # outputs
        outputs["V_spline"] = V_spline.flatten()
        outputs["P_spline"] = P_spline.flatten()
        outputs["Omega_spline"] = Omega_spline.flatten()

    def compute_partials(self, inputs, partials):
        linspace_with_deriv
        V_spline, dy_dstart, dy_dstop = linspace_with_deriv(inputs["v_min"][0], inputs["v_max"][0], self.n_pc_spline)
        partials["V_spline", "v_min"] = dy_dstart
        partials["V_spline", "v_max"] = dy_dstop


# Class to define a constraint so that the blade cannot operate in stall conditions
class NoStallConstraint(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        self.n_span = n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        self.n_aoa = n_aoa = modeling_options["WISDEM"]["RotorSE"]["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = modeling_options["WISDEM"]["RotorSE"]["n_Re"]  # Number of Reynolds, so far hard set at 1

        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input("aoa_along_span", val=np.zeros(n_span), units="deg", desc="Angle of attack along blade span")
        self.add_input("stall_margin", val=3.0, units="deg", desc="Minimum margin from the stall angle")
        self.add_input(
            "min_s",
            val=0.25,
            desc="Minimum nondimensional coordinate along blade span where to define the constraint (blade root typically stalls)",
        )
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re)), desc="lift coefficients, spanwise")
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re)), desc="drag coefficients, spanwise")
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re)), desc="moment coefficients, spanwise")
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg", desc="angle of attack grid for polars")

        self.add_output(
            "no_stall_constraint",
            val=np.zeros(n_span),
            desc="Constraint, ratio between angle of attack plus a margin and stall angle",
        )
        self.add_output(
            "stall_angle_along_span", val=np.zeros(n_span), units="deg", desc="Stall angle along blade span"
        )

    def compute(self, inputs, outputs):
        i_min = np.argmin(np.abs(inputs["min_s"] - inputs["s"]))
        n_span = len(inputs["s"])

        for i in range(n_span):
            unsteady = eval_unsteady(
                inputs["airfoils_aoa"],
                inputs["airfoils_cl"][i, :, 0],
                inputs["airfoils_cd"][i, :, 0],
                inputs["airfoils_cm"][i, :, 0],
            )
            outputs["stall_angle_along_span"][i] = unsteady["alpha1"]
            if outputs["stall_angle_along_span"][i] == 0:
                outputs["stall_angle_along_span"][i] = 1e-6  # To avoid nan

        for i in range(i_min, n_span):
            outputs["no_stall_constraint"][i] = (inputs["aoa_along_span"][i] + inputs["stall_margin"][0]) / outputs[
                "stall_angle_along_span"
            ][i]

            if outputs["stall_angle_along_span"][i] <= 1.0e-6:
                outputs["no_stall_constraint"][i] = 0.0

            logger.debug(
                "Blade is violating the minimum margin to stall at span location %.2f %%" % (inputs["s"][i] * 100.0)
            )


class AEP(ExplicitComponent):
    def initialize(self):
        self.options.declare("nspline")

    def setup(self):
        n_pc_spline = self.options["nspline"]
        """integrate to find annual energy production"""

        # inputs
        self.add_input(
            "CDF_V",
            val=np.zeros(n_pc_spline),
            units="m/s",
            desc="cumulative distribution function evaluated at each wind speed",
        )
        self.add_input("P", val=np.zeros(n_pc_spline), units="W", desc="power curve (power)")
        self.add_input(
            "lossFactor", val=1.0, desc="multiplicative factor for availability and other losses (soiling, array, etc.)"
        )

        # outputs
        self.add_output("AEP", val=0.0, units="kW*h", desc="annual energy production")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        lossFactor = inputs["lossFactor"][0]
        P = inputs["P"]
        CDF_V = inputs["CDF_V"]

        factor = lossFactor / 1e3 * 365.0 * 24.0
        try:
            # Numpy v1/2 clash
            outputs["AEP"] = factor * np.trapezoid(P, CDF_V)  # in kWh
        except AttributeError:
            outputs["AEP"] = factor * np.trapz(P, CDF_V)  # in kWh
        """
        dAEP_dP, dAEP_dCDF = trapezoid_deriv(P, CDF_V)
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([outputs['AEP']/lossFactor])
        self.J = {}
        self.J['AEP', 'CDF_V'] = np.reshape(dAEP_dCDF, (1, len(dAEP_dCDF)))
        self.J['AEP', 'P'] = np.reshape(dAEP_dP, (1, len(dAEP_dP)))
        self.J['AEP', 'lossFactor'] = dAEP_dlossFactor

    def compute_partials(self, inputs, J):
        J = self.J
        """


def compute_P_and_eff(aeroPower, ratedPower, Omega_rpm, drivetrainType, drivetrainEff):
    if not np.any(drivetrainEff):
        drivetrainType = drivetrainType.upper()
        if drivetrainType == "GEARED":
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif drivetrainType == "SINGLE_STAGE":
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif drivetrainType == "MULTI_DRIVE":
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif drivetrainType in ["PM_DIRECT_DRIVE", "DIRECT_DRIVE", "DIRECT DRIVE"]:
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899

        elif drivetrainType == "CONSTANT_EFF":
            constant = 0.00
            linear = 0.07
            quadratic = 0.0
        else:
            raise ValueError("The drivetrain model is not supported! Please check rotor_power.py")

        Pbar0 = aeroPower / ratedPower

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant / Pbar + linear + quadratic * Pbar)
        eff = np.maximum(eff, 1e-3)
    else:
        # Use table lookup from rpm to calculate total efficiency
        eff = np.interp(Omega_rpm, drivetrainEff[:, 0], drivetrainEff[:, 1])

    return aeroPower * eff, eff


def eval_unsteady(alpha, cl, cd, cm):
    # calculate unsteady coefficients from polars for OpenFAST's Aerodyn

    unsteady = {}
    Re = 1e6  # Does not factor into any calculations
    mypolar = Polar(Re=Re, alpha=alpha, cl=cl, cd=cd, cm=cm, compute_params=True, radians=False)
    (alpha0, alpha1, alpha2, cnSlope, cn1, cn2, cd0, cm0) = mypolar.unsteadyParams()
    unsteady["alpha0"] = alpha0
    unsteady["alpha1"] = alpha1
    unsteady["alpha2"] = alpha2
    unsteady["Cd0"] = cd0
    unsteady["Cm0"] = cm0
    unsteady["Cn1"] = cn1
    unsteady["Cn2"] = cn2
    unsteady["C_nalpha"] = cnSlope
    unsteady["eta_e"] = 1
    unsteady["T_f0"] = "Default"
    unsteady["T_V0"] = "Default"
    unsteady["T_p"] = "Default"
    unsteady["T_VL"] = "Default"
    unsteady["b1"] = "Default"
    unsteady["b2"] = "Default"
    unsteady["b5"] = "Default"
    unsteady["A1"] = "Default"
    unsteady["A2"] = "Default"
    unsteady["A5"] = "Default"
    unsteady["S1"] = 0
    unsteady["S2"] = 0
    unsteady["S3"] = 0
    unsteady["S4"] = 0
    unsteady["St_sh"] = "Default"
    unsteady["k0"] = 0
    unsteady["k1"] = 0
    unsteady["k2"] = 0
    unsteady["k3"] = 0
    unsteady["k1_hat"] = 0
    unsteady["x_cp_bar"] = "Default"
    unsteady["UACutout"] = "Default"
    unsteady["filtCutOff"] = "Default"
    unsteady["Alpha"] = alpha
    unsteady["Cl"] = cl
    unsteady["Cd"] = cd
    unsteady["Cm"] = cm

    return unsteady
