import os

import numpy as np
import openmdao.api as om
from scipy.interpolate import PchipInterpolator
from wisdem.optimization_drivers.nsga2_driver import NSGA2Driver

class PoseOptimization(object):
    def __init__(self, wt_init, modeling_options, analysis_options):
        self.wt_init = wt_init
        self.modeling = modeling_options
        self.opt = analysis_options

        self.nlopt_methods = [
            "GN_DIRECT",
            "GN_DIRECT_L",
            "GN_DIRECT_L_NOSCAL",
            "GN_ORIG_DIRECT",
            "GN_ORIG_DIRECT_L",
            "GN_AGS",
            "GN_ISRES",
            "LN_COBYLA",
            "LD_MMA",
            "LD_CCSAQ",
            "LD_SLSQP",
        ]

        self.scipy_methods = [
            "SLSQP",
            "Nelder-Mead",
            "COBYLA",
        ]

        self.pyoptsparse_methods = [
            "SNOPT",
            "CONMIN",
            # "NSGA2",
        ]

        self.floating_solve_component = 'floatingse'
        self.floating_period_solve_component = 'floatingse'

    def _get_step_size(self):
        # If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.
        return (
            1.0e-6
            if not "step_size" in self.opt["driver"]["optimization"]
            else self.opt["driver"]["optimization"]["step_size"]
        )

    def _set_optimizer_properties(self, wt_opt, options_keys=[], opt_settings_keys=[], mapped_keys={}):
        """
        Set the optimizer properties, both the `driver.options` and
        `driver.opt_settings`. See OpenMDAO documentation on drivers
        to determine which settings are set by either options or
        opt_settings.

        Parameters
        ----------
        wt_opt : OpenMDAO problem object
            The wind turbine problem object.
        options_keys : list
            List of keys for driver options to be set.
        opt_settings_keys: list
            List of keys for driver opt_settings to be set.
        mapped_keys: dict
            Key pairs where the yaml name differs from what's expected
            by the driver. Specifically, the key is what's given in the yaml
            and the value is what's expected by the driver.

        Returns
        -------
        wt_opt : OpenMDAO problem object
            The updated wind turbine problem object with driver settings applied.
        """

        opt_options = self.opt["driver"]["optimization"]

        # Loop through all of the options provided and set them in the OM driver object
        for key in options_keys:
            if key in opt_options:
                if key in mapped_keys:
                    wt_opt.driver.options[mapped_keys[key]] = opt_options[key]
                else:
                    wt_opt.driver.options[key] = opt_options[key]

        # Loop through all of the opt_settings provided and set them in the OM driver object
        for key in opt_settings_keys:
            if key in opt_options:
                if key in mapped_keys:
                    wt_opt.driver.opt_settings[mapped_keys[key]] = opt_options[key]
                else:
                    wt_opt.driver.opt_settings[key] = opt_options[key]

        return wt_opt

    def set_driver(self, wt_opt):
        folder_output = self.opt["general"]["folder_output"]

        if self.opt["driver"]["optimization"]["flag"]:
            opt_options = self.opt["driver"]["optimization"]
            step_size = self._get_step_size()

            if opt_options["step_calc"] == "None":
                step_calc = None
            else:
                step_calc = opt_options["step_calc"]
            wt_opt.model.approx_totals(method="fd", step=step_size, form=opt_options["form"], step_calc=step_calc)

            # Set optimization solver and options. First, Scipy's SLSQP and COBYLA
            if opt_options["solver"] in self.scipy_methods:
                wt_opt.driver = om.ScipyOptimizeDriver()
                wt_opt.driver.options["optimizer"] = opt_options["solver"]

                options_keys = ["tol", "max_iter", "disp"]
                opt_settings_keys = ["rhobeg", "catol", "adaptive"]
                mapped_keys = {"max_iter": "maxiter"}
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys, opt_settings_keys, mapped_keys)

            # The next two optimization methods require pyOptSparse.
            elif opt_options["solver"] in self.pyoptsparse_methods:
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, but you have not installed pyOptSparse. Please do so and rerun."
                    )
                wt_opt.driver = pyOptSparseDriver()
                try:
                    wt_opt.driver.options["optimizer"] = opt_options["solver"]
                except:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, but you have not installed it within pyOptSparse. Please build {opt_options['solver']} and rerun."
                    )

                # Most of the pyOptSparse options have special syntax when setting them,
                # so here we set them by hand instead of using `_set_optimizer_properties` for SNOPT and CONMIN.
                if opt_options["solver"] == "CONMIN":
                    wt_opt.driver.opt_settings["ITMAX"] = opt_options["max_iter"]

                if opt_options["solver"] == "NSGA2":
                    opt_settings_keys = [
                        "PopSize",
                        "maxGen",
                        "pCross_real",
                        "pMut_real",
                        "eta_c",
                        "eta_m",
                        "pCross_bin",
                        "pMut_bin",
                        "PrintOut",
                        "seed",
                        "xinit",
                    ]
                    wt_opt = self._set_optimizer_properties(wt_opt, opt_settings_keys=opt_settings_keys)

                elif opt_options["solver"] == "SNOPT":
                    wt_opt.driver.opt_settings["Major optimality tolerance"] = float(opt_options["tol"])
                    wt_opt.driver.opt_settings["Major iterations limit"] = int(opt_options["max_major_iter"])
                    wt_opt.driver.opt_settings["Iterations limit"] = int(opt_options["max_minor_iter"])
                    wt_opt.driver.opt_settings["Major feasibility tolerance"] = float(opt_options["tol"])
                    if "time_limit" in opt_options:
                        wt_opt.driver.opt_settings["Time limit"] = int(opt_options["time_limit"])
                    if "major_step_limit" in opt_options:
                        wt_opt.driver.opt_settings["Major step limit"] = float(opt_options["major_step_limit"])
                    if "function_precision" in opt_options:
                        wt_opt.driver.opt_settings["Function precision"] = float(opt_options["function_precision"])
                    if "verify_level" in opt_options:
                        wt_opt.driver.opt_settings["Verify level"] = opt_options["verify_level"]
                    else:
                        wt_opt.driver.opt_settings["Verify level"] = -1

                # below are pyoptsparse options
                wt_opt.driver.options["output_dir"] = folder_output # Directory location of pyopt_sparse output files.Default is {prob_name}_out/reports. OpenMDAO overwrties SNOPT output file locations.
                if "hist_file_name" in opt_options:
                    wt_opt.driver.options["hist_file"] = os.path.join(folder_output, opt_options["hist_file_name"]) # File location for saving pyopt_sparse optimization history. Default is None for no output.
                if "hotstart_file" in opt_options:
                    wt_opt.driver.options["hotstart_file"] = opt_options["hotstart_file"] # File location of a pyopt_sparse optimization history to use to hot start the optimization. Default is None.

            elif opt_options["solver"] == "NSGA2":
                wt_opt.driver = NSGA2Driver()
                options_keys = [
                    "max_gen",
                    "pop_size",
                    "run_parallel",
                    "procs_per_model",
                    "penalty_parameter",
                    "penalty_exponent",
                    "Pc",
                    "eta_c",
                    "Pm",
                    "eta_m",
                ]
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys)

            elif opt_options["solver"] == "GA":
                wt_opt.driver = om.SimpleGADriver()
                options_keys = [
                    "Pc",
                    "Pm",
                    "bits",
                    "compute_pareto",
                    "cross_bits",
                    "elitism",
                    "gray",
                    "max_gen",
                    "multi_obj_exponent",
                    "multi_obj_weights",
                    "penalty_exponent",
                    "penalty_parameter",
                    "pop_size",
                    "procs_per_model",
                    "run_parallel",
                ]
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys)

            elif opt_options["solver"] == "DE":
                wt_opt.driver = om.DifferentialEvolutionDriver()
                # wt_opt.driver.options["optimizer"] = opt_options["solver"]
                options_keys = [
                    "F", # differential rate
                    "Pc", # crossover probability
                    "debug_type", # what driver variables to print at each iteration
                    "invalid_desvar_behavior", # behavior of driver if initial DV violated bounds
                    "max_gen", # number of generations before termination
                    "multi_obj_exponent", # weighting exponent for multiobjective
                    "multi_obj_weights", # weights of objectives for multiobjective optimization (abs name -> weight dict)
                    "penalty_exponent", # penalty function exponent
                    "penalty_parameter", # penalty function parameter
                    "pop_size", # number of points in the GA (default to 20x inputs if 0)
                    "procs_per_model", # processors to give each model under mpi
                    "run_parallel", # true -> execut points in a generation in parallel
                ]
                mapped_keys = {"debug_print": "debug_type"}
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys, mapped_keys=mapped_keys)

            elif opt_options["solver"] in self.nlopt_methods:
                try:
                    from wisdem.optimization_drivers.nlopt_driver import NLoptDriver
                except:
                    raise ImportError(
                        "You requested an optimization method from NLopt, but need to first install NLopt to use this method."
                    )

                wt_opt.driver = NLoptDriver()
                wt_opt.driver.options["optimizer"] = opt_options["solver"]
                options_keys = ["tol", "xtol", "max_iter", "max_time", "numgen"]
                mapped_keys = {"max_iter": "maxiter", "max_time": "maxtime"}
                wt_opt = self._set_optimizer_properties(wt_opt, options_keys, mapped_keys=mapped_keys)

            else:
                raise ValueError(f"The {self.opt['driver']['optimization']['solver']} optimizer is not yet supported!")

            if opt_options["debug_print"]:
                wt_opt.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs", "totals"]

        elif self.opt["driver"]["design_of_experiments"]["flag"]:
            doe_options = self.opt["driver"]["design_of_experiments"]
            if doe_options["generator"].lower() == "uniform":
                generator = om.UniformGenerator(
                    num_samples=int(doe_options["num_samples"]),
                    seed=doe_options["seed"],
                )
            elif doe_options["generator"].lower() == "fullfact":
                generator = om.FullFactorialGenerator(levels=int(doe_options["num_samples"]))
            elif doe_options["generator"].lower() == "plackettburman":
                generator = om.PlackettBurmanGenerator()
            elif doe_options["generator"].lower() == "boxbehnken":
                generator = om.BoxBehnkenGenerator()
            elif doe_options["generator"].lower() == "latinhypercube":
                generator = om.LatinHypercubeGenerator(
                    samples=int(doe_options["num_samples"]),
                    criterion=doe_options["criterion"],
                    seed=doe_options["seed"],
                )
            else:
                raise Exception("The generator type {} is unsupported.".format(doe_options["generator"]))

            # Initialize driver
            wt_opt.driver = om.DOEDriver(generator)

            if doe_options["debug_print"]:
                wt_opt.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

            # options
            wt_opt.driver.options["run_parallel"] = doe_options["run_parallel"]

        elif self.opt["driver"]["step_size_study"]["flag"]:
            pass

        else:
            print("WARNING: Design variables are set to be optimized or studied, but no driver is selected.")
            print("         If you want to run an optimization, please enable a driver.")

        return wt_opt

    def set_objective(self, wt_opt):

        if isinstance(self.opt['merit_figure_user'], dict):
            self.opt['merit_figure_user'] = [self.opt['merit_figure_user']]

        for merit_figure_user in self.opt['merit_figure_user']:
            wt_opt = self.set_merit_figure_user(wt_opt, merit_figure_user)


        # make merit figure a list if it is not already
        if isinstance(self.opt['merit_figure'], str):
            self.opt['merit_figure'] = [self.opt['merit_figure']]

        for merit_figure in self.opt['merit_figure']:
            if merit_figure.lower() != 'none':
                wt_opt = self.set_merit_figure(wt_opt, merit_figure)

        return wt_opt

    def set_merit_figure_user(self, wt_opt, merit_figure_user):
        # Set merit figure. Each objective has its own scaling.  Check first for user override
        if merit_figure_user["name"] != "":
            coeff = -1.0 if merit_figure_user["max_flag"] else 1.0
            wt_opt.model.add_objective(merit_figure_user["name"],
                                       ref=coeff*np.abs(merit_figure_user["ref"]))

        return wt_opt


    def set_merit_figure(self, wt_opt, merit_figure):
        # Set a single merit figure
        # merit_figure is a single string

        if merit_figure.lower() == "aep":
            wt_opt.model.add_objective("rotorse.rp.AEP", ref=-1.0e6)

        elif merit_figure == "blade_mass":
            wt_opt.model.add_objective("rotorse.blade_mass", ref=1.0e6)

        elif merit_figure == "blade_cost":
            wt_opt.model.add_objective("rotorse.total_bc.total_blade_cost", ref=1.0e6)

        elif merit_figure.lower() == "lcoe":
            wt_opt.model.add_objective("financese.lcoe", ref=0.1)

        elif merit_figure == "blade_tip_deflection":
            wt_opt.model.add_objective("tcons.tip_deflection_ratio")

        elif merit_figure == "tower_mass":
            wt_opt.model.add_objective("towerse.tower_mass", ref=1e6)

        elif merit_figure == "tower_cost":
            wt_opt.model.add_objective("tcc.tower_cost", ref=1e6)

        elif merit_figure == "monopile_mass":
            wt_opt.model.add_objective("fixedse.monopile_mass", ref=1e6)

        elif merit_figure == "monopile_cost":
            wt_opt.model.add_objective("fixedse.monopile_cost", ref=1e6)

        elif merit_figure == "jacket_mass":
            wt_opt.model.add_objective("fixedse.jacket_mass", ref=1e6)

        elif merit_figure == "structural_mass":
            if not self.modeling["flags"]["floating"]:
                wt_opt.model.add_objective("fixedse.structural_mass", ref=1e6)
            else:
                wt_opt.model.add_objective("floatingse.system_structural_mass", ref=1e6)

        elif merit_figure == "hub_mass":
            wt_opt.model.add_objective("drivese.hub_system_mass", ref=1e5)

        elif merit_figure == "nacelle_mass":
            wt_opt.model.add_objective("drivese.nacelle_mass", ref=1e6)

        elif merit_figure == "nacelle_cost":
            wt_opt.model.add_objective("tcc.nacelle_cost", ref=1e6)

        elif merit_figure == "platform_hull_mass":
            wt_opt.model.add_objective("floatingse.platform_hull_mass", ref=1e6)

        elif merit_figure == "platform_mass":
            wt_opt.model.add_objective("floatingse.platform_mass", ref=1e6)

        elif merit_figure == "platform_cost":
            wt_opt.model.add_objective("floatingse.platform_cost", ref=1e6)

        elif merit_figure == "mooring_mass":
            wt_opt.model.add_objective("floatingse.mooring_mass", ref=1e4)

        elif merit_figure == "mooring_cost":
            wt_opt.model.add_objective("floatingse.mooring_cost", ref=1e4)

        elif merit_figure == "Cp":
            if self.modeling["flags"]["blade"]:
                wt_opt.model.add_objective("rotorse.rp.powercurve.Cp_regII", ref=-1.0)
            else:
                wt_opt.model.add_objective("rotorse.ccblade.CP", ref=-1.0)

        elif merit_figure in ["turbine_cost", "turbine_capex"]:
            wt_opt.model.add_objective("tcc.turbine_cost_kW", ref=1e3)

        elif merit_figure in ["bos", "bos_cost", "bos_capex"]:
            if self.modeling["flags"]["offshore"]:
                wt_opt.model.add_objective("orbit.total_capex_kW", ref=1e3)
            else:
                wt_opt.model.add_objective("landbosse.total_capex_kW", ref=1e3)

        elif merit_figure == "inverse_design":
            wt_opt.model.add_objective("inverse_design.objective")

        else:
            raise ValueError("The merit figure " + merit_figure + " is unknown or not supported.")

        return wt_opt

    def set_design_variables(self, wt_opt, wt_init):
        # Set optimization design variables.
        rotorD_opt = self.opt["design_variables"]["rotor_diameter"]
        blade_opt = self.opt["design_variables"]["blade"]
        tower_opt = self.opt["design_variables"]["tower"]
        monopile_opt = self.opt["design_variables"]["monopile"]
        jacket_opt = self.opt["design_variables"]["jacket"]
        control_opt = self.opt["design_variables"]["control"]
        hub_opt = self.opt["design_variables"]["hub"]
        drive_opt = self.opt["design_variables"]["drivetrain"]
        float_opt = self.opt["design_variables"]["floating"]
        mooring_opt = self.opt["design_variables"]["mooring"]

        # -- Rotor & Blade --
        if rotorD_opt["flag"]:
            wt_opt.model.add_design_var(
                "configuration.rotor_diameter_user", lower=rotorD_opt["minimum"], upper=rotorD_opt["maximum"], ref=1.0e2
            )

        twist_options = blade_opt["aero_shape"]["twist"]
        if twist_options["flag"]:
            if blade_opt["aero_shape"]["twist"]["index_end"] > blade_opt["aero_shape"]["twist"]["n_opt"]:
                raise Exception(
                    "Check the analysis options yaml, index_end of the blade twist is higher than the number of DVs n_opt"
                )
            elif blade_opt["aero_shape"]["twist"]["index_end"] == 0:
                blade_opt["aero_shape"]["twist"]["index_end"] = blade_opt["aero_shape"]["twist"]["n_opt"]
            indices_twist = range(twist_options["index_start"], twist_options["index_end"])
            s_opt_twist = np.linspace(0.0, 1.0, blade_opt["aero_shape"]["twist"]["n_opt"])
            twist_interpolator = PchipInterpolator(
                wt_init["components"]["blade"]["outer_shape"]["twist"]["grid"],
                wt_init["components"]["blade"]["outer_shape"]["twist"]["values"],
            )
            init_twist_opt = twist_interpolator(s_opt_twist)
            wt_opt.model.add_design_var(
                "blade.opt_var.twist_opt",
                indices=indices_twist,
                lower=np.deg2rad(init_twist_opt[indices_twist] - blade_opt["aero_shape"]["twist"]["max_decrease"]),
                upper=np.deg2rad(init_twist_opt[indices_twist] + blade_opt["aero_shape"]["twist"]["max_increase"]),
            )

        chord_options = blade_opt["aero_shape"]["chord"]
        if chord_options["flag"]:
            if blade_opt["aero_shape"]["chord"]["index_end"] > blade_opt["aero_shape"]["chord"]["n_opt"]:
                raise Exception(
                    "Check the analysis options yaml, index_end of the blade chord is higher than the number of DVs n_opt"
                )
            elif blade_opt["aero_shape"]["chord"]["index_end"] == 0:
                blade_opt["aero_shape"]["chord"]["index_end"] = blade_opt["aero_shape"]["chord"]["n_opt"]
            indices_chord = range(chord_options["index_start"], chord_options["index_end"])
            s_opt_chord = np.linspace(0.0, 1.0, blade_opt["aero_shape"]["chord"]["n_opt"])
            chord_interpolator = PchipInterpolator(
                wt_init["components"]["blade"]["outer_shape"]["chord"]["grid"],
                wt_init["components"]["blade"]["outer_shape"]["chord"]["values"],
            )
            init_chord_opt = chord_interpolator(s_opt_chord)
            wt_opt.model.add_design_var(
                "blade.opt_var.chord_opt",
                indices=indices_chord,
                lower=init_chord_opt[indices_chord] * chord_options["max_decrease"],
                upper=init_chord_opt[indices_chord] * chord_options["max_increase"],
            )

        if blade_opt["aero_shape"]["af_positions"]["flag"]:
            n_af_master = self.modeling["WISDEM"]["RotorSE"]["n_af_master"]
            indices_af = range(blade_opt["aero_shape"]["af_positions"]["af_start"], n_af_master - 1)
            af_pos_init = np.zeros(n_af_master)
            for i in range(n_af_master):
                af_pos_init[i] = wt_init["components"]["blade"]["outer_shape"]["airfoils"][i]["spanwise_position"]
            step_size = self._get_step_size()
            lb_af = np.zeros(n_af_master)
            ub_af = np.zeros(n_af_master)
            for i in range(1, indices_af[0]):
                lb_af[i] = ub_af[i] = af_pos_init[i]
            for i in indices_af:
                lb_af[i] = 0.5 * (af_pos_init[i - 1] + af_pos_init[i]) + step_size
                ub_af[i] = 0.5 * (af_pos_init[i + 1] + af_pos_init[i]) - step_size
            lb_af[-1] = ub_af[-1] = 1.0
            wt_opt.model.add_design_var(
                "blade.opt_var.af_position", indices=indices_af, lower=lb_af[indices_af], upper=ub_af[indices_af]
            )

        if "structure" in blade_opt and len(blade_opt["structure"]) > 0:
            if self.modeling["user_elastic"]["blade"]:
                raise Exception("Blade structural design variables not available for user-defined blade elastic model. Please modify the modeling or optimization options.")
            layers = wt_init["components"]["blade"]["structure"]["layers"]
            for i in range(len(blade_opt["structure"])):
                k = blade_opt["layer_index_opt"][i]
                layer_interp = PchipInterpolator(
                        layers[k]["thickness"]["grid"],
                        layers[k]["thickness"]["values"],
                        extrapolate=False
                    )
                s_opt_layer = np.linspace(0.0, 1.0, blade_opt["n_opt_struct"][k])
                init_layer_opt = layer_interp(s_opt_layer)
                # Only set an optimization variable if the layer has non zero initial thickness
                indices_i = [
                    idx for idx in range(blade_opt["structure"][i]["index_start"], blade_opt["structure"][i]["index_end"])
                    if init_layer_opt[idx] >= 1.e-6
                ]
                wt_opt.model.add_design_var(
                    "blade.opt_var.layer_%d_opt"%k,
                    indices=indices_i,
                    lower=init_layer_opt[indices_i] * blade_opt["structure"][i]["max_decrease"],
                    upper=init_layer_opt[indices_i] * blade_opt["structure"][i]["max_increase"],
                    ref=1.0e-2,
                )

        # -- Tower --
        if tower_opt["outer_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "tower.diameter",
                lower=tower_opt["outer_diameter"]["lower_bound"],
                upper=tower_opt["outer_diameter"]["upper_bound"],
                ref=5.0,
            )

        if tower_opt["layer_thickness"]["flag"]:
            wt_opt.model.add_design_var(
                "tower.layer_thickness",
                lower=tower_opt["layer_thickness"]["lower_bound"],
                upper=tower_opt["layer_thickness"]["upper_bound"],
                ref=1e-2,
            )

        if tower_opt["E"]["flag"]:
            ivc = wt_opt.model.add_subsystem("E_ivc", om.IndepVarComp(), promotes=[])
            ivc.add_output("E_user", val=10.0, units="Pa")
            wt_opt.model.add_design_var(
                "E_ivc.E_user",
                lower=tower_opt["E"]["lower_bound"],
                upper=tower_opt["E"]["upper_bound"],
                ref=1e9,
            )
            wt_opt.model.connect("E_ivc.E_user", "towerse.E_user")

            if self.modeling["flags"]["monopile"] or self.modeling["flags"]["jacket"]:
                wt_opt.model.connect("E_ivc.E_user", "fixedse.E_user")

        for idx, material in enumerate(wt_init["materials"]):
            if material["name"] == "steel":
                tower_material_index = idx

        if tower_opt["rho"]["flag"]:
            wt_opt.model.add_design_var(
                "materials.rho",
                lower=tower_opt["rho"]["lower_bound"],
                upper=tower_opt["rho"]["upper_bound"],
                ref=1e3,
                indices=[tower_material_index],
            )

        # -- Monopile --
        if monopile_opt["outer_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "monopile.diameter",
                lower=monopile_opt["outer_diameter"]["lower_bound"],
                upper=monopile_opt["outer_diameter"]["upper_bound"],
                ref=5.0,
            )

        if monopile_opt["layer_thickness"]["flag"]:
            wt_opt.model.add_design_var(
                "monopile.layer_thickness",
                lower=monopile_opt["layer_thickness"]["lower_bound"],
                upper=monopile_opt["layer_thickness"]["upper_bound"],
                ref=1e-2,
            )

        # -- Jacket --
        if jacket_opt["foot_head_ratio"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.foot_head_ratio",
                lower=jacket_opt["foot_head_ratio"]["lower_bound"],
                upper=jacket_opt["foot_head_ratio"]["upper_bound"],
            )

        if jacket_opt["r_head"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.r_head",
                lower=jacket_opt["r_head"]["lower_bound"],
                upper=jacket_opt["r_head"]["upper_bound"],
                ref=5.0,
            )

        if jacket_opt["leg_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.leg_diameter",
                lower=jacket_opt["leg_diameter"]["lower_bound"],
                upper=jacket_opt["leg_diameter"]["upper_bound"],
            )

        if jacket_opt["leg_thickness"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.leg_thickness",
                lower=jacket_opt["leg_thickness"]["lower_bound"],
                upper=jacket_opt["leg_thickness"]["upper_bound"],
            )

        if jacket_opt["brace_diameters"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.brace_diameters",
                lower=jacket_opt["brace_diameters"]["lower_bound"],
                upper=jacket_opt["brace_diameters"]["upper_bound"],
            )

        if jacket_opt["brace_thicknesses"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.brace_thicknesses",
                lower=jacket_opt["brace_thicknesses"]["lower_bound"],
                upper=jacket_opt["brace_thicknesses"]["upper_bound"],
            )

        if jacket_opt["height"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.height",
                lower=jacket_opt["height"]["lower_bound"],
                upper=jacket_opt["height"]["upper_bound"],
            )

        if jacket_opt["bay_spacing"]["flag"]:
            wt_opt.model.add_design_var(
                "jacket.bay_spacing",
                lower=jacket_opt["bay_spacing"]["lower_bound"],
                upper=jacket_opt["bay_spacing"]["upper_bound"],
            )

        # -- Control --
        if control_opt["tsr"]["flag"]:
            wt_opt.model.add_design_var(
                "control.rated_TSR", lower=control_opt["tsr"]["minimum"], upper=control_opt["tsr"]["maximum"], ref=1e1
            )
        # -- Hub & Drivetrain --
        if hub_opt["cone"]["flag"]:
            wt_opt.model.add_design_var(
                "hub.cone", lower=hub_opt["cone"]["lower_bound"], upper=hub_opt["cone"]["upper_bound"], ref=1e-2
            )
        if hub_opt["hub_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "hub.diameter",
                lower=hub_opt["hub_diameter"]["lower_bound"],
                upper=hub_opt["hub_diameter"]["upper_bound"],
            )
        if drive_opt["uptilt"]["flag"]:
            wt_opt.model.add_design_var(
                "drivetrain.uptilt",
                lower=drive_opt["uptilt"]["lower_bound"],
                upper=drive_opt["uptilt"]["upper_bound"],
                ref=1e-2,
            )

        if drive_opt["generator_length"]["flag"]:
            wt_opt.model.add_design_var(
                "drivetrain.L_generator",
                lower=drive_opt["generator_length"]["lower_bound"],
                upper=drive_opt["generator_length"]["upper_bound"],
            )

        for k in [
            "overhang",
            "distance_tt_hub",
            "distance_hub_mb",
            "distance_mb_mb",
            "gear_ratio",
            "bedplate_flange_width",
            "lss_diameter",
            "hss_diameter",
            "nose_diameter",
        ]:
            if drive_opt[k]["flag"]:
                wt_opt.model.add_design_var(
                    "drivetrain." + k, lower=drive_opt[k]["lower_bound"], upper=drive_opt[k]["upper_bound"]
                )

        for k in [
            "bedplate_wall_thickness",
            "bedplate_web_thickness",
            "bedplate_flange_thickness",
            "lss_wall_thickness",
            "hss_wall_thickness",
            "nose_wall_thickness",
        ]:
            if drive_opt[k]["flag"]:
                wt_opt.model.add_design_var("drivetrain." + k, lower=drive_opt[k]["lower_bound"], upper=drive_opt[k]["upper_bound"], ref=1e-2)

        # -- Floating --
        count = 0
        jointz = float_opt["joints"]["z_coordinate"]
        jointr = float_opt["joints"]["r_coordinate"]
        for k in range(len(jointz)):
            if jointz[k]["flag"]:
                wt_opt.model.add_design_var(f"floating.jointdv_{count}", lower=jointz[k]["lower_bound"], upper=jointz[k]["upper_bound"])
                count += 1

        for k in range(len(jointr)):
            if jointr[k]["flag"]:
                wt_opt.model.add_design_var(f"floating.jointdv_{count}", lower=jointr[k]["lower_bound"], upper=jointr[k]["upper_bound"])
                count += 1

        for kgrp in float_opt["members"]["groups"]:
            if kgrp["flag"]:
                memname = kgrp["names"][0]
                idx = self.modeling["floating"]["members"]["name2idx"][memname]
                imem = self.modeling["floating"]["members"]["name"].index(memname)
                istruct = wt_init["components"]["floating_platform"]["members"][imem]["structure"]

                if "diameter" in kgrp:
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.outer_diameter_in",
                        lower=kgrp["diameter"]["lower_bound"],
                        upper=kgrp["diameter"]["upper_bound"],
                    )
                if "side_length_a" in kgrp:
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.side_length_a_in",
                        lower=kgrp["side_length_a"]["lower_bound"],
                        upper=kgrp["side_length_a"]["upper_bound"],
                    )
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.side_length_b_in",
                        lower=kgrp["side_length_b"]["lower_bound"],
                        upper=kgrp["side_length_b"]["upper_bound"],
                    )
                if "thickness" in kgrp:
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.layer_thickness_in",
                        lower=kgrp["thickness"]["lower_bound"],
                        upper=kgrp["thickness"]["upper_bound"],
                    )
                if "ballast" in kgrp and len(istruct["ballast"]) > 0:
                    V_ballast = np.zeros(len(istruct["ballast"]))
                    for j in range(V_ballast.size):
                        if "volume" in istruct["ballast"][j]:
                            V_ballast[j] = istruct["ballast"][j]["volume"]
                    iball = np.where(V_ballast > 0.0)[0]
                    if iball.size > 0:
                        wt_opt.model.add_design_var(
                            f"floating.memgrp{idx}.ballast_volume",
                            lower=kgrp["ballast"]["lower_bound"],
                            upper=kgrp["ballast"]["upper_bound"],
                            indices=iball,
                            ref=1e3,
                        )
                if "stiffeners" in kgrp:
                    if "ring" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["ring"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["ring"]:
                            wt_opt.model.add_design_var(
                                f"floating.memgrp{idx}.ring_stiffener_spacing",
                                lower=kgrp["stiffeners"]["ring"]["spacing"]["lower_bound"],
                                upper=kgrp["stiffeners"]["ring"]["spacing"]["upper_bound"],
                            )
                    if "longitudinal" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["longitudinal"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["longitudinal"]:
                            wt_opt.model.add_design_var(
                                f"floating.memgrp{idx}.axial_stiffener_spacing",
                                lower=kgrp["stiffeners"]["longitudinal"]["spacing"]["lower_bound"],
                                upper=kgrp["stiffeners"]["longitudinal"]["spacing"]["upper_bound"],
                            )
                if "axial_joints" in kgrp:
                    aidx = []
                    lower = []
                    upper = []
                    mem_axial_names = self.modeling["floating"]["members"]["axial_joint_name_member_" + memname]
                    for agrp in kgrp["axial_joints"]:
                        tryidx = None
                        agrp_names = agrp["names"]
                        lower.append(agrp["lower_bound"])
                        upper.append(agrp["upper_bound"])
                        for iname in agrp_names:
                            try:
                                tryidx = mem_axial_names.index(iname)
                                aidx.append(tryidx)
                                break
                            except Exception:
                                continue
                        if tryidx is None:
                            raise ValueError(
                                f"None of these axial joint names were found in member, {memname}:{agrp_names}"
                            )
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.grid_axial_joints", lower=lower, upper=upper, indices=aidx
                    )

        # -- Mooring --
        if mooring_opt["line_length"]["flag"]:
            wt_opt.model.add_design_var(
                "mooring.unstretched_length_in",
                lower=mooring_opt["line_length"]["lower_bound"],
                upper=mooring_opt["line_length"]["upper_bound"],
                ref=1e2,
            )

        if mooring_opt["line_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "mooring.line_diameter_in",
                lower=mooring_opt["line_diameter"]["lower_bound"],
                upper=mooring_opt["line_diameter"]["upper_bound"],
                ref=1e-1,
            )

        if mooring_opt["line_mass_density_coeff"]["flag"]:
            wt_opt.model.add_design_var(
                "mooring.line_mass_density_coeff",
                lower=mooring_opt["line_mass_density_coeff"]["lower_bound"],
                upper=mooring_opt["line_mass_density_coeff"]["upper_bound"],
            )

        if mooring_opt["line_stiffness_coeff"]["flag"]:
            wt_opt.model.add_design_var(
                "mooring.line_stiffness_coeff",
                lower=mooring_opt["line_stiffness_coeff"]["lower_bound"],
                upper=mooring_opt["line_stiffness_coeff"]["upper_bound"],
            )


        # -- User DVs --
        if "user" in self.opt["design_variables"]:
            user_defined = self.opt["design_variables"]["user"]
            for i in range(len(user_defined)):
                name_i = user_defined[i]["name"]

                if "lower_bound" in user_defined[i]:
                    lower_i = user_defined[i]["lower_bound"]
                elif "lower" in user_defined[i]:
                    lower_i = user_defined[i]["lower"]
                else:
                    lower_i = None

                if "upper_bound" in user_defined[i]:
                    upper_i = user_defined[i]["upper_bound"]
                elif "upper" in user_defined[i]:
                    upper_i = user_defined[i]["upper"]
                else:
                    upper_i = None

                if "ref" in user_defined[i]:
                    ref_i = user_defined[i]["ref"]
                else:
                    ref_i = None

                if "indices" in user_defined[i]:
                    indices_i = user_defined[i]["indices"]
                else:
                    indices_i = None

                wt_opt.model.add_design_var(
                    name_i,
                    lower=lower_i,
                    upper=upper_i,
                    ref=ref_i,
                    indices=indices_i,
                )

        return wt_opt

    def set_constraints(self, wt_opt):
        blade_opt = self.opt["design_variables"]["blade"]

        # Set non-linear blade constraints
        blade_constr = self.opt["constraints"]["blade"]

        if blade_constr["strains_spar_cap_ss"]["flag"]:
            indices_strains_spar_cap_ss = range(
                    blade_constr["strains_spar_cap_ss"]["index_start"], blade_constr["strains_spar_cap_ss"]["index_end"]
                )
            wt_opt.model.add_constraint(
                    "rotorse.rs.constr.constr_max_strainU_spar", indices=indices_strains_spar_cap_ss, upper=1.0
                )
        if blade_constr["strains_spar_cap_ps"]["flag"]:
            indices_strains_spar_cap_ps = range(
                    blade_constr["strains_spar_cap_ps"]["index_start"], blade_constr["strains_spar_cap_ps"]["index_end"]
                )
            wt_opt.model.add_constraint(
                    "rotorse.rs.constr.constr_max_strainL_spar", indices=indices_strains_spar_cap_ps, upper=1.0
                )
        if blade_constr["strains_te_ss"]["flag"]:
            indices_strains_te_ss = range(
                    blade_constr["strains_te_ss"]["index_start"], blade_constr["strains_te_ss"]["index_end"]
                )
            wt_opt.model.add_constraint(
                    "rotorse.rs.constr.constr_max_strainU_te", indices=indices_strains_te_ss, upper=1.0
                )
        if blade_constr["strains_te_ps"]["flag"]:
            indices_strains_te_ps = range(
                    blade_constr["strains_te_ps"]["index_start"], blade_constr["strains_te_ps"]["index_end"]
                )
            wt_opt.model.add_constraint(
                    "rotorse.rs.constr.constr_max_strainL_te", indices=indices_strains_te_ps, upper=1.0
                )
        if blade_constr["stall"]["flag"]:
            wt_opt.model.add_constraint("rotorse.stall_check.no_stall_constraint", upper=1.0)
            if not blade_opt["aero_shape"]["twist"]["flag"]:
                print(
                    "WARNING: the margin to stall is set to be constrained, but twist is not an active design variable."
                )
        if blade_constr["tip_deflection"]["flag"]:
            wt_opt.model.add_constraint("tcons.tip_deflection_ratio", upper=1.0)
        if blade_constr["chord"]["flag"]:
            if blade_opt["aero_shape"]["chord"]["flag"]:
                wt_opt.model.add_constraint("blade.pa.max_chord_constr", upper=1.0)
            else:
                print(
                    "WARNING: the max chord is set to be constrained, but chord is not an active design variable. The constraint is not enforced."
                )
        if blade_constr["chord_slope"]["flag"]:
            if blade_opt["aero_shape"]["chord"]["flag"]:
                wt_opt.model.add_constraint("blade.pa.slope_chord_constr", upper=0.0)
            else:
                print(
                    "WARNING: the slope of the chord is set to be constrained, but chord is not an active design variable. The constraint is not enforced."
                )
        if blade_constr["twist_slope"]["flag"]:
            if blade_opt["aero_shape"]["twist"]["flag"]:
                wt_opt.model.add_constraint("blade.pa.slope_twist_constr", upper=0.0)
            else:
                print(
                    "WARNING: the slope of the twist is set to be constrained, but twist is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["root_circle_diameter"]["flag"]:
            if blade_opt["aero_shape"]["chord"]["flag"] and blade_opt["aero_shape"]["chord"]["index_start"] == 0.0:
                wt_opt.model.add_constraint(
                    "rotorse.rs.brs.ratio", upper=blade_constr["root_circle_diameter"]["max_ratio"]
                )
            else:
                print(
                    "WARNING: the blade root size is set to be constrained, but chord at blade root is not an active design variable. The constraint is not enforced."
                )
        if blade_constr["frequency"]["flap_3P"]:
            wt_opt.model.add_constraint("rotorse.rs.constr.constr_flap_f_margin", upper=0.0)
        if blade_constr["frequency"]["edge_3P"]:
            wt_opt.model.add_constraint("rotorse.rs.constr.constr_edge_f_margin", upper=0.0)
        if blade_constr["frequency"]["first_flap"]["flag"]:
            target = blade_constr["frequency"]["first_flap"]["target"]
            error = blade_constr["frequency"]["first_flap"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("rotorse.rs.frame.flap_mode_freqs", indices=[0], upper = upper_value, lower = lower_value)
        # if blade_constr["frequency"]["second_flap"]["flag"]:
        #     target = blade_constr["frequency"]["second_flap"]["target"]
        #     error = blade_constr["frequency"]["second_flap"]["acceptable_error"]
        #     upper_value = target + error
        #     lower_value = target - error
        #     wt_opt.model.add_constraint("rotorse.rs.frame.flap_mode_freqs", indices=[1], upper = upper_value, lower = lower_value, alias = '_2F')
        if blade_constr["frequency"]["first_edge"]["flag"]:
            target = blade_constr["frequency"]["first_edge"]["target"]
            error = blade_constr["frequency"]["first_edge"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("rotorse.rs.frame.edge_mode_freqs", indices=[0], upper = upper_value, lower = lower_value)
        # if blade_constr["frequency"]["second_edge"]["flag"]:
        #     target = blade_constr["frequency"]["second_edge"]["target"]
        #     error = blade_constr["frequency"]["second_edge"]["acceptable_error"]
        #     upper_value = target + error
        #     lower_value = target - error
        #     wt_opt.model.add_constraint("rotorse.rs.frame.edge_mode_freqs", indices=[1], upper = upper_value, lower = lower_value, alias = '_2E')
        if blade_constr["frequency"]["first_torsion"]["flag"]:
            target = blade_constr["frequency"]["first_torsion"]["target"]
            error = blade_constr["frequency"]["first_torsion"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("rotorse.rs.frame.tors_mode_freqs", indices=[0], upper = upper_value, lower = lower_value)
        if blade_constr["mass"]["flag"]:
            target = blade_constr["mass"]["target"]
            error = blade_constr["mass"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("rotorse.blade_mass", upper = upper_value, lower = lower_value, ref = 1.e+5)
        if blade_constr["rail_transport"]["8_axle"]:
            wt_opt.model.add_constraint("rotorse.re.rail.constr_LV_8axle_horiz", lower=0.8, upper=1.0)
            wt_opt.model.add_constraint("rotorse.re.rail.constr_strainPS", upper=1.0)
            wt_opt.model.add_constraint("rotorse.re.rail.constr_strainSS", upper=1.0)
        elif blade_constr["rail_transport"]["4_axle"]:
            wt_opt.model.add_constraint("rotorse.re.rail.constr_LV_4axle_horiz", upper=1.0)
            wt_opt.model.add_constraint("rotorse.re.rail.constr_strainPS", upper=1.0)
            wt_opt.model.add_constraint("rotorse.re.rail.constr_strainSS", upper=1.0)

        if blade_constr["rated_velocity"]["flag"]:
            target = blade_constr["rated_velocity"]["target"]
            error = blade_constr["rated_velocity"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("rotorse.rp.powercurve.rated_V", upper = upper_value, lower = lower_value, ref = 10.0)

        if self.opt["constraints"]["blade"]["moment_coefficient"]["flag"]:
            wt_opt.model.add_constraint(
                "rotorse.ccblade.CM",
                lower=self.opt["constraints"]["blade"]["moment_coefficient"]["min"],
                upper=self.opt["constraints"]["blade"]["moment_coefficient"]["max"],
            )
        if (
            self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cl"]
            or self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cd"]
        ):
            data_target = np.loadtxt(self.opt["constraints"]["blade"]["match_cl_cd"]["filename"])
            eta_opt = np.linspace(0.0, 1.0, self.opt["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"])
            target_cl = PchipInterpolator(data_target[:, 0], data_target[:, 3])(eta_opt)
            target_cd = PchipInterpolator(data_target[:, 0], data_target[:, 4])(eta_opt)
            eps_cl = 1.0e-2
            if self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cl"]:
                wt_opt.model.add_constraint(
                    "rotorse.ccblade.cl_n_opt", lower=target_cl - eps_cl, upper=target_cl + eps_cl
                )
            if self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cd"]:
                wt_opt.model.add_constraint(
                    "rotorse.ccblade.cd_n_opt", lower=target_cd - eps_cl, upper=target_cd + eps_cl
                )
        if (
            self.opt["constraints"]["blade"]["match_L_D"]["flag_L"]
            or self.opt["constraints"]["blade"]["match_L_D"]["flag_D"]
        ):
            data_target = np.loadtxt(self.opt["constraints"]["blade"]["match_L_D"]["filename"])
            eta_opt = np.linspace(0.0, 1.0, self.opt["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"])
            target_L = PchipInterpolator(data_target[:, 0], data_target[:, 7])(eta_opt)
            target_D = PchipInterpolator(data_target[:, 0], data_target[:, 8])(eta_opt)
        eps_L = 1.0e2
        if self.opt["constraints"]["blade"]["match_L_D"]["flag_L"]:
            wt_opt.model.add_constraint("rotorse.ccblade.L_n_opt", lower=target_L - eps_L, upper=target_L + eps_L)
        if self.opt["constraints"]["blade"]["match_L_D"]["flag_D"]:
            wt_opt.model.add_constraint("rotorse.ccblade.D_n_opt", lower=target_D - eps_L, upper=target_D + eps_L)

        if self.opt["constraints"]["blade"]["AEP"]["flag"]:
            wt_opt.model.add_constraint(
                "rotorse.rp.AEP", lower=self.opt["constraints"]["blade"]["AEP"]["min"], ref=-1.0e6
            )

        # to constrain C_T and forcibly design higher thrust turbines
        if blade_constr["thrust_coeff"]["flag"]:
            if "lower_bound" in blade_constr["thrust_coeff"]:
                wt_opt.model.add_constraint(
                    "rotorse.rp.powercurve.Ct_regII", lower=blade_constr["thrust_coeff"]["lower_bound"]
                )
            elif "upper_bound" in blade_constr["thrust_coeff"]:
                wt_opt.model.add_constraint(
                    "rotorse.rp.powercurve.Ct_regII", upper=blade_constr["thrust_coeff"]["upper_bound"]
                )
            else:
                raise Exception("thrust coefficient constraint requested but no upper or lower bound found.")

        # Tower and monopile contraints
        tower_constr = self.opt["constraints"]["tower"]
        monopile_constr = self.opt["constraints"]["monopile"]

        if tower_constr["height_constraint"]["flag"]:
            wt_opt.model.add_constraint(
                "towerse.height_constraint",
                lower=tower_constr["height_constraint"]["lower_bound"],
                upper=tower_constr["height_constraint"]["upper_bound"],
            )

        if tower_constr["stress"]["flag"] and not monopile_constr["stress"]["flag"]:
            wt_opt.model.add_constraint("towerse.post.constr_stress", upper=1.0)

        if tower_constr["global_buckling"]["flag"] and not monopile_constr["global_buckling"]["flag"]:
            wt_opt.model.add_constraint("towerse.post.constr_global_buckling", upper=1.0)

        if tower_constr["shell_buckling"]["flag"] and not monopile_constr["shell_buckling"]["flag"]:
            wt_opt.model.add_constraint("towerse.post.constr_shell_buckling", upper=1.0)

        if tower_constr["d_to_t"]["flag"]:
            wt_opt.model.add_constraint(
                "towerse.constr_d_to_t",
                lower=tower_constr["d_to_t"]["lower_bound"],
                upper=tower_constr["d_to_t"]["upper_bound"],
            )

        if tower_constr["taper"]["flag"]:
            wt_opt.model.add_constraint("towerse.constr_taper", lower=tower_constr["taper"]["lower_bound"])

        if tower_constr["slope"]["flag"]:
            wt_opt.model.add_constraint("towerse.slope", upper=1.0)

        if tower_constr["thickness_slope"]["flag"]:
            wt_opt.model.add_constraint("towerse.thickness_slope", upper=1.0)

        if tower_constr["frequency"]["flag"]:
            wt_opt.model.add_constraint("tcons.constr_tower_f_1Pmargin", upper=0.0)
            wt_opt.model.add_constraint("tcons.constr_tower_f_NPmargin", upper=0.0)

        elif tower_constr["frequency_1"]["flag"]:
            # Cannot set both 1P/3P and hard freq values, so else-if statement here
            varstr = (
                "floatingse.structural_frequencies"
                if self.modeling["flags"]["floating"]
                else "towerse.tower.structural_frequencies"
            )
            wt_opt.model.add_constraint(
                varstr,
                indices=[0],
                lower=tower_constr["frequency_1"]["lower_bound"],
                upper=tower_constr["frequency_1"]["upper_bound"],
            )

        if tower_constr["mass"]["flag"]:
            target = tower_constr["mass"]["target"]
            error = tower_constr["mass"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("towerse.tower_mass", upper = upper_value, lower = lower_value, ref = 1.e+6)

        if monopile_constr["stress"]["flag"] and tower_constr["stress"]["flag"]:
            wt_opt.model.add_constraint("fixedse.post_monopile_tower.constr_stress", upper=1.0)
        elif monopile_constr["stress"]["flag"]:
            wt_opt.model.add_constraint("fixedse.post.constr_stress", upper=1.0)

        if monopile_constr["global_buckling"]["flag"] and tower_constr["global_buckling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.post_monopile_tower.constr_global_buckling", upper=1.0)
        elif monopile_constr["global_buckling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.post.constr_global_buckling", upper=1.0)

        if monopile_constr["shell_buckling"]["flag"] and tower_constr["shell_buckling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.post_monopile_tower.constr_shell_buckling", upper=1.0)
        elif monopile_constr["shell_buckling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.post.constr_shell_buckling", upper=1.0)

        if monopile_constr["d_to_t"]["flag"]:
            wt_opt.model.add_constraint(
                "fixedse.constr_d_to_t",
                lower=monopile_constr["d_to_t"]["lower_bound"],
                upper=monopile_constr["d_to_t"]["upper_bound"],
            )

        if monopile_constr["taper"]["flag"]:
            wt_opt.model.add_constraint("fixedse.constr_taper", lower=monopile_constr["taper"]["lower_bound"])

        if monopile_constr["slope"]["flag"]:
            wt_opt.model.add_constraint("fixedse.slope", upper=1.0)

        if monopile_constr["thickness_slope"]["flag"]:
            wt_opt.model.add_constraint("fixedse.thickness_slope", upper=1.0)

        if monopile_constr["frequency_1"]["flag"]:
            wt_opt.model.add_constraint(
                "fixedse.structural_frequencies",
                indices=[0],
                lower=monopile_constr["frequency_1"]["lower_bound"],
                upper=monopile_constr["frequency_1"]["upper_bound"],
            )

        if monopile_constr["pile_depth"]["flag"]:
            wt_opt.model.add_constraint("fixedse.suctionpile_depth", lower=monopile_constr["pile_depth"]["lower_bound"])

        if monopile_constr["tower_diameter_coupling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.constr_diam_consistency", upper=1.0)

        if monopile_constr["mass"]["flag"]:
            target = monopile_constr["mass"]["target"]
            error = monopile_constr["mass"]["acceptable_error"]
            upper_value = target + error
            lower_value = target - error
            wt_opt.model.add_constraint("fixedse.monopile_mass", upper = upper_value, lower = lower_value, ref = 1.e+6)

        # Jacket constraints
        jacket_constr = self.opt["constraints"]["jacket"]

        if jacket_constr["stress"]["flag"]:
            wt_opt.model.add_constraint("fixedse.constr_stress", upper=1.0)

        if jacket_constr["global_buckling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.constr_global_buckling", upper=1.0)

        if jacket_constr["shell_buckling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.constr_shell_buckling", upper=1.0)

        if jacket_constr["tower_diameter_coupling"]["flag"]:
            wt_opt.model.add_constraint("fixedse.constr_diam_consistency", upper=1.0)

        if jacket_constr["frequency_1"]["flag"]:
            wt_opt.model.add_constraint(
                "fixedse.structural_frequencies",
                indices=[0],
                lower=jacket_constr["frequency_1"]["lower_bound"],
                upper=jacket_constr["frequency_1"]["upper_bound"],
            )

        # Hub and drivetrain constraints
        hub_constr = self.opt["constraints"]["hub"]
        drive_constr = self.opt["constraints"]["drivetrain"]

        if hub_constr["hub_diameter"]["flag"]:
            wt_opt.model.add_constraint("drivese.constr_hub_diameter", lower=0.0)

        for k in ["lss", "hss", "bedplate", "mb1", "mb2"]:
            if drive_constr[k]["flag"]:
                mystr = k + "_vonmises" if k in ["lss", "hss", "bedplate"] else k + "_defl"
                wt_opt.model.add_constraint("drivese.constr_" + mystr, upper=1.0)

        for k in ["shaft", "stator"]:
            for m in ["deflection", "angle"]:
                if drive_constr[f"{k}_{m}"]["flag"]:
                    wt_opt.model.add_constraint(f"drivese.constr_{k}_{m}", upper=1.0)

        for k in ["length", "height", "access", "ecc"]:
            if drive_constr[k]["flag"]:
                wt_opt.model.add_constraint("drivese.constr_" + k, lower=0.0)

        # Floating platform and mooring constraints
        float_constr = self.opt["constraints"]["floating"]

        if float_constr["buoyancy"]["flag"]:
            wt_opt.model.add_constraint("floatingse.variable_ballast_mass", lower=0.0)

        if float_constr["fixed_ballast_capacity"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_fixed_margin", upper=1.0)

        if float_constr["variable_ballast_capacity"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_variable_margin", lower=0.0, upper=1.0)

        if float_constr["metacentric_height"]["flag"]:
            wt_opt.model.add_constraint(
                "floatingse.metacentric_height_roll", lower=float_constr["metacentric_height"]["lower_bound"]
            )
            wt_opt.model.add_constraint(
                "floatingse.metacentric_height_pitch", lower=float_constr["metacentric_height"]["lower_bound"]
            )

        if float_constr["freeboard_margin"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_freeboard_heel_margin", upper=1.0)

        if float_constr["draft_margin"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_draft_heel_margin", upper=1.0)

        if float_constr["fairlead_depth"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_fairlead_wave", upper=1.0)

        if float_constr["mooring_heel"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_mooring_heel", upper=1.0)

        if float_constr["mooring_surge"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_mooring_surge", upper=1.0)

        if float_constr["mooring_tension"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_axial_load", upper=1.0)

        if float_constr["mooring_length"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_mooring_length", upper=1.0)

        if float_constr["anchor_vertical"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_anchor_vertical", lower=0.0)

        if float_constr["anchor_lateral"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_anchor_lateral", lower=0.0)

        if float_constr["stress"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_platform_stress", upper=1.0)

        if float_constr["shell_buckling"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_platform_shell_buckling", upper=1.0)

        if float_constr["global_buckling"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_platform_global_buckling", upper=1.0)

        for modestr in ["surge", "sway", "heave", "roll", "pitch", "yaw"]:
            if float_constr[f"{modestr}_period"]["flag"]:
                wt_opt.model.add_constraint(
                    f"{self.floating_period_solve_component}.{modestr}_period",
                    lower=float_constr[f"{modestr}_period"]["lower_bound"],
                    upper=float_constr[f"{modestr}_period"]["upper_bound"],
                )

        # User constraints
        user_constr = self.opt["constraints"]["user"]
        for k in range(len(user_constr)):
            var_k = user_constr[k]["name"]

            if "lower_bound" in user_constr[k]:
                lower_k = user_constr[k]["lower_bound"]
            elif "lower" in user_constr[k]:
                lower_k = user_constr[k]["lower"]
            else:
                lower_k = None

            if "upper_bound" in user_constr[k]:
                upper_k = user_constr[k]["upper_bound"]
            elif "upper" in user_constr[k]:
                upper_k = user_constr[k]["upper"]
            else:
                upper_k = None

            if "indices" in user_constr[k]:
                idx_k = user_constr[k]["indices"]
            else:
                idx_k = None

            if "ref" in user_constr[k]:
                ref_k = user_constr[k]["ref"]
            else:
                ref_k = None

            if lower_k is None and upper_k is None:
                raise Exception(f"Must include a lower_bound and/or an upper bound for {var_k}")

            wt_opt.model.add_constraint(var_k, lower=lower_k, upper=upper_k, indices=idx_k, ref=ref_k)

        return wt_opt

    def set_recorders(self, wt_opt):
        folder_output = self.opt["general"]["folder_output"]

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        if self.opt["recorder"]["flag"]:
            recorder = om.SqliteRecorder(os.path.join(folder_output, self.opt["recorder"]["file_name"]))
            wt_opt.driver.add_recorder(recorder)
            wt_opt.add_recorder(recorder)

            wt_opt.driver.recording_options["excludes"] = ["*_df"]
            wt_opt.driver.recording_options["record_constraints"] = True
            wt_opt.driver.recording_options["record_desvars"] = True
            wt_opt.driver.recording_options["record_objectives"] = True

            if self.opt["recorder"]["includes"]:
                wt_opt.driver.recording_options["includes"] = self.opt["recorder"]["includes"]

        return wt_opt

    def set_initial(self, wt_opt, wt_init):
        blade_opt = self.opt["design_variables"]["blade"]

        if self.modeling["flags"]["blade"]:
            wt_opt["blade.opt_var.s_opt_twist"] = np.linspace(0.0, 1.0, blade_opt["aero_shape"]["twist"]["n_opt"])
            twist_interpolator = PchipInterpolator(
                wt_init["components"]["blade"]["outer_shape"]["twist"]["grid"],
                wt_init["components"]["blade"]["outer_shape"]["twist"]["values"],
            )
            init_twist_opt = twist_interpolator(wt_opt["blade.opt_var.s_opt_twist"])
            wt_opt["blade.opt_var.twist_opt"] = init_twist_opt
            wt_opt["blade.opt_var.s_opt_chord"] = np.linspace(0.0, 1.0, blade_opt["aero_shape"]["chord"]["n_opt"])
            chord_interpolator = PchipInterpolator(
                wt_init["components"]["blade"]["outer_shape"]["chord"]["grid"],
                wt_init["components"]["blade"]["outer_shape"]["chord"]["values"],
            )
            init_chord_opt = chord_interpolator(wt_opt["blade.opt_var.s_opt_chord"])
            wt_opt["blade.opt_var.chord_opt"] = init_chord_opt

            if not self.modeling["user_elastic"]["blade"]:
                # YL: no internal structure optimization when using user-defined blade elastic properties
                layers = wt_init["components"]["blade"]["structure"]["layers"]
                for i in range(self.modeling["WISDEM"]["RotorSE"]["n_layers"]):
                    wt_opt["blade.opt_var.s_opt_layer_%d"%i] = np.linspace(
                        0.0, 1.0, blade_opt["n_opt_struct"][i]
                    )
                    thick_interp = PchipInterpolator(
                                layers[i]["thickness"]["grid"],
                                layers[i]["thickness"]["values"],
                                extrapolate=False)
                    init_opt = thick_interp(wt_opt["blade.opt_var.s_opt_layer_%d"%i])
                    wt_opt["blade.opt_var.layer_%d_opt"%i] = np.nan_to_num(init_opt, nan=0.)

                if self.modeling["WISDEM"]["RotorSE"]["flag"]:
                    blade_constr = self.opt["constraints"]["blade"]
                    wt_opt["rotorse.rs.constr.max_strainU_spar"] = blade_constr["strains_spar_cap_ss"]["max"]
                    wt_opt["rotorse.rs.constr.max_strainL_spar"] = blade_constr["strains_spar_cap_ps"]["max"]
                    if blade_constr["rail_transport"]["flag"]:
                        wt_opt["rotorse.re.rail.max_strains"] = min(
                            blade_constr["strains_spar_cap_ss"]["max"], blade_constr["strains_spar_cap_ps"]["max"]
                        )

                    wt_opt["rotorse.rs.constr.max_strainU_te"] = blade_constr["strains_te_ss"]["max"]
                    wt_opt["rotorse.rs.constr.max_strainL_te"] = blade_constr["strains_te_ps"]["max"]
                    wt_opt["rotorse.stall_check.stall_margin"] = blade_constr["stall"]["margin"]
                    if self.modeling["flags"]["tower"]:
                        wt_opt["tcons.max_allowable_td_ratio"] = blade_constr["tip_deflection"]["margin"]

        if self.modeling["flags"]["drivetrain"]:
            drive_constr = self.opt["constraints"]["drivetrain"]

            wt_opt["drivese.shaft_deflection_allowable"] = drive_constr["shaft_deflection"]["upper_bound"]
            wt_opt["drivese.shaft_angle_allowable"] = np.deg2rad(drive_constr["shaft_angle"]["upper_bound"])

            wt_opt["drivese.stator_deflection_allowable"] = drive_constr["stator_deflection"]["upper_bound"]
            wt_opt["drivese.stator_angle_allowable"] = np.deg2rad(drive_constr["stator_angle"]["upper_bound"])

            if self.modeling["WISDEM"]["DriveSE"]["direct"]:
                wt_opt["drivese.access_diameter"] = drive_constr["access"]["lower_bound"]

        if self.modeling["flags"]["floating"]:
            float_constr = self.opt["constraints"]["floating"]
            wt_opt["floatingse.max_surge_fraction"] = float_constr["max_surge"]["upper_bound"]
            wt_opt.set_val("floatingse.operational_heel", float_constr["operational_heel"]["upper_bound"], units="deg")
            wt_opt.set_val("floatingse.survival_heel", float_constr["survival_heel"]["upper_bound"], units="deg")

        return wt_opt

    def set_restart(self, wt_opt):
        if "warmstart_file" in self.opt["driver"]["optimization"]:
            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict

            db = SqliteDict(self.opt["driver"]["optimization"]["warmstart_file"])

            # Grab the last iteration's design variables
            last_key = db["last"]
            desvars = db[last_key]["xuser"]

            # Obtain the already-setup OM problem's design variables
            if wt_opt.model._static_mode:
                design_vars = wt_opt.model._static_design_vars
            else:
                design_vars = wt_opt.model._design_vars

            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = wt_opt.model._var_allprocs_prom2abs_list["output"]
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]

                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]["scaler"]
                adder = design_vars[prom_key]["adder"]

                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0

                scaled_dv = desvars[key] / scaler - adder

                # Special handling for blade twist as we only have the
                # last few control points as design variables
                if "twist_opt" in key:
                    wt_opt[key][2:] = scaled_dv
                else:
                    wt_opt[key][:] = scaled_dv

        return wt_opt
