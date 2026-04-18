import os
import copy
from pprint import pprint
from wisdem.inputs.validation import write_yaml
from wisdem.inputs.validation import simple_types

import numpy as np

from wisdem.optimization_drivers.nsga2.algo_nsga2 import NSGA2 as NSGA2_implementation

try:
    from pyDOE3 import lhs
except ModuleNotFoundError:
    lhs = None

from openmdao.core.constants import INF_BOUND
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.concurrent_utils import concurrent_eval
from openmdao.utils.mpi import MPI
from openmdao.core.analysis_error import AnalysisError


class NSGA2Driver(Driver):
    """
    Driver for a simple genetic algorithm.

    # Parameters
    # ----------
    # **kwargs : dict of keyword arguments
    #     Keyword arguments that will be mapped into the Driver options.
    #
    # Attributes
    # ----------
    # _problem_comm : MPI.Comm or None
    #     The MPI communicator for the Problem.
    # _concurrent_pop_size : int
    #     Number of points to run concurrently when model is a parallel one.
    # _concurrent_color : int
    #     Color of current rank when running a parallel model.
    # _desvar_idx : dict
    #     Keeps track of the indices for each desvar, since GeneticAlgorithm sees an array of
    #     design variables.
    # _ga : <GeneticAlgorithm>
    #     Main genetic algorithm lies here.
    # _randomstate : np.random.RandomState, int
    #       Random state (or seed-number) which controls the seed and random draws.
    # _nfit : int
    #       Number of successful function evaluations.

    """

    def __init__(self, **kwargs):
        """
        initialize the NSGA2 driver.
        """

        # TODO: is this necessary???
        if lhs is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                "which can be installed with one of the following commands:\n"
                "    pip install openmdao[doe]\n"
                "    pip install pyDOE3"
            )

        super().__init__(**kwargs)

        # what we support
        self.supports["optimization"] = True
        self.supports["inequality_constraints"] = True
        self.supports["multiple_objectives"] = True
        self.supports["two_sided_constraints"] = True

        # what we don't support yet
        self.supports["integer_design_vars"] = False  # TODO: implement
        self.supports["equality_constraints"] = False
        self.supports["linear_constraints"] = False
        self.supports["simultaneous_derivatives"] = False
        self.supports["active_set"] = False
        self.supports["distributed_design_vars"] = False
        self.supports._read_only = True

        self._desvar_idx = {}

        # random state can be set for predictability during testing
        self._randomstate = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

        self._nfit = 0  # Number of successful function evaluations

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare(
            "max_gen",
            default=100,
            desc="Number of generations before termination.",
        )
        self.options.declare(
            "pop_size",
            default=0,
            desc="Number of points in the GA. Set to 0 and it will be computed " "as four times the number of bits.",
        )
        self.options.declare(
            "run_parallel",
            types=bool,
            default=False,
            desc="Set to True to execute the points in a generation in parallel.",
        )
        self.options.declare(
            "procs_per_model",
            default=1,
            lower=1,
            desc="Number of processors to give each model under MPI.",
        )
        self.options.declare(
            "penalty_parameter",
            default=0.0,
            lower=0.0,
            desc="Penalty function parameter.",
        )
        self.options.declare("penalty_exponent", default=1.0, desc="Penalty function exponent.")
        self.options.declare(
            "Pc",
            default=0.9,
            lower=0.0,
            upper=1.0,
            desc="Crossover rate.",
        )
        self.options.declare(
            "eta_c",
            default=20.0,
            lower=0.0,
            desc="Distribution index for crossover.",
        )
        self.options.declare(
            "Pm",
            default=0.1,
            lower=0.0,
            upper=1.0,
            allow_none=True,
            desc="Mutation rate.",
        )
        self.options.declare(
            "eta_m",
            default=20.0,
            lower=0.0,
            desc="Distribution index for mutation.",
        )

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        # check design vars and constraints for invalid bounds
        for name, meta in self._designvars.items():
            lower, upper = meta["lower"], meta["upper"]
            for param in (lower, upper):
                if param is None or np.all(np.abs(param) >= INF_BOUND):
                    msg = (
                        f"Invalid bounds for design variable '{name}'. When using "
                        f"{self.__class__.__name__}, values for both 'lower' and 'upper' "
                        f"must be specified between +/-INF_BOUND ({INF_BOUND}), "
                        f"but they are: lower={lower}, upper={upper}."
                    )
                    raise ValueError(msg)

        for name, meta in self._cons.items():
            equals, lower, upper = meta["equals"], meta["lower"], meta["upper"]
            if (
                (equals is None or np.all(np.abs(equals) >= INF_BOUND))
                and (lower is None or np.all(np.abs(lower) >= INF_BOUND))
                and (upper is None or np.all(np.abs(upper) >= INF_BOUND))
            ):
                msg = (
                    f"Invalid bounds for constraint '{name}'. "
                    f"When using {self.__class__.__name__}, the value for 'equals', "
                    f"'lower' or 'upper' must be specified between +/-INF_BOUND "
                    f"({INF_BOUND}), but they are: "
                    f"equals={equals}, lower={lower}, upper={upper}."
                )
                raise ValueError(msg)

        model_mpi = None
        comm = problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options["run_parallel"]:
            comm = None

        self.config_mpi = (comm, model_mpi)

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Here, we generate the model communicators.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        self._problem_comm = comm

        procs_per_model = self.options["procs_per_model"]
        if MPI and self.options["run_parallel"]:

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError(
                    "The total number of processors is not evenly divisible by the "
                    "specified number of processors per model.\n Provide a "
                    "number of processors that is a multiple of %d, or "
                    "specify a number of processors per model that divides "
                    "into %d." % (procs_per_model, full_size)
                )
            color = comm.rank % size
            model_comm = comm.Split(color)

            # everything we need to figure out which case to run
            self._concurrent_pop_size = size
            self._concurrent_color = color

            return model_comm

        self._concurrent_pop_size = 0
        self._concurrent_color = 0
        return comm

    def _setup_recording(self):
        """
        Set up case recording.
        """

        if MPI:
            run_parallel = self.options["run_parallel"]
            procs_per_model = self.options["procs_per_model"]

            for recorder in self._rec_mgr:
                if run_parallel:
                    # write cases only on procs up to the number of parallel models
                    # (i.e. on the root procs for the cases)
                    if procs_per_model == 1:
                        recorder.record_on_process = True
                    else:
                        size = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < size:
                            recorder.record_on_process = True

                elif self._problem_comm.rank == 0:
                    # if not running cases in parallel, then just record on proc 0
                    recorder.record_on_process = True

        super()._setup_recording()

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "NSGA2"

    def run(self):
        """
        Execute the genetic algorithm.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """

        self.result.reset()
        model = self._problem().model

        pop_size = self.options["pop_size"]
        max_gen = self.options["max_gen"]

        Pc = self.options["Pc"]  # if None, it will be calculated in execute_ga()
        eta_c = self.options["eta_c"]  # if None, it will be calculated in execute_ga()
        Pm = self.options["Pm"]  # if None, it will be calculated in execute_ga()
        eta_m = self.options["eta_m"]  # if None, it will be calculated in execute_ga()

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        # if compute_pareto:
        #     self._ga.nobj = len(self._objs)

        # size design variables
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()

        count = 0
        for name, meta in desvars.items():
            if name in self._designvars_discrete:
                val = desvar_vals[name]
                if np.ndim(val) == 0:
                    size = 1
                else:
                    size = len(val)
            else:
                size = meta["size"]
            self._desvar_idx[name] = (count, count + size)
            count += size

        lower_bound = np.empty((count,))
        upper_bound = np.empty((count,))
        outer_bound = np.full((count,), np.inf)
        x0 = np.empty(count)

        # figure out bounds vectors and initial design vars
        for name, meta in desvars.items():
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta["lower"]
            upper_bound[i:j] = meta["upper"]
            x0[i:j] = desvar_vals[name]

        # bits of resolution
        resolver = model._resolver  # TODO: delete?

        for name, meta in desvars.items():
            i, j = self._desvar_idx[name]

            if resolver.is_abs(name, "output"):
                prom_name = resolver.abs2prom(name, "output")
            else:
                prom_name = name

        # automatic population size
        if pop_size == 0:
            pop_size = 10 * count  # 10 per DV DOF

        if pop_size % 2 != 0:
            raise ValueError(
                f"NSGA2 requires an even population size, but pop_size={pop_size}. "
                "Please set pop_size to an even number."
            )

        # generate initial population using Latin Hypercube Sampling
        design_vars_init = lower_bound + (upper_bound - lower_bound) * lhs(
            count,
            pop_size,
            "center",
        )
        self.population_init = design_vars_init  # save the initial population for inspection

        # create a new NSGA2 instance
        self.icase = 0
        self.optimizer_nsga2 = NSGA2_implementation(
            design_vars_init,
            self.objective_callback,
            len(self._objs),
            len(self._cons),
            design_vars_l=lower_bound,
            design_vars_u=upper_bound,
            params_override=(Pc, eta_c, Pm, eta_m),
            comm_mpi=(self.config_mpi[0] if MPI and self.options["run_parallel"] else None),
            model_mpi=self.config_mpi[1],
            # verbose=True,
            verbose=False,
        )
        self.optimizer_nsga2.get_fronts()  # evaluate the initial fronts

        # get the debug output file for the openmdao nsga2 driver
        nsga2_output_dir = model.get_outputs_dir()
        nsga2_debug_collection = dict()

        rv = self.optimizer_nsga2.get_fronts(
            compute_constrs=True,
            feasibility_dominates=True,
        )
        nsga2_debug_collection[-1] = {
            "generation": -1,
            "design_vars_fronts": rv[1],
            "objs_fronts": rv[2],
            "constrs_fronts": rv[3],
        }
        # create a yaml file at the path
        os.makedirs(nsga2_output_dir, exist_ok=True)
        write_yaml(nsga2_debug_collection, nsga2_output_dir / "nsga2_debug.yaml")

        # iterate over the specified generations
        for generation in range(max_gen + 1):
            # iterate the population
            self.optimizer_nsga2.iterate_population()

            rv = self.optimizer_nsga2.get_fronts(
                compute_constrs=True,
                feasibility_dominates=True,
            )
            nsga2_debug_collection[generation] = {
                "generation": generation,
                "design_vars_fronts": rv[1],
                "objs_fronts": rv[2],
                "constrs_fronts": rv[3],
            }
            # create a yaml file at the path
            write_yaml(nsga2_debug_collection, nsga2_output_dir / "nsga2_debug.yaml")
            print(f"generation: {generation} of {max_gen}")

        # Save the non-dominated Pareto front — the whole point of NSGA2
        self.optimizer_nsga2.sort_data()

        rv = self.optimizer_nsga2.get_fronts(compute_constrs=True, feasibility_dominates=True)
        design_vars_fronts = rv[1]
        objs_fronts = rv[2]
        constrs_fronts = rv[3]
        self.desvar_nd = copy.deepcopy(design_vars_fronts[0])
        self.constr_nd = copy.deepcopy(constrs_fronts[0])
        self.obj_nd = copy.deepcopy(objs_fronts[0])

        # Set framework state to the median Pareto point and re-run
        median_idx = len(design_vars_fronts[0]) // 2
        desvar_new = design_vars_fronts[0][median_idx, :]
        for name in desvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, desvar_new[i:j])
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self._run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False

    def objective_callback(self, x):

        model = self._problem().model  # get the model
        success = 1  # flag

        objs = self.get_objective_values()  # extract the objectives
        nr_objectives = len(objs)  # count 'em

        constrs = self.get_constraint_values()  # extract the constraints
        nr_constraits = len(constrs)  # count 'em

        # verify if this is single-objective use
        if nr_objectives > 1:
            is_single_objective = False
        else:
            for obj in objs.items():
                is_single_objective = len(obj) == 1
                break

        # set the DVs
        out_of_bounds = False
        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

            # Check that design variables are within bounds
            if (
                not (self._designvars[name]["lower"] <= x[i:j]).all()
                or not (x[i:j] <= self._designvars[name]["upper"]).all()
            ):
                out_of_bounds = True
                break

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = INF_BOUND

        # execute the model under a debugger
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1

            if not out_of_bounds:
                try:
                    self._run_solve_nonlinear()
                except AnalysisError:
                    # tell the optimizer that this is a bad point
                    model._clear_iprint()
                    success = 0

                # get the objective values
                obj_values = self.get_objective_values()
                constr_values_raw = self.get_constraint_values()
            else:
                # if out of bounds, set the objective to a very large number and skip
                obj_values = {name: np.inf for name in self._objs}
                constr_values_raw = (
                    self.get_constraint_values()
                )  # get the constraint values, which should be all zeros, but since fitness is inf, it hopefully doesn't matter

            if is_single_objective:
                obj = list(obj_values.values())[0]
            else:
                obj = np.array([val for val in obj_values.values()]).flatten()

            constr_adjusted = []  # convert all bounds to leq zero
            for name, meta in self._cons.items():
                if (meta["lower"] <= -INF_BOUND / 10) and (
                    meta["upper"] <= INF_BOUND / 10
                ):  # within an order of magnitude of the inf bound
                    constr_adjusted.append(np.array(meta["upper"] - constr_values_raw[name]).flatten())
                elif (meta["lower"] >= -INF_BOUND / 10) and (
                    meta["upper"] >= INF_BOUND / 10
                ):  # within an order of magnitude of the inf bound
                    constr_adjusted.append(np.array(constr_values_raw[name] - meta["lower"]).flatten())
                elif (meta["lower"] >= -INF_BOUND / 10) and (
                    meta["upper"] <= INF_BOUND / 10
                ):  # within an order of magnitude of the inf bound
                    # add as sequential one-sided constraints
                    constr_adjusted.append(np.array(meta["upper"] - constr_values_raw[name]).flatten())
                    constr_adjusted.append(np.array(constr_values_raw[name] - meta["lower"]).flatten())
                else:
                    raise ValueError(
                        f"you've attempted to constraint {name} between numerically infinite values in both directions: \n{meta}"
                    )
            if len(constr_adjusted):
                constr = np.hstack(constr_adjusted)
            else:
                constr = np.array([])

        if self.options["penalty_parameter"] != 0:
            raise NotImplementedError("penalty-driven constraints not implemented.")

        return np.array(obj.tolist() + constr.tolist())
