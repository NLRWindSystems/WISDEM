"""
OpenMDAO Wrapper for the NLopt package of optimizers.

More info at https://nlopt.readthedocs.io/
"""

import numpy as np
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.mpi import MPI

try:
    from openmdao.utils.class_util import weak_method_wrapper as weak_method_wrapper
except ImportError:
    from openmdao.utils.class_util import WeakMethodWrapper as weak_method_wrapper


try:
    import nlopt
except ImportError:
    nlopt = None


# All optimizers in NLopt that we support and their corresponding package name.
# Other optimizers could be added, but we've focused on those that can
# handle either inequality or equality constraints.
if nlopt is not None:
    optimizer_methods = {
        "GN_DIRECT": nlopt.GN_DIRECT,
        "GN_DIRECT_L": nlopt.GN_DIRECT_L,
        "GN_DIRECT_L_NOSCAL": nlopt.GN_DIRECT_L_NOSCAL,
        "GN_ORIG_DIRECT": nlopt.GN_ORIG_DIRECT,
        "GN_ORIG_DIRECT_L": nlopt.GN_ORIG_DIRECT_L,
        "GN_AGS": nlopt.GN_AGS,
        "GN_ISRES": nlopt.GN_ISRES,
        "LN_COBYLA": nlopt.LN_COBYLA,
        "LD_MMA": nlopt.LD_MMA,
        "LD_CCSAQ": nlopt.LD_CCSAQ,
        "LD_SLSQP": nlopt.LD_SLSQP,
    }
else:
    optimizer_methods = {}

_optimizers = set(optimizer_methods)

# Define subsets of optimizers that support different functions
_gradient_optimizers = {"LD_MMA", "LD_SLSQP", "LD_CCSAQ"}
_bounds_optimizers = _optimizers
_constraint_optimizers = {
    "LD_SLSQP",
    "LN_COBYLA",
    "LD_MMA",
    "LD_CCSAQ",
    "GN_ORIG_DIRECT",
    "GN_ORIG_DIRECT_L",
    "GN_AGS",
    "GN_ISRES",
}
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers
_eq_constraint_optimizers = {"LD_SLSQP", "LN_COBYLA", "GN_ISRES"}
_global_optimizers = {
    "GN_DIRECT",
    "GN_DIRECT_L",
    "GN_ORIG_DIRECT",
    "GN_ORIG_DIRECT_L",
    "GN_DIRECT_L_NOSCAL",
    "GN_AGS",
    "GN_ISRES",
}

CITATIONS = """
@article{johnson_nlopt
 author = {Johnson, Steven G.},
 title = "{The NLopt nonlinear-optimization package}",
 url = {http://github.com/stevengj/nlopt},
 }
"""


class NLoptDriver(Driver):
    """
    Driver wrapper for NLopt.

    NLopt is an open-source nonlinear optimization framework.

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from NLopt.optimize call.
    _con_cache : dict
        Cached result of constraint evaluations because NLopt asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : OrderedDict
        Cached result of nonlinear constraint derivatives because NLopt asks for them in a separate
        function.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _obj_and_nlcons : list
        List of objective + nonlinear constraints. Used to compute total derivatives
        for all except linear constraints.
    _dvlist : list
        Copy of _designvars.
    _lincongrad_cache : np.ndarray
        Pre-calculated gradients of linear constraints.
    """

    def __init__(self, **kwargs):
        """
        Initialize the NLoptDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        if nlopt is None:
            raise RuntimeError("NLoptDriver is not available, NLopt is not installed.")

        super().__init__(**kwargs)

        # What we support
        self.supports["inequality_constraints"] = True
        self.supports["equality_constraints"] = True
        self.supports["two_sided_constraints"] = True
        self.supports["linear_constraints"] = True
        self.supports["simultaneous_derivatives"] = True

        # What we don't support
        self.supports["multiple_objectives"] = False
        self.supports["active_set"] = False
        self.supports["integer_design_vars"] = False
        self.supports._read_only = True

        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.iter_count = 0
        self._exc_info = None

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare("optimizer", "LD_SLSQP", values=_optimizers, desc="Name of optimizer to use")
        self.options.declare(
            "tol",
            1.0e-6,
            lower=0.0,
            desc="Tolerance for termination. Based on "
            + "relative function value change. Uses the "
            + "method `set_ftol_rel()` from NLOpt.",
        )
        self.options.declare("maxiter", 200, lower=0, desc="Maximum number of iterations.")
        self.options.declare("numgen", 10000, lower=0, desc="Maximum number of iterations.")
        self.options.declare(
            "maxtime",
            1.0e10,
            lower=0.0,
            desc="Maximum time in seconds to perform optimization.",
        )
        self.options.declare(
            "xtol",
            1.0e-12,
            lower=0.0,
            desc="Tolerance for termination. Based on "
            + "relative function value change. Uses the "
            + "method `set_xtol_rel()` from NLOpt.",
        )

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "NLopt_" + self.options["optimizer"]

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super()._setup_driver(problem)
        opt = self.options["optimizer"]

        self.supports._read_only = False
        self.supports["gradients"] = opt in _gradient_optimizers
        self.supports["inequality_constraints"] = opt in _constraint_optimizers
        self.supports["two_sided_constraints"] = opt in _constraint_optimizers
        self.supports["equality_constraints"] = opt in _eq_constraint_optimizers
        self.supports._read_only = True

        # Raises error if multiple objectives are not supported, but more objectives were defined.
        if not self.supports["multiple_objectives"] and len(self._objs) > 1:
            msg = "{} currently does not support multiple objectives."
            raise RuntimeError(msg.format(self.msginfo))

        self._model_ran = False
        self._setup_tot_jac_sparsity()
        
    def run(self):
        """
        Optimize the problem using selected NLopt optimizer.
        """
        self.result.reset()
        problem = self._problem()
        opt = self.options["optimizer"]
        self.iter_count = 0
        self._total_jac = None
        self._total_jac_linear = None
        self._desvar_array_cache = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        # Initial Run
        model_ran = False
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()
            model_ran = True
            self.iter_count += 1

        self._model_ran = model_ran
        self._coloring_info.run_model = not model_ran

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

        # Size Problem
        ndesvar = 0
        for desvar in self._designvars.values():
            size = desvar['size']
            ndesvar += size
        x_init = np.empty(ndesvar)

        if ndesvar == 0:
            raise RuntimeError('Problem has no design variables.')

        # Initialize the NLopt problem with the method and number of design vars
        opt_prob = nlopt.opt(optimizer_methods[opt], int(ndesvar))

        # Initial Design Vars
        i = 0
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = []
        else:
            bounds = None

        dv_bounds = self._autoscaler.get_bounds_scaling('design_var')

        for name, meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            x_init[i:i + size] = desvar_vals[name]
            i += size

            # Bounds if our optimizer supports them
            if use_bounds:
                meta_low = dv_bounds[name].lower
                meta_high = dv_bounds[name].upper
                for j in range(size):
                    p_low = None if meta_low is None else meta_low[j]
                    p_high = None if meta_high is None else meta_high[j]

                    if p_low is not None and np.isneginf(p_low):
                        p_low = None
                    if p_high is not None and np.isposinf(p_high):
                        p_high = None

                    bounds.append((p_low, p_high))

        # Actually add the bounds to the optimization problem.
        if bounds is not None:
            lower_bound, upper_bound = zip(*bounds)
            lower = np.array([x if x is not None else -np.inf for x in lower_bound])
            upper = np.array([x if x is not None else np.inf for x in upper_bound])
            opt_prob.set_lower_bounds(lower)
            opt_prob.set_upper_bounds(upper)

        # Constraints
        nl_i = 1  # start at 1 since row 0 is the objective.  Constraints start at row 1.
        lin_i = 0  # counter for linear constraint jacobian
        lincons = []  # list of linear constraints
        self._obj_and_nlcons = list(self._objs)

        if opt in _constraint_optimizers:
            # get list of linear constraints and precalculate gradients for them (if any)
            if opt in _constraint_grad_optimizers:
                lincons = [name for name, meta in self._cons.items() if meta.get('linear')]
            else:
                lincons = []

            if lincons:
                self._lincongrad_cache = self._compute_totals(of=lincons, wrt=self._dvlist, return_format='array')
            else:
                self._lincongrad_cache = None

            con_bounds = self._autoscaler.get_bounds_scaling('constraint')

            # map constraints to index and instantiate constraints for scipy
            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    meta['size'] = size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']
                upper = con_bounds[name].upper
                lower = con_bounds[name].lower
                equals = con_bounds[name].equals if meta['equals'] is not None else None
                linear = name in lincons

                if linear:
                    self._con_idx[name] = lin_i
                    lin_i += size
                else:
                    self._obj_and_nlcons.append(name)
                    self._con_idx[name] = nl_i
                    nl_i += size

                # Loop over every index separately,
                # because it's easier to defined each
                # constraint by index.
                for j in range(size):
                    # Equality constraints are added as two inequality constraints
                    if equals is not None:
                        args = [name, False, j]
                        try:
                            opt_prob.add_equality_constraint(
                                signature_extender(weak_method_wrapper(self, "_confunc"), args)
                            )
                        except ValueError:
                            msg = (
                                "The selected optimizer, {}, does not support"
                                + " equality constraints. Select from {}."
                            )
                            raise NotImplementedError(msg.format(opt, _eq_constraint_optimizers))

                    else:
                        # Double-sided constraints are accepted by the algorithm
                        args = [name, False, j]
                        opt_prob.add_inequality_constraint(
                            signature_extender(weak_method_wrapper(self, "_confunc"), args)
                        )

                        upper_j = np.inf if upper is None else upper[j]
                        lower_j = -np.inf if lower is None else lower[j]

                        dblcon = np.isfinite(upper_j) and np.isfinite(lower_j)

                        # Add extra constraint if double-sided
                        if dblcon:
                            args = [name, True, j]
                            opt_prob.add_inequality_constraint(
                                signature_extender(weak_method_wrapper(self, "_confunc"), args)
                            )

            # precalculate gradients of linear constraints
            if lincons:
                self._lincongrad_cache = self._compute_totals(of=lincons, wrt=self._dvlist, return_format="array")
            else:
                self._lincongrad_cache = None

        # compute dynamic simul deriv coloring if option is set
        problem.get_total_coloring(self._coloring_info, run_model=False)

        # Finalize the optimization problem setup and actually perform optimization
        try:
            if opt in _optimizers:
                opt_prob.set_min_objective(self._objfunc)
                opt_prob.set_ftol_rel(self.options["tol"])
                opt_prob.set_xtol_rel(self.options["xtol"])
                opt_prob.set_maxeval(int(self.options["maxiter"]))
                opt_prob.set_maxtime(self.options["maxtime"])
                opt_prob.set_population(int(self.options["maxiter"] / self.options["numgen"]))
                opt_prob.optimize(x_init)

            else:
                msg = 'Optimizer "{}" is not implemented yet. Choose from: {}'
                raise NotImplementedError(msg.format(opt, _optimizers))

        # If an exception was swallowed in one of our callbacks, we want to raise it
        except Exception:
            if self._exc_info is not None:
                self._reraise()
            else:
                raise

        if self._exc_info is not None:
            self._reraise()

            
    def _objfunc(self, x_new, grad):
        """
        Evaluate and return the objective function.

        Model is executed here.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.
        grad : ndarray
            Empty array that is modified in-place with gradient information for
            the new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        
        model = self._problem().model
        dv_vec = self._vectors['design_var']
        f_new = 1e10

        try:

            # Broadcast same x_new to all ranks
            if MPI:
                model.comm.Bcast(x_new, root=0)
                model.comm.Bcast(grad, root=0)

            # Update the cached design variable vector
            dv_vec.set_data(x_new, driver_scaling=True)

            # Design variables in dv_vec are now in optimizer scaled space and in driver-units.
            self._set_design_vars(driver_scaling=True)

            with RecordingDebugging(self._get_name(), self.iter_count, self):
                self.iter_count += 1
                self._run_solve_nonlinear()
                self._model_ran = True

            # Get the objective function evaluations
            f_new = list(self.get_objective_values().values())[0]

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = msg

        try:
            if grad.size > 0:
                self._grad_cache = self._compute_totals(
                    of=self._obj_and_nlcons, wrt=self._dvlist, return_format="array"
                )
                # Bug in OpenMDAO return_format that sometimes gives dict instead
                if isinstance(self._grad_cache, dict):
                    temp = []
                    for k in self._grad_cache.keys():
                        val = self._grad_cache[k].flatten()
                        temp = np.r_[temp, val]
                    self._grad_cache = temp.reshape((-1, grad.size))
                grad[:] = self._grad_cache[0, :]

        except Exception as msg:
            self._exc_info = msg

        return float(f_new[0])

    def _confunc(self, x_new, grad, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        Note that this function is called for each constraint, so the model is only run when the
        objective is evaluated.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.
        grad : ndarray
            Empty array that is modified in-place with gradient information for
            the new design point.
        name : string
            Name of the constraint to be evaluated.
        dbl : bool
            True if double sided constraint.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        if self._exc_info is not None:
            self._reraise()

        cons = self._con_cache
        meta = self._cons[name]

        con_bounds = self._autoscaler.get_bounds_scaling('constraint')

        if meta["linear"]:
            grad_cache = self._lincongrad_cache
        else:
            grad_cache = self._grad_cache

        grad_idx = self._con_idx[name] + idx

        # Equality constraints
        if meta['equals'] is not None:
            eq = con_bounds[name].equals
            if grad.size > 0:
                grad[:] = grad_cache[grad_idx, :]
            return cons[name][idx] - eq[idx]
        
        # Equality constraints
        #equals = meta["equals"]
        #if equals is not None:
        #    if isinstance(equals, np.ndarray):
        #        equals = equals[idx]
        #    if grad.size > 0:
        #        grad[:] = grad_cache[grad_idx, :]
        #    return cons[name][idx] - equals

        # Note, NLopt defines constraints to be satisfied when negative,
        # which is the same as OpenMDAO.
        upper_arr = con_bounds[name].upper
        lower_arr = con_bounds[name].lower
        upper = np.inf if upper_arr is None else upper_arr[idx]
        lower = -np.inf if lower_arr is None else lower_arr[idx]

        if dbl or np.isneginf(lower):
            if grad.size > 0:
                grad[:] = grad_cache[grad_idx, :]
            return cons[name][idx] - upper
        else:
            if grad.size > 0:
                grad[:] = -grad_cache[grad_idx, :]
            return lower - cons[name][idx]

    def _reraise(self):
        """
        Reraise any exception encountered when NLopt calls back into our method.
        """
        exc = self._exc_info
        raise exc


def signature_extender(fcn, extra_args):
    """
    Closure function, which appends extra arguments to the original function call.

    The first argument is the design vector and the second is the gradient vector.
    The possible extra arguments from the callback
    of :func:`NLopt.optimize` are not passed to the function.

    Parameters
    ----------
    fcn : callable
        Function, which takes the design vector as the first argument.
    extra_args : tuple or list
        Extra arguments for the function

    Returns
    -------
    callable
        The function with the signature expected by the driver.
    """

    def closure(x, grad, *args):
        return fcn(x, grad, *extra_args)

    return closure
