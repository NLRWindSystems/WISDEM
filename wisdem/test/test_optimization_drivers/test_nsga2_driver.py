"""Tests for the NSGA2Driver and its integration with gc_PoseOptimization schema fields."""

import unittest
import numpy as np
import openmdao.api as om

from wisdem.optimization_drivers.nsga2_driver import NSGA2Driver


def _make_two_obj_problem():
    """Build a simple 2-objective OpenMDAO problem for testing NSGA2Driver.

    Minimize f1 = x^2  and  f2 = (x - 2)^2  subject to x in [0, 3].
    Pareto front: x in [0, 2], trading off f1 vs f2.
    """

    class TwoObj(om.ExplicitComponent):
        def setup(self):
            self.add_input("x", val=1.0)
            self.add_output("f1", val=0.0)
            self.add_output("f2", val=0.0)

        def compute(self, inputs, outputs):
            x = inputs["x"][0]
            outputs["f1"] = x ** 2
            outputs["f2"] = (x - 2.0) ** 2

    prob = om.Problem()
    prob.model.add_subsystem("comp", TwoObj(), promotes=["*"])
    prob.model.add_design_var("x", lower=0.0, upper=3.0)
    prob.model.add_objective("f1")
    prob.model.add_objective("f2")
    return prob



class TestNSGA2DriverRun(unittest.TestCase):
    """Integration tests: run the driver on a minimal 2-objective problem."""

    def _run(self, pop_size=10, max_gen=2, Pc=0.9, Pm=0.1):
        prob = _make_two_obj_problem()
        prob.driver = NSGA2Driver()
        prob.driver.options["pop_size"] = pop_size
        prob.driver.options["max_gen"] = max_gen
        prob.driver.options["Pc"] = Pc
        prob.driver.options["Pm"] = Pm
        prob.setup()
        prob.run_driver()
        return prob

    def test_run_completes(self):
        """Driver runs to completion without error."""
        prob = self._run()
        self.assertIsNotNone(prob["x"])

    def test_pareto_front_populated(self):
        """desvar_nd and obj_nd are always populated after run."""
        prob = self._run()
        driver = prob.driver
        self.assertTrue(hasattr(driver, "desvar_nd"), "desvar_nd not set")
        self.assertTrue(hasattr(driver, "obj_nd"), "obj_nd not set")
        self.assertGreater(len(driver.desvar_nd), 0)
        self.assertGreater(len(driver.obj_nd), 0)

    def test_pop_size_sets_population(self):
        """pop_size controls the initial population size."""
        for size in [10, 20]:
            prob = _make_two_obj_problem()
            prob.driver = NSGA2Driver()
            prob.driver.options["pop_size"] = size
            prob.driver.options["max_gen"] = 1
            prob.setup()
            prob.run_driver()
            self.assertEqual(prob.driver.population_init.shape[0], size)

    def test_odd_pop_size_raises(self):
        """Odd pop_size raises a clear ValueError before entering the algorithm."""
        prob = _make_two_obj_problem()
        prob.driver = NSGA2Driver()
        prob.driver.options["pop_size"] = 11
        prob.setup()
        with self.assertRaises(ValueError, msg="pop_size=11"):
            prob.run_driver()

    def test_pareto_front_feasible(self):
        """All points on the Pareto front satisfy the design variable bounds."""
        # pop_size must be even for NSGA2 SBX crossover
        prob = self._run(pop_size=16, max_gen=3)
        driver = prob.driver
        for row in driver.desvar_nd:
            self.assertGreaterEqual(row[0], 0.0 - 1e-8)
            self.assertLessEqual(row[0], 3.0 + 1e-8)


class TestNSGA2PoseOptimizationIntegration(unittest.TestCase):
    """Verify that gc_PoseOptimization._set_optimizer_properties correctly maps
    schema fields to NSGA2Driver options."""

    def test_set_optimizer_properties(self):
        """_set_optimizer_properties transfers pop_size, Pc, Pm, run_parallel from schema to driver options."""
        from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

        opt = {
            "general": {"folder_output": "outputs"},
            "driver": {
                "optimization": {
                    "flag": True,
                    "solver": "NSGA2",
                    "tol": 1e-6,
                    "max_iter": 10,
                    "step_size": 1e-3,
                    "form": "central",
                    "step_calc": "None",
                    "debug_print": False,
                    "pop_size": 30,
                    "max_gen": 2,
                    "run_parallel": False,
                    "Pc": 0.75,
                    "Pm": 0.05,
                },
                "design_of_experiments": {"flag": False},
            },
        }

        pose = PoseOptimization(wt_init={}, modeling_options={}, analysis_options=opt)

        prob = _make_two_obj_problem()
        prob.driver = NSGA2Driver()
        prob = pose._set_optimizer_properties(prob, options_keys=["pop_size", "max_gen", "run_parallel", "Pc", "Pm"])

        self.assertEqual(prob.driver.options["pop_size"], 30)
        self.assertEqual(prob.driver.options["max_gen"], 2)
        self.assertFalse(prob.driver.options["run_parallel"])
        self.assertAlmostEqual(prob.driver.options["Pc"], 0.75)
        self.assertAlmostEqual(prob.driver.options["Pm"], 0.05)


if __name__ == "__main__":
    unittest.main()
