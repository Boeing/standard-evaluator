"""Unit and integration tests for set_variable_bounds and build_opt_problem.

Tests the workflow of setting bounds on a JoinedInfo interface description
and building an OptProblem from it for use with OpenMDAOEvaluator.
"""

import copy

import numpy as np
import pandas as pd
import openmdao.api as om

import standard_evaluator as se
from standard_evaluator.evaluators import OpenMDAOEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_aero_assembly():
    """Build a simple aero assembly and return the problem + info."""
    aero = om.Group()
    aero.add_subsystem('pressure_calc',
        om.ExecComp('pressure = 0.5 * rho * v**2', pressure={'tags': 'internal'}),
        promotes_inputs=['rho', 'v'])
    aero.add_subsystem('lift_calc',
        om.ExecComp('lift = cl * q * area', q={'tags': 'internal'}),
        promotes_inputs=['cl', 'area'], promotes_outputs=['lift'])
    aero.add_subsystem('drag_calc',
        om.ExecComp('drag = cd * q * area', q={'tags': 'internal'}),
        promotes_inputs=['cd', 'area'], promotes_outputs=['drag'])
    aero.connect('pressure_calc.pressure', 'lift_calc.q')
    aero.connect('pressure_calc.pressure', 'drag_calc.q')

    assembly = om.Group()
    assembly.add_subsystem('aero', aero, promotes=['*'])
    prob = om.Problem(model=assembly)
    prob.setup()
    prob.final_setup()
    prob.run_model()

    info = se.get_interface(assembly)
    aero_info = copy.deepcopy(info.components['aero'])
    return prob, aero_info


# ---------------------------------------------------------------------------
# Tests: set_variable_bounds
# ---------------------------------------------------------------------------

class TestSetVariableBounds:
    """Test the set_variable_bounds utility function."""

    def test_sets_bounds_on_matching_inputs(self):
        """Bounds are set on variables whose names match the dict keys."""
        _, aero_info = _build_aero_assembly()
        se.set_variable_bounds(aero_info, {'rho': (0.5, 2.0), 'v': (10.0, 100.0)})

        rho_var = next(v for v in aero_info.inputs if v.name == 'rho')
        v_var = next(v for v in aero_info.inputs if v.name == 'v')
        assert rho_var.bounds == (0.5, 2.0)
        assert v_var.bounds == (10.0, 100.0)

    def test_leaves_unmatched_inputs_unchanged(self):
        """Variables not in the bounds dict keep their original bounds."""
        _, aero_info = _build_aero_assembly()
        se.set_variable_bounds(aero_info, {'rho': (0.5, 2.0)})

        area_var = next(v for v in aero_info.inputs if v.name == 'area')
        assert area_var.bounds == (-np.inf, np.inf)

    def test_modifies_in_place(self):
        """The function modifies the info object in place (no return value)."""
        _, aero_info = _build_aero_assembly()
        result = se.set_variable_bounds(aero_info, {'rho': (1.0, 3.0)})
        assert result is None
        rho_var = next(v for v in aero_info.inputs if v.name == 'rho')
        assert rho_var.bounds == (1.0, 3.0)

    def test_empty_bounds_dict_is_noop(self):
        """An empty bounds dict changes nothing."""
        _, aero_info = _build_aero_assembly()
        original_bounds = [(v.name, v.bounds) for v in aero_info.inputs]
        se.set_variable_bounds(aero_info, {})
        new_bounds = [(v.name, v.bounds) for v in aero_info.inputs]
        assert original_bounds == new_bounds

    def test_nonexistent_name_is_ignored(self):
        """Names in the bounds dict that don't match any input are silently ignored."""
        _, aero_info = _build_aero_assembly()
        se.set_variable_bounds(aero_info, {'nonexistent_var': (0, 1)})
        # No error raised, all bounds unchanged
        for v in aero_info.inputs:
            assert v.bounds == (-np.inf, np.inf)


# ---------------------------------------------------------------------------
# Tests: build_opt_problem
# ---------------------------------------------------------------------------

class TestBuildOptProblem:
    """Test the build_opt_problem utility function."""

    def test_creates_opt_problem_from_info(self):
        """Returns an OptProblem with non-internal variables and responses."""
        _, aero_info = _build_aero_assembly()
        opt = se.build_opt_problem(aero_info)

        assert isinstance(opt, se.OptProblem)
        var_names = [v.name for v in opt.variables]
        resp_names = [r.name for r in opt.responses]
        # Internal variables (drag_calc.q, lift_calc.q) should be excluded
        assert 'rho' in var_names
        assert 'v' in var_names
        assert 'cl' in var_names
        assert 'cd' in var_names
        assert 'area' in var_names
        assert 'drag_calc.q' not in var_names
        assert 'lift_calc.q' not in var_names
        # Internal responses (pressure_calc.pressure) should be excluded
        assert 'drag' in resp_names
        assert 'lift' in resp_names
        assert 'pressure_calc.pressure' not in resp_names

    def test_preserves_bounds(self):
        """Bounds set on the info are reflected in the OptProblem variables."""
        _, aero_info = _build_aero_assembly()
        se.set_variable_bounds(aero_info, {'rho': (0.5, 2.0), 'cl': (0.1, 1.5)})
        opt = se.build_opt_problem(aero_info)

        rho_var = next(v for v in opt.variables if v.name == 'rho')
        cl_var = next(v for v in opt.variables if v.name == 'cl')
        assert rho_var.bounds == (0.5, 2.0)
        assert cl_var.bounds == (0.1, 1.5)

    def test_does_not_modify_original_info(self):
        """build_opt_problem deep-copies; mutating the result doesn't affect the info."""
        _, aero_info = _build_aero_assembly()
        se.set_variable_bounds(aero_info, {'rho': (0.5, 2.0)})
        opt = se.build_opt_problem(aero_info)

        # Mutate the opt_problem variable
        opt.variables[0].bounds = (99.0, 100.0)

        # Original info should be unchanged
        rho_var = next(v for v in aero_info.inputs if v.name == 'rho')
        assert rho_var.bounds == (0.5, 2.0)

    def test_uses_info_name(self):
        """The OptProblem name comes from the info's name field."""
        _, aero_info = _build_aero_assembly()
        opt = se.build_opt_problem(aero_info)
        assert opt.name == aero_info.name or opt.name == "from_interface"


# ---------------------------------------------------------------------------
# Tests: OpenMDAOEvaluator with opt_problem parameter
# ---------------------------------------------------------------------------

class TestOpenMDAOEvaluatorWithOptProblem:
    """Test the opt_problem parameter on OpenMDAOEvaluator."""

    def test_uses_provided_opt_problem(self):
        """When opt_problem is passed, the evaluator uses it directly."""
        _, aero_info = _build_aero_assembly()
        se.set_variable_bounds(aero_info, {
            'rho': (0.5, 2.0), 'v': (10.0, 100.0),
            'cl': (0.1, 1.5), 'cd': (0.01, 0.1), 'area': (5.0, 50.0),
        })
        opt = se.build_opt_problem(aero_info)
        aero_prob = se.create_problem(aero_info)

        evaluator = OpenMDAOEvaluator(aero_prob, opt_problem=opt)

        rho_var = next(v for v in evaluator.opt_problem.variables if v.name == 'rho')
        assert rho_var.bounds == (0.5, 2.0)

    def test_evaluator_can_evaluate(self):
        """The evaluator with an explicit opt_problem can run evaluations."""
        _, aero_info = _build_aero_assembly()
        bounds = {
            'rho': (0.5, 2.0), 'v': (10.0, 100.0),
            'cl': (0.1, 1.5), 'cd': (0.01, 0.1), 'area': (5.0, 50.0),
        }
        se.set_variable_bounds(aero_info, bounds)
        opt = se.build_opt_problem(aero_info)
        aero_prob = se.create_problem(aero_info)

        evaluator = OpenMDAOEvaluator(aero_prob, opt_problem=opt)

        # Build test data sampling within bounds
        np.random.seed(42)
        test_data = {}
        for v in evaluator.opt_problem.variables:
            low, high = v.bounds
            if np.isinf(low) or np.isinf(high):
                test_data[v.name] = np.ones(3) * 1.0
            else:
                test_data[v.name] = np.random.uniform(low, high, size=3)

        sites = pd.DataFrame(test_data)
        evaluator(sites)

        assert 'lift' in sites.columns
        assert 'drag' in sites.columns
        assert all(sites['lift'] > 0)
        assert all(sites['drag'] > 0)

    def test_backward_compatible_without_opt_problem(self):
        """Existing usage without opt_problem still works."""
        _, aero_info = _build_aero_assembly()
        aero_prob = se.create_problem(aero_info)

        evaluator = OpenMDAOEvaluator(
            aero_prob, scan_model=True, use_defined_problem=False
        )
        assert evaluator.opt_problem is not None
        assert len(evaluator.inputs) > 0
