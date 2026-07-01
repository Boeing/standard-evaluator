"""Integration test for mixed scalar/array round-trip.

Tests the full pipeline:
  OpenMDAO model → OpenMDAOEvaluator → verify correct OptProblem
  Evaluator with mixed types → EvaluatorOpenMdaoComponent → run in OpenMDAO group
  → verify outputs match direct model execution.

Requirements: 3.5, 6.4, 7.3
"""

import numpy as np
import pandas as pd
import openmdao.api as om

from standard_evaluator.components import EvaluatorOpenMdaoComponent
from standard_evaluator.evaluators import OpenMDAOEvaluator
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.problem import ArrayVariable, FloatVariable, OptProblem


# ---------------------------------------------------------------------------
# Test OpenMDAO component with both scalar and array I/O
# ---------------------------------------------------------------------------


class MixedScalarArrayComp(om.ExplicitComponent):
    """OpenMDAO component with both scalar and array inputs/outputs.

    Inputs:
        x (scalar): a float value
        arr (array, shape (3,)): a 3-element array

    Outputs:
        y (scalar): x ** 2
        result (array, shape (3,)): arr * 2 + x
    """

    def setup(self):
        self.add_input("x", val=1.0)
        self.add_input("arr", val=np.array([1.0, 2.0, 3.0]))

        self.add_output("y", val=0.0)
        self.add_output("result", val=np.zeros(3))

    def compute(self, inputs, outputs):
        x = inputs["x"][0]
        arr = inputs["arr"]
        outputs["y"] = x ** 2
        outputs["result"] = arr * 2 + x


class MixedEvaluator(Evaluator):
    """Evaluator implementing y = x**2 and result = arr*2 + x with mixed types.

    This is the same logic as MixedScalarArrayComp but implemented as a
    Standard Evaluator for use with EvaluatorOpenMdaoComponent.
    """

    def _evaluate(self, sites: pd.DataFrame) -> None:
        # Pre-initialize output columns (array columns need object dtype)
        sites["y"] = 0.0
        sites["result"] = pd.array([None] * len(sites), dtype=object)
        for idx in range(len(sites)):
            x = sites.iloc[idx]["x"]
            arr = sites.iloc[idx]["arr"]
            sites.at[idx, "y"] = float(x) ** 2
            sites.at[idx, "result"] = np.asarray(arr) * 2 + float(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_direct_problem():
    """Build a direct OpenMDAO problem with the mixed component, design vars, and responses."""
    prob = om.Problem()
    prob.model.add_subsystem("comp", MixedScalarArrayComp(), promotes=["*"])

    prob.model.add_design_var("x", lower=-10.0, upper=10.0)
    prob.model.add_design_var("arr", lower=-5.0, upper=5.0)

    prob.model.add_objective("y")
    prob.model.add_constraint("result", lower=-100.0, upper=100.0)

    prob.setup()
    prob.run_model()
    return prob


def _make_mixed_evaluator() -> MixedEvaluator:
    """Create a MixedEvaluator with mixed scalar/array inputs and outputs."""
    opt_problem = OptProblem(
        name="mixed_test",
        variables=[
            FloatVariable(
                name="x",
                bounds=(-10.0, 10.0),
                default=1.0,
            ),
            ArrayVariable(
                name="arr",
                shape=(3,),
                bounds=(-5.0, 5.0),
                default=np.array([1.0, 2.0, 3.0]),
            ),
        ],
        responses=[
            FloatVariable(
                name="y",
                bounds=(-np.inf, np.inf),
            ),
            ArrayVariable(
                name="result",
                shape=(3,),
                bounds=(-np.inf, np.inf),
            ),
        ],
        objectives=["y"],
        constraints=[],
    )
    return MixedEvaluator(opt_problem=opt_problem)


# ---------------------------------------------------------------------------
# Integration test: OpenMDAOEvaluator scanning mixed models
# ---------------------------------------------------------------------------


class TestOpenMDAOEvaluatorMixedModel:
    """Verify OpenMDAOEvaluator produces correct OptProblem from mixed models.

    Requirements: 3.5, 7.3
    """

    def test_evaluator_produces_correct_opt_problem_types(self):
        """OpenMDAOEvaluator produces FloatVariable for scalars and ArrayVariable for arrays."""
        prob = _build_direct_problem()
        evaluator = OpenMDAOEvaluator(prob, scan_model=True)
        opt = evaluator.opt_problem

        var_dict = {v.name: v for v in opt.variables}
        resp_dict = {r.name: r for r in opt.responses}

        # Scalar input x → FloatVariable
        assert "x" in var_dict
        assert isinstance(var_dict["x"], FloatVariable)
        assert not isinstance(var_dict["x"], ArrayVariable)

        # Array input arr → ArrayVariable with shape (3,)
        assert "arr" in var_dict
        assert isinstance(var_dict["arr"], ArrayVariable)
        assert var_dict["arr"].shape == (3,)

        # Scalar output y → FloatVariable
        assert "y" in resp_dict
        assert isinstance(resp_dict["y"], FloatVariable)
        assert not isinstance(resp_dict["y"], ArrayVariable)

        # Array output result → ArrayVariable with shape (3,)
        assert "result" in resp_dict
        assert isinstance(resp_dict["result"], ArrayVariable)
        assert resp_dict["result"].shape == (3,)

    def test_evaluator_produces_correct_design_var_bounds(self):
        """OpenMDAOEvaluator correctly maps bounds from design variables."""
        prob = _build_direct_problem()
        evaluator = OpenMDAOEvaluator(prob, scan_model=True)
        opt = evaluator.opt_problem

        var_dict = {v.name: v for v in opt.variables}

        # Scalar variable bounds
        assert var_dict["x"].bounds[0] == -10.0
        assert var_dict["x"].bounds[1] == 10.0

        # Array variable bounds
        arr_var = var_dict["arr"]
        np.testing.assert_array_equal(
            np.asarray(arr_var.bounds[0]), np.full(3, -5.0)
        )
        np.testing.assert_array_equal(
            np.asarray(arr_var.bounds[1]), np.full(3, 5.0)
        )

    def test_evaluator_evaluates_mixed_model_correctly(self):
        """OpenMDAOEvaluator evaluates mixed scalar/array sites correctly.

        Requirements: 3.5
        """
        prob = _build_direct_problem()
        evaluator = OpenMDAOEvaluator(prob, scan_model=True)

        # Create sites DataFrame with mixed types
        x_val = 3.0
        arr_val = np.array([1.5, -2.0, 4.0])

        sites = pd.DataFrame(
            {"x": [x_val], "arr": [arr_val]}
        )
        # Pre-create ALL output columns (the evaluator needs them to exist)
        # Array-valued columns need object dtype; scalar columns get float64
        for output_name in evaluator.outputs:
            resp_lookup = {r.name: r for r in evaluator.opt_problem.responses}
            if output_name in resp_lookup and isinstance(resp_lookup[output_name], ArrayVariable):
                sites[output_name] = pd.array([None] * len(sites), dtype=object)
            else:
                sites[output_name] = np.nan

        evaluator(sites)

        # Expected: y = x^2 = 9.0, result = arr*2 + x = [6.0, -1.0, 11.0]
        # The promoted names (from add_objective/add_constraint) are "y" and "result"
        np.testing.assert_allclose(sites["y"].iloc[0], 9.0, rtol=1e-12)
        np.testing.assert_allclose(
            sites["result"].iloc[0], np.array([6.0, -1.0, 11.0]), rtol=1e-12
        )

    def test_mixed_model_scalar_vars_are_float_variables(self):
        """In a mixed model, scalar variables remain FloatVariable (not ArrayVariable).

        Requirements: 7.3
        """
        prob = _build_direct_problem()
        evaluator = OpenMDAOEvaluator(prob, scan_model=True)
        opt = evaluator.opt_problem

        var_dict = {v.name: v for v in opt.variables}
        # x is scalar (shape (1,)) so must be FloatVariable
        x_var = var_dict["x"]
        assert type(x_var) is FloatVariable


# ---------------------------------------------------------------------------
# Integration test: EvaluatorOpenMdaoComponent with mixed types
# ---------------------------------------------------------------------------


class TestEvaluatorComponentMixedRoundTrip:
    """Full round-trip: evaluator with mixed types → component → OpenMDAO group.

    Requirements: 6.4, 7.3
    """

    def test_round_trip_outputs_match_direct_execution(self):
        """EvaluatorOpenMdaoComponent wrapping a mixed evaluator produces correct outputs.

        Requirements: 3.5, 6.4, 7.3
        """
        evaluator = _make_mixed_evaluator()

        # Build OpenMDAO problem with the wrapped evaluator
        wrapper_prob = om.Problem()
        wrapper_prob.model.add_subsystem(
            "wrapped", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        wrapper_prob.setup()

        # Set inputs
        x_val = 3.0
        arr_val = np.array([1.5, -2.0, 4.0])
        wrapper_prob.set_val("x", x_val)
        wrapper_prob.set_val("arr", arr_val)
        wrapper_prob.run_model()

        # Expected: y = x^2 = 9.0, result = arr*2 + x = [6.0, -1.0, 11.0]
        expected_y = x_val ** 2
        expected_result = arr_val * 2 + x_val

        np.testing.assert_allclose(
            wrapper_prob.get_val("y"), expected_y, rtol=1e-12
        )
        np.testing.assert_allclose(
            wrapper_prob.get_val("result"), expected_result, rtol=1e-12
        )

    def test_round_trip_with_different_inputs(self):
        """Round-trip produces correct results for different input combinations."""
        evaluator = _make_mixed_evaluator()

        wrapper_prob = om.Problem()
        wrapper_prob.model.add_subsystem(
            "wrapped", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        wrapper_prob.setup()

        test_cases = [
            (0.0, np.array([0.0, 0.0, 0.0])),
            (-5.0, np.array([1.0, 2.0, 3.0])),
            (2.5, np.array([-1.0, 0.5, 3.5])),
        ]

        for x_val, arr_val in test_cases:
            expected_y = x_val ** 2
            expected_result = arr_val * 2 + x_val

            wrapper_prob.set_val("x", x_val)
            wrapper_prob.set_val("arr", arr_val)
            wrapper_prob.run_model()

            np.testing.assert_allclose(
                wrapper_prob.get_val("y"), expected_y, rtol=1e-12,
                err_msg=f"y mismatch for x={x_val}"
            )
            np.testing.assert_allclose(
                wrapper_prob.get_val("result"), expected_result, rtol=1e-12,
                err_msg=f"result mismatch for x={x_val}, arr={arr_val}"
            )

    def test_component_registers_correct_shapes(self):
        """Component registers scalar inputs as shape (1,) and array inputs as shape (3,).

        Requirements: 7.3
        """
        evaluator = _make_mixed_evaluator()

        wrapper_prob = om.Problem()
        wrapper_prob.model.add_subsystem(
            "wrapped", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        wrapper_prob.setup()

        comp = wrapper_prob.model.wrapped
        input_meta = {
            meta["prom_name"]: meta
            for _, meta in comp.get_io_metadata(
                iotypes="input", metadata_keys=["val", "shape"]
            ).items()
        }

        # x is scalar → shape (1,)
        assert input_meta["x"]["shape"] == (1,)
        # arr is array → shape (3,)
        assert input_meta["arr"]["shape"] == (3,)

        output_meta = {
            meta["prom_name"]: meta
            for _, meta in comp.get_io_metadata(
                iotypes="output", metadata_keys=["val", "shape"]
            ).items()
        }

        # y is scalar → shape (1,)
        assert output_meta["y"]["shape"] == (1,)
        # result is array → shape (3,)
        assert output_meta["result"]["shape"] == (3,)

    def test_full_pipeline_openmdao_to_evaluator_to_component(self):
        """Full integration: OpenMDAO model → evaluator scan → same math in component.

        Creates an OpenMDAO model, wraps with OpenMDAOEvaluator (verifying
        the OptProblem), then uses a matching evaluator in a component to
        verify the full round-trip produces identical results.

        Requirements: 3.5, 6.4, 7.3
        """
        # Step 1: Build and run the direct OpenMDAO problem
        direct_prob = _build_direct_problem()
        x_val = 3.0
        arr_val = np.array([1.5, -2.0, 4.0])

        direct_prob.set_val("x", x_val)
        direct_prob.set_val("arr", arr_val)
        direct_prob.run_model()

        expected_y = direct_prob.get_val("y").copy()
        expected_result = direct_prob.get_val("result").copy()

        # Step 2: Verify OpenMDAOEvaluator scans the model correctly
        om_evaluator = OpenMDAOEvaluator(direct_prob, scan_model=True)
        opt = om_evaluator.opt_problem
        var_dict = {v.name: v for v in opt.variables}
        resp_dict = {r.name: r for r in opt.responses}

        assert isinstance(var_dict["x"], FloatVariable)
        assert not isinstance(var_dict["x"], ArrayVariable)
        assert isinstance(var_dict["arr"], ArrayVariable)
        assert isinstance(resp_dict["result"], ArrayVariable)

        # Step 3: Use a matching evaluator (same opt_problem structure, same math)
        # wrapped in EvaluatorOpenMdaoComponent and run in a new OpenMDAO group
        mixed_evaluator = _make_mixed_evaluator()

        wrapper_prob = om.Problem()
        wrapper_prob.model.add_subsystem(
            "wrapped", EvaluatorOpenMdaoComponent(mixed_evaluator), promotes=["*"]
        )
        wrapper_prob.setup()

        wrapper_prob.set_val("x", x_val)
        wrapper_prob.set_val("arr", arr_val)
        wrapper_prob.run_model()

        # Step 4: Verify outputs match direct OpenMDAO execution
        np.testing.assert_allclose(
            wrapper_prob.get_val("y"), expected_y, rtol=1e-12
        )
        np.testing.assert_allclose(
            wrapper_prob.get_val("result"), expected_result, rtol=1e-12
        )
