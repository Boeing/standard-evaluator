"""Unit tests for EvaluatorOpenMdaoComponent.

Tests cover import resolution, constructor error handling, kwarg conflict
detection, discrete input/output rejection, response scaling validation,
and exception propagation.

Requirements: 1.1, 1.2, 1.3, 2.2, 2.4, 3.2, 3.3, 6.6, 7.4, 7.5, 7.6
"""

import numpy as np
import pandas as pd
import pytest

import openmdao.api as om

from standard_evaluator.components import EvaluatorOpenMdaoComponent
from standard_evaluator.components.evaluator_om_component import (
    EvaluatorOpenMdaoComponent as DirectImport,
)
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.problem import FloatVariable
from standard_evaluator.evaluators.test import HS100
import standard_evaluator as se


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleEvaluator(Evaluator):
    """Minimal evaluator for unit tests."""

    def _evaluate(self, sites: pd.DataFrame) -> None:
        sites["f"] = sites["x1"] + sites["x2"]


class FailingEvaluator(Evaluator):
    """Evaluator that always raises during evaluation."""

    def _evaluate(self, sites: pd.DataFrame) -> None:
        raise RuntimeError("Intentional failure during evaluation")


def _make_simple_evaluator() -> SimpleEvaluator:
    """Create a simple evaluator with 2 variables and 1 response."""
    opt_problem = se.utilities.create_opt_problem(num_independent=2, num_dependent=1)
    opt_problem.variables[0].name = "x1"
    opt_problem.variables[0].bounds = [-5.0, 5.0]
    opt_problem.variables[0].default = 1.0
    opt_problem.variables[1].name = "x2"
    opt_problem.variables[1].bounds = [-5.0, 5.0]
    opt_problem.variables[1].default = 2.0
    opt_problem.responses[0].name = "f"
    opt_problem.responses[0].bounds = (-np.inf, np.inf)
    opt_problem.objectives = ["f"]
    opt_problem.constraints = []
    return SimpleEvaluator(opt_problem=opt_problem)


def _make_evaluator_with_zero_scale() -> SimpleEvaluator:
    """Create an evaluator where a response has scale=0.0.

    Uses model_construct to bypass pydantic validation so we can test
    the component's own scale=0.0 guard in setup().
    """
    opt_problem = se.utilities.create_opt_problem(num_independent=2, num_dependent=1)
    opt_problem.variables[0].name = "x1"
    opt_problem.variables[0].bounds = [-5.0, 5.0]
    opt_problem.variables[0].default = 1.0
    opt_problem.variables[1].name = "x2"
    opt_problem.variables[1].bounds = [-5.0, 5.0]
    opt_problem.variables[1].default = 2.0
    # Bypass pydantic validation to inject scale=0.0
    bad_response = FloatVariable.model_construct(
        name="f",
        bounds=(-np.inf, np.inf),
        shift=0.0,
        scale=0.0,
        default=None,
        units=None,
        description="",
        options={},
        class_type="float",
    )
    opt_problem.responses[0] = bad_response
    opt_problem.objectives = ["f"]
    opt_problem.constraints = []
    return SimpleEvaluator(opt_problem=opt_problem)


def _make_failing_evaluator() -> FailingEvaluator:
    """Create an evaluator that raises during compute."""
    opt_problem = se.utilities.create_opt_problem(num_independent=2, num_dependent=1)
    opt_problem.variables[0].name = "x1"
    opt_problem.variables[0].bounds = [-5.0, 5.0]
    opt_problem.variables[0].default = 1.0
    opt_problem.variables[1].name = "x2"
    opt_problem.variables[1].bounds = [-5.0, 5.0]
    opt_problem.variables[1].default = 2.0
    opt_problem.responses[0].name = "f"
    opt_problem.responses[0].bounds = (-np.inf, np.inf)
    opt_problem.objectives = ["f"]
    opt_problem.constraints = []
    return FailingEvaluator(opt_problem=opt_problem)


# ---------------------------------------------------------------------------
# Requirement 1.1, 1.2, 1.3 — Import resolution
# ---------------------------------------------------------------------------


class TestImportResolution:
    """Verify that the component can be imported from the expected locations."""

    def test_import_from_components_package(self):
        """from standard_evaluator.components import EvaluatorOpenMdaoComponent"""
        assert EvaluatorOpenMdaoComponent is not None

    def test_import_from_module_directly(self):
        """from standard_evaluator.components.evaluator_om_component import ..."""
        assert DirectImport is not None
        assert DirectImport is EvaluatorOpenMdaoComponent

    def test_in_all_list(self):
        """EvaluatorOpenMdaoComponent is in __all__."""
        assert EvaluatorOpenMdaoComponent.__name__ in se.components.__all__


# ---------------------------------------------------------------------------
# Requirement 2.4 — TypeError when non-Evaluator is passed
# ---------------------------------------------------------------------------


class TestTypeErrorForNonEvaluator:
    """Constructor raises TypeError if the first arg is not an Evaluator."""

    def test_raises_on_string(self):
        with pytest.raises(TypeError, match="Evaluator"):
            EvaluatorOpenMdaoComponent("not an evaluator")

    def test_raises_on_integer(self):
        with pytest.raises(TypeError, match="Evaluator"):
            EvaluatorOpenMdaoComponent(42)

    def test_raises_on_dict(self):
        with pytest.raises(TypeError, match="Evaluator"):
            EvaluatorOpenMdaoComponent({"key": "value"})

    def test_raises_on_none_like_object(self):
        """An object that is not None but also not an Evaluator."""

        class FakeEvaluator:
            pass

        with pytest.raises(TypeError, match="Evaluator"):
            EvaluatorOpenMdaoComponent(FakeEvaluator())


# ---------------------------------------------------------------------------
# Requirement 3.2 — AttributeError when no evaluator and no evaluator_options
# ---------------------------------------------------------------------------


class TestAttributeErrorNoEvaluatorOptions:
    """Constructor raises AttributeError when neither evaluator nor options given."""

    def test_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="evaluator_options"):
            EvaluatorOpenMdaoComponent()

    def test_raises_with_irrelevant_kwargs(self):
        """Other kwargs present but not evaluator_options."""
        with pytest.raises(AttributeError, match="evaluator_options"):
            EvaluatorOpenMdaoComponent(evaluator=None, some_other_key="value")


# ---------------------------------------------------------------------------
# Requirement 2.2 — Kwarg conflict detection
# ---------------------------------------------------------------------------


class TestKwargConflictDetection:
    """Constructor raises a clear error on conflicting kwargs."""

    def test_raises_on_eval_kwarg(self):
        """'eval' conflicts with the stored evaluator attribute."""
        evaluator = _make_simple_evaluator()
        with pytest.raises(ValueError, match="eval"):
            EvaluatorOpenMdaoComponent(evaluator, eval="conflict")


# ---------------------------------------------------------------------------
# Requirement 7.4 — ValueError for discrete_inputs not None
# ---------------------------------------------------------------------------


class TestDiscreteInputsRejected:
    """compute() raises ValueError if discrete_inputs is not None."""

    def test_raises_value_error(self):
        evaluator = _make_simple_evaluator()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        comp = prob.model.comp
        # Call compute directly with discrete_inputs
        inputs = {"x1": np.array([1.0]), "x2": np.array([2.0])}
        outputs = {"f": np.array([0.0])}
        with pytest.raises(ValueError, match="discrete inputs"):
            comp.compute(inputs, outputs, discrete_inputs={"d": 1})


# ---------------------------------------------------------------------------
# Requirement 7.5 — ValueError for discrete_outputs not None
# ---------------------------------------------------------------------------


class TestDiscreteOutputsRejected:
    """compute() raises ValueError if discrete_outputs is not None."""

    def test_raises_value_error(self):
        evaluator = _make_simple_evaluator()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        comp = prob.model.comp
        inputs = {"x1": np.array([1.0]), "x2": np.array([2.0])}
        outputs = {"f": np.array([0.0])}
        with pytest.raises(ValueError, match="discrete outputs"):
            comp.compute(inputs, outputs, discrete_outputs={"d": 1})


# ---------------------------------------------------------------------------
# Requirement 6.6 — ValueError for response with scale=0.0
# ---------------------------------------------------------------------------


class TestScaleZeroRaisesValueError:
    """setup() raises ValueError when a response has scale=0.0."""

    def test_raises_value_error(self):
        evaluator = _make_evaluator_with_zero_scale()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        with pytest.raises(ValueError, match="scale value of 0.0"):
            prob.setup()


# ---------------------------------------------------------------------------
# Requirement 3.3 — Exception propagation from SurrogateModel.from_dict
# ---------------------------------------------------------------------------


class TestFromDictExceptionPropagation:
    """Exceptions from SurrogateModel.from_dict propagate to the caller."""

    def test_type_error_on_non_dict_options(self):
        """Passing a non-dict as evaluator_options triggers from_dict TypeError."""
        with pytest.raises(TypeError):
            EvaluatorOpenMdaoComponent(evaluator_options="not a dict")

    def test_key_error_on_missing_keys(self):
        """A dict missing required keys raises KeyError from from_dict."""
        with pytest.raises((KeyError, TypeError)):
            EvaluatorOpenMdaoComponent(evaluator_options={"incomplete": True})


# ---------------------------------------------------------------------------
# Requirement 7.6 — Exception propagation from evaluator during compute
# ---------------------------------------------------------------------------


class TestEvaluatorExceptionPropagation:
    """Exceptions raised by the evaluator during compute propagate unchanged."""

    def test_runtime_error_propagates(self):
        evaluator = _make_failing_evaluator()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        comp = prob.model.comp
        inputs = {"x1": np.array([1.0]), "x2": np.array([2.0])}
        outputs = {"f": np.array([0.0])}
        with pytest.raises(RuntimeError, match="Intentional failure"):
            comp.compute(inputs, outputs)


# ---------------------------------------------------------------------------
# Requirement 7.2, 7.4 — Backward compatibility for scalar-only OptProblems
# ---------------------------------------------------------------------------


class TestScalarOnlyComponentBackwardCompatibility:
    """Verify scalar-only OptProblems register identical inputs/outputs.

    Requirements: 7.2, 7.4
    """

    def test_scalar_inputs_registered_correctly(self):
        """Scalar-only OptProblem registers inputs with correct val and no shape argument."""
        evaluator = _make_simple_evaluator()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Check that inputs are registered as scalars
        comp = prob.model.comp
        # OpenMDAO stores input metadata; check the defaults
        x1_meta = comp.get_io_metadata(iotypes="input", metadata_keys=["val", "shape"])
        input_meta = {
            meta["prom_name"]: meta
            for _, meta in x1_meta.items()
        }
        assert "x1" in input_meta
        assert "x2" in input_meta
        # Scalar inputs should have shape (1,)
        assert input_meta["x1"]["shape"] == (1,)
        assert input_meta["x2"]["shape"] == (1,)
        # Check default values
        np.testing.assert_allclose(input_meta["x1"]["val"], [1.0])
        np.testing.assert_allclose(input_meta["x2"]["val"], [2.0])

    def test_scalar_outputs_registered_correctly(self):
        """Scalar-only OptProblem registers outputs with correct bounds and scaling."""
        evaluator = _make_simple_evaluator()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        comp = prob.model.comp
        output_meta = {
            meta["prom_name"]: meta
            for _, meta in comp.get_io_metadata(
                iotypes="output", metadata_keys=["val", "shape"]
            ).items()
        }
        assert "f" in output_meta
        # Scalar output should have shape (1,)
        assert output_meta["f"]["shape"] == (1,)

    def test_scalar_compute_produces_correct_output(self):
        """Scalar-only component compute produces identical results."""
        evaluator = _make_simple_evaluator()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()
        prob.set_val("x1", 3.0)
        prob.set_val("x2", 4.0)
        prob.run_model()

        # f = x1 + x2 = 7.0
        np.testing.assert_allclose(prob.get_val("f"), 7.0)

    def test_no_new_mandatory_constructor_params(self):
        """EvaluatorOpenMdaoComponent still works with just the evaluator argument.

        Requirements: 7.4
        """
        evaluator = _make_simple_evaluator()
        # This should work with just the single positional argument
        comp = EvaluatorOpenMdaoComponent(evaluator)
        assert comp is not None
        assert comp.eval is not None

    def test_scalar_only_with_hs100_evaluator(self):
        """Full scalar-only backward compat using HS100 (a real evaluator with scaling)."""
        evaluator = HS100()
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Set initial guess
        prob.set_val("x1", 1.0)
        prob.set_val("x2", 2.0)
        prob.set_val("x3", 0.0)
        prob.set_val("x4", 4.0)
        prob.set_val("x5", 0.0)
        prob.set_val("x6", 1.0)
        prob.set_val("x7", 1.0)
        prob.run_model()

        # HS100 known value at initial guess: f(1,2,0,4,0,1,1) = 714
        np.testing.assert_allclose(prob.get_val("f"), 714.0, rtol=1e-10)
