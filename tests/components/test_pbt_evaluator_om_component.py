"""Property-based tests for EvaluatorOpenMdaoComponent.

Uses Hypothesis to validate correctness properties of the OpenMDAO component
wrapper across a range of inputs and evaluator configurations.

# Feature: openmdao-component-migration
"""

import numpy as np
import pandas as pd
import pytest

import openmdao.api as om
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from standard_evaluator.components import EvaluatorOpenMdaoComponent
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.evaluators.test import (
    HS100,
    Rosenbrock,
    Sphere,
    ConstrainedBetts,
)
from standard_evaluator.problem import OptProblem, FloatVariable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleEvaluator(Evaluator):
    """Minimal evaluator for property-based tests."""

    __test__ = False

    def _evaluate(self, sites: pd.DataFrame) -> None:
        for resp in self.outputs:
            sites[resp] = sites[self.inputs[0]]


def _create_evaluator_instance(evaluator_cls):
    """Create a TestEvaluator instance, handling classes that require kwargs."""
    if evaluator_cls in (Rosenbrock, Sphere):
        return evaluator_cls(num_independent=2, num_dependent=1)
    return evaluator_cls()


def _get_variable_bounds(evaluator):
    """Extract variable bounds as a list of (lower, upper) tuples."""
    bounds = []
    for var in evaluator.opt_problem.variables:
        lower, upper = var.bounds
        bounds.append((float(lower), float(upper)))
    return bounds


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for finite float bounds where lower < upper
_finite_bounds_strategy = st.tuples(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
).map(lambda t: (min(t), max(t))).filter(lambda t: t[0] < t[1])

# Strategy for optional defaults
_default_strategy = st.one_of(
    st.none(),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

# Strategy for mutations to apply to opt_problem after component construction
_mutation_strategy = st.sampled_from([
    "rename_variable",
    "change_bounds",
    "change_default",
    "add_variable",
    "change_response_name",
])


@st.composite
def evaluator_with_varied_opt_problem(draw):
    """Generate a SimpleEvaluator with a randomized opt_problem structure.

    Generates 1-3 variables and 1-2 responses with varied bounds and defaults.
    """
    num_vars = draw(st.integers(min_value=1, max_value=3))
    num_resp = draw(st.integers(min_value=1, max_value=2))

    # Build variables with random bounds and defaults
    variables = []
    for i in range(num_vars):
        bounds = draw(_finite_bounds_strategy)
        default = draw(_default_strategy)
        # Clamp default within bounds if provided
        if default is not None:
            default = max(bounds[0], min(bounds[1], default))
        variables.append(
            FloatVariable(name=f"x{i}", bounds=bounds, default=default)
        )

    # Build responses
    responses = []
    for i in range(num_resp):
        responses.append(FloatVariable(name=f"f{i}"))

    opt_problem = OptProblem(
        variables=variables,
        responses=responses,
        objectives=[responses[0].name],
    )

    return SimpleEvaluator(opt_problem=opt_problem)


# ---------------------------------------------------------------------------
# Property 1: Deep Copy Isolation
# Feature: openmdao-component-migration, Property 1: Deep Copy Isolation
# ---------------------------------------------------------------------------


def _apply_mutation(evaluator: Evaluator, mutation: str) -> None:
    """Apply a mutation to the evaluator's opt_problem to test isolation."""
    opt = evaluator.opt_problem

    if mutation == "rename_variable":
        opt.variables[0].name = "mutated_var_name"
    elif mutation == "change_bounds":
        opt.variables[0].bounds = (-999.0, 999.0)
    elif mutation == "change_default":
        opt.variables[0].default = 42.42
    elif mutation == "add_variable":
        new_var = FloatVariable(name="new_injected_var", bounds=(-1.0, 1.0))
        opt.variables.append(new_var)
    elif mutation == "change_response_name":
        opt.responses[0].name = "mutated_resp_name"


class TestDeepCopyIsolation:
    """Property 1: Deep Copy Isolation.

    For any Evaluator instance used to construct the component, mutating the
    original Evaluator's opt_problem after construction SHALL NOT affect the
    component's stored evaluator.

    **Validates: Requirements 2.1**
    """

    @given(
        evaluator=evaluator_with_varied_opt_problem(),
        mutation=_mutation_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_deep_copy_isolation(self, evaluator, mutation):
        """Mutating the original evaluator does not affect the component."""
        # Snapshot the component's evaluator state after construction
        component = EvaluatorOpenMdaoComponent(evaluator)
        component_vars_snapshot = [
            v.model_dump() for v in component.eval.opt_problem.variables
        ]
        component_resps_snapshot = [
            r.model_dump() for r in component.eval.opt_problem.responses
        ]

        # Mutate the original evaluator's opt_problem
        _apply_mutation(evaluator, mutation)

        # Verify the component's stored evaluator is completely unaffected
        component_vars_after = [
            v.model_dump() for v in component.eval.opt_problem.variables
        ]
        component_resps_after = [
            r.model_dump() for r in component.eval.opt_problem.responses
        ]

        assert component_vars_snapshot == component_vars_after, (
            f"Component variables were affected by mutation '{mutation}' "
            f"on original evaluator"
        )
        assert component_resps_snapshot == component_resps_after, (
            f"Component responses were affected by mutation '{mutation}' "
            f"on original evaluator"
        )


# ---------------------------------------------------------------------------
# Property 5: Response Scaling Formula
# Feature: openmdao-component-migration, Property 5: Response Scaling Formula
# ---------------------------------------------------------------------------


class TestResponseScalingFormula:
    """Property 5: Response Scaling Formula.

    For any response Variable with shift and scale values (where scale != 0),
    the OpenMDAO output ref0 SHALL equal -shift and ref SHALL equal
    (1.0 / scale) + ref0.

    **Validates: Requirements 6.5**
    """

    @given(
        shift=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_scaling_formula(self, shift: float, scale: float):
        """ref0 = -shift and ref = (1.0/scale) + ref0 for any non-zero scale."""
        # scale must not be zero
        assume(abs(scale) > 1e-15)

        # Build a response variable with the given shift and scale
        # Use model_construct to bypass pydantic validation for arbitrary values
        response = FloatVariable.model_construct(
            name="f",
            bounds=(-np.inf, np.inf),
            shift=shift,
            scale=scale,
            default=None,
            units=None,
            description="",
            options={},
            class_type="float",
        )

        opt_problem = OptProblem(
            variables=[FloatVariable(name="x1", bounds=(0.0, 10.0), default=5.0)],
            responses=[response],
        )
        evaluator = SimpleEvaluator(opt_problem=opt_problem)

        # Set up the OpenMDAO problem
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Access output metadata to verify ref0 and ref
        meta = prob.model.comp.get_io_metadata(iotypes="output")
        output_meta = meta["f"]

        # Expected values per the formula
        expected_ref0 = -shift
        expected_ref = (1.0 / scale) + expected_ref0

        # Verify the scaling formula
        assert output_meta["ref0"] == pytest.approx(expected_ref0, rel=1e-12, abs=1e-15), (
            f"ref0 mismatch: got {output_meta['ref0']}, expected {expected_ref0} "
            f"(shift={shift})"
        )
        assert output_meta["ref"] == pytest.approx(expected_ref, rel=1e-12, abs=1e-15), (
            f"ref mismatch: got {output_meta['ref']}, expected {expected_ref} "
            f"(shift={shift}, scale={scale})"
        )

# ---------------------------------------------------------------------------
# Strategies for Property 3
# ---------------------------------------------------------------------------

# Strategy for variable names (valid identifiers, no whitespace)
_var_names = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)


@st.composite
def float_variable_with_explicit_default(draw):
    """Generate a FloatVariable with an explicit default value (not None).

    The default may fall inside or outside the bounds.
    """
    name = draw(_var_names)
    lower = draw(
        st.floats(min_value=-1e6, max_value=1e6 - 1,
                  allow_nan=False, allow_infinity=False)
    )
    upper = draw(
        st.floats(min_value=lower + 0.01, max_value=1e6,
                  allow_nan=False, allow_infinity=False)
    )
    assume(upper > lower)
    default_val = draw(
        st.floats(min_value=-1e8, max_value=1e8,
                  allow_nan=False, allow_infinity=False)
    )
    return FloatVariable(name=name, bounds=(lower, upper), default=default_val)


@st.composite
def float_variable_without_explicit_default(draw):
    """Generate a FloatVariable with default=None (use midpoint of bounds)."""
    name = draw(_var_names)
    lower = draw(
        st.floats(min_value=-1e6, max_value=1e6 - 1,
                  allow_nan=False, allow_infinity=False)
    )
    upper = draw(
        st.floats(min_value=lower + 0.01, max_value=1e6,
                  allow_nan=False, allow_infinity=False)
    )
    assume(upper > lower)
    return FloatVariable(name=name, bounds=(lower, upper), default=None)


# ---------------------------------------------------------------------------
# Property 3: Variable-to-Input Default Mapping
# Feature: openmdao-component-migration, Property 3: Variable-to-Input Default Mapping
# ---------------------------------------------------------------------------


class DummyDefaultEvaluator(Evaluator):
    """Minimal evaluator that returns zeros for all responses."""

    __test__ = False

    def _evaluate(self, sites: pd.DataFrame) -> None:
        for resp in self.outputs:
            sites[resp] = 0.0


class TestVariableToInputDefaultMapping:
    """Property 3: Variable-to-Input Default Mapping.

    For any Variable in the evaluator's opt_problem.variables, the
    corresponding OpenMDAO input default value SHALL be variable.default if
    variable.default is not None, otherwise it SHALL be the midpoint of
    the variable's bounds (lower + upper) / 2.

    **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
    """

    @given(var=float_variable_with_explicit_default())
    @settings(max_examples=100, deadline=None)
    def test_explicit_default_used_as_input_value(self, var: FloatVariable):
        """When var.default is not None, the input default equals var.default.

        **Validates: Requirements 5.2, 5.3, 5.5**
        """
        # Build evaluator with a single variable and a single response
        opt_problem = OptProblem(
            variables=[var],
            responses=[
                FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))
            ],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        # Create OpenMDAO problem and setup
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Verify the input default matches var.default
        actual_default = prob.get_val(var.name).flatten()[0]
        expected_default = var.default
        assert np.isclose(actual_default, expected_default, rtol=1e-10), (
            f"Expected input default={expected_default} for variable "
            f"'{var.name}' with explicit default, got {actual_default}"
        )

    @given(var=float_variable_without_explicit_default())
    @settings(max_examples=100, deadline=None)
    def test_midpoint_default_when_no_explicit_default(
        self, var: FloatVariable
    ):
        """When var.default is None, the input default equals (lower+upper)/2.

        **Validates: Requirements 5.2, 5.4**
        """
        # Build evaluator with a single variable and a single response
        opt_problem = OptProblem(
            variables=[var],
            responses=[
                FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))
            ],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        # Create OpenMDAO problem and setup
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Verify the input default is the midpoint of bounds
        lower, upper = var.bounds
        expected_default = (lower + upper) / 2.0
        actual_default = prob.get_val(var.name).flatten()[0]
        assert np.isclose(actual_default, expected_default, rtol=1e-10), (
            f"Expected input default={expected_default} (midpoint of "
            f"[{lower}, {upper}]) for variable '{var.name}' with no "
            f"explicit default, got {actual_default}"
        )

    @given(var=float_variable_with_explicit_default())
    @settings(max_examples=100, deadline=None)
    def test_explicit_default_ignores_bounds_for_calculation(
        self, var: FloatVariable
    ):
        """When both default and bounds exist, the explicit default is used
        regardless of whether it falls inside or outside the bounds.

        **Validates: Requirements 5.3, 5.5**
        """
        # Build evaluator with a single variable and a single response
        opt_problem = OptProblem(
            variables=[var],
            responses=[
                FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))
            ],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        # Create OpenMDAO problem and setup
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # The input default should be var.default, NOT the midpoint
        actual_default = prob.get_val(var.name).flatten()[0]
        lower, upper = var.bounds
        midpoint = (lower + upper) / 2.0

        # It should equal the explicit default
        assert np.isclose(actual_default, var.default, rtol=1e-10)

        # Verify it's not accidentally the midpoint
        # (unless they happen to be equal)
        if not np.isclose(var.default, midpoint, rtol=1e-10):
            assert not np.isclose(actual_default, midpoint, rtol=1e-10), (
                f"Input default should be the explicit default "
                f"{var.default}, not the midpoint {midpoint}"
            )


# ---------------------------------------------------------------------------
# Property 6: Compute Round Trip
# Feature: openmdao-component-migration, Property 6: Compute Round Trip
# ---------------------------------------------------------------------------


def input_values_strategy(evaluator):
    """Create a strategy that generates input values within variable bounds."""
    var_bounds = _get_variable_bounds(evaluator)
    strategies = []
    for lower, upper in var_bounds:
        strategies.append(
            st.floats(
                min_value=lower,
                max_value=upper,
                allow_nan=False,
                allow_infinity=False,
            )
        )
    return st.tuples(*strategies)


_EVALUATOR_CLASSES = [HS100, Rosenbrock, Sphere, ConstrainedBetts]


@pytest.mark.parametrize(
    "evaluator_cls", _EVALUATOR_CLASSES, ids=lambda cls: cls.__name__
)
def test_compute_round_trip(evaluator_cls):
    """Property 6: Compute Round Trip

    For any TestEvaluator, when the component is constructed, set up in an
    OpenMDAO problem, and run_model() is called with generated inputs within
    variable bounds, the output values for every response SHALL match (within
    relative tolerance 1e-10) the values obtained by invoking the same evaluator
    directly with a DataFrame of the same inputs.

    **Validates: Requirements 7.1, 7.2, 7.3, 8.2, 8.3**
    """
    evaluator = _create_evaluator_instance(evaluator_cls)
    strategy = input_values_strategy(evaluator)

    @given(input_vals=strategy)
    @settings(max_examples=100, deadline=None)
    def check_round_trip(input_vals):
        # Create fresh OpenMDAO problem for each example
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Set input values from the generated tuple
        var_names = [var.name for var in evaluator.opt_problem.variables]
        for name, val in zip(var_names, input_vals):
            prob.set_val(name, val)

        # Run the model through OpenMDAO
        prob.run_model()

        # Call evaluator directly with same inputs as DataFrame
        input_dict = {name: [val] for name, val in zip(var_names, input_vals)}
        data_frame = pd.DataFrame(data=input_dict)
        evaluator(data_frame)

        # Assert all outputs match within 1e-10 tolerance
        for response in evaluator.outputs:
            om_value = prob.get_val(response).flatten()[0]
            direct_value = float(data_frame[response].iloc[0])
            np.testing.assert_allclose(
                om_value,
                direct_value,
                rtol=1e-10,
                err_msg=(
                    f"Mismatch for response '{response}' with evaluator "
                    f"{evaluator_cls.__name__}: OpenMDAO={om_value}, "
                    f"Direct={direct_value}"
                ),
            )

    check_round_trip()


# ---------------------------------------------------------------------------
# Strategies for Property 4
# ---------------------------------------------------------------------------

# Strategy for finite float values suitable for bounds
_finite_floats = st.floats(
    min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False
)


@st.composite
def response_bounds_strategy(draw):
    """Generate a tuple of (lower, upper) bounds with mix of finite/infinite.

    Covers four cases:
    - Both bounds finite (lower <= upper)
    - Lower bound = -inf, upper finite
    - Lower finite, upper = +inf
    - Both bounds infinite
    """
    case = draw(st.sampled_from([
        "both_finite", "lower_inf", "upper_inf", "both_inf"
    ]))

    if case == "both_finite":
        lower = draw(_finite_floats)
        upper = draw(_finite_floats.filter(lambda x: x >= lower))
        return (lower, upper)
    elif case == "lower_inf":
        upper = draw(_finite_floats)
        return (-np.inf, upper)
    elif case == "upper_inf":
        lower = draw(_finite_floats)
        return (lower, np.inf)
    else:  # both_inf
        return (-np.inf, np.inf)


@st.composite
def response_variable_strategy(draw, name=None):
    """Generate a FloatVariable suitable for use as a response."""
    if name is None:
        name = draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L",), whitelist_characters="_"
                ),
                min_size=1,
                max_size=8,
            ).filter(lambda s: s[0].isalpha() or s[0] == "_")
        )
    bounds = draw(response_bounds_strategy())
    shift = draw(_finite_floats)
    scale = draw(_finite_floats.filter(lambda x: x != 0.0))
    return FloatVariable(
        name=name,
        bounds=bounds,
        shift=shift,
        scale=scale,
        default=None,
    )


@st.composite
def response_list_strategy(draw):
    """Generate a list of 1-5 response FloatVariables with unique names."""
    num_responses = draw(st.integers(min_value=1, max_value=5))
    names = draw(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L",), whitelist_characters="_"
                ),
                min_size=1,
                max_size=8,
            ).filter(lambda s: s[0].isalpha() or s[0] == "_"),
            min_size=num_responses,
            max_size=num_responses,
            unique=True,
        )
    )
    responses = []
    for name in names:
        resp = draw(response_variable_strategy(name=name))
        responses.append(resp)
    return responses


# ---------------------------------------------------------------------------
# Property 4: Response-to-Output Bounds Mapping
# Feature: openmdao-component-migration, Property 4: Response-to-Output Bounds Mapping
# ---------------------------------------------------------------------------


class DummyResponseEvaluator(Evaluator):
    """Minimal evaluator for testing response-to-output bounds mapping."""

    __test__ = False

    def _evaluate(self, sites: pd.DataFrame) -> None:
        for resp in self.opt_problem.responses:
            sites[resp.name] = 0.0


class TestResponseToOutputBoundsMapping:
    """Property 4: Response-to-Output Bounds Mapping.

    For any response Variable in opt_problem.responses, the OpenMDAO output
    bounds SHALL be: the actual bound value when finite, or None when the
    bound is infinite (-inf for lower, +inf for upper).

    **Validates: Requirements 6.3, 6.4**
    """

    @given(responses=response_list_strategy())
    @settings(max_examples=100, deadline=None)
    def test_response_to_output_bounds_mapping(self, responses):
        """Verify OpenMDAO output bounds are correct for generated responses.

        For each response:
        - If lower bound is finite, OpenMDAO lower should equal the bound value
        - If lower bound is -inf, OpenMDAO lower should be None
        - If upper bound is finite, OpenMDAO upper should equal the bound value
        - If upper bound is +inf, OpenMDAO upper should be None
        """
        # Create a minimal opt_problem with one input variable and the
        # generated responses
        input_var = FloatVariable(
            name="x_input", bounds=(-1.0, 1.0), default=0.0
        )
        opt_problem = OptProblem(
            variables=[input_var],
            responses=responses,
        )

        # Create evaluator and set up OpenMDAO problem
        evaluator = DummyResponseEvaluator(opt_problem=opt_problem)

        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", EvaluatorOpenMdaoComponent(evaluator), promotes=["*"]
        )
        prob.setup()

        # Verify bounds for each response
        comp = prob.model.comp
        meta = comp.get_io_metadata(iotypes="output")

        for resp in responses:
            lower_bound, upper_bound = resp.bounds

            # Find the metadata entry for this response
            output_key = None
            for key in meta:
                if key.endswith(f".{resp.name}") or key == resp.name:
                    output_key = key
                    break

            assert output_key is not None, (
                f"Response '{resp.name}' not found in component outputs"
            )

            output_meta = meta[output_key]

            # Check lower bound: finite -> actual value, -inf -> None
            if np.isfinite(lower_bound):
                assert output_meta["lower"] == lower_bound, (
                    f"Response '{resp.name}': expected lower={lower_bound}, "
                    f"got {output_meta['lower']}"
                )
            else:
                assert output_meta["lower"] is None, (
                    f"Response '{resp.name}': expected lower=None for -inf, "
                    f"got {output_meta['lower']}"
                )

            # Check upper bound: finite -> actual value, +inf -> None
            if np.isfinite(upper_bound):
                assert output_meta["upper"] == upper_bound, (
                    f"Response '{resp.name}': expected upper={upper_bound}, "
                    f"got {output_meta['upper']}"
                )
            else:
                assert output_meta["upper"] is None, (
                    f"Response '{resp.name}': expected upper=None for +inf, "
                    f"got {output_meta['upper']}"
                )
