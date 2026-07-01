"""Property-based tests for EvaluatorOpenMdaoComponent.

Uses Hypothesis to validate correctness properties of the OpenMDAO component
wrapper across a range of inputs and evaluator configurations.

# Feature: openmdao-component-migration
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

import openmdao.api as om
import openmdao.utils.general_utils as om_utils
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from standard_evaluator.components import EvaluatorOpenMdaoComponent
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.evaluators.open_mdao_evaluator import OpenMDAOEvaluator
from standard_evaluator.evaluators.test import (
    HS100,
    Rosenbrock,
    Sphere,
    ConstrainedBetts,
)
from standard_evaluator.problem import OptProblem, FloatVariable, ArrayVariable


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


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 2
# ---------------------------------------------------------------------------

# Strategy for optional units strings
_optional_units_strategy = st.one_of(
    st.none(),
    st.sampled_from(["m", "kg", "s", "m/s", "N", "Pa", "degC", "rad"]),
)

# Strategy for optional scalar val (as a 1-element array, per OpenMDAO convention)
_scalar_val_strategy = st.one_of(
    st.none(),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    .map(lambda v: np.array([v])),
)

# Strategy for non-zero finite floats (valid for scale/scaler values)
_nonzero_finite_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
).filter(lambda x: abs(x) > 1e-15)

# Strategy for non-zero finite floats for ref (ref cannot be 0 in OpenMDAO)
_nonzero_ref_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
).filter(lambda x: abs(x) > 1e-10)


@st.composite
def scalar_info_dict_strategy(draw):
    """Generate an info dict for a scalar variable with shape == (1,).

    Represents the dictionary returned by OpenMDAO's list_inputs/list_outputs
    for a scalar variable. Respects OpenMDAO constraints:
    - ref/ref0 are mutually exclusive with adder/scaler
    - ref cannot be 0 (causes division by zero)
    - scaler cannot be 0 (FloatVariable rejects zero scale)
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    units = draw(_optional_units_strategy)
    val = draw(_scalar_val_strategy)

    info = {
        "prom_name": name,
        "shape": (1,),
    }

    if units is not None:
        info["units"] = units
    if val is not None:
        info["val"] = val

    # OpenMDAO constraint: ref/ref0 are mutually exclusive with adder/scaler
    # Choose one of: no scaling, ref-based, adder/scaler-based
    scaling_mode = draw(st.sampled_from(["none", "ref_based", "adder_scaler_based"]))

    if scaling_mode == "ref_based":
        # OpenMDAO constraint: scaler = 1/(ref - ref0), so ref != ref0
        # When ref is absent, OpenMDAO defaults ref=1.0
        # When ref0 is absent, OpenMDAO defaults ref0=0.0
        use_ref = draw(st.booleans())
        use_ref0 = draw(st.booleans())
        if use_ref:
            info["ref"] = draw(_nonzero_ref_float)
        if use_ref0:
            info["ref0"] = draw(st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ))
        # Ensure at least one is set
        if not use_ref and not use_ref0:
            info["ref"] = draw(_nonzero_ref_float)
        # Ensure ref != ref0 to avoid division by zero in determine_adder_scaler
        # scaler = 1.0 / (ref + adder) where adder = -ref0
        # So we need ref - ref0 != 0, i.e., ref != ref0
        effective_ref = info.get("ref", 1.0)
        effective_ref0 = info.get("ref0", 0.0)
        assume(abs(effective_ref - effective_ref0) > 1e-10)
    elif scaling_mode == "adder_scaler_based":
        use_adder = draw(st.booleans())
        use_scaler = draw(st.booleans())
        if use_adder:
            info["adder"] = draw(st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ))
        if use_scaler:
            # scaler must be non-zero (FloatVariable validation rejects zero scale)
            info["scaler"] = draw(_nonzero_finite_float)
        # Ensure at least one is set
        if not use_adder and not use_scaler:
            info["scaler"] = draw(_nonzero_finite_float)
    # else: no scaling metadata at all

    return info


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 1
# ---------------------------------------------------------------------------

# Strategy for array shapes that are NOT (1,)
_array_shape_strategy = st.one_of(
    # 1-D shapes with size > 1
    st.integers(min_value=2, max_value=10).map(lambda n: (n,)),
    # 2-D shapes
    st.tuples(
        st.integers(min_value=2, max_value=5),
        st.integers(min_value=2, max_value=5),
    ),
    # 3-D shapes
    st.tuples(
        st.integers(min_value=2, max_value=3),
        st.integers(min_value=2, max_value=3),
        st.integers(min_value=2, max_value=3),
    ),
)


@st.composite
def array_info_dict_strategy(draw):
    """Generate an info dict for an array variable with shape != (1,).

    Represents the dictionary returned by OpenMDAO's list_inputs/list_outputs
    for an array variable. Includes optional scaling metadata and val array.
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    shape = draw(_array_shape_strategy)
    units = draw(_optional_units_strategy)

    info = {
        "prom_name": name,
        "shape": shape,
    }

    if units is not None:
        info["units"] = units

    # Optionally include a val array
    include_val = draw(st.booleans())
    if include_val:
        flat_size = 1
        for d in shape:
            flat_size *= d
        val_elements = draw(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        info["val"] = np.array(val_elements).reshape(shape)

    # Choose scaling mode: no scaling, ref-based, or adder/scaler-based
    scaling_mode = draw(st.sampled_from(["none", "ref_based", "adder_scaler_based"]))

    if scaling_mode == "ref_based":
        use_ref = draw(st.booleans())
        use_ref0 = draw(st.booleans())
        if use_ref:
            info["ref"] = draw(_nonzero_ref_float)
        if use_ref0:
            info["ref0"] = draw(st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ))
        # Ensure at least one is set
        if not use_ref and not use_ref0:
            info["ref"] = draw(_nonzero_ref_float)
        # Ensure ref != ref0 to avoid division by zero
        effective_ref = info.get("ref", 1.0)
        effective_ref0 = info.get("ref0", 0.0)
        assume(abs(effective_ref - effective_ref0) > 1e-10)
    elif scaling_mode == "adder_scaler_based":
        use_adder = draw(st.booleans())
        use_scaler = draw(st.booleans())
        if use_adder:
            info["adder"] = draw(st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ))
        if use_scaler:
            info["scaler"] = draw(_nonzero_finite_float)
        # Ensure at least one is set
        if not use_adder and not use_scaler:
            info["scaler"] = draw(_nonzero_finite_float)
    # else: no scaling metadata at all

    return info


# ---------------------------------------------------------------------------
# Property 1: Scan produces ArrayVariable with correct fields for array shapes
# Feature: array-variable-support, Property 1
# ---------------------------------------------------------------------------


class TestExpandInfoArrayVariable:
    """Property 1: Scan produces ArrayVariable with correct fields for array shapes.

    For any info dictionary with shape != (1,) and arbitrary scaling metadata
    (ref, ref0, adder, scaler), val array, and units string,
    _expand_info_to_variable SHALL produce an ArrayVariable whose shape matches
    the input shape, whose shift and scale equal the output of
    om_utils.determine_adder_scaler(ref0, ref, adder, scaler), whose default
    equals the val array (or np.zeros(shape) if val is absent), and whose units
    equals the input units (or None if absent).

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6, 1.7**
    """

    @given(info=array_info_dict_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_produces_array_variable(self, info):
        """Property 1: Scan produces ArrayVariable for array shapes."""
        # Call the function under test
        evaluator = OpenMDAOEvaluator.__new__(OpenMDAOEvaluator)
        result = evaluator._expand_info_to_variable(info)

        # Must be an ArrayVariable
        assert isinstance(result, ArrayVariable), (
            f"Expected ArrayVariable for shape {info['shape']}, got {type(result).__name__}"
        )

        # Shape must match
        assert result.shape == info["shape"], (
            f"Shape mismatch: got {result.shape}, expected {info['shape']}"
        )

        # Name must match
        assert result.name == info["prom_name"], (
            f"Name mismatch: got {result.name!r}, expected {info['prom_name']!r}"
        )

        # Units must match
        expected_units = info.get("units", None)
        assert result.units == expected_units, (
            f"Units mismatch: got {result.units!r}, expected {expected_units!r}"
        )

        # Verify default
        if "val" in info:
            np.testing.assert_array_equal(
                result.default, info["val"],
                err_msg="Default mismatch with provided val"
            )
        else:
            np.testing.assert_array_equal(
                result.default, np.zeros(info["shape"]),
                err_msg="Default should be zeros when val absent"
            )

        # Verify shift and scale match expected values
        ref = info.get("ref", None)
        ref0 = info.get("ref0", None)
        adder = info.get("adder", None)
        scaler = info.get("scaler", None)

        if ref is not None or ref0 is not None or adder is not None or scaler is not None:
            expected_adder, expected_scaler = om_utils.determine_adder_scaler(
                ref0, ref, adder, scaler
            )
            expected_shift = expected_adder if expected_adder is not None else 0.0
            expected_scale = expected_scaler if expected_scaler is not None else 1.0
        else:
            expected_shift = 0.0
            expected_scale = 1.0

        # ArrayVariable.check_shape expands scalar shift/scale to full arrays
        if np.isscalar(expected_shift):
            expected_shift_array = np.full(info["shape"], expected_shift)
        else:
            expected_shift_array = np.asarray(expected_shift)
        if np.isscalar(expected_scale):
            expected_scale_array = np.full(info["shape"], expected_scale)
        else:
            expected_scale_array = np.asarray(expected_scale)

        np.testing.assert_allclose(
            result.shift, expected_shift_array, rtol=1e-12,
            err_msg=f"Shift mismatch: got {result.shift}, expected {expected_shift_array}"
        )
        np.testing.assert_allclose(
            result.scale, expected_scale_array, rtol=1e-12,
            err_msg=f"Scale mismatch: got {result.scale}, expected {expected_scale_array}"
        )

        # Bounds must be (-inf, +inf) expanded to arrays
        np.testing.assert_array_equal(
            result.bounds[0], np.full(info["shape"], -np.inf),
            err_msg="Lower bounds should be -inf"
        )
        np.testing.assert_array_equal(
            result.bounds[1], np.full(info["shape"], np.inf),
            err_msg="Upper bounds should be +inf"
        )


# ---------------------------------------------------------------------------
# Property 2: Scan produces FloatVariable with correct fields for scalar shapes
# Feature: array-variable-support, Property 2
# ---------------------------------------------------------------------------


class TestExpandInfoFloatVariable:
    """Property 2: Scan produces FloatVariable with correct fields for scalar shapes.

    For any info dictionary with shape == (1,) and arbitrary scaling metadata
    (ref, ref0, adder, scaler), val entry, and optional units string,
    _expand_info_to_variable SHALL produce a FloatVariable (not ArrayVariable)
    with scalar shift, scale, and default values, and whose units equals the
    input units string (or None if absent or None).

    **Validates: Requirements 1.5, 7.3**
    """

    @given(info=scalar_info_dict_strategy())
    @settings(max_examples=100, deadline=None)
    def test_scalar_produces_float_variable(self, info):
        """Property 2: Scan produces FloatVariable for scalar shapes."""
        # Call the function under test
        evaluator = OpenMDAOEvaluator.__new__(OpenMDAOEvaluator)
        result = evaluator._expand_info_to_variable(info)

        # Must be a FloatVariable, NOT an ArrayVariable
        assert isinstance(result, FloatVariable), (
            f"Expected FloatVariable for shape (1,), got {type(result).__name__}"
        )
        assert not isinstance(result, ArrayVariable), (
            "Expected FloatVariable (not ArrayVariable) for shape (1,), "
            "got ArrayVariable"
        )

        # Name must match
        assert result.name == info["prom_name"], (
            f"Name mismatch: got {result.name!r}, expected {info['prom_name']!r}"
        )

        # Units must match
        expected_units = info.get("units", None)
        assert result.units == expected_units, (
            f"Units mismatch: got {result.units!r}, expected {expected_units!r}"
        )

        # Default must be scalar
        if "val" in info:
            expected_default = info["val"][0]
        else:
            expected_default = 0.0
        assert np.isclose(result.default, expected_default, equal_nan=True), (
            f"Default mismatch: got {result.default}, expected {expected_default}"
        )

        # Shift and scale must be scalar floats (not arrays)
        assert isinstance(result.shift, (int, float)), (
            f"Shift should be scalar, got {type(result.shift).__name__}"
        )
        assert isinstance(result.scale, (int, float)), (
            f"Scale should be scalar, got {type(result.scale).__name__}"
        )

        # Verify shift and scale match determine_adder_scaler output
        ref = info.get("ref", None)
        ref0 = info.get("ref0", None)
        adder = info.get("adder", None)
        scaler = info.get("scaler", None)
        expected_adder, expected_scaler = om_utils.determine_adder_scaler(
            ref0, ref, adder, scaler
        )
        expected_shift = expected_adder if expected_adder is not None else 0.0
        expected_scale = expected_scaler if expected_scaler is not None else 1.0

        assert np.isclose(result.shift, expected_shift, equal_nan=True), (
            f"Shift mismatch: got {result.shift}, expected {expected_shift}"
        )
        assert np.isclose(result.scale, expected_scale, equal_nan=True), (
            f"Scale mismatch: got {result.scale}, expected {expected_scale}"
        )

        # Bounds must be (-inf, inf)
        assert result.bounds[0] == -np.inf, (
            f"Lower bound should be -inf, got {result.bounds[0]}"
        )
        assert result.bounds[1] == np.inf, (
            f"Upper bound should be +inf, got {result.bounds[1]}"
        )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 3
# ---------------------------------------------------------------------------


@st.composite
def map_elements_array_metadata_strategy(draw):
    """Generate a metadata dict for _map_elements with size > 1.

    Represents the dictionary returned by OpenMDAO's get_design_vars() or
    get_responses() for an array variable (size > 1). Includes optional
    bounds, scaling, and units fields.
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))

    # Size must be > 1 for array path
    size = draw(st.integers(min_value=2, max_value=20))

    # Shape can be 1-D matching size or multi-dimensional
    shape_mode = draw(st.sampled_from(["1d", "2d"]))
    if shape_mode == "1d":
        shape = (size,)
    else:
        # Pick two factors (may not exactly divide, so just use (size,) fallback)
        dim1 = draw(st.integers(min_value=2, max_value=min(size, 5)))
        dim2 = size // dim1
        if dim1 * dim2 == size and dim2 >= 2:
            shape = (dim1, dim2)
        else:
            shape = (size,)

    units = draw(_optional_units_strategy)

    meta = {
        "size": size,
        "shape": shape,
    }

    if units is not None:
        meta["units"] = units

    # Optionally include lower bound (scalar or array)
    lower_mode = draw(st.sampled_from(["absent", "scalar", "array"]))
    if lower_mode == "scalar":
        meta["lower"] = draw(st.floats(
            min_value=-1e6, max_value=1e5, allow_nan=False, allow_infinity=False
        ))
    elif lower_mode == "array":
        flat_size = 1
        for d in shape:
            flat_size *= d
        lower_elements = draw(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e5, allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        meta["lower"] = np.array(lower_elements, dtype=np.float64).reshape(shape)

    # Optionally include upper bound (scalar or array)
    upper_mode = draw(st.sampled_from(["absent", "scalar", "array"]))
    if upper_mode == "scalar":
        upper_val = draw(st.floats(
            min_value=-1e5, max_value=1e6, allow_nan=False, allow_infinity=False
        ))
        # Ensure upper >= lower for any scalar/scalar combo
        if lower_mode == "scalar" and "lower" in meta:
            upper_val = max(upper_val, meta["lower"])
        meta["upper"] = upper_val
    elif upper_mode == "array":
        flat_size = 1
        for d in shape:
            flat_size *= d
        # Generate upper elements that are >= lower if lower is also an array
        if lower_mode == "array" and "lower" in meta:
            lower_flat = meta["lower"].flatten()
            upper_elements = []
            for lv in lower_flat:
                uv = draw(st.floats(
                    min_value=float(lv), max_value=1e6,
                    allow_nan=False, allow_infinity=False
                ))
                upper_elements.append(uv)
        else:
            upper_elements = draw(
                st.lists(
                    st.floats(min_value=-1e5, max_value=1e6, allow_nan=False, allow_infinity=False),
                    min_size=flat_size,
                    max_size=flat_size,
                )
            )
        meta["upper"] = np.array(upper_elements, dtype=np.float64).reshape(shape)

    # Ensure lower <= upper when both are present as scalars
    if "lower" in meta and "upper" in meta:
        if np.isscalar(meta["lower"]) and np.isscalar(meta["upper"]):
            assume(meta["lower"] <= meta["upper"])
        elif isinstance(meta.get("lower"), np.ndarray) and isinstance(
            meta.get("upper"), np.ndarray
        ):
            assume(np.all(meta["lower"] <= meta["upper"]))
        elif np.isscalar(meta.get("lower")) and isinstance(
            meta.get("upper"), np.ndarray
        ):
            assume(np.all(meta["lower"] <= meta["upper"]))
        elif isinstance(meta.get("lower"), np.ndarray) and np.isscalar(
            meta.get("upper")
        ):
            assume(np.all(meta["lower"] <= meta["upper"]))

    # Optionally include scaler (non-zero)
    include_scaler = draw(st.booleans())
    if include_scaler:
        meta["scaler"] = draw(_nonzero_finite_float)

    # Optionally include adder
    include_adder = draw(st.booleans())
    if include_adder:
        meta["adder"] = draw(st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ))

    return name, meta


# ---------------------------------------------------------------------------
# Property 3: Map produces ArrayVariable for size > 1
# Feature: array-variable-support, Property 3: Map produces ArrayVariable for size > 1
# ---------------------------------------------------------------------------


class TestMapElementsArrayVariable:
    """Property 3: Map produces ArrayVariable for size > 1.

    For any design-variable or response metadata dictionary with size > 1 and
    an optional units field, _map_elements SHALL produce an ArrayVariable whose
    shape matches the metadata shape, whose bounds match the metadata
    lower/upper (defaulting independently to -inf/+inf when absent), whose
    scale matches the metadata scaler, whose shift matches the metadata adder,
    and whose units equals the metadata units string when not None (or None
    when absent/None).

    **Validates: Requirements 2.1, 2.2, 2.4, 2.5**
    """

    @given(data=map_elements_array_metadata_strategy())
    @settings(max_examples=100, deadline=None)
    def test_map_elements_produces_array_variable(self, data):
        """# Feature: array-variable-support, Property 3: Map produces ArrayVariable for size > 1"""
        name, meta = data

        # Build the info dict as _map_elements expects: {name: metadata}
        info = {name: meta}

        # Call the function under test
        evaluator = OpenMDAOEvaluator.__new__(OpenMDAOEvaluator)
        result = evaluator._map_elements(info)

        # Should produce exactly one variable
        assert len(result) == 1, (
            f"Expected 1 result from _map_elements, got {len(result)}"
        )

        var = result[0]

        # Must be an ArrayVariable
        assert isinstance(var, ArrayVariable), (
            f"Expected ArrayVariable for size {meta['size']}, got {type(var).__name__}"
        )

        # Name must match
        assert var.name == name, (
            f"Name mismatch: got {var.name!r}, expected {name!r}"
        )

        # Shape must match metadata shape
        assert var.shape == meta["shape"], (
            f"Shape mismatch: got {var.shape}, expected {meta['shape']}"
        )

        # Units must match
        expected_units = meta.get("units", None)
        assert var.units == expected_units, (
            f"Units mismatch: got {var.units!r}, expected {expected_units!r}"
        )

        # Verify bounds: lower defaults to -inf, upper defaults to +inf
        expected_lower = meta.get("lower", -np.inf)
        if expected_lower is None:
            expected_lower = -np.inf
        expected_upper = meta.get("upper", np.inf)
        if expected_upper is None:
            expected_upper = np.inf

        # ArrayVariable check_shape expands scalar bounds to full arrays
        if np.isscalar(expected_lower):
            expected_lower_array = np.full(meta["shape"], expected_lower, dtype=np.float64)
        else:
            expected_lower_array = np.asarray(expected_lower, dtype=np.float64)
        if np.isscalar(expected_upper):
            expected_upper_array = np.full(meta["shape"], expected_upper, dtype=np.float64)
        else:
            expected_upper_array = np.asarray(expected_upper, dtype=np.float64)

        np.testing.assert_array_equal(
            var.bounds[0], expected_lower_array,
            err_msg="Lower bounds mismatch"
        )
        np.testing.assert_array_equal(
            var.bounds[1], expected_upper_array,
            err_msg="Upper bounds mismatch"
        )

        # Verify scale: defaults to 1.0 if absent or None
        expected_scale = meta.get("scaler", 1.0)
        if expected_scale is None:
            expected_scale = 1.0
        # ArrayVariable check_shape expands scalar scale to full array
        if np.isscalar(expected_scale):
            expected_scale_array = np.full(meta["shape"], expected_scale, dtype=np.float64)
        else:
            expected_scale_array = np.asarray(expected_scale, dtype=np.float64)

        np.testing.assert_allclose(
            var.scale, expected_scale_array, rtol=1e-12,
            err_msg=f"Scale mismatch: got {var.scale}, expected {expected_scale_array}"
        )

        # Verify shift: defaults to 0.0 if absent or None
        expected_shift = meta.get("adder", 0.0)
        if expected_shift is None:
            expected_shift = 0.0
        # ArrayVariable check_shape expands scalar shift to full array
        if np.isscalar(expected_shift):
            expected_shift_array = np.full(meta["shape"], expected_shift, dtype=np.float64)
        else:
            expected_shift_array = np.asarray(expected_shift, dtype=np.float64)

        np.testing.assert_allclose(
            var.shift, expected_shift_array, rtol=1e-12,
            err_msg=f"Shift mismatch: got {var.shift}, expected {expected_shift_array}"
        )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 4
# ---------------------------------------------------------------------------


@st.composite
def scalar_map_metadata_strategy(draw):
    """Generate metadata dict for _map_elements with size == 1 or size absent.

    Represents a single entry from OpenMDAO's get_design_vars()/get_responses()
    for a scalar variable. The metadata may include lower, upper, scaler, and units.
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))

    # size is either 1 or absent (both should produce FloatVariable)
    include_size = draw(st.booleans())
    meta = {}
    if include_size:
        meta["size"] = 1

    # Optional lower bound: finite float, -inf, or absent
    lower_case = draw(st.sampled_from(["finite", "neg_inf", "absent", "none"]))
    if lower_case == "finite":
        meta["lower"] = draw(st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ))
    elif lower_case == "neg_inf":
        meta["lower"] = -np.inf
    elif lower_case == "none":
        meta["lower"] = None
    # else "absent": key not present

    # Optional upper bound: finite float, +inf, or absent
    upper_case = draw(st.sampled_from(["finite", "pos_inf", "absent", "none"]))
    if upper_case == "finite":
        meta["upper"] = draw(st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ))
    elif upper_case == "pos_inf":
        meta["upper"] = np.inf
    elif upper_case == "none":
        meta["upper"] = None
    # else "absent": key not present

    # Ensure lower <= upper when both are finite
    effective_lower = meta.get("lower", -np.inf)
    if effective_lower is None:
        effective_lower = -np.inf
    effective_upper = meta.get("upper", np.inf)
    if effective_upper is None:
        effective_upper = np.inf
    if np.isfinite(effective_lower) and np.isfinite(effective_upper):
        assume(effective_lower <= effective_upper)

    # Optional scaler (non-zero)
    include_scaler = draw(st.booleans())
    if include_scaler:
        meta["scaler"] = draw(_nonzero_finite_float)

    # Optional units
    units = draw(_optional_units_strategy)
    if units is not None:
        meta["units"] = units

    return name, meta


# ---------------------------------------------------------------------------
# Property 4: Map produces FloatVariable for size <= 1
# Feature: array-variable-support, Property 4: Map produces FloatVariable for size <= 1
# ---------------------------------------------------------------------------


class TestMapElementsFloatVariable:
    """Property 4: Map produces FloatVariable for size <= 1.

    For any design-variable or response metadata dictionary with size == 1 or
    size absent and an optional units field, _map_elements SHALL produce a
    FloatVariable with bounds from metadata (defaulting to [-inf, +inf] when
    absent) and units set to the metadata units string when not None (or None
    when absent/None).

    **Validates: Requirements 2.3, 2.5**
    """

    @given(data=scalar_map_metadata_strategy())
    @settings(max_examples=100, deadline=None)
    def test_map_produces_float_variable_for_scalar(self, data):
        """Property 4: Map produces FloatVariable for size <= 1."""
        name, meta = data

        # Call _map_elements with a single-entry info dict
        evaluator = OpenMDAOEvaluator.__new__(OpenMDAOEvaluator)
        info = {name: meta}
        result = evaluator._map_elements(info)

        # Should produce exactly one variable
        assert len(result) == 1, (
            f"Expected 1 result, got {len(result)}"
        )

        var = result[0]

        # Must be FloatVariable, NOT ArrayVariable
        assert isinstance(var, FloatVariable), (
            f"Expected FloatVariable for size=1/absent, got {type(var).__name__}"
        )
        assert not isinstance(var, ArrayVariable), (
            "Expected FloatVariable (not ArrayVariable) for size=1/absent, "
            "got ArrayVariable"
        )

        # Name must match
        assert var.name == name, (
            f"Name mismatch: got {var.name!r}, expected {name!r}"
        )

        # Verify bounds: default independently to -inf/+inf when absent or None
        effective_lower = meta.get("lower", -np.inf)
        if effective_lower is None:
            effective_lower = -np.inf
        effective_upper = meta.get("upper", np.inf)
        if effective_upper is None:
            effective_upper = np.inf

        assert var.bounds[0] == pytest.approx(float(effective_lower)), (
            f"Lower bound mismatch: got {var.bounds[0]}, expected {effective_lower}"
        )
        assert var.bounds[1] == pytest.approx(float(effective_upper)), (
            f"Upper bound mismatch: got {var.bounds[1]}, expected {effective_upper}"
        )

        # Verify units
        expected_units = meta.get("units", None)
        assert var.units == expected_units, (
            f"Units mismatch: got {var.units!r}, expected {expected_units!r}"
        )

        # Verify scale matches scaler from metadata (default 1.0)
        expected_scale = meta.get("scaler", 1.0)
        if expected_scale is None:
            expected_scale = 1.0
        assert var.scale == pytest.approx(expected_scale), (
            f"Scale mismatch: got {var.scale}, expected {expected_scale}"
        )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 5
# ---------------------------------------------------------------------------


@st.composite
def mixed_opt_problem_and_sites_strategy(draw):
    """Generate an OptProblem with a mix of FloatVariable and ArrayVariable
    inputs and outputs, along with a matching DataFrame of valid sites.

    Returns a tuple of (opt_problem, sites_df, expected_get_val_returns)
    where expected_get_val_returns maps response names to values the mock
    should return from get_val.
    """
    # Generate 1-3 variables: at least one float and one array
    num_float_vars = draw(st.integers(min_value=1, max_value=2))
    num_array_vars = draw(st.integers(min_value=1, max_value=2))

    variables = []
    # Float variables
    for i in range(num_float_vars):
        name = f"fv_{i}"
        variables.append(
            FloatVariable(name=name, bounds=(-10.0, 10.0), default=0.0)
        )
    # Array variables
    for i in range(num_array_vars):
        name = f"av_{i}"
        shape = draw(st.sampled_from([(2,), (3,), (2, 2)]))
        variables.append(
            ArrayVariable(
                name=name,
                shape=shape,
                bounds=(-10.0, 10.0),
                default=np.zeros(shape),
            )
        )

    # Generate 1-2 responses: at least one float and one array
    num_float_resps = draw(st.integers(min_value=1, max_value=2))
    num_array_resps = draw(st.integers(min_value=1, max_value=2))

    responses = []
    # Float responses
    for i in range(num_float_resps):
        name = f"fr_{i}"
        responses.append(
            FloatVariable(name=name, bounds=(-np.inf, np.inf))
        )
    # Array responses
    for i in range(num_array_resps):
        name = f"ar_{i}"
        shape = draw(st.sampled_from([(2,), (3,), (2, 2)]))
        responses.append(
            ArrayVariable(
                name=name,
                shape=shape,
                bounds=(-np.inf, np.inf),
            )
        )

    opt_problem = OptProblem(
        variables=variables,
        responses=responses,
        objectives=[responses[0].name],
    )

    # Generate 1-3 sites (rows)
    num_sites = draw(st.integers(min_value=1, max_value=3))

    # Build DataFrame with appropriate data for each variable
    data = {}
    for var in variables:
        if isinstance(var, ArrayVariable):
            # Each cell contains a NumPy array
            arrays = []
            for _ in range(num_sites):
                flat_size = int(np.prod(var.shape))
                elements = draw(
                    st.lists(
                        st.floats(min_value=-5.0, max_value=5.0,
                                  allow_nan=False, allow_infinity=False),
                        min_size=flat_size,
                        max_size=flat_size,
                    )
                )
                arrays.append(np.array(elements).reshape(var.shape))
            data[var.name] = arrays
        else:
            # Each cell contains a scalar
            scalars = draw(
                st.lists(
                    st.floats(min_value=-5.0, max_value=5.0,
                              allow_nan=False, allow_infinity=False),
                    min_size=num_sites,
                    max_size=num_sites,
                )
            )
            data[var.name] = scalars

    # Add response columns (will be overwritten by _evaluate)
    for resp in responses:
        if isinstance(resp, ArrayVariable):
            data[resp.name] = [None] * num_sites
        else:
            data[resp.name] = [None] * num_sites

    sites_df = pd.DataFrame(data)

    # Define what get_val should return for each response
    get_val_returns = {}
    for resp in responses:
        if isinstance(resp, ArrayVariable):
            # Return a NumPy array matching the response shape
            flat_size = int(np.prod(resp.shape))
            elements = draw(
                st.lists(
                    st.floats(min_value=-10.0, max_value=10.0,
                              allow_nan=False, allow_infinity=False),
                    min_size=flat_size,
                    max_size=flat_size,
                )
            )
            get_val_returns[resp.name] = np.array(elements).reshape(resp.shape)
        else:
            # Return a scalar value (as OpenMDAO would via get_val)
            val = draw(
                st.floats(min_value=-10.0, max_value=10.0,
                          allow_nan=False, allow_infinity=False)
            )
            get_val_returns[resp.name] = val

    return opt_problem, sites_df, get_val_returns


# ---------------------------------------------------------------------------
# Property 5: Evaluate correctly passes and extracts mixed scalar/array data
# Feature: array-variable-support, Property 5
# ---------------------------------------------------------------------------


class TestEvaluateMixedDataFlow:
    """Property 5: Evaluate correctly passes and extracts mixed scalar/array data.

    For any OptProblem containing a mix of FloatVariable and ArrayVariable
    inputs and outputs, and for any DataFrame of valid sites, _evaluate SHALL
    call set_val with the full NumPy array for ArrayVariable inputs (preserving
    shape) and with scalar values for FloatVariable inputs, and SHALL store the
    full NumPy array from get_val in the DataFrame cell for ArrayVariable
    outputs and scalar values for FloatVariable outputs.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """

    @given(data=mixed_opt_problem_and_sites_strategy())
    @settings(max_examples=100, deadline=None)
    def test_evaluate_mixed_data_flow(self, data):
        """Property 5: Evaluate passes and extracts mixed scalar/array data."""
        opt_problem, sites_df, get_val_returns = data

        # Create a partially-initialized OpenMDAOEvaluator without calling __init__
        evaluator = OpenMDAOEvaluator.__new__(OpenMDAOEvaluator)

        # Set up required attributes that _evaluate uses
        evaluator._opt_problem = opt_problem
        evaluator._inputs = [v.name for v in opt_problem.variables]
        evaluator._outputs = [r.name for r in opt_problem.responses]

        # Create a mock om_problem
        mock_om_problem = MagicMock()
        mock_om_problem.set_val = MagicMock()
        mock_om_problem.run_model = MagicMock()
        # get_val returns the appropriate value based on the response name
        mock_om_problem.get_val = MagicMock(
            side_effect=lambda name: get_val_returns[name]
        )
        evaluator.om_problem = mock_om_problem

        # Run the method under test
        evaluator._evaluate(sites_df)

        num_sites = len(sites_df)

        # Verify set_val calls: check that ArrayVariable inputs get arrays
        # and FloatVariable inputs get scalars
        all_set_val_calls = mock_om_problem.set_val.call_args_list

        for site_idx in range(num_sites):
            for var in opt_problem.variables:
                # Find the set_val call for this variable at this site
                # Calls are in order: site0_var0, site0_var1, ..., site1_var0, ...
                call_idx = site_idx * len(opt_problem.variables) + \
                    opt_problem.variables.index(var)
                actual_call = all_set_val_calls[call_idx]
                call_name = actual_call[0][0]  # First positional arg
                call_value = actual_call[0][1]  # Second positional arg

                assert call_name == var.name, (
                    f"Expected set_val call for '{var.name}', got '{call_name}'"
                )

                if isinstance(var, ArrayVariable):
                    # ArrayVariable: must be called with a NumPy array
                    assert isinstance(call_value, np.ndarray), (
                        f"Expected NumPy array for ArrayVariable '{var.name}', "
                        f"got {type(call_value).__name__}"
                    )
                    expected_array = sites_df.iloc[site_idx][var.name]
                    np.testing.assert_array_equal(
                        call_value, expected_array,
                        err_msg=f"Array value mismatch for '{var.name}' at site {site_idx}"
                    )
                    # Verify shape is preserved
                    assert call_value.shape == var.shape, (
                        f"Shape mismatch for '{var.name}': got {call_value.shape}, "
                        f"expected {var.shape}"
                    )
                else:
                    # FloatVariable: must be called with a scalar
                    assert np.isscalar(call_value) or (
                        isinstance(call_value, (int, float))
                    ), (
                        f"Expected scalar for FloatVariable '{var.name}', "
                        f"got {type(call_value).__name__}"
                    )

        # Verify run_model was called once per site
        assert mock_om_problem.run_model.call_count == num_sites, (
            f"Expected run_model called {num_sites} times, "
            f"got {mock_om_problem.run_model.call_count}"
        )

        # Verify DataFrame output cells: ArrayVariable responses get arrays,
        # FloatVariable responses get scalars
        for site_idx in range(num_sites):
            for resp in opt_problem.responses:
                cell_value = sites_df.iloc[site_idx][resp.name]

                if isinstance(resp, ArrayVariable):
                    # ArrayVariable response: cell should contain a NumPy array
                    assert isinstance(cell_value, np.ndarray), (
                        f"Expected NumPy array in DataFrame for ArrayVariable "
                        f"response '{resp.name}' at site {site_idx}, "
                        f"got {type(cell_value).__name__}"
                    )
                    np.testing.assert_array_equal(
                        cell_value, get_val_returns[resp.name],
                        err_msg=f"Array output mismatch for '{resp.name}' at site {site_idx}"
                    )
                else:
                    # FloatVariable response: cell should contain a scalar
                    expected_scalar = get_val_returns[resp.name]
                    assert np.isscalar(cell_value) or isinstance(cell_value, (int, float)), (
                        f"Expected scalar in DataFrame for FloatVariable "
                        f"response '{resp.name}' at site {site_idx}, "
                        f"got {type(cell_value).__name__}"
                    )
                    assert cell_value == pytest.approx(expected_scalar), (
                        f"Scalar output mismatch for '{resp.name}' at site {site_idx}: "
                        f"got {cell_value}, expected {expected_scalar}"
                    )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 6
# ---------------------------------------------------------------------------


@st.composite
def array_variable_for_input_strategy(draw):
    """Generate an ArrayVariable suitable for use as a component input.

    Generates ArrayVariables with finite bounds (lower < upper element-wise),
    optional default (None or a valid array), and optional units.
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    shape = draw(_array_shape_strategy)
    units = draw(_optional_units_strategy)

    flat_size = int(np.prod(shape))

    # Generate finite bounds where lower < upper element-wise
    lower_elements = draw(
        st.lists(
            st.floats(min_value=-1e4, max_value=1e4 - 1,
                      allow_nan=False, allow_infinity=False),
            min_size=flat_size,
            max_size=flat_size,
        )
    )
    upper_elements = []
    for lv in lower_elements:
        uv = draw(st.floats(
            min_value=lv + 0.01, max_value=1e4,
            allow_nan=False, allow_infinity=False,
        ))
        upper_elements.append(uv)

    lower = np.array(lower_elements, dtype=np.float64).reshape(shape)
    upper = np.array(upper_elements, dtype=np.float64).reshape(shape)

    # Optional default: None or a valid array
    include_default = draw(st.booleans())
    if include_default:
        default_elements = draw(
            st.lists(
                st.floats(min_value=-1e4, max_value=1e4,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        default = np.array(default_elements, dtype=np.float64).reshape(shape)
    else:
        default = None

    # Non-zero scale (required by ArrayVariable validator)
    scale = draw(
        st.floats(min_value=0.1, max_value=10.0,
                  allow_nan=False, allow_infinity=False)
    )

    var = ArrayVariable(
        name=name,
        shape=shape,
        bounds=(lower, upper),
        default=default,
        scale=scale,
        units=units,
    )
    return var


@st.composite
def float_variable_for_input_strategy(draw):
    """Generate a FloatVariable suitable for use as a component input.

    Generates FloatVariables with finite bounds, optional default, and optional units.
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    units = draw(_optional_units_strategy)

    lower = draw(st.floats(
        min_value=-1e4, max_value=1e4 - 1,
        allow_nan=False, allow_infinity=False,
    ))
    upper = draw(st.floats(
        min_value=lower + 0.01, max_value=1e4,
        allow_nan=False, allow_infinity=False,
    ))

    # Optional default
    include_default = draw(st.booleans())
    if include_default:
        default = draw(st.floats(
            min_value=-1e4, max_value=1e4,
            allow_nan=False, allow_infinity=False,
        ))
    else:
        default = None

    var = FloatVariable(
        name=name,
        bounds=(lower, upper),
        default=default,
        units=units,
    )
    return var


@st.composite
def variable_for_input_strategy(draw):
    """Generate either an ArrayVariable or FloatVariable for input testing."""
    use_array = draw(st.booleans())
    if use_array:
        return draw(array_variable_for_input_strategy())
    else:
        return draw(float_variable_for_input_strategy())


# ---------------------------------------------------------------------------
# Property 6: Component setup registers inputs correctly with units
# Feature: array-variable-support, Property 6: Component setup registers inputs correctly with units
# ---------------------------------------------------------------------------


class TestComponentInputRegistration:
    """Property 6: Component setup registers inputs correctly with units.

    For any variable (either ArrayVariable or FloatVariable) in an OptProblem's
    variables list, the component's setup SHALL call add_input with the
    appropriate shape and val, and SHALL pass units to add_input only when the
    variable's units is not None.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.6, 4.7**
    """

    @given(var=array_variable_for_input_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_variable_with_default_registers_correctly(self, var):
        """ArrayVariable with non-None default: val = default, shape = var.shape.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        assume(var.default is not None)

        # Build an OptProblem with this variable and a dummy response
        opt_problem = OptProblem(
            variables=[var],
            responses=[FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        # Create the component and mock add_input
        component = EvaluatorOpenMdaoComponent(evaluator)
        add_input_calls = []

        def mock_add_input(name, **kwargs):
            add_input_calls.append((name, kwargs))

        component.add_input = mock_add_input
        # Mock add_output to avoid side effects
        component.add_output = lambda name, **kwargs: None

        # Call setup
        component.setup()

        # Find the call for our variable
        matching_calls = [(n, kw) for n, kw in add_input_calls if n == var.name]
        assert len(matching_calls) == 1, (
            f"Expected exactly 1 add_input call for '{var.name}', "
            f"got {len(matching_calls)}"
        )

        call_name, call_kwargs = matching_calls[0]

        # Verify shape
        assert "shape" in call_kwargs, (
            f"ArrayVariable '{var.name}' should have shape in add_input kwargs"
        )
        assert call_kwargs["shape"] == var.shape, (
            f"Shape mismatch: got {call_kwargs['shape']}, expected {var.shape}"
        )

        # Verify val = default
        assert "val" in call_kwargs, (
            f"ArrayVariable '{var.name}' should have val in add_input kwargs"
        )
        np.testing.assert_array_equal(
            call_kwargs["val"], var.default,
            err_msg=f"val should equal default for ArrayVariable '{var.name}'"
        )

        # Verify units handling
        if var.units is not None:
            assert "units" in call_kwargs, (
                f"ArrayVariable '{var.name}' with units={var.units!r} "
                f"should have units in add_input kwargs"
            )
            assert call_kwargs["units"] == var.units, (
                f"Units mismatch: got {call_kwargs['units']!r}, expected {var.units!r}"
            )
        else:
            assert "units" not in call_kwargs, (
                f"ArrayVariable '{var.name}' with units=None should NOT "
                f"have units in add_input kwargs"
            )

    @given(var=array_variable_for_input_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_variable_without_default_uses_midpoint(self, var):
        """ArrayVariable with None default: val = midpoint of bounds, shape = var.shape.

        **Validates: Requirements 4.7**
        """
        # Force default to None by creating a new variable without default
        var_no_default = ArrayVariable(
            name=var.name,
            shape=var.shape,
            bounds=var.bounds,
            default=None,
            scale=var.scale,
            units=var.units,
        )

        opt_problem = OptProblem(
            variables=[var_no_default],
            responses=[FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_input_calls = []

        def mock_add_input(name, **kwargs):
            add_input_calls.append((name, kwargs))

        component.add_input = mock_add_input
        component.add_output = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_input_calls if n == var_no_default.name]
        assert len(matching_calls) == 1

        call_name, call_kwargs = matching_calls[0]

        # Verify shape
        assert call_kwargs["shape"] == var_no_default.shape

        # Verify val = midpoint of bounds
        lower, upper = var_no_default.bounds
        expected_val = (lower + upper) / 2.0
        np.testing.assert_allclose(
            call_kwargs["val"], expected_val, rtol=1e-12,
            err_msg=f"val should equal midpoint of bounds for ArrayVariable "
                    f"'{var_no_default.name}' with None default"
        )

    @given(var=float_variable_for_input_strategy())
    @settings(max_examples=100, deadline=None)
    def test_float_variable_with_default_registers_correctly(self, var):
        """FloatVariable with non-None default: val = default (scalar).

        **Validates: Requirements 4.5, 4.6**
        """
        assume(var.default is not None)

        opt_problem = OptProblem(
            variables=[var],
            responses=[FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_input_calls = []

        def mock_add_input(name, **kwargs):
            add_input_calls.append((name, kwargs))

        component.add_input = mock_add_input
        component.add_output = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_input_calls if n == var.name]
        assert len(matching_calls) == 1

        call_name, call_kwargs = matching_calls[0]

        # FloatVariable should NOT have shape
        assert "shape" not in call_kwargs, (
            f"FloatVariable '{var.name}' should NOT have shape in add_input kwargs"
        )

        # Verify val = default
        assert "val" in call_kwargs, (
            f"FloatVariable '{var.name}' should have val in add_input kwargs"
        )
        assert np.isclose(call_kwargs["val"], var.default, rtol=1e-12), (
            f"val mismatch: got {call_kwargs['val']}, expected {var.default}"
        )

        # Verify units handling
        if var.units is not None:
            assert "units" in call_kwargs, (
                f"FloatVariable '{var.name}' with units={var.units!r} "
                f"should have units in add_input kwargs"
            )
            assert call_kwargs["units"] == var.units
        else:
            assert "units" not in call_kwargs, (
                f"FloatVariable '{var.name}' with units=None should NOT "
                f"have units in add_input kwargs"
            )

    @given(var=float_variable_for_input_strategy())
    @settings(max_examples=100, deadline=None)
    def test_float_variable_without_default_uses_midpoint(self, var):
        """FloatVariable with None default: val = midpoint of bounds.

        **Validates: Requirements 4.5, 4.6**
        """
        # Create variable without default
        var_no_default = FloatVariable(
            name=var.name,
            bounds=var.bounds,
            default=None,
            units=var.units,
        )

        opt_problem = OptProblem(
            variables=[var_no_default],
            responses=[FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_input_calls = []

        def mock_add_input(name, **kwargs):
            add_input_calls.append((name, kwargs))

        component.add_input = mock_add_input
        component.add_output = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_input_calls if n == var_no_default.name]
        assert len(matching_calls) == 1

        call_name, call_kwargs = matching_calls[0]

        # Verify val = midpoint
        lower, upper = var_no_default.bounds
        expected_val = (lower + upper) / 2.0
        assert np.isclose(call_kwargs["val"], expected_val, rtol=1e-12), (
            f"val should equal midpoint ({expected_val}) for FloatVariable "
            f"'{var_no_default.name}' with None default, got {call_kwargs['val']}"
        )

        # Verify units handling
        if var_no_default.units is not None:
            assert "units" in call_kwargs
            assert call_kwargs["units"] == var_no_default.units
        else:
            assert "units" not in call_kwargs

    @given(var=variable_for_input_strategy())
    @settings(max_examples=100, deadline=None)
    def test_units_passed_only_when_not_none(self, var):
        """Units are passed to add_input only when not None, for any variable type.

        **Validates: Requirements 4.2, 4.3, 4.5, 4.6**
        """
        opt_problem = OptProblem(
            variables=[var],
            responses=[FloatVariable(name="resp_out", bounds=(-np.inf, np.inf))],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_input_calls = []

        def mock_add_input(name, **kwargs):
            add_input_calls.append((name, kwargs))

        component.add_input = mock_add_input
        component.add_output = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_input_calls if n == var.name]
        assert len(matching_calls) == 1

        _, call_kwargs = matching_calls[0]

        if var.units is not None:
            assert "units" in call_kwargs, (
                f"Variable '{var.name}' (type={type(var).__name__}) with "
                f"units={var.units!r} should have units passed to add_input"
            )
            assert call_kwargs["units"] == var.units
        else:
            assert "units" not in call_kwargs, (
                f"Variable '{var.name}' (type={type(var).__name__}) with "
                f"units=None should NOT have units passed to add_input"
            )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 9
# ---------------------------------------------------------------------------

# Strategy for non-zero scale values (avoid values too close to zero for
# numerical stability in the 1/scale computation)
_nonzero_scale_for_roundtrip = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
).filter(lambda x: abs(x) > 1e-6)

# Strategy for shift values. We bound shift magnitude so that catastrophic
# cancellation in ref - ref0 does not exceed rtol=1e-10. With double precision
# eps ~1e-16, we need |shift| * eps / |1/scale| << 1e-10.
# Keeping |shift| <= 1e4 and |scale| >= 1e-6 ensures |shift * scale| <= 1e10,
# giving relative error ~ eps * |shift * scale| = 1e-16 * 1e10 = 1e-6 << 1e-10? No.
# Actually: error(ref - ref0) ~ eps * max(|ref|, |ref0|) and
# ref - ref0 = 1/scale, so rel_error(1/scale) ~ eps * |ref0| / |1/scale|
#            = eps * |shift| * |scale|.
# For rtol=1e-10 we need eps * |shift| * |scale| < 1e-10,
# i.e., |shift| * |scale| < 1e-10 / 1e-16 = 1e6.
# With |shift| <= 1e3 and |scale| <= 1e3, |shift*scale| <= 1e6. Safe.
_shift_for_roundtrip = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)


@st.composite
def scaling_roundtrip_arrays_strategy(draw):
    """Generate non-zero scale arrays and shift arrays for the round-trip test.

    Returns (shift_array, scale_array) where all scale elements are non-zero.
    Generates 1-D or 2-D shaped arrays.
    """
    # Choose shape: 1-D or 2-D
    shape = draw(st.one_of(
        st.integers(min_value=1, max_value=10).map(lambda n: (n,)),
        st.tuples(
            st.integers(min_value=2, max_value=4),
            st.integers(min_value=2, max_value=4),
        ),
    ))

    flat_size = int(np.prod(shape))

    # Generate shift array (any finite floats)
    shift_elements = draw(
        st.lists(
            _shift_for_roundtrip,
            min_size=flat_size,
            max_size=flat_size,
        )
    )
    shift_array = np.array(shift_elements, dtype=np.float64).reshape(shape)

    # Generate scale array (all elements non-zero)
    scale_elements = draw(
        st.lists(
            _nonzero_scale_for_roundtrip,
            min_size=flat_size,
            max_size=flat_size,
        )
    )
    scale_array = np.array(scale_elements, dtype=np.float64).reshape(shape)

    return shift_array, scale_array


# ---------------------------------------------------------------------------
# Property 9: Scaling conversion round-trip
# Feature: array-variable-support, Property 9: Scaling conversion round-trip
# ---------------------------------------------------------------------------


class TestScalingConversionRoundTrip:
    """Property 9: Scaling conversion round-trip.

    For any non-zero scale array and shift array, converting from (shift, scale)
    to (ref0, ref) via ref0 = -shift; ref = (1/scale) + ref0 and back via
    shift = -ref0; scale = 1/(ref - ref0) SHALL produce the original shift and
    scale values (within floating-point tolerance).

    **Validates: Requirements 5.5**
    """

    @given(data=scaling_roundtrip_arrays_strategy())
    @settings(max_examples=100, deadline=None)
    def test_scaling_conversion_round_trip(self, data):
        """# Feature: array-variable-support, Property 9: Scaling conversion round-trip"""
        shift, scale = data

        # Forward conversion: (shift, scale) -> (ref0, ref)
        ref0 = -shift
        ref = (1.0 / scale) + ref0

        # Reverse conversion: (ref0, ref) -> (shift_back, scale_back)
        shift_back = -ref0
        scale_back = 1.0 / (ref - ref0)

        # Assert round-trip produces original values within tolerance.
        # The input domain is constrained so that |shift| * |scale| <= 1e6,
        # ensuring floating-point cancellation in (ref - ref0) stays well
        # within rtol=1e-10 for double-precision arithmetic.
        np.testing.assert_allclose(
            shift_back, shift, rtol=1e-10,
            err_msg=(
                f"Shift round-trip mismatch: original shift={shift}, "
                f"recovered shift_back={shift_back}"
            ),
        )
        np.testing.assert_allclose(
            scale_back, scale, rtol=1e-10,
            err_msg=(
                f"Scale round-trip mismatch: original scale={scale}, "
                f"recovered scale_back={scale_back}"
            ),
        )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 7
# ---------------------------------------------------------------------------


@st.composite
def array_variable_for_output_strategy(draw):
    """Generate an ArrayVariable suitable for use as a component output/response.

    Generates ArrayVariables with various bound configurations (all-inf, uniform,
    mixed), non-zero scale arrays, optional shift, and optional units.
    No zero scale elements are generated (would cause ValueError).
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    shape = draw(_array_shape_strategy)
    units = draw(_optional_units_strategy)

    flat_size = int(np.prod(shape))

    # Generate bounds with different configurations
    lower_mode = draw(st.sampled_from(["all_neginf", "uniform_finite", "mixed"]))
    if lower_mode == "all_neginf":
        lower = np.full(shape, -np.inf)
    elif lower_mode == "uniform_finite":
        scalar_lower = draw(st.floats(
            min_value=-1e4, max_value=0.0,
            allow_nan=False, allow_infinity=False,
        ))
        lower = np.full(shape, scalar_lower)
    else:  # mixed
        lower_elements = draw(
            st.lists(
                st.floats(min_value=-1e4, max_value=0.0,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        lower = np.array(lower_elements, dtype=np.float64).reshape(shape)
        # Ensure at least two different values for "mixed"
        if np.all(lower == lower.flat[0]):
            # Tweak one element to be different
            lower.flat[-1] = lower.flat[0] - 1.0

    upper_mode = draw(st.sampled_from(["all_posinf", "uniform_finite", "mixed"]))
    if upper_mode == "all_posinf":
        upper = np.full(shape, np.inf)
    elif upper_mode == "uniform_finite":
        scalar_upper = draw(st.floats(
            min_value=0.01, max_value=1e4,
            allow_nan=False, allow_infinity=False,
        ))
        upper = np.full(shape, scalar_upper)
    else:  # mixed
        upper_elements = draw(
            st.lists(
                st.floats(min_value=0.01, max_value=1e4,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        upper = np.array(upper_elements, dtype=np.float64).reshape(shape)
        # Ensure at least two different values for "mixed"
        if np.all(upper == upper.flat[0]):
            upper.flat[-1] = upper.flat[0] + 1.0

    # Ensure lower <= upper element-wise (for finite elements)
    finite_mask = np.isfinite(lower) & np.isfinite(upper)
    if np.any(finite_mask):
        # Where both finite, ensure lower <= upper
        violations = finite_mask & (lower > upper)
        if np.any(violations):
            # Swap where violated
            temp = lower[violations].copy()
            lower[violations] = upper[violations]
            upper[violations] = temp

    # Generate non-zero scale (no zeros allowed)
    scale_mode = draw(st.sampled_from(["default", "uniform_nonzero", "array_nonzero"]))
    if scale_mode == "default":
        scale = np.ones(shape)
    elif scale_mode == "uniform_nonzero":
        scalar_scale = draw(st.floats(
            min_value=0.01, max_value=10.0,
            allow_nan=False, allow_infinity=False,
        ))
        scale = np.full(shape, scalar_scale)
    else:  # array_nonzero
        scale_elements = draw(
            st.lists(
                st.floats(min_value=0.01, max_value=10.0,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        scale = np.array(scale_elements, dtype=np.float64).reshape(shape)

    # Generate shift
    shift_mode = draw(st.sampled_from(["default", "uniform", "array"]))
    if shift_mode == "default":
        shift = np.zeros(shape)
    elif shift_mode == "uniform":
        scalar_shift = draw(st.floats(
            min_value=-1e4, max_value=1e4,
            allow_nan=False, allow_infinity=False,
        ))
        shift = np.full(shape, scalar_shift)
    else:  # array
        shift_elements = draw(
            st.lists(
                st.floats(min_value=-1e4, max_value=1e4,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        shift = np.array(shift_elements, dtype=np.float64).reshape(shape)

    var = ArrayVariable(
        name=name,
        shape=shape,
        bounds=(lower, upper),
        shift=shift,
        scale=scale,
        units=units,
    )
    return var


@st.composite
def float_variable_for_output_strategy(draw):
    """Generate a FloatVariable suitable for use as a component output/response.

    Generates FloatVariables with optional shift/scale (non-zero), optional
    bounds (finite or infinite), and optional units.
    """
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True))
    units = draw(_optional_units_strategy)

    # Bounds: mix of finite and infinite
    lower_mode = draw(st.sampled_from(["finite", "neginf"]))
    if lower_mode == "finite":
        lower = draw(st.floats(
            min_value=-1e4, max_value=0.0,
            allow_nan=False, allow_infinity=False,
        ))
    else:
        lower = -np.inf

    upper_mode = draw(st.sampled_from(["finite", "posinf"]))
    if upper_mode == "finite":
        upper = draw(st.floats(
            min_value=0.01, max_value=1e4,
            allow_nan=False, allow_infinity=False,
        ))
    else:
        upper = np.inf

    # Ensure lower <= upper when both finite
    if np.isfinite(lower) and np.isfinite(upper):
        assume(lower <= upper)

    # Non-zero scale
    scale = draw(st.floats(
        min_value=-1e4, max_value=1e4,
        allow_nan=False, allow_infinity=False,
    ).filter(lambda x: abs(x) > 1e-10))

    # Shift
    shift = draw(st.floats(
        min_value=-1e4, max_value=1e4,
        allow_nan=False, allow_infinity=False,
    ))

    var = FloatVariable(
        name=name,
        bounds=(lower, upper),
        shift=shift,
        scale=scale,
        units=units,
    )
    return var


@st.composite
def variable_for_output_strategy(draw):
    """Generate either an ArrayVariable or FloatVariable for output testing."""
    use_array = draw(st.booleans())
    if use_array:
        return draw(array_variable_for_output_strategy())
    else:
        return draw(float_variable_for_output_strategy())


# ---------------------------------------------------------------------------
# Property 7: Component setup registers outputs correctly with units
# Feature: array-variable-support, Property 7
# ---------------------------------------------------------------------------


class TestComponentOutputRegistration:
    """Property 7: Component setup registers outputs correctly with units.

    For any response (either ArrayVariable or FloatVariable) in an OptProblem's
    responses list with no zero scale elements, the component's setup SHALL call
    add_output with the appropriate shape, val, bounds, and scaling, and SHALL
    pass units to add_output only when the response's units is not None.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    """

    @given(resp=array_variable_for_output_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_variable_output_val_and_shape(self, resp):
        """ArrayVariable output: val = np.zeros(shape), shape = resp.shape.

        **Validates: Requirements 5.1**
        """
        opt_problem = OptProblem(
            variables=[FloatVariable(name="x_in", bounds=(-1.0, 1.0), default=0.0)],
            responses=[resp],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_output_calls = []

        def mock_add_output(name, **kwargs):
            add_output_calls.append((name, kwargs))

        component.add_output = mock_add_output
        component.add_input = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_output_calls if n == resp.name]
        assert len(matching_calls) == 1, (
            f"Expected exactly 1 add_output call for '{resp.name}', "
            f"got {len(matching_calls)}"
        )

        _, call_kwargs = matching_calls[0]

        # Verify shape
        assert "shape" in call_kwargs, (
            f"ArrayVariable '{resp.name}' should have shape in add_output kwargs"
        )
        assert call_kwargs["shape"] == resp.shape, (
            f"Shape mismatch: got {call_kwargs['shape']}, expected {resp.shape}"
        )

        # Verify val = np.zeros(shape)
        assert "val" in call_kwargs, (
            f"ArrayVariable '{resp.name}' should have val in add_output kwargs"
        )
        np.testing.assert_array_equal(
            call_kwargs["val"], np.zeros(resp.shape),
            err_msg=f"val should be np.zeros({resp.shape}) for ArrayVariable output"
        )

    @given(resp=array_variable_for_output_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_variable_output_lower_bound(self, resp):
        """ArrayVariable output lower bound: scalar if uniform, array if mixed, omitted if all-inf.

        **Validates: Requirements 5.5**
        """
        opt_problem = OptProblem(
            variables=[FloatVariable(name="x_in", bounds=(-1.0, 1.0), default=0.0)],
            responses=[resp],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_output_calls = []

        def mock_add_output(name, **kwargs):
            add_output_calls.append((name, kwargs))

        component.add_output = mock_add_output
        component.add_input = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_output_calls if n == resp.name]
        assert len(matching_calls) == 1
        _, call_kwargs = matching_calls[0]

        lower_bound = np.asarray(resp.bounds[0])

        if np.all(np.isneginf(lower_bound)):
            # All -inf: lower should be omitted
            assert "lower" not in call_kwargs, (
                f"ArrayVariable '{resp.name}' with all-neginf lower bounds "
                f"should NOT have 'lower' in add_output kwargs"
            )
        elif np.all(lower_bound == lower_bound.flat[0]):
            # All same finite value: pass scalar
            assert "lower" in call_kwargs, (
                f"ArrayVariable '{resp.name}' with uniform finite lower "
                f"should have 'lower' in add_output kwargs"
            )
            assert np.isscalar(call_kwargs["lower"]) or isinstance(call_kwargs["lower"], float), (
                f"Lower bound should be scalar for uniform values, "
                f"got {type(call_kwargs['lower'])}"
            )
            assert call_kwargs["lower"] == pytest.approx(float(lower_bound.flat[0])), (
                f"Lower bound mismatch: got {call_kwargs['lower']}, "
                f"expected {float(lower_bound.flat[0])}"
            )
        else:
            # Mixed values: pass array
            assert "lower" in call_kwargs, (
                f"ArrayVariable '{resp.name}' with mixed lower bounds "
                f"should have 'lower' in add_output kwargs"
            )
            np.testing.assert_array_equal(
                call_kwargs["lower"], lower_bound,
                err_msg="Lower bound array mismatch"
            )

    @given(resp=array_variable_for_output_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_variable_output_upper_bound(self, resp):
        """ArrayVariable output upper bound: scalar if uniform, array if mixed, omitted if all-inf.

        **Validates: Requirements 5.5**
        """
        opt_problem = OptProblem(
            variables=[FloatVariable(name="x_in", bounds=(-1.0, 1.0), default=0.0)],
            responses=[resp],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_output_calls = []

        def mock_add_output(name, **kwargs):
            add_output_calls.append((name, kwargs))

        component.add_output = mock_add_output
        component.add_input = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_output_calls if n == resp.name]
        assert len(matching_calls) == 1
        _, call_kwargs = matching_calls[0]

        upper_bound = np.asarray(resp.bounds[1])

        if np.all(np.isposinf(upper_bound)):
            # All +inf: upper should be omitted
            assert "upper" not in call_kwargs, (
                f"ArrayVariable '{resp.name}' with all-posinf upper bounds "
                f"should NOT have 'upper' in add_output kwargs"
            )
        elif np.all(upper_bound == upper_bound.flat[0]):
            # All same finite value: pass scalar
            assert "upper" in call_kwargs, (
                f"ArrayVariable '{resp.name}' with uniform finite upper "
                f"should have 'upper' in add_output kwargs"
            )
            assert np.isscalar(call_kwargs["upper"]) or isinstance(call_kwargs["upper"], float), (
                f"Upper bound should be scalar for uniform values, "
                f"got {type(call_kwargs['upper'])}"
            )
            assert call_kwargs["upper"] == pytest.approx(float(upper_bound.flat[0])), (
                f"Upper bound mismatch: got {call_kwargs['upper']}, "
                f"expected {float(upper_bound.flat[0])}"
            )
        else:
            # Mixed values: pass array
            assert "upper" in call_kwargs, (
                f"ArrayVariable '{resp.name}' with mixed upper bounds "
                f"should have 'upper' in add_output kwargs"
            )
            np.testing.assert_array_equal(
                call_kwargs["upper"], upper_bound,
                err_msg="Upper bound array mismatch"
            )

    @given(resp=array_variable_for_output_strategy())
    @settings(max_examples=100, deadline=None)
    def test_array_variable_output_scaling(self, resp):
        """ArrayVariable output scaling: ref0/ref from shift/scale.

        Computed when non-default, omitted when default.

        **Validates: Requirements 5.5**
        """
        opt_problem = OptProblem(
            variables=[FloatVariable(name="x_in", bounds=(-1.0, 1.0), default=0.0)],
            responses=[resp],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_output_calls = []

        def mock_add_output(name, **kwargs):
            add_output_calls.append((name, kwargs))

        component.add_output = mock_add_output
        component.add_input = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_output_calls if n == resp.name]
        assert len(matching_calls) == 1
        _, call_kwargs = matching_calls[0]

        shift_arr = np.asarray(resp.shift) if resp.shift is not None else np.zeros(resp.shape)
        scale_arr = np.asarray(resp.scale) if resp.scale is not None else np.ones(resp.shape)

        is_default_scaling = np.all(shift_arr == 0.0) and np.all(scale_arr == 1.0)

        if is_default_scaling:
            # Default scaling: ref0 and ref should be omitted
            assert "ref0" not in call_kwargs, (
                f"ArrayVariable '{resp.name}' with default scaling should "
                f"NOT have ref0 in add_output kwargs"
            )
            assert "ref" not in call_kwargs, (
                f"ArrayVariable '{resp.name}' with default scaling should "
                f"NOT have ref in add_output kwargs"
            )
        else:
            # Non-default: ref0 = -shift, ref = (1/scale) + ref0
            expected_ref0 = -shift_arr
            expected_ref = (1.0 / scale_arr) + expected_ref0

            assert "ref0" in call_kwargs, (
                f"ArrayVariable '{resp.name}' with non-default scaling should "
                f"have ref0 in add_output kwargs"
            )
            assert "ref" in call_kwargs, (
                f"ArrayVariable '{resp.name}' with non-default scaling should "
                f"have ref in add_output kwargs"
            )

            np.testing.assert_allclose(
                call_kwargs["ref0"], expected_ref0, rtol=1e-12,
                err_msg=f"ref0 mismatch for '{resp.name}'"
            )
            np.testing.assert_allclose(
                call_kwargs["ref"], expected_ref, rtol=1e-12,
                err_msg=f"ref mismatch for '{resp.name}'"
            )

    @given(resp=float_variable_for_output_strategy())
    @settings(max_examples=100, deadline=None)
    def test_float_variable_output_registers_correctly(self, resp):
        """FloatVariable output: val = 0.0, bounds converted, scaling computed.

        **Validates: Requirements 5.3, 5.4**
        """
        opt_problem = OptProblem(
            variables=[FloatVariable(name="x_in", bounds=(-1.0, 1.0), default=0.0)],
            responses=[resp],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_output_calls = []

        def mock_add_output(name, **kwargs):
            add_output_calls.append((name, kwargs))

        component.add_output = mock_add_output
        component.add_input = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_output_calls if n == resp.name]
        assert len(matching_calls) == 1
        _, call_kwargs = matching_calls[0]

        # Verify val = 0.0
        assert "val" in call_kwargs
        assert call_kwargs["val"] == 0.0, (
            f"FloatVariable output val should be 0.0, got {call_kwargs['val']}"
        )

        # Verify lower bound: finite -> value, -inf -> None
        lower, upper = resp.bounds
        if np.isfinite(lower):
            assert call_kwargs["lower"] == pytest.approx(lower), (
                f"Lower bound mismatch: got {call_kwargs['lower']}, expected {lower}"
            )
        else:
            assert call_kwargs["lower"] is None, (
                f"Lower bound should be None for -inf, got {call_kwargs['lower']}"
            )

        # Verify upper bound: finite -> value, +inf -> None
        if np.isfinite(upper):
            assert call_kwargs["upper"] == pytest.approx(upper), (
                f"Upper bound mismatch: got {call_kwargs['upper']}, expected {upper}"
            )
        else:
            assert call_kwargs["upper"] is None, (
                f"Upper bound should be None for +inf, got {call_kwargs['upper']}"
            )

        # Verify scaling: ref0 = -shift, ref = (1/scale) + ref0
        shift = resp.shift if resp.shift is not None else 0.0
        scale = resp.scale if resp.scale is not None else 1.0
        expected_ref0 = -shift
        expected_ref = (1.0 / scale) + expected_ref0

        assert call_kwargs["ref0"] == pytest.approx(expected_ref0, rel=1e-12, abs=1e-15), (
            f"ref0 mismatch: got {call_kwargs['ref0']}, expected {expected_ref0}"
        )
        assert call_kwargs["ref"] == pytest.approx(expected_ref, rel=1e-12, abs=1e-15), (
            f"ref mismatch: got {call_kwargs['ref']}, expected {expected_ref}"
        )

    @given(resp=variable_for_output_strategy())
    @settings(max_examples=100, deadline=None)
    def test_units_passed_only_when_not_none(self, resp):
        """Units are passed to add_output only when not None, for any response type.

        **Validates: Requirements 5.2, 5.3, 5.4**
        """
        opt_problem = OptProblem(
            variables=[FloatVariable(name="x_in", bounds=(-1.0, 1.0), default=0.0)],
            responses=[resp],
        )
        evaluator = DummyDefaultEvaluator(opt_problem=opt_problem)

        component = EvaluatorOpenMdaoComponent(evaluator)
        add_output_calls = []

        def mock_add_output(name, **kwargs):
            add_output_calls.append((name, kwargs))

        component.add_output = mock_add_output
        component.add_input = lambda name, **kwargs: None

        component.setup()

        matching_calls = [(n, kw) for n, kw in add_output_calls if n == resp.name]
        assert len(matching_calls) == 1
        _, call_kwargs = matching_calls[0]

        if resp.units is not None:
            assert "units" in call_kwargs, (
                f"Response '{resp.name}' (type={type(resp).__name__}) with "
                f"units={resp.units!r} should have units passed to add_output"
            )
            assert call_kwargs["units"] == resp.units
        else:
            assert "units" not in call_kwargs, (
                f"Response '{resp.name}' (type={type(resp).__name__}) with "
                f"units=None should NOT have units passed to add_output"
            )


# ---------------------------------------------------------------------------
# Strategies for array-variable-support Property 8
# ---------------------------------------------------------------------------


@st.composite
def mixed_opt_problem_for_compute_strategy(draw):
    """Generate an OptProblem with mixed FloatVariable/ArrayVariable variables
    and responses, along with corresponding input values and expected evaluator
    response values for testing the compute method.

    Returns a tuple of (opt_problem, input_values, expected_responses) where:
    - opt_problem: an OptProblem with at least one FloatVariable and one ArrayVariable
    - input_values: dict mapping variable names to input values (scalars or arrays)
    - expected_responses: dict mapping response names to values the mock evaluator returns
    """
    # Generate 1-2 float variables and 1-2 array variables
    num_float_vars = draw(st.integers(min_value=1, max_value=2))
    num_array_vars = draw(st.integers(min_value=1, max_value=2))

    variables = []
    input_values = {}

    # Float variables
    for i in range(num_float_vars):
        name = f"fv_{i}"
        lower = draw(st.floats(
            min_value=-100.0, max_value=0.0,
            allow_nan=False, allow_infinity=False,
        ))
        upper = draw(st.floats(
            min_value=lower + 0.1, max_value=100.0,
            allow_nan=False, allow_infinity=False,
        ))
        variables.append(
            FloatVariable(name=name, bounds=(lower, upper), default=0.0)
        )
        # Generate a scalar input value
        val = draw(st.floats(
            min_value=lower, max_value=upper,
            allow_nan=False, allow_infinity=False,
        ))
        input_values[name] = np.array([val])  # OpenMDAO inputs are 1-element arrays for scalars

    # Array variables
    for i in range(num_array_vars):
        name = f"av_{i}"
        shape = draw(st.sampled_from([(2,), (3,), (2, 2)]))
        flat_size = int(np.prod(shape))
        variables.append(
            ArrayVariable(
                name=name,
                shape=shape,
                bounds=(-100.0, 100.0),
                default=np.zeros(shape),
            )
        )
        # Generate an array input value
        elements = draw(
            st.lists(
                st.floats(min_value=-50.0, max_value=50.0,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        input_values[name] = np.array(elements, dtype=np.float64).reshape(shape)

    # Generate 1-2 float responses and 1-2 array responses
    num_float_resps = draw(st.integers(min_value=1, max_value=2))
    num_array_resps = draw(st.integers(min_value=1, max_value=2))

    responses = []
    expected_responses = {}

    # Float responses
    for i in range(num_float_resps):
        name = f"fr_{i}"
        responses.append(
            FloatVariable(name=name, bounds=(-np.inf, np.inf))
        )
        # Expected scalar response value
        val = draw(st.floats(
            min_value=-100.0, max_value=100.0,
            allow_nan=False, allow_infinity=False,
        ))
        expected_responses[name] = val

    # Array responses
    for i in range(num_array_resps):
        name = f"ar_{i}"
        shape = draw(st.sampled_from([(2,), (3,), (2, 2)]))
        flat_size = int(np.prod(shape))
        responses.append(
            ArrayVariable(
                name=name,
                shape=shape,
                bounds=(-np.inf, np.inf),
            )
        )
        # Expected array response value
        elements = draw(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0,
                          allow_nan=False, allow_infinity=False),
                min_size=flat_size,
                max_size=flat_size,
            )
        )
        expected_responses[name] = np.array(elements, dtype=np.float64).reshape(shape)

    opt_problem = OptProblem(
        variables=variables,
        responses=responses,
        objectives=[responses[0].name],
    )

    return opt_problem, input_values, expected_responses


# ---------------------------------------------------------------------------
# Property 8: Component compute marshals mixed scalar/array data correctly
# Feature: array-variable-support, Property 8
# ---------------------------------------------------------------------------


class MockComputeEvaluator(Evaluator):
    """Evaluator that captures the DataFrame it receives and writes known responses.

    Used to verify that compute() correctly marshals data to/from the evaluator.
    """

    __test__ = False

    def __init__(self, opt_problem, expected_responses):
        super().__init__(opt_problem=opt_problem)
        self.expected_responses = expected_responses
        self.captured_dataframe = None

    def _evaluate(self, sites: pd.DataFrame) -> None:
        # Capture the DataFrame for assertions
        self.captured_dataframe = sites.copy()
        # Write known response values
        for name, value in self.expected_responses.items():
            sites[name] = [value]


class TestComponentComputeMarshaling:
    """Property 8: Component compute marshals mixed scalar/array data correctly.

    For any OptProblem containing a mix of FloatVariable and ArrayVariable
    variables and responses, the component's compute SHALL place full NumPy
    arrays from inputs[name] into the DataFrame for ArrayVariable inputs and
    scalars for FloatVariable inputs, and SHALL extract full NumPy arrays from
    the DataFrame to outputs[name] for ArrayVariable responses and scalars for
    FloatVariable responses.

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """

    @given(data=mixed_opt_problem_for_compute_strategy())
    @settings(max_examples=100, deadline=None)
    def test_compute_marshals_mixed_data(self, data):
        """Property 8: Component compute marshals mixed scalar/array data."""
        opt_problem, input_values, expected_responses = data

        # Create mock evaluator that captures inputs and returns known responses
        evaluator = MockComputeEvaluator(
            opt_problem=opt_problem,
            expected_responses=expected_responses,
        )

        # Create the component
        component = EvaluatorOpenMdaoComponent(evaluator)

        # Mock add_input/add_output so setup() doesn't interact with OpenMDAO
        component.add_input = lambda name, **kwargs: None
        component.add_output = lambda name, **kwargs: None
        component.setup()

        # Build mock inputs dict (simulating OpenMDAO's inputs dict behavior)
        mock_inputs = input_values.copy()

        # Build mock outputs dict that will be populated by compute
        mock_outputs = {}

        # Call compute
        component.compute(mock_inputs, mock_outputs)

        # Verify INPUT marshaling: check what the evaluator received in its DataFrame
        captured_df = component.eval.captured_dataframe
        assert captured_df is not None, "Evaluator should have been called"
        assert len(captured_df) == 1, "Should have exactly 1 row"

        var_lookup = {v.name: v for v in opt_problem.variables}

        for var_name, var in var_lookup.items():
            cell_value = captured_df.iloc[0][var_name]

            if isinstance(var, ArrayVariable):
                # ArrayVariable inputs: DataFrame cell should contain a NumPy array
                assert isinstance(cell_value, np.ndarray), (
                    f"Expected NumPy array in DataFrame for ArrayVariable "
                    f"input '{var_name}', got {type(cell_value).__name__}"
                )
                np.testing.assert_array_equal(
                    cell_value, input_values[var_name],
                    err_msg=(
                        f"Array input mismatch for '{var_name}': "
                        f"expected shape {input_values[var_name].shape}"
                    ),
                )
                # Shape must be preserved
                assert cell_value.shape == var.shape, (
                    f"Shape mismatch for ArrayVariable input '{var_name}': "
                    f"got {cell_value.shape}, expected {var.shape}"
                )
            else:
                # FloatVariable inputs: DataFrame cell should contain a scalar
                assert np.isscalar(cell_value) or (
                    isinstance(cell_value, np.ndarray) and cell_value.ndim == 0
                ), (
                    f"Expected scalar in DataFrame for FloatVariable "
                    f"input '{var_name}', got {type(cell_value).__name__} "
                    f"with value {cell_value}"
                )
                # Value should match (OpenMDAO passes 1-element arrays for scalars)
                expected_scalar = float(input_values[var_name].flat[0])
                assert float(cell_value) == pytest.approx(expected_scalar), (
                    f"Scalar input mismatch for '{var_name}': "
                    f"got {cell_value}, expected {expected_scalar}"
                )

        # Verify OUTPUT marshaling: check what compute placed in outputs dict
        resp_lookup = {r.name: r for r in opt_problem.responses}

        for resp_name, resp in resp_lookup.items():
            assert resp_name in mock_outputs, (
                f"Response '{resp_name}' should be in outputs dict"
            )
            output_value = mock_outputs[resp_name]

            if isinstance(resp, ArrayVariable):
                # ArrayVariable responses: outputs should contain a NumPy array
                assert isinstance(output_value, np.ndarray), (
                    f"Expected NumPy array in outputs for ArrayVariable "
                    f"response '{resp_name}', got {type(output_value).__name__}"
                )
                np.testing.assert_array_equal(
                    output_value, expected_responses[resp_name],
                    err_msg=(
                        f"Array output mismatch for '{resp_name}': "
                        f"expected shape {expected_responses[resp_name].shape}"
                    ),
                )
                # Shape must match the declared shape
                assert output_value.shape == resp.shape, (
                    f"Shape mismatch for ArrayVariable response '{resp_name}': "
                    f"got {output_value.shape}, expected {resp.shape}"
                )
            else:
                # FloatVariable responses: outputs should contain a scalar
                expected_scalar = expected_responses[resp_name]
                actual_scalar = float(output_value) if isinstance(
                    output_value, np.ndarray
                ) else float(output_value)
                assert actual_scalar == pytest.approx(expected_scalar), (
                    f"Scalar output mismatch for '{resp_name}': "
                    f"got {actual_scalar}, expected {expected_scalar}"
                )
