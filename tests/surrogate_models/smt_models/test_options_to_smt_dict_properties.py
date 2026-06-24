"""Property-based test for _options_to_smt_dict conversion correctness.

Feature: surrogate-model-migration
Property 5: _options_to_smt_dict conversion correctness

**Validates: Requirements 5.7**
"""

import numpy as np
from pathlib import Path
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from standard_evaluator.problem import OptProblem, FloatVariable
from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModelOptions,
    _options_to_smt_dict,
)


# --- Strategies ---


@st.composite
def smt_options_and_problem(draw):
    """Generate a valid AbstractSmtModelOptions instance paired with an OptProblem.

    Strategy:
    - Generate between 1-5 variables with finite bounds
    - Generate 1-2 responses
    - Randomize boolean print toggles and use_xlimits flag
    - Optionally set a data_dir path
    """
    # Generate variables with finite bounds
    num_vars = draw(st.integers(min_value=1, max_value=5))
    variables = []
    for i in range(num_vars):
        lb = draw(st.floats(min_value=-100.0, max_value=50.0, allow_nan=False, allow_infinity=False))
        ub = draw(st.floats(min_value=lb + 0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
        assume(ub > lb)
        variables.append(FloatVariable(name=f"x{i}", bounds=(lb, ub)))

    # Generate responses
    num_responses = draw(st.integers(min_value=1, max_value=2))
    responses = [FloatVariable(name=f"y{i}") for i in range(num_responses)]

    opt_problem = OptProblem(variables=variables, responses=responses)

    # Generate options with random boolean toggles
    print_global = draw(st.booleans())
    print_training = draw(st.booleans())
    print_prediction = draw(st.booleans())
    print_problem = draw(st.booleans())
    print_solver = draw(st.booleans())
    use_xlimits = draw(st.booleans())

    # Optionally set data_dir
    data_dir = draw(st.one_of(st.none(), st.just(Path("/tmp/test_data"))))

    options = AbstractSmtModelOptions(
        print_global=print_global,
        print_training=print_training,
        print_prediction=print_prediction,
        print_problem=print_problem,
        print_solver=print_solver,
        use_xlimits=use_xlimits,
        data_dir=data_dir,
    )

    # nonconstant_variables: all variable names (no constants in this strategy)
    nonconstant_variables = [v.name for v in variables]

    return options, opt_problem, nonconstant_variables


@given(data=smt_options_and_problem())
@settings(max_examples=100, deadline=None)
def test_excluded_keys_not_in_output(data):
    """Property 5: The output dict does NOT contain 'parameters' or 'use_xlimits' keys.
    The 'data_dir' Pydantic field is excluded from the model_dump iteration, but when
    data_dir is set, a string-converted version is re-added for the SMT constructor.

    Feature: surrogate-model-migration
    Property 5: _options_to_smt_dict conversion correctness

    **Validates: Requirements 5.7**
    """
    options, opt_problem, nonconstant_variables = data

    result = _options_to_smt_dict(options, opt_problem, nonconstant_variables)

    # parameters and use_xlimits must NEVER appear
    assert "parameters" not in result, (
        f"'parameters' key should not be in result, but found: {result.get('parameters')}"
    )
    assert "use_xlimits" not in result, (
        f"'use_xlimits' key should not be in result, but found: {result.get('use_xlimits')}"
    )

    # data_dir: when options.data_dir is None, it should not be in result;
    # when set, it should be a string (not a Path object)
    if options.data_dir is None:
        assert "data_dir" not in result, (
            f"'data_dir' should not be in result when options.data_dir is None"
        )
    else:
        assert "data_dir" in result, (
            f"'data_dir' should be in result when options.data_dir is set"
        )
        assert isinstance(result["data_dir"], str), (
            f"'data_dir' should be a string in result, got {type(result['data_dir'])}"
        )
        assert result["data_dir"] == str(options.data_dir), (
            f"'data_dir' value mismatch: expected {str(options.data_dir)!r}, "
            f"got {result['data_dir']!r}"
        )


@given(data=smt_options_and_problem())
@settings(max_examples=100, deadline=None)
def test_xlimits_computed_from_opt_problem_when_use_xlimits_true(data):
    """Property 5: When use_xlimits is True, xlimits is computed from OptProblem
    variable bounds for non-constant variables.

    Feature: surrogate-model-migration
    Property 5: _options_to_smt_dict conversion correctness

    **Validates: Requirements 5.7**
    """
    options, opt_problem, nonconstant_variables = data

    # Force use_xlimits to True for this test
    options_with_xlimits = options.model_copy(update={"use_xlimits": True})

    result = _options_to_smt_dict(options_with_xlimits, opt_problem, nonconstant_variables)

    # xlimits should be present
    assert "xlimits" in result, (
        "Expected 'xlimits' key in result when use_xlimits=True, "
        f"but keys are: {list(result.keys())}"
    )

    xlimits = result["xlimits"]

    # Build expected xlimits from OptProblem
    expected_xlimits = []
    for var in opt_problem.variables:
        if var.name in nonconstant_variables:
            expected_xlimits.append(list(var.bounds))
    expected_xlimits = np.array(expected_xlimits)

    np.testing.assert_array_almost_equal(
        xlimits,
        expected_xlimits,
        decimal=12,
        err_msg=(
            f"xlimits mismatch: got {xlimits}, expected {expected_xlimits}"
        ),
    )


@given(data=smt_options_and_problem())
@settings(max_examples=100, deadline=None)
def test_all_other_option_fields_present_with_correct_values(data):
    """Property 5: All option fields other than 'parameters', 'use_xlimits', and
    'data_dir' are present in the result with their correct values.

    Feature: surrogate-model-migration
    Property 5: _options_to_smt_dict conversion correctness

    **Validates: Requirements 5.7**
    """
    options, opt_problem, nonconstant_variables = data

    result = _options_to_smt_dict(options, opt_problem, nonconstant_variables)

    # The fields excluded from model_dump iteration
    excluded_fields = {"parameters", "use_xlimits", "data_dir"}
    options_dict = options.model_dump()

    for field_name, value in options_dict.items():
        if field_name in excluded_fields:
            continue
        assert field_name in result, (
            f"Expected field '{field_name}' in result dict, "
            f"but keys are: {list(result.keys())}"
        )
        assert result[field_name] == value, (
            f"Field '{field_name}' has wrong value: "
            f"expected {value!r}, got {result[field_name]!r}"
        )
