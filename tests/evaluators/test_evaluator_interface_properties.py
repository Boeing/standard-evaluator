"""Property-based tests for the simplified Evaluator interface.

Property 1: OptProblem initialization preserves interface
Property 2: EvaluatorInfo initialization preserves interface

Validates: Requirements 2.1, 2.2
"""

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from standard_evaluator import EvaluatorInfo, FloatVariable, OptProblem
from standard_evaluator.evaluators.abstract_evaluator import Evaluator


# ---------------------------------------------------------------------------
# Minimal concrete Evaluator subclass for testing
# ---------------------------------------------------------------------------


class MinimalEvaluator(Evaluator):
    """Minimal concrete Evaluator subclass that does nothing in _evaluate."""

    def _evaluate(self, sites: pd.DataFrame) -> None:
        pass


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Strategy for generating valid unique names (alphanumeric, non-empty)
_name_alphabet = st.characters(whitelist_categories=("L", "N"))


@composite
def unique_names(draw, min_size=1, max_size=5):
    """Generate a list of unique non-empty alphanumeric strings."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    names = draw(
        st.lists(
            st.text(alphabet=_name_alphabet, min_size=1, max_size=10),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return names


@composite
def valid_opt_problem(draw):
    """Generate a random valid OptProblem with unique variable and response names.

    - At least 1 variable and 1 response
    - All names are unique across variables and responses
    - Uses FloatVariable with valid non-degenerate bounds
    """
    n_vars = draw(st.integers(min_value=1, max_value=6))
    n_responses = draw(st.integers(min_value=1, max_value=4))

    # Generate unique names for all variables and responses combined
    all_names = draw(
        st.lists(
            st.text(alphabet=_name_alphabet, min_size=1, max_size=10),
            min_size=n_vars + n_responses,
            max_size=n_vars + n_responses,
            unique=True,
        )
    )

    var_names = all_names[:n_vars]
    resp_names = all_names[n_vars:]

    # Build variables with valid bounds
    variables = []
    for name in var_names:
        lower = draw(
            st.floats(min_value=-100.0, max_value=99.0,
                      allow_nan=False, allow_infinity=False)
        )
        upper = draw(
            st.floats(min_value=lower + 0.1, max_value=100.0,
                      allow_nan=False, allow_infinity=False)
        )
        variables.append(FloatVariable(name=name, bounds=(lower, upper)))

    # Build responses
    responses = []
    for name in resp_names:
        responses.append(FloatVariable(name=name))

    opt_problem = OptProblem(
        name="test_problem",
        variables=variables,
        responses=responses,
    )

    return opt_problem


# ---------------------------------------------------------------------------
# Property 1: OptProblem initialization preserves interface
# ---------------------------------------------------------------------------


class TestOptProblemInitializationPreservesInterface:
    """Property 1: OptProblem initialization preserves interface.

    For any valid OptProblem instance, constructing an Evaluator subclass with
    that opt_problem results in evaluator.inputs and evaluator.outputs matching
    the variable and response names from the OptProblem.

    **Validates: Requirements 2.1**
    """

    @given(opt_problem=valid_opt_problem())
    @settings(max_examples=100)
    def test_inputs_length_matches_variables(self, opt_problem):
        """evaluator.inputs has the same length as opt_problem.variables.

        **Validates: Requirements 2.1**
        """
        evaluator = MinimalEvaluator(opt_problem=opt_problem)
        assert len(evaluator.inputs) == len(opt_problem.variables)

    @given(opt_problem=valid_opt_problem())
    @settings(max_examples=100)
    def test_outputs_length_matches_responses(self, opt_problem):
        """evaluator.outputs has the same length as opt_problem.responses.

        **Validates: Requirements 2.1**
        """
        evaluator = MinimalEvaluator(opt_problem=opt_problem)
        assert len(evaluator.outputs) == len(opt_problem.responses)

    @given(opt_problem=valid_opt_problem())
    @settings(max_examples=100)
    def test_input_names_match_variable_names(self, opt_problem):
        """The names in evaluator.inputs match the variable names from OptProblem.

        **Validates: Requirements 2.1**
        """
        evaluator = MinimalEvaluator(opt_problem=opt_problem)
        expected_names = [v.name for v in opt_problem.variables]
        assert evaluator.inputs == expected_names

    @given(opt_problem=valid_opt_problem())
    @settings(max_examples=100)
    def test_output_names_match_response_names(self, opt_problem):
        """The names in evaluator.outputs match the response names from OptProblem.

        **Validates: Requirements 2.1**
        """
        evaluator = MinimalEvaluator(opt_problem=opt_problem)
        expected_names = [r.name for r in opt_problem.responses]
        assert evaluator.outputs == expected_names


# ---------------------------------------------------------------------------
# Strategy for generating valid EvaluatorInfo instances
# ---------------------------------------------------------------------------


@composite
def valid_evaluator_info(draw):
    """Generate a random valid EvaluatorInfo with unique input and output names.

    - At least 1 input and 1 output
    - All names are unique across inputs and outputs
    - Uses FloatVariable with valid non-degenerate bounds
    """
    n_inputs = draw(st.integers(min_value=1, max_value=6))
    n_outputs = draw(st.integers(min_value=1, max_value=4))

    # Generate unique names for all inputs and outputs combined
    all_names = draw(
        st.lists(
            st.text(alphabet=_name_alphabet, min_size=1, max_size=10),
            min_size=n_inputs + n_outputs,
            max_size=n_inputs + n_outputs,
            unique=True,
        )
    )

    input_names = all_names[:n_inputs]
    output_names = all_names[n_inputs:]

    # Build inputs with valid bounds
    inputs = []
    for name in input_names:
        lower = draw(
            st.floats(min_value=-100.0, max_value=99.0,
                      allow_nan=False, allow_infinity=False)
        )
        upper = draw(
            st.floats(min_value=lower + 0.1, max_value=100.0,
                      allow_nan=False, allow_infinity=False)
        )
        inputs.append(FloatVariable(name=name, bounds=(lower, upper)))

    # Build outputs
    outputs = []
    for name in output_names:
        outputs.append(FloatVariable(name=name))

    evaluator_info = EvaluatorInfo(
        name="test_evaluator",
        inputs=inputs,
        outputs=outputs,
    )

    return evaluator_info


# ---------------------------------------------------------------------------
# Property 2: EvaluatorInfo initialization preserves interface
# ---------------------------------------------------------------------------


class TestEvaluatorInfoInitializationPreservesInterface:
    """Property 2: EvaluatorInfo initialization preserves interface.

    For any valid EvaluatorInfo instance, constructing an Evaluator subclass with
    that interface results in evaluator.inputs and evaluator.outputs matching
    the input and output names from the EvaluatorInfo.

    **Validates: Requirements 2.2**
    """

    @given(evaluator_info=valid_evaluator_info())
    @settings(max_examples=100)
    def test_inputs_length_matches_evaluator_info_inputs(self, evaluator_info):
        """evaluator.inputs has the same length as EvaluatorInfo's inputs.

        **Validates: Requirements 2.2**
        """
        evaluator = MinimalEvaluator(interface=evaluator_info)
        assert len(evaluator.inputs) == len(evaluator_info.inputs)

    @given(evaluator_info=valid_evaluator_info())
    @settings(max_examples=100)
    def test_outputs_length_matches_evaluator_info_outputs(self, evaluator_info):
        """evaluator.outputs has the same length as EvaluatorInfo's outputs.

        **Validates: Requirements 2.2**
        """
        evaluator = MinimalEvaluator(interface=evaluator_info)
        assert len(evaluator.outputs) == len(evaluator_info.outputs)

    @given(evaluator_info=valid_evaluator_info())
    @settings(max_examples=100)
    def test_input_names_match_evaluator_info_input_names(self, evaluator_info):
        """The names in evaluator.inputs match the input names from EvaluatorInfo.

        **Validates: Requirements 2.2**
        """
        evaluator = MinimalEvaluator(interface=evaluator_info)
        expected_names = [v.name for v in evaluator_info.inputs]
        assert evaluator.inputs == expected_names

    @given(evaluator_info=valid_evaluator_info())
    @settings(max_examples=100)
    def test_output_names_match_evaluator_info_output_names(self, evaluator_info):
        """The names in evaluator.outputs match the output names from EvaluatorInfo.

        **Validates: Requirements 2.2**
        """
        evaluator = MinimalEvaluator(interface=evaluator_info)
        expected_names = [v.name for v in evaluator_info.outputs]
        assert evaluator.outputs == expected_names


# ---------------------------------------------------------------------------
# Property 3: create_opt_problem utility produces valid OptProblem
# ---------------------------------------------------------------------------


class TestCreateOptProblemProducesValidOptProblem:
    """Property 3: create_opt_problem utility produces valid OptProblem.

    For any pair of positive integers (num_independent, num_dependent),
    calling create_opt_problem(num_independent, num_dependent) returns a valid
    OptProblem with exactly num_independent variables and num_dependent responses.

    **Validates: Requirements 8.1**
    """

    @given(
        num_independent=st.integers(min_value=1, max_value=50),
        num_dependent=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_variable_count_matches_num_independent(self, num_independent, num_dependent):
        """The resulting OptProblem has exactly num_independent variables.

        **Validates: Requirements 8.1**
        """
        from standard_evaluator.utilities import create_opt_problem

        result = create_opt_problem(num_independent, num_dependent)
        assert len(result.variables) == num_independent

    @given(
        num_independent=st.integers(min_value=1, max_value=50),
        num_dependent=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_response_count_matches_num_dependent(self, num_independent, num_dependent):
        """The resulting OptProblem has exactly num_dependent responses.

        **Validates: Requirements 8.1**
        """
        from standard_evaluator.utilities import create_opt_problem

        result = create_opt_problem(num_independent, num_dependent)
        assert len(result.responses) == num_dependent

    @given(
        num_independent=st.integers(min_value=1, max_value=50),
        num_dependent=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_variable_names_follow_pattern(self, num_independent, num_dependent):
        """Variable names follow the pattern "x0", "x1", etc.

        **Validates: Requirements 8.1**
        """
        from standard_evaluator.utilities import create_opt_problem

        result = create_opt_problem(num_independent, num_dependent)
        expected_names = [f"x{i}" for i in range(num_independent)]
        actual_names = [v.name for v in result.variables]
        assert actual_names == expected_names

    @given(
        num_independent=st.integers(min_value=1, max_value=50),
        num_dependent=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_response_names_follow_pattern(self, num_independent, num_dependent):
        """Response names follow the pattern "f0", "f1", etc.

        **Validates: Requirements 8.1**
        """
        from standard_evaluator.utilities import create_opt_problem

        result = create_opt_problem(num_independent, num_dependent)
        expected_names = [f"f{i}" for i in range(num_dependent)]
        actual_names = [r.name for r in result.responses]
        assert actual_names == expected_names
