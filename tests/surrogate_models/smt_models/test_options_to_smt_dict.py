"""Unit tests for _options_to_smt_dict helper function.

Tests the Pydantic-based options-to-dict conversion that replaces
the old options_dict_to_dict function.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModelOptions,
    _options_to_smt_dict,
)


class TestOptionsToSmtDict:
    """Tests for _options_to_smt_dict conversion helper."""

    def test_excludes_parameters_field(self):
        """Parameters field should never appear in the output dict."""
        options = AbstractSmtModelOptions()
        result = _options_to_smt_dict(options)
        assert "parameters" not in result

    def test_excludes_use_xlimits_field(self):
        """use_xlimits field should never appear in the output dict."""
        options = AbstractSmtModelOptions(use_xlimits=True)
        result = _options_to_smt_dict(options)
        assert "use_xlimits" not in result

    def test_excludes_data_dir_when_none(self):
        """data_dir should not appear in output when None."""
        options = AbstractSmtModelOptions(data_dir=None)
        result = _options_to_smt_dict(options)
        assert "data_dir" not in result

    def test_includes_data_dir_as_string_when_non_none(self):
        """data_dir should appear as a string when non-None."""
        options = AbstractSmtModelOptions(data_dir=Path("/tmp/my_data"))
        result = _options_to_smt_dict(options)
        assert "data_dir" in result
        assert result["data_dir"] == str(Path("/tmp/my_data"))
        assert isinstance(result["data_dir"], str)

    def test_includes_all_non_excluded_fields(self):
        """All non-excluded fields should appear in output with their values."""
        options = AbstractSmtModelOptions(
            print_global=False,
            print_training=False,
            print_prediction=True,
            print_problem=True,
            print_solver=False,
        )
        result = _options_to_smt_dict(options)
        assert result["print_global"] is False
        assert result["print_training"] is False
        assert result["print_prediction"] is True
        assert result["print_problem"] is True
        assert result["print_solver"] is False

    def test_default_options_produce_correct_dict(self):
        """Default options should produce a dict with all print flags True."""
        options = AbstractSmtModelOptions()
        result = _options_to_smt_dict(options)
        assert result["print_global"] is True
        assert result["print_training"] is True
        assert result["print_prediction"] is True
        assert result["print_problem"] is True
        assert result["print_solver"] is True

    def test_xlimits_not_added_when_use_xlimits_false(self):
        """xlimits should not be in output when use_xlimits is False."""
        options = AbstractSmtModelOptions(use_xlimits=False)
        opt_problem = MagicMock()
        result = _options_to_smt_dict(options, opt_problem=opt_problem)
        assert "xlimits" not in result

    def test_xlimits_not_added_when_no_opt_problem(self):
        """xlimits should not be in output when opt_problem is None."""
        options = AbstractSmtModelOptions(use_xlimits=True)
        result = _options_to_smt_dict(options, opt_problem=None)
        assert "xlimits" not in result

    def test_xlimits_computed_from_opt_problem_all_variables(self):
        """xlimits should contain bounds from all variables when nonconstant_variables is None."""
        options = AbstractSmtModelOptions(use_xlimits=True)

        # Create mock variables with bounds
        var1 = MagicMock()
        var1.name = "x0"
        var1.bounds = (-1.0, 1.0)

        var2 = MagicMock()
        var2.name = "x1"
        var2.bounds = (0.0, 5.0)

        opt_problem = MagicMock()
        opt_problem.variables = [var1, var2]

        result = _options_to_smt_dict(
            options, opt_problem=opt_problem, nonconstant_variables=None
        )
        assert "xlimits" in result
        expected = np.array([[-1.0, 1.0], [0.0, 5.0]])
        np.testing.assert_array_equal(result["xlimits"], expected)

    def test_xlimits_filters_by_nonconstant_variables(self):
        """xlimits should only include variables in the nonconstant_variables list."""
        options = AbstractSmtModelOptions(use_xlimits=True)

        # Create mock variables with bounds
        var1 = MagicMock()
        var1.name = "x0"
        var1.bounds = (-1.0, 1.0)

        var2 = MagicMock()
        var2.name = "x1"
        var2.bounds = (0.0, 5.0)

        var3 = MagicMock()
        var3.name = "x_const"
        var3.bounds = (3.0, 3.0)  # constant variable

        opt_problem = MagicMock()
        opt_problem.variables = [var1, var2, var3]

        # Only x0 and x1 are non-constant
        result = _options_to_smt_dict(
            options,
            opt_problem=opt_problem,
            nonconstant_variables=["x0", "x1"],
        )
        assert "xlimits" in result
        expected = np.array([[-1.0, 1.0], [0.0, 5.0]])
        np.testing.assert_array_equal(result["xlimits"], expected)

    def test_xlimits_is_numpy_array(self):
        """xlimits should be a numpy array."""
        options = AbstractSmtModelOptions(use_xlimits=True)

        var1 = MagicMock()
        var1.name = "x0"
        var1.bounds = (-2.0, 2.0)

        opt_problem = MagicMock()
        opt_problem.variables = [var1]

        result = _options_to_smt_dict(
            options, opt_problem=opt_problem, nonconstant_variables=["x0"]
        )
        assert isinstance(result["xlimits"], np.ndarray)

    def test_data_dir_path_to_string_conversion(self):
        """data_dir Path objects should be converted to strings."""
        options = AbstractSmtModelOptions(data_dir=Path("some/relative/path"))
        result = _options_to_smt_dict(options)
        assert result["data_dir"] == str(Path("some/relative/path"))

    def test_empty_nonconstant_variables_produces_empty_xlimits(self):
        """When nonconstant_variables is empty, xlimits should be empty."""
        options = AbstractSmtModelOptions(use_xlimits=True)

        var1 = MagicMock()
        var1.name = "x_const"
        var1.bounds = (1.0, 1.0)

        opt_problem = MagicMock()
        opt_problem.variables = [var1]

        result = _options_to_smt_dict(
            options,
            opt_problem=opt_problem,
            nonconstant_variables=[],
        )
        assert "xlimits" in result
        # Empty array should still be a numpy array
        assert isinstance(result["xlimits"], np.ndarray)
        assert result["xlimits"].shape[0] == 0


# --- Property-Based Tests ---

from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite


# --- Strategies for generating valid AbstractSmtModelOptions instances ---

# Strategy for Path-like strings (used for data_dir)
path_strings = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=50,
    ).map(Path),
)

# Strategy for boolean fields
bool_strategy = st.booleans()


@composite
def abstract_smt_model_options(draw):
    """Generate random AbstractSmtModelOptions instances with various field combinations."""
    return AbstractSmtModelOptions(
        print_global=draw(bool_strategy),
        print_training=draw(bool_strategy),
        print_prediction=draw(bool_strategy),
        print_problem=draw(bool_strategy),
        print_solver=draw(bool_strategy),
        use_xlimits=draw(bool_strategy),
        data_dir=draw(path_strings),
    )


@composite
def mock_opt_problem_with_variables(draw):
    """Generate a mock OptProblem with random variables and bounds."""
    n_vars = draw(st.integers(min_value=1, max_value=5))
    variables = []
    for i in range(n_vars):
        var = MagicMock()
        var.name = f"x{i}"
        lb = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        ub = draw(st.floats(min_value=lb + 0.01, max_value=200.0, allow_nan=False, allow_infinity=False))
        var.bounds = (lb, ub)
        variables.append(var)
    opt_problem = MagicMock()
    opt_problem.variables = variables
    return opt_problem


class TestOptionsToSmtDictProperty:
    """Property 8: Options-to-SMT-Dict Conversion.

    For any valid AbstractSmtModelOptions subclass instance with arbitrary field values:
    1. The dict produced SHALL NOT contain keys "parameters" or "use_xlimits"
    2. IF data_dir is None, the dict SHALL NOT contain key "data_dir"
    3. IF data_dir is a non-None Path, the dict SHALL contain "data_dir" mapped to str(path)
    4. IF use_xlimits is True and an OptProblem is provided, the dict SHALL contain "xlimits"

    Feature: surrogate-model-pydantic-options, Property 8: Options-to-SMT-Dict Conversion

    **Validates: Requirements 11.1, 11.2, 11.3, 9.7**
    """

    @settings(max_examples=100)
    @given(options=abstract_smt_model_options())
    def test_output_never_contains_parameters_or_use_xlimits(self, options):
        """Output dict never contains 'parameters' or 'use_xlimits'.

        Feature: surrogate-model-pydantic-options, Property 8: Options-to-SMT-Dict Conversion

        **Validates: Requirements 11.1, 11.2, 11.3, 9.7**
        """
        result = _options_to_smt_dict(options)
        assert "parameters" not in result
        assert "use_xlimits" not in result

    @settings(max_examples=100)
    @given(options=abstract_smt_model_options())
    def test_data_dir_handling(self, options):
        """data_dir absent when None, present as str when Path.

        Feature: surrogate-model-pydantic-options, Property 8: Options-to-SMT-Dict Conversion

        **Validates: Requirements 11.1, 11.2, 11.3, 9.7**
        """
        result = _options_to_smt_dict(options)
        if options.data_dir is None:
            assert "data_dir" not in result
        else:
            assert "data_dir" in result
            assert result["data_dir"] == str(options.data_dir)
            assert isinstance(result["data_dir"], str)

    @settings(max_examples=100)
    @given(opt_problem=mock_opt_problem_with_variables())
    def test_xlimits_computed_when_use_xlimits_true(self, opt_problem):
        """xlimits computed correctly when use_xlimits=True with OptProblem.

        Feature: surrogate-model-pydantic-options, Property 8: Options-to-SMT-Dict Conversion

        **Validates: Requirements 11.1, 11.2, 11.3, 9.7**
        """
        options = AbstractSmtModelOptions(use_xlimits=True)
        nonconstant_vars = [v.name for v in opt_problem.variables]

        result = _options_to_smt_dict(
            options, opt_problem=opt_problem, nonconstant_variables=nonconstant_vars
        )

        assert "xlimits" in result
        assert isinstance(result["xlimits"], np.ndarray)
        assert result["xlimits"].shape == (len(opt_problem.variables), 2)

        # Verify each row matches the variable bounds
        for i, var in enumerate(opt_problem.variables):
            expected_bounds = list(var.bounds)
            np.testing.assert_array_almost_equal(
                result["xlimits"][i], expected_bounds
            )

    @settings(max_examples=100)
    @given(opt_problem=mock_opt_problem_with_variables())
    def test_xlimits_absent_when_use_xlimits_false(self, opt_problem):
        """xlimits NOT present when use_xlimits=False, even with OptProblem.

        Feature: surrogate-model-pydantic-options, Property 8: Options-to-SMT-Dict Conversion

        **Validates: Requirements 11.1, 11.2, 11.3, 9.7**
        """
        options = AbstractSmtModelOptions(use_xlimits=False)
        result = _options_to_smt_dict(options, opt_problem=opt_problem)
        assert "xlimits" not in result

    @settings(max_examples=100)
    @given(options=abstract_smt_model_options())
    def test_xlimits_absent_when_no_opt_problem(self, options):
        """xlimits NOT present when opt_problem is None, regardless of use_xlimits.

        Feature: surrogate-model-pydantic-options, Property 8: Options-to-SMT-Dict Conversion

        **Validates: Requirements 11.1, 11.2, 11.3, 9.7**
        """
        result = _options_to_smt_dict(options, opt_problem=None)
        assert "xlimits" not in result
