"""Property-based tests for OptProblem build_maps and _setup_partials.

Validates: Requirements 17.1, 17.2, 17.4, 17.7, 17.8
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.strategies import composite

from standard_evaluator import (
    FloatVariable,
    IntVariable,
    ArrayVariable,
    OptProblem,
)


# ---------------------------------------------------------------------------
# Hypothesis Strategies for generating valid OptProblem configurations
# ---------------------------------------------------------------------------


@composite
def float_variable_strategy(draw, fixed=False):
    """Generate a FloatVariable with valid bounds."""
    name = draw(st.from_regex(r"[a-z][a-z0-9]{0,5}", fullmatch=True))
    if fixed:
        # Fixed variable: lower == upper
        val = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        return FloatVariable(name=name, bounds=[val, val])
    else:
        lower = draw(st.floats(min_value=-100.0, max_value=99.0, allow_nan=False, allow_infinity=False))
        upper = draw(st.floats(min_value=lower + 0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
        return FloatVariable(name=name, bounds=[lower, upper])


@composite
def int_variable_strategy(draw, fixed=False):
    """Generate an IntVariable with valid bounds."""
    name = draw(st.from_regex(r"[a-z][a-z0-9]{0,5}", fullmatch=True))
    if fixed:
        val = draw(st.integers(min_value=-50, max_value=50))
        return IntVariable(name=name, bounds=[val, val])
    else:
        lower = draw(st.integers(min_value=-50, max_value=49))
        upper = draw(st.integers(min_value=lower + 1, max_value=50))
        return IntVariable(name=name, bounds=[lower, upper])


@composite
def array_variable_strategy(draw, fixed=False):
    """Generate an ArrayVariable with valid bounds and a small shape."""
    name = draw(st.from_regex(r"[a-z][a-z0-9]{0,5}", fullmatch=True))
    ndims = draw(st.integers(min_value=1, max_value=2))
    shape = tuple(draw(st.integers(min_value=1, max_value=3)) for _ in range(ndims))
    if fixed:
        val = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        lower = np.full(shape, val)
        upper = np.full(shape, val)
    else:
        lower_val = draw(st.floats(min_value=-10.0, max_value=9.0, allow_nan=False, allow_infinity=False))
        upper_val = draw(st.floats(min_value=lower_val + 0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
        lower = np.full(shape, lower_val)
        upper = np.full(shape, upper_val)
    return ArrayVariable(name=name, bounds=[lower, upper], shape=shape)


@composite
def variable_strategy(draw, fixed=False):
    """Generate any valid variable type."""
    var_type = draw(st.sampled_from(["float", "int", "array"]))
    if var_type == "float":
        return draw(float_variable_strategy(fixed=fixed))
    elif var_type == "int":
        return draw(int_variable_strategy(fixed=fixed))
    else:
        return draw(array_variable_strategy(fixed=fixed))


def _count_flat_elements(var):
    """Count the number of flattened elements for a variable."""
    if isinstance(var, ArrayVariable):
        return int(np.prod(var.shape))
    return 1


@composite
def opt_problem_strategy(draw, min_vars=1, max_vars=4, min_responses=1, max_responses=3,
                         force_fixed=False, force_objectives=False, force_constraints=False):
    """Generate a valid OptProblem with unique variable/response names."""
    n_vars = draw(st.integers(min_value=min_vars, max_value=max_vars))
    n_responses = draw(st.integers(min_value=min_responses, max_value=max_responses))

    # Generate variables with unique names
    used_names = set()
    variables = []
    for i in range(n_vars):
        # Use deterministic name prefixes to guarantee uniqueness
        name_prefix = f"v{i}"
        if force_fixed and i == 0:
            # Force at least one fixed variable
            var = draw(float_variable_strategy(fixed=True))
        else:
            var = draw(variable_strategy(fixed=False))
        # Override name to guarantee uniqueness
        var_name = f"{name_prefix}{var.name}"
        if isinstance(var, ArrayVariable):
            variables.append(ArrayVariable(name=var_name, bounds=var.bounds, shape=var.shape))
        elif isinstance(var, IntVariable):
            variables.append(IntVariable(name=var_name, bounds=var.bounds))
        else:
            variables.append(FloatVariable(name=var_name, bounds=var.bounds))
        used_names.add(var_name)

    # Generate responses with unique names (no ArrayVariable responses to keep objectives/constraints simple)
    responses = []
    for i in range(n_responses):
        name = f"r{i}"
        while name in used_names:
            name = f"r{i}x"
        responses.append(FloatVariable(name=name, bounds=[-100.0, 100.0]))
        used_names.add(name)

    # Build objectives and constraints from response names
    response_names = [r.name for r in responses]
    objectives = []
    constraints = []

    if force_objectives and len(response_names) > 0:
        # Pick at least one objective
        n_objs = draw(st.integers(min_value=1, max_value=len(response_names)))
        objectives = draw(st.lists(
            st.sampled_from(response_names),
            min_size=n_objs, max_size=n_objs, unique=True
        ))
    elif len(response_names) > 0:
        objectives = draw(st.lists(
            st.sampled_from(response_names),
            min_size=0, max_size=len(response_names), unique=True
        ))

    if force_constraints and len(response_names) > 0:
        remaining = [r for r in response_names if r not in objectives]
        if remaining:
            n_cons = draw(st.integers(min_value=1, max_value=len(remaining)))
            constraints = draw(st.lists(
                st.sampled_from(remaining),
                min_size=n_cons, max_size=n_cons, unique=True
            ))
    elif len(response_names) > 0:
        # Constraints must be responses (can overlap with objectives)
        constraints = draw(st.lists(
            st.sampled_from(response_names),
            min_size=0, max_size=len(response_names), unique=True
        ))

    prob = OptProblem(
        name=draw(st.from_regex(r"[a-z][a-z0-9]{0,7}", fullmatch=True)),
        variables=variables,
        responses=responses,
        objectives=objectives,
        constraints=constraints,
    )
    return prob


@composite
def opt_problem_with_fixed_vars_strategy(draw):
    """Generate an OptProblem that has at least one fixed variable (lower==upper)."""
    n_free = draw(st.integers(min_value=1, max_value=3))
    n_fixed = draw(st.integers(min_value=1, max_value=2))
    n_responses = draw(st.integers(min_value=1, max_value=3))

    variables = []
    used_names = set()

    # Create fixed variables
    for i in range(n_fixed):
        name = f"fixed{i}"
        val = draw(st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False))
        variables.append(FloatVariable(name=name, bounds=[val, val]))
        used_names.add(name)

    # Create free variables
    for i in range(n_free):
        name = f"free{i}"
        lower = draw(st.floats(min_value=-50.0, max_value=49.0, allow_nan=False, allow_infinity=False))
        upper = draw(st.floats(min_value=lower + 0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
        variables.append(FloatVariable(name=name, bounds=[lower, upper]))
        used_names.add(name)

    # Create responses
    responses = []
    for i in range(n_responses):
        name = f"resp{i}"
        responses.append(FloatVariable(name=name, bounds=[-100.0, 100.0]))
        used_names.add(name)

    response_names = [r.name for r in responses]
    objectives = draw(st.lists(
        st.sampled_from(response_names),
        min_size=0, max_size=len(response_names), unique=True
    ))
    constraints = draw(st.lists(
        st.sampled_from(response_names),
        min_size=0, max_size=len(response_names), unique=True
    ))

    return OptProblem(
        name="fixedtest",
        variables=variables,
        responses=responses,
        objectives=objectives,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Property 8: build_maps structural correctness
# ---------------------------------------------------------------------------


class TestBuildMapsStructuralCorrectness:
    """Property 8: build_maps structural correctness.

    For any valid OptProblem, var_map has columns (name, multi, flat, fixed, jac_row, grad_row)
    and res_map has columns (name, multi, flat, objective, constraint, grad_col, jac_col).

    **Validates: Requirements 17.1**
    """

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_var_map_has_correct_columns(self, prob):
        """var_map contains exactly the required columns."""
        var_map = prob.var_map
        expected_columns = {"name", "multi", "flat", "fixed", "jac_row", "grad_row"}
        assert set(var_map.columns) == expected_columns

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_res_map_has_correct_columns(self, prob):
        """res_map contains exactly the required columns."""
        res_map = prob.res_map
        expected_columns = {"name", "multi", "flat", "objective", "constraint", "grad_col", "jac_col"}
        assert set(res_map.columns) == expected_columns

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_var_map_row_count_matches_flattened_elements(self, prob):
        """var_map has exactly F rows where F is the total flattened variable count."""
        expected_flat_count = sum(_count_flat_elements(v) for v in prob.variables)
        assert len(prob.var_map) == expected_flat_count

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_res_map_row_count_matches_flattened_elements(self, prob):
        """res_map has exactly R rows where R is the total flattened response count."""
        expected_flat_count = sum(_count_flat_elements(r) for r in prob.responses)
        assert len(prob.res_map) == expected_flat_count

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_var_map_flat_indices_contiguous(self, prob):
        """var_map flat column contains contiguous 0-based indices."""
        var_map = prob.var_map
        n = len(var_map)
        expected = list(range(n))
        assert list(var_map["flat"]) == expected

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_res_map_flat_indices_contiguous(self, prob):
        """res_map flat column contains contiguous 0-based indices."""
        res_map = prob.res_map
        n = len(res_map)
        expected = list(range(n))
        assert list(res_map["flat"]) == expected


# ---------------------------------------------------------------------------
# Property 9: Fixed variable exclusion in var_map
# ---------------------------------------------------------------------------


class TestFixedVariableExclusion:
    """Property 9: Fixed variable exclusion in var_map.

    Variables where lower==upper are marked fixed=True; their jac_row and grad_row are None.

    **Validates: Requirements 17.7**
    """

    @given(prob=opt_problem_with_fixed_vars_strategy())
    @settings(max_examples=100, deadline=None)
    def test_fixed_variables_marked_correctly(self, prob):
        """Variables with lower==upper have fixed=True in var_map."""
        var_map = prob.var_map
        for var in prob.variables:
            is_fixed = np.array_equal(var.bounds[0], var.bounds[1])
            var_rows = var_map[var_map["name"] == var.name]
            for _, row in var_rows.iterrows():
                assert row["fixed"] == is_fixed, (
                    f"Variable '{var.name}' should be fixed={is_fixed}"
                )

    @given(prob=opt_problem_with_fixed_vars_strategy())
    @settings(max_examples=100, deadline=None)
    def test_fixed_variables_have_none_jac_grad_rows(self, prob):
        """Fixed variables have jac_row=None and grad_row=None."""
        var_map = prob.var_map
        fixed_rows = var_map[var_map["fixed"] == True]
        assert len(fixed_rows) > 0, "Should have at least one fixed variable"
        for _, row in fixed_rows.iterrows():
            assert row["jac_row"] is None, (
                f"Fixed variable '{row['name']}' should have jac_row=None"
            )
            assert row["grad_row"] is None, (
                f"Fixed variable '{row['name']}' should have grad_row=None"
            )

    @given(prob=opt_problem_with_fixed_vars_strategy())
    @settings(max_examples=100, deadline=None)
    def test_fixed_variables_excluded_from_free_positions(self, prob):
        """Fixed variable elements do not appear in free_flat_var_positions."""
        var_map = prob.var_map
        fixed_flat_indices = set(var_map[var_map["fixed"] == True]["flat"].tolist())
        free_positions = set(prob.free_flat_var_positions.tolist())
        # No overlap between fixed indices and free positions
        assert fixed_flat_indices.isdisjoint(free_positions), (
            f"Fixed indices {fixed_flat_indices} should not appear in free positions {free_positions}"
        )

    @given(prob=opt_problem_with_fixed_vars_strategy())
    @settings(max_examples=100, deadline=None)
    def test_free_variables_have_assigned_jac_grad_rows(self, prob):
        """Free variables have integer jac_row and grad_row values."""
        var_map = prob.var_map
        free_rows = var_map[var_map["fixed"] == False]
        for _, row in free_rows.iterrows():
            assert isinstance(row["jac_row"], (int, np.integer)), (
                f"Free variable '{row['name']}' should have int jac_row, got {type(row['jac_row'])}"
            )
            assert isinstance(row["grad_row"], (int, np.integer)), (
                f"Free variable '{row['name']}' should have int grad_row, got {type(row['grad_row'])}"
            )


# ---------------------------------------------------------------------------
# Property 10: OptProblem maps initialized after construction
# ---------------------------------------------------------------------------


class TestMapsInitializedAfterConstruction:
    """Property 10: OptProblem maps initialized after construction.

    After constructing any valid OptProblem, var_map and res_map are not None.

    **Validates: Requirements 17.2, 17.4**
    """

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_var_map_is_not_none(self, prob):
        """var_map is a non-None DataFrame after construction."""
        assert prob.var_map is not None
        assert isinstance(prob.var_map, pd.DataFrame)

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_res_map_is_not_none(self, prob):
        """res_map is a non-None DataFrame after construction."""
        assert prob.res_map is not None
        assert isinstance(prob.res_map, pd.DataFrame)

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_flat_partials_res_indices_populated(self, prob):
        """flat_partials_res_indices is a numpy array after construction."""
        assert prob.flat_partials_res_indices is not None
        assert isinstance(prob.flat_partials_res_indices, np.ndarray)
        assert prob.flat_partials_res_indices.ndim == 1

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_num_partials_responses_is_int(self, prob):
        """num_partials_responses is a non-negative int."""
        assert prob.num_partials_responses is not None
        assert isinstance(prob.num_partials_responses, int)
        assert prob.num_partials_responses >= 0

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_free_flat_var_mask_correct_length(self, prob):
        """free_flat_var_mask is a boolean array with length matching var_map rows."""
        mask = prob.free_flat_var_mask
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(prob.var_map)

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_free_flat_var_positions_correct_type(self, prob):
        """free_flat_var_positions is an integer numpy array."""
        positions = prob.free_flat_var_positions
        assert positions is not None
        assert isinstance(positions, np.ndarray)
        assert positions.ndim == 1

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_num_flat_vars_consistent_with_positions(self, prob):
        """num_flat_vars equals the length of free_flat_var_positions."""
        assert prob.num_flat_vars == len(prob.free_flat_var_positions)

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_full_to_free_var_index_correct_length(self, prob):
        """full_to_free_var_index is an int array with length matching var_map rows."""
        ftf = prob.full_to_free_var_index
        assert ftf is not None
        assert isinstance(ftf, np.ndarray)
        assert len(ftf) == len(prob.var_map)

    @given(prob=opt_problem_strategy(force_objectives=True))
    @settings(max_examples=100, deadline=None)
    def test_num_partials_responses_matches_objectives_and_constraints(self, prob):
        """num_partials_responses equals count of responses that are objectives or constraints."""
        res_map = prob.res_map
        expected = int((res_map["objective"] | res_map["constraint"]).sum())
        assert prob.num_partials_responses == expected


# ---------------------------------------------------------------------------
# Property 11: OptProblem backward compatibility
# ---------------------------------------------------------------------------


class TestOptProblemBackwardCompatibility:
    """Property 11: OptProblem backward compatibility.

    Existing properties (variable_names, response_names, calculate_default, etc.)
    still work after adding maps.

    **Validates: Requirements 17.8**
    """

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_variable_names_still_works(self, prob):
        """variable_names() returns correct list of variable names."""
        names = prob.variable_names()
        assert isinstance(names, list)
        assert len(names) == len(prob.variables)
        for var, name in zip(prob.variables, names):
            assert name == var.name

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_response_names_still_works(self, prob):
        """response_names() returns correct list of response names."""
        names = prob.response_names()
        assert isinstance(names, list)
        assert len(names) == len(prob.responses)
        for resp, name in zip(prob.responses, names):
            assert name == resp.name

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_calculate_default_still_works(self, prob):
        """calculate_default() runs without error after map initialization."""
        # Should not raise
        prob.calculate_default()
        # After calculate_default, variables with finite bounds should have defaults
        for var in prob.variables:
            if isinstance(var, ArrayVariable):
                # ArrayVariable bounds are arrays; check all elements are finite
                if np.all(np.isfinite(var.bounds[0])) and np.all(np.isfinite(var.bounds[1])):
                    assert var.default is not None
            elif isinstance(var, FloatVariable):
                if np.isfinite(var.bounds[0]) and np.isfinite(var.bounds[1]):
                    assert var.default is not None
            elif isinstance(var, IntVariable):
                if np.isfinite(var.bounds[0]) and np.isfinite(var.bounds[1]):
                    assert var.default is not None

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_set_defaults_still_works(self, prob):
        """set_defaults() can update variable defaults."""
        # Pick first scalar FloatVariable (not ArrayVariable or IntVariable) if available
        for var in prob.variables:
            if isinstance(var, (ArrayVariable, IntVariable)):
                continue
            if isinstance(var, FloatVariable):
                new_val = (var.bounds[0] + var.bounds[1]) / 2.0
                prob.set_defaults({var.name: new_val})
                assert prob.variables[prob.variable_names().index(var.name)].default == new_val
                return
        # If no scalar FloatVariable, try IntVariable (not ArrayVariable)
        for var in prob.variables:
            if isinstance(var, ArrayVariable):
                continue
            if isinstance(var, IntVariable):
                new_val = (var.bounds[0] + var.bounds[1]) // 2
                prob.set_defaults({var.name: new_val})
                assert prob.variables[prob.variable_names().index(var.name)].default == new_val
                return

    @given(prob=opt_problem_strategy())
    @settings(max_examples=100, deadline=None)
    def test_unroll_names_still_works(self, prob):
        """unroll_names() produces correct flattened name list."""
        unrolled = prob.unroll_names(prob.variables)
        assert isinstance(unrolled, list)
        # Total count should equal sum of flat elements
        expected_count = sum(_count_flat_elements(v) for v in prob.variables)
        assert len(unrolled) == expected_count

    @given(data=st.data())
    @settings(max_examples=50, deadline=None)
    def test_check_problem_validator_still_raises_for_invalid_objectives(self, data):
        """check_problem validator still raises ValueError for invalid objectives."""
        variables = [FloatVariable(name="x0", bounds=[0.0, 1.0])]
        responses = [FloatVariable(name="y0", bounds=[0.0, 1.0])]
        with pytest.raises(ValueError, match="not defined as a variable or response"):
            OptProblem(
                name="invalid",
                variables=variables,
                responses=responses,
                objectives=["nonexistent"],
            )

    @given(data=st.data())
    @settings(max_examples=50, deadline=None)
    def test_check_problem_validator_still_raises_for_variable_as_constraint(self, data):
        """check_problem validator raises ValueError when a variable is used as constraint."""
        variables = [FloatVariable(name="x0", bounds=[0.0, 1.0])]
        responses = [FloatVariable(name="y0", bounds=[0.0, 1.0])]
        with pytest.raises(ValueError, match="defined as a constraint"):
            OptProblem(
                name="invalid",
                variables=variables,
                responses=responses,
                constraints=["x0"],
            )
