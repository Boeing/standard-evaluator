"""Property-based tests for ShiftScaleEvaluator.

Property 5: ShiftScaleEvaluator transform equivalence
Property 6: ShiftScaleEvaluator Jacobian consistency

**Validates: Requirements 6.4, 6.7**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.strategies import composite

from standard_evaluator import FloatVariable, OptProblem
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.evaluators.evaluator import PyEvaluator
from standard_evaluator.evaluators.shift_scale_evaluator import ShiftScaleEvaluator


# ---------------------------------------------------------------------------
# Helper: A simple polynomial evaluator with known analytic Jacobian
# ---------------------------------------------------------------------------


class PolynomialEvaluator(Evaluator):
    """A test evaluator that computes polynomial functions with analytic Jacobian.

    For n inputs x0, x1, ..., x(n-1) and m outputs y0, y1, ..., y(m-1):
        y_j = sum_i (coeffs[j][i] * x_i^2 + coeffs[j][i+n] * x_i)

    The Jacobian d(y_j)/d(x_i) = 2 * coeffs[j][i] * x_i + coeffs[j][i+n]
    """

    def __init__(self, opt_problem: OptProblem, coeffs: np.ndarray):
        """
        Args:
            opt_problem: Problem definition with variables and responses.
            coeffs: Shape (n_outputs, 2 * n_inputs) coefficient matrix.
                    First n_inputs columns are quadratic coefficients,
                    last n_inputs columns are linear coefficients.
        """
        self._coeffs = coeffs
        super().__init__(opt_problem=opt_problem)

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Evaluate polynomial outputs."""
        n_inputs = len(self.inputs)
        x = sites[self.inputs].to_numpy(dtype=np.float64)
        for j, resp_name in enumerate(self.outputs):
            quad_coeffs = self._coeffs[j, :n_inputs]
            lin_coeffs = self._coeffs[j, n_inputs:]
            # y_j = sum(a_i * x_i^2 + b_i * x_i)
            sites[resp_name] = np.sum(
                quad_coeffs * x**2 + lin_coeffs * x, axis=1
            )

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute the analytic Jacobian.

        Args:
            x: Input array of shape (n_sites, n_inputs).

        Returns:
            Jacobian of shape (n_sites, n_outputs, n_inputs).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n_sites = x.shape[0]
        n_inputs = x.shape[1]
        n_outputs = len(self.outputs)
        jac = np.zeros((n_sites, n_outputs, n_inputs))
        for j in range(n_outputs):
            quad_coeffs = self._coeffs[j, :n_inputs]
            lin_coeffs = self._coeffs[j, n_inputs:]
            # d(y_j)/d(x_i) = 2 * a_i * x_i + b_i
            jac[:, j, :] = 2.0 * quad_coeffs * x + lin_coeffs
        return jac


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@composite
def shift_scale_problem_strategy(draw, n_inputs=None, n_outputs=None):
    """Generate an OptProblem with non-zero shift and scale values.

    Returns a tuple of (opt_problem, coefficients) where coefficients
    define the polynomial evaluator behavior.
    """
    if n_inputs is None:
        n_inputs = draw(st.integers(min_value=1, max_value=4))
    if n_outputs is None:
        n_outputs = draw(st.integers(min_value=1, max_value=3))

    variables = []
    for i in range(n_inputs):
        lower = draw(st.floats(min_value=-10.0, max_value=-0.5,
                               allow_nan=False, allow_infinity=False))
        upper = draw(st.floats(min_value=0.5, max_value=10.0,
                               allow_nan=False, allow_infinity=False))
        # Shift and scale: design_to_optimizer = (x + shift) * scale
        # Non-zero scale, reasonable magnitude
        shift = draw(st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False))
        scale = draw(st.floats(min_value=0.1, max_value=5.0,
                               allow_nan=False, allow_infinity=False))
        # Randomly negate scale to test negative scales
        if draw(st.booleans()):
            scale = -scale
        variables.append(FloatVariable(
            name=f"x{i}", bounds=[lower, upper], shift=shift, scale=scale
        ))

    responses = []
    for j in range(n_outputs):
        shift = draw(st.floats(min_value=-5.0, max_value=5.0,
                               allow_nan=False, allow_infinity=False))
        scale = draw(st.floats(min_value=0.1, max_value=5.0,
                               allow_nan=False, allow_infinity=False))
        if draw(st.booleans()):
            scale = -scale
        responses.append(FloatVariable(
            name=f"y{j}", bounds=[-100.0, 100.0], shift=shift, scale=scale
        ))

    opt_problem = OptProblem(
        name="test_poly",
        variables=variables,
        responses=responses,
        objectives=[f"y{j}" for j in range(n_outputs)],
    )

    # Generate coefficients for the polynomial evaluator
    coeffs = draw(st.lists(
        st.lists(
            st.floats(min_value=-2.0, max_value=2.0,
                      allow_nan=False, allow_infinity=False),
            min_size=2 * n_inputs, max_size=2 * n_inputs,
        ),
        min_size=n_outputs, max_size=n_outputs,
    ))
    coeffs = np.array(coeffs)

    return opt_problem, coeffs


@composite
def input_sites_strategy(draw, opt_problem):
    """Generate a DataFrame of input sites within the variable bounds."""
    n_sites = draw(st.integers(min_value=1, max_value=5))
    data = {}
    for var in opt_problem.variables:
        lower, upper = var.bounds
        values = draw(st.lists(
            st.floats(min_value=float(lower), max_value=float(upper),
                      allow_nan=False, allow_infinity=False),
            min_size=n_sites, max_size=n_sites,
        ))
        data[var.name] = values
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Property 5: ShiftScaleEvaluator transform equivalence
# ---------------------------------------------------------------------------


class TestShiftScaleTransformEquivalence:
    """Property 5: ShiftScaleEvaluator transform equivalence.

    For any evaluator wrapped in ShiftScaleEvaluator, evaluating sites in
    optimizer space and transforming back to design space yields the same
    responses as evaluating in design space directly.

    **Validates: Requirements 6.4**
    """

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_transform_equivalence(self, data):
        """Evaluating via ShiftScaleEvaluator matches manual shift/scale application."""
        # Generate problem and coefficients
        opt_problem, coeffs = data.draw(shift_scale_problem_strategy())
        n_inputs = len(opt_problem.variables)
        n_outputs = len(opt_problem.responses)

        # Create the inner evaluator in design space
        # The inner evaluator needs a problem without shift/scale (identity transform)
        design_vars = []
        for var in opt_problem.variables:
            design_vars.append(FloatVariable(
                name=var.name, bounds=list(var.bounds), shift=0.0, scale=1.0
            ))
        design_responses = []
        for resp in opt_problem.responses:
            design_responses.append(FloatVariable(
                name=resp.name, bounds=list(resp.bounds), shift=0.0, scale=1.0
            ))
        design_problem = OptProblem(
            name="design",
            variables=design_vars,
            responses=design_responses,
            objectives=[r.name for r in design_responses],
        )

        inner_evaluator = PolynomialEvaluator(
            opt_problem=design_problem, coeffs=coeffs
        )

        # Create ShiftScaleEvaluator wrapping the inner evaluator
        shift_scale_eval = ShiftScaleEvaluator(
            evaluate=inner_evaluator, opt_problem=opt_problem
        )

        # Generate input sites in DESIGN space (within original bounds)
        design_sites = data.draw(input_sites_strategy(opt_problem))
        assume(len(design_sites) > 0)

        # Evaluate directly in design space
        design_sites_copy = design_sites.copy()
        inner_evaluator(design_sites_copy)

        # Transform input sites to optimizer space, evaluate via ShiftScaleEvaluator
        optimizer_sites = shift_scale_eval.design_to_optimizer_space(design_sites.copy())

        # Evaluate in optimizer space via ShiftScaleEvaluator
        shift_scale_eval(optimizer_sites)

        # Transform responses back from optimizer space to design space
        optimizer_responses = optimizer_sites[shift_scale_eval.outputs].copy()
        design_responses_from_optimizer = (
            shift_scale_eval.optimizer_to_design_space(optimizer_responses)
        )

        # Compare: design space direct vs. transformed-back from optimizer space
        for resp_name in inner_evaluator.outputs:
            direct_values = design_sites_copy[resp_name].to_numpy(dtype=np.float64)
            transformed_values = design_responses_from_optimizer[resp_name].to_numpy(
                dtype=np.float64
            )
            np.testing.assert_allclose(
                transformed_values, direct_values, rtol=1e-10, atol=1e-10,
                err_msg=f"Response '{resp_name}' mismatch between direct "
                        f"design-space evaluation and transformed optimizer-space evaluation"
            )


# ---------------------------------------------------------------------------
# Property 6: ShiftScaleEvaluator Jacobian consistency
# ---------------------------------------------------------------------------


class TestShiftScaleJacobianConsistency:
    """Property 6: ShiftScaleEvaluator Jacobian consistency.

    The shifted/scaled Jacobian is consistent with the chain rule:
    J_scaled = output_scale * J_design / input_scale

    **Validates: Requirements 6.7**
    """

    @given(data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_jacobian_chain_rule(self, data):
        """ShiftScaleEvaluator Jacobian matches chain-rule expectation."""
        # Generate problem with 1-3 inputs and 1-2 outputs for manageable Jacobians
        opt_problem, coeffs = data.draw(
            shift_scale_problem_strategy(n_inputs=None, n_outputs=None)
        )
        n_inputs = len(opt_problem.variables)
        n_outputs = len(opt_problem.responses)

        # Create the inner evaluator in design space (no shift/scale)
        design_vars = []
        for var in opt_problem.variables:
            design_vars.append(FloatVariable(
                name=var.name, bounds=list(var.bounds), shift=0.0, scale=1.0
            ))
        design_responses = []
        for resp in opt_problem.responses:
            design_responses.append(FloatVariable(
                name=resp.name, bounds=list(resp.bounds), shift=0.0, scale=1.0
            ))
        design_problem = OptProblem(
            name="design",
            variables=design_vars,
            responses=design_responses,
            objectives=[r.name for r in design_responses],
        )

        inner_evaluator = PolynomialEvaluator(
            opt_problem=design_problem, coeffs=coeffs
        )

        # Create ShiftScaleEvaluator
        shift_scale_eval = ShiftScaleEvaluator(
            evaluate=inner_evaluator, opt_problem=opt_problem
        )

        # Generate a single input point in design space (within bounds)
        point_data = {}
        for var in opt_problem.variables:
            lower, upper = var.bounds
            val = data.draw(st.floats(
                min_value=float(lower) + 0.01 * abs(float(upper) - float(lower)),
                max_value=float(upper) - 0.01 * abs(float(upper) - float(lower)),
                allow_nan=False, allow_infinity=False,
            ))
            point_data[var.name] = [val]

        design_point = pd.DataFrame(point_data)

        # Transform design point to optimizer space
        optimizer_point = shift_scale_eval.design_to_optimizer_space(design_point)
        x_optimizer = optimizer_point[shift_scale_eval.inputs].to_numpy(
            dtype=np.float64
        ).flatten()

        # Get shift/scale arrays
        input_shifts = np.array([var.shift for var in opt_problem.variables],
                               dtype=np.float64)
        input_scales = np.array([var.scale for var in opt_problem.variables],
                               dtype=np.float64)
        output_scales = np.array([resp.scale for resp in opt_problem.responses],
                                dtype=np.float64)

        # Compute the ShiftScaleEvaluator's Jacobian at the optimizer-space point
        scaled_jac = shift_scale_eval.jacobian(x_optimizer)

        # Compute the inner evaluator's Jacobian at the design-space point
        x_design = design_point[inner_evaluator.inputs].to_numpy(
            dtype=np.float64
        )
        design_jac = inner_evaluator.jacobian(x_design)

        # Expected: J_scaled = output_scale * J_design / input_scale
        # Axes: (n_sites, n_outputs, n_inputs)
        expected_jac = (
            design_jac
            * (1.0 / input_scales[np.newaxis, np.newaxis, :])
            * output_scales[np.newaxis, :, np.newaxis]
        )

        np.testing.assert_allclose(
            scaled_jac, expected_jac, rtol=1e-8, atol=1e-10,
            err_msg="ShiftScaleEvaluator Jacobian does not match chain rule "
                    "expectation: J_scaled = output_scale * J_design / input_scale"
        )

    @given(data=st.data())
    @settings(max_examples=50, deadline=None)
    def test_jacobian_vs_finite_difference(self, data):
        """ShiftScaleEvaluator Jacobian agrees with finite-difference approximation."""
        # Use small fixed problem for cleaner finite-difference
        opt_problem, coeffs = data.draw(
            shift_scale_problem_strategy(n_inputs=2, n_outputs=1)
        )
        n_inputs = len(opt_problem.variables)
        n_outputs = len(opt_problem.responses)

        # Create the inner evaluator in design space (no shift/scale)
        design_vars = []
        for var in opt_problem.variables:
            design_vars.append(FloatVariable(
                name=var.name, bounds=list(var.bounds), shift=0.0, scale=1.0
            ))
        design_responses = []
        for resp in opt_problem.responses:
            design_responses.append(FloatVariable(
                name=resp.name, bounds=list(resp.bounds), shift=0.0, scale=1.0
            ))
        design_problem = OptProblem(
            name="design",
            variables=design_vars,
            responses=design_responses,
            objectives=[r.name for r in design_responses],
        )

        inner_evaluator = PolynomialEvaluator(
            opt_problem=design_problem, coeffs=coeffs
        )

        # Create ShiftScaleEvaluator
        shift_scale_eval = ShiftScaleEvaluator(
            evaluate=inner_evaluator, opt_problem=opt_problem
        )

        # Generate a point well within bounds (avoid boundary effects)
        point_data = {}
        for var in opt_problem.variables:
            lower, upper = var.bounds
            margin = 0.1 * abs(float(upper) - float(lower))
            val = data.draw(st.floats(
                min_value=float(lower) + margin,
                max_value=float(upper) - margin,
                allow_nan=False, allow_infinity=False,
            ))
            point_data[var.name] = [val]

        design_point = pd.DataFrame(point_data)
        optimizer_point = shift_scale_eval.design_to_optimizer_space(design_point)
        x_opt = optimizer_point[shift_scale_eval.inputs].to_numpy(
            dtype=np.float64
        ).flatten()

        # Get analytic Jacobian from ShiftScaleEvaluator
        analytic_jac = shift_scale_eval.jacobian(x_opt)

        # Compute finite-difference Jacobian in optimizer space
        delta = 1e-6
        n_sites = 1
        fd_jac = np.zeros((n_sites, n_outputs, n_inputs))

        for i in range(n_inputs):
            x_plus = x_opt.copy()
            x_minus = x_opt.copy()
            x_plus[i] += delta
            x_minus[i] -= delta

            # Evaluate at x_plus
            df_plus = pd.DataFrame(
                [x_plus], columns=shift_scale_eval.inputs
            )
            shift_scale_eval(df_plus)
            f_plus = df_plus[shift_scale_eval.outputs].to_numpy(dtype=np.float64)

            # Evaluate at x_minus
            df_minus = pd.DataFrame(
                [x_minus], columns=shift_scale_eval.inputs
            )
            shift_scale_eval(df_minus)
            f_minus = df_minus[shift_scale_eval.outputs].to_numpy(dtype=np.float64)

            fd_jac[0, :, i] = (f_plus[0] - f_minus[0]) / (2.0 * delta)

        np.testing.assert_allclose(
            analytic_jac, fd_jac, rtol=1e-4, atol=1e-4,
            err_msg="ShiftScaleEvaluator Jacobian does not agree with "
                    "finite-difference approximation within tolerance 1e-4"
        )
