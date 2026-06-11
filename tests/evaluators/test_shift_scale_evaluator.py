import pytest
import numpy as np
import pandas as pd
import numdifftools as nd
from pandas.testing import assert_frame_equal
from standard_evaluator.evaluators import (
    ShiftScaleEvaluator,
    Evaluator,
    NumpyEvaluator,
)
from standard_evaluator.evaluators.test import HS100
import standard_evaluator as se
from standard_evaluator import (
    OptProblem,
    EvaluatorInfo,
    IntVariable,
    FloatVariable,
    CategoricalVariable,
)

@pytest.fixture
def hs100() -> Evaluator:
    # Define variables and responses
    return HS100()

@pytest.fixture
def opt_prob(hs100: Evaluator) -> OptProblem:
    # Define variables and responses
    full_problem = hs100.opt_problem
    # Set shift and scale for variables
    for var in full_problem.variables:
        if var.name == "x1":
            var.shift = 5.0    
    return full_problem

@pytest.fixture
def shifted_initial_guess_vals(opt_prob: OptProblem) -> pd.DataFrame:
    # Extract variable and response names from OptProblem object
    variable_names = [var.name for var in opt_prob.variables]
    response_names = [resp.name for resp in opt_prob.responses]
    return pd.DataFrame(
        data=[[0.6, 0.2, 0.0, 0.4, 0.0, 0.1, 0.1, 0.714, 0.013, 2.65, 1.71, 0.04]],
        columns=variable_names + response_names,
    )

@pytest.fixture
def simple_evaluator_with_jacobian() -> NumpyEvaluator:

    class SimpleEvaluator(NumpyEvaluator):

        def __init__(self, **kwargs):
            opt_problem = self._def_opt_problem()
            # Pass opt_problem to the base class constructor
            super().__init__(opt_problem=opt_problem, **kwargs)

        def eval_np(self, input_data: np.ndarray, names: list = None) -> np.ndarray:
            f_evals = input_data[:, 0] ** 2 + input_data[:, 1] ** 2
            g_evals = input_data[:, 0] - input_data[:, 1] ** 3

            evals = np.zeros((input_data.shape[0], 2))
            evals[:, 0] = f_evals
            evals[:, 1] = g_evals

            return evals
        
        def _def_opt_problem(self) -> OptProblem:

            # Create opt problem
            new_prob = se.utilities.create_opt_problem(num_independent=2, num_dependent=2)

            var_names = ["x", "y"]
            var_bounds = ([-1.0, 1.0], [-1.0, 1.0])

            for var, local_bound, name in zip(
                new_prob.variables, var_bounds, var_names
            ):
                var.bounds = local_bound
                var.name = name

            resp_names = ["f", "g"]

            for resp, name in zip(
                new_prob.responses, resp_names
            ):
                resp.name = name

            # Define objectives and constraints
            new_prob.objectives = []
            new_prob.constraints = []

            return new_prob

        def jacobian(self, x: np.ndarray) -> np.ndarray:

            jac = np.zeros((x.shape[0], 2, 2))

            jac[:, 0, 0] = 2.0 * x[:, 0]
            jac[:, 0, 1] = 2.0 * x[:, 1]
            jac[:, 1, 0] = 1.0
            jac[:, 1, 1] = -3.0 * x[:, 1] ** 2

            return jac

    return SimpleEvaluator()


class TestShiftScaleEvaluator:

    def test_value_checks(self, opt_prob: dict) -> None:
        """Check that that error is thrown correctly"""
        # Check that a problem dictionary is passed in
        with pytest.raises(TypeError):
            ShiftScaleEvaluator(opt_problem="str")
        # Check that an evaluator is passed in
        with pytest.raises(TypeError):
            ShiftScaleEvaluator( evaluate="str", opt_problem=opt_prob)

    def test_basics(
        self, hs100: Evaluator, opt_prob: OptProblem, shifted_initial_guess_vals: pd.DataFrame
    ) -> None:            
        """
        Test the main behavior of the ShiftScaleEvaluator including initial guess handling,
        evaluation, and transformation between optimization and design spaces.

        Parameters
        ----------
        hs100 : Evaluator
            An example evaluator instance used for testing.
        opt_prob : OptProblem
            The optimization problem definition in simulation space.
        shifted_initial_guess_vals : pd.DataFrame
            The expected initial guess values in the optimization space after shifting and scaling.

        Steps performed:
        - Create a ShiftScaleEvaluator wrapping the example evaluator with the given opt_problem.
        - Obtain the shifted and scaled initial guess from the ShiftScaleEvaluator.
        - Evaluate the shifted initial guess using the ShiftScaleEvaluator.
        - Assert that the evaluated shifted initial guess matches the expected values.
        - Convert the shifted initial guess back to the design space (unshift and unscale).
        - Obtain and evaluate the initial guess from the original evaluator.
        - Assert that the unshifted initial guess matches the original evaluator's initial guess.
        """
        # Create the ShiftScaleEvaluator
        shifted_hs100 = ShiftScaleEvaluator(evaluate=hs100, opt_problem=opt_prob, )
        # Get the shifted and scaled initial guess
        shifted_initial = shifted_hs100.initial_guess()
        # Evaluate the shifted and scaled initial guess
        shifted_hs100(shifted_initial)
        # Compare to the expected values
        assert_frame_equal(shifted_initial, shifted_initial_guess_vals)
        # Unshift and unscale the shifted and scaled initial guess
        unshifted_initial = shifted_hs100.optimizer_to_design_space(shifted_initial)
        # Get the initial guess from the original function, and evaluate it
        hs100_initial = hs100.initial_guess()
        hs100(hs100_initial)
        # Compare the evaluated initial guess from the original function with
        # the unshifted and unscaled initial guess
        assert_frame_equal(unshifted_initial, hs100_initial)

    def test_interface_conversion_to_opt_problem(self, hs100: Evaluator):
        """
        Test that providing an EvaluatorInfo interface converts correctly to an OptProblem
        and that ShiftScaleEvaluator initializes without error.
        """
        # Create an EvaluatorInfo from hs100 evaluator
        interface = hs100.interface if hasattr(hs100, "interface") else None

        # Initialize ShiftScaleEvaluator with interface only
        evaluator = ShiftScaleEvaluator(evaluate=hs100, interface=interface)

        # Check that the internal opt_problem is set and has variables and responses
        assert hasattr(evaluator, "_shift_scale")
        opt_prob = evaluator.opt_problem
        assert len(opt_prob.variables) == len(interface.inputs)
        assert len(opt_prob.responses) == len(interface.outputs)        

    def test_jacobian(self, simple_evaluator_with_jacobian: NumpyEvaluator, opt_prob: OptProblem):
        """Test shift scale on evaluator with a jacobian method
        Parameters
        ----------
        simple_evaluator_with_jacobian : NumpyEvaluator
            An evaluator instance that implements both eval_np and jacobian methods.
        opt_prob : OptProblem
            The optimization problem definition to be used with the evaluator.

        Test details:
        - Instantiate the evaluator and obtain its OptProblem.
        - Set shift and scale values for variables and responses in the OptProblem.
        - Create a ShiftScaleEvaluator wrapping the evaluator and the shifted/scaled OptProblem.
        - Define test points in the optimization space.
        - Evaluate the function at the test points and compare to expected shifted/scaled values.
        - Verify that the ShiftScaleEvaluator has a jacobian method.
        - Compute the expected Jacobian matrix considering the shift and scale factors.
        - Compare the computed Jacobian from the ShiftScaleEvaluator to the expected Jacobian.
        - Use finite difference approximation to numerically estimate the Jacobian.
        - Assert that the finite difference Jacobian matches both the expected and computed Jacobians
        within specified tolerances.
        """        

        # Initialize the evaluator
        evaluator = simple_evaluator_with_jacobian

        # Adjust problem to have shift scale
        opt_prob = evaluator.opt_problem

        # Set shift and scale for variables
        for var in opt_prob.variables:
            if var.name == "x":
                var.shift = 1.0
                var.scale = 5.0
            elif var.name == "y":
                var.shift = -1.0
                var.scale = 10.0

        # Set shift and scale for responses
        for resp in opt_prob.responses:
            if resp.name == "f":
                resp.shift = 2.0
                resp.scale = 0.5
            if resp.name == "g":
                resp.shift = -2.0
                resp.scale = 2.0                

        # Create the ShiftScaleEvaluator
        ss_evaluator = ShiftScaleEvaluator(evaluate=evaluator, opt_problem=opt_prob)                

        # Define some test points
        test_points = np.array(
            [[0.0, -20.0], [0.0, 0.0], [10.0, 0.0], [10.0, -20.0], [5.0, -10.0]]
        )

        # Check evaluation
        expected_evals = np.array(
            [[2.0, -4.0], [2.0, -8.0], [2.0, -4.0], [2.0, 0.0], [1.0, -4.0]]
        )

        ss_evals = ss_evaluator.eval_np(test_points)

        tol = 1e-8
        assert np.linalg.norm(expected_evals - ss_evals) < tol

        # Check jacobian
        assert hasattr(ss_evaluator, "jacobian")

        dx, dy, df, dg = 5.0, 10.0, 0.5, 2.0

        expected_jac = np.array(
            [
                [[-2.0 / dx * df, -2.0 / dy * df], [1.0 / dx * dg, -3.0 / dy * dg]],
                [[-2.0 / dx * df, 2.0 / dy * df], [1.0 / dx * dg, -3.0 / dy * dg]],
                [[2.0 / dx * df, 2.0 / dy * df], [1.0 / dx * dg, -3.0 / dy * dg]],
                [[2.0 / dx * df, -2.0 / dy * df], [1.0 / dx * dg, -3.0 / dy * dg]],
                [[0.0 / dx * df, 0.0 / dy * df], [1.0 / dx * dg, 0.0 / dy * dg]],
            ]
        )

        ss_jac = ss_evaluator.jacobian(test_points)

        tol = 1e-6
        assert np.linalg.norm(expected_jac - ss_jac) < tol

        # Check jacobian against finite-differencing the evaluator

        def eval_wrapper(x):
            return ss_evaluator.eval_np(x.reshape((1, -1))).flatten()

        stepsize = 1e-5
        approx_jac_evaluator = nd.Jacobian(eval_wrapper, step=stepsize)

        approx_jacs = []
        for pt in test_points:
            approx_jacs.append(approx_jac_evaluator(pt))
        approx_jac = np.array(approx_jacs)

        tol = 1e-4
        assert np.linalg.norm(expected_jac - approx_jac) < tol
        assert np.linalg.norm(ss_jac - approx_jac) < tol
