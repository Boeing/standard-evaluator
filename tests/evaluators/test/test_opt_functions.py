"""Testing of the optpytest classes"""

import pytest
import pydantic
import pandas as pd
import numpy as np
import re

from standard_evaluator.evaluators import TestEvaluator
from standard_evaluator.evaluators.test import HS100, PowellSingularFunction
from standard_evaluator import OptProblem, FloatVariable
import standard_evaluator as se
from pydantic import BaseModel, field_validator, ValidationError


def test_hs100():
    """Test the HS100 test function"""
    # Instantiate the test function
    test_func = HS100()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert initial_guess.iloc[0]["f"] == pytest.approx(714.0)
    assert initial_guess.iloc[0]["c1"] == pytest.approx(13.0)
    assert initial_guess.iloc[0]["c2"] == pytest.approx(265.0)
    assert initial_guess.iloc[0]["c3"] == pytest.approx(171.0)
    assert initial_guess.iloc[0]["c4"] == pytest.approx(4.0)

    assert test_func.inputs == ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]

    assert test_func.outputs == ["f", "c1", "c2", "c3", "c4"]
    # Verify opt_problem variables
    opt = test_func.opt_problem
    var_map = {v.name: v for v in opt.variables}
    expected_vars = {
        "x1": {"bounds": (-10.0, 10.075), "default": 1.0, "scale": 0.1, "shift": 0.0},
        "x2": {"bounds": (-10.0, 10.075), "default": 2.0, "scale": 0.1, "shift": 0.0},
        "x3": {"bounds": (-10.0, 10.075), "default": 0.0, "scale": 0.1, "shift": 0.0},
        "x4": {"bounds": (-10.0, 10.075), "default": 4.0, "scale": 0.1, "shift": 0.0},
        "x5": {"bounds": (-10.0, 10.075), "default": 0.0, "scale": 0.1, "shift": 0.0},
        "x6": {"bounds": (-10.0, 10.075), "default": 1.0, "scale": 0.1, "shift": 0.0},
        "x7": {"bounds": (-10.0, 10.075), "default": 1.0, "scale": 0.1, "shift": 0.0},
    }
    for name, expected in expected_vars.items():
        var = var_map[name]
        assert var.bounds == expected["bounds"]
        assert var.default == expected["default"]
        assert var.scale == expected["scale"]
        assert var.shift == expected["shift"]

    # Verify opt_problem responses
    resp_map = {r.name: r for r in opt.responses}
    expected_resps = {
        "f": {"bounds": (-np.inf, np.inf), "scale": 0.001, "shift": 0.0},
        "c1": {"bounds": (0.0, np.inf), "scale": 0.001, "shift": 0.0},
        "c2": {"bounds": (0.0, np.inf), "scale": 0.01, "shift": 0.0},
        "c3": {"bounds": (0.0, np.inf), "scale": 0.01, "shift": 0.0},
        "c4": {"bounds": (0.0, np.inf), "scale": 0.01, "shift": 0.0},
    }
    for name, expected in expected_resps.items():
        resp = resp_map[name]
        assert resp.bounds == expected["bounds"]
        assert resp.scale == expected["scale"]
        assert resp.shift == expected["shift"]

    assert opt.objectives == ["f"]
    assert opt.constraints == ["c1", "c2", "c3", "c4"]


def test_hs100_eval_np():
    """Test the eval_np function of the HS100 test function"""
    # Instantiate the test function
    test_func = HS100()
    # Get the initial guess
    initial_guess = np.array(test_func.initial_guess()[test_func.inputs])
    # Evaluate the initial guess
    result = test_func.eval_np(initial_guess)
    # Check some specific responses
    default_result = np.array([[714.0, 13.0, 265.0, 171.0, 4.0]])
    np.testing.assert_allclose(
        result,
        default_result,
    )


def test_hs100_opt():
    "Test the optimal solution of the HS100 test function"
    # Instantiate the test function
    test_func = HS100()
    # Get the known solution
    known_sol = test_func.known_solution

    assert len(known_sol) == 1
    # Check some specific responses
    assert known_sol.iloc[0]["f"] == pytest.approx(680.6300573)


def test_hs100_de_eval():
    "Test the evaluator method that can be used in conjunction with DE5"
    # Instantiate the test function
    test_func = HS100()
    # Get the initial guess
    initial_guess = test_func.initial_guess()

    # Convert the initial guess data frame into a list of list like
    # DE5 would expect. Note that we only get the variable values.
    initial_guess_list = initial_guess[test_func.inputs].to_numpy().tolist()

    initial_guess_responses = test_func.eval_list(initial_guess_list)

    # Check some specific responses
    assert initial_guess_responses[0][0] == pytest.approx(714.0)


def test_hs100_failed_call():
    """Test the HS100 test function"""
    # Instantiate the test function
    test_func = HS100()
    site = pd.DataFrame(
        [[1.0, 35.0, 1.0, 3.0, 6.0, 0.7, 0.9]],
        columns=test_func.inputs,
        dtype="float64",
    )
    site.drop(columns=["x2"], inplace=True)
    with pytest.raises(ValueError):
        test_func(site)


def test_powell_singular_function():
    """Test the Powell Singular test function"""
    # Instantiate the test function
    test_func = PowellSingularFunction()
    # Get the initial guess
    initial_guess = test_func.initial_guess()
    # Evaluate the initial guess
    test_func(initial_guess)
    # Check some specific responses
    assert initial_guess.iloc[0].f == pytest.approx(95.0)

    assert test_func.inputs == ["x1", "x2", "x3", "x4"]

    assert test_func.outputs == ["f"]


def test_no_variables_descendent():

    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self) -> OptProblem:
            """Create and return an OptProblem instance."""
            se.utilities.create_opt_problem()

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(TypeError):
        DummyTestFunction()


def test_variables_no_info_descendent():
    class OptProblemModel(BaseModel):
        problem: "OptProblem"  # Expecting an OptProblem instance

    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self):
            # Validate test_prob using Pydantic model
            OptProblemModel(
                problem={"variables": {"x1"}}
            )  # This will raise ValidationError if type is wrong

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(pydantic.ValidationError):
        DummyTestFunction()


def test_variables_no_bounds_descendent():
    class OptProblemParams(BaseModel):
        num_independent: int
        num_dependent: int  # Required field

    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self) -> OptProblem:
            OptProblemParams(num_independent=1)

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(pydantic.ValidationError, match="Field required"):
        DummyTestFunction()


def test_variables_bounds_wrong_type_descendent():
    class BoundsModel(BaseModel):
        bounds: tuple[float, float]

    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self) -> OptProblem:
            # Create the basic problem
            se.utilities.create_opt_problem(num_independent=1, num_dependent=1)
            var_names = ["x1"]
            var_bounds = 0.0

            # Validate bounds using Pydantic model (raises ValidationError if invalid)
            BoundsModel(bounds=var_bounds)

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(pydantic.ValidationError, match="Input should be a valid tuple"):
        DummyTestFunction()


def test_variables_bounds_wrong_size_descendent():
    class BoundsModel(BaseModel):
        bounds: list[float]

        @field_validator("bounds")
        def check_bounds_length(cls, bounds_value):
            if len(bounds_value) != 2:
                raise ValueError(
                    f'Bounds for variable "x1" must have two elements. Received {len(bounds_value)}'
                )

    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self) -> OptProblem:
            se.utilities.create_opt_problem(num_independent=1, num_dependent=1)
            var_names = ["x1"]
            var_bounds = [0.0]  # Invalid bounds length

            # Validate bounds using Pydantic model (raises ValidationError if invalid)
            BoundsModel(bounds=var_bounds)

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(
        pydantic.ValidationError,
        match='Bounds for variable "x1" must have two elements. Received 1',
    ):
        DummyTestFunction()


def test_no_responses_descendent():
    class OptProblemParams(BaseModel):
        num_independent: int
        num_dependent: int  # Required field

    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self) -> OptProblem:
            OptProblemParams(num_independent=1)

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(
        pydantic.ValidationError,
        match="1 validation error for OptProblemParams\nnum_dependent\n  Field required",
    ):
        DummyTestFunction()


def test_responses_no_info_descendent():
    class DummyTestFunction(TestEvaluator):
        """Dummy test function that does not work"""

        def _create_opt_problem(self) -> OptProblem:
            new_prob = se.utilities.create_opt_problem(
                num_independent=1, num_dependent=1
            )

            new_prob.responses = "f"

            return new_prob

        def initial_guess(self) -> pd.DataFrame:
            """Provide an initial guess for an optimizer

            :return: DataFrame providing the initial guess to use with an optimizer
            :rtype: pd.DataFrame
            """
            return self._create_df([1.0, 2.0, 2.0, 4.0, 0.0, 1.0, 1.0])

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    with pytest.raises(ValidationError, match=r"Input should be a valid list"): DummyTestFunction()


def test_default_initial_guess():
    class DummyTestFunction(TestEvaluator):
        """Test that the default initial guess calculation works"""

        def _create_opt_problem(self) -> OptProblem:
            new_prob = se.utilities.create_opt_problem(
                num_independent=5, num_dependent=1
            )
            var_names = ["x1", "x2", "x3", "x4", "x5"]
            var_bounds = (
                [0.0, 8.0],
                [1.0, 5.0],
                [-np.inf, 5.0],
                [-np.inf, np.inf],
                [2.0, np.inf],
            )
            for var, name, local_bound in zip(
                new_prob.variables, var_names, var_bounds
            ):
                var.name = name
                var.bounds = local_bound

            resp_name = ["white"]
            for resp, name in zip(new_prob.responses, resp_name):
                resp.name = name

            new_prob.objectives = ["white"]
            new_prob.constraints = []

            return new_prob

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    func = DummyTestFunction()
    initial_guess = func.initial_guess()
    assert initial_guess.iloc[0].x1 == pytest.approx(4.0)
    assert initial_guess.iloc[0].x2 == pytest.approx(3.0)
    assert initial_guess.iloc[0].x3 == pytest.approx(5.0)
    assert initial_guess.iloc[0].x4 == pytest.approx(0.0)
    assert initial_guess.iloc[0].x5 == pytest.approx(2.0)


def test_default_known_solution():
    class DummyTestFunction(TestEvaluator):
        """Test that the default initial guess calculation works"""

        def _create_opt_problem(self) -> OptProblem:
            new_prob = se.utilities.create_opt_problem(
                num_independent=2, num_dependent=1
            )
            var_names = ["x1", "x2"]
            var_bounds = ([0.0, 8.0], [1.0, 5.0])
            for var, name, local_bound in zip(
                new_prob.variables, var_names, var_bounds
            ):
                var.name = name
                var.bounds = local_bound

            resp_name = ["white"]
            for resp, name in zip(new_prob.responses, resp_name):
                resp.name = name

            new_prob.objectives = ["white"]
            new_prob.constraints = []

            return new_prob

        def _evaluate(self, sites: pd.DataFrame) -> None:
            """Dummy call

            :param df: The dataframe that contains the input values, and is updated with the responses
            :type df: DataFrame
            """

    func = DummyTestFunction()
    func.known_solution is None
