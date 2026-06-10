import sys
import numpy as np
import pandas as pd
import pytest

from standard_evaluator.evaluators import TestEvaluator
import standard_evaluator as se
from standard_evaluator import OptProblem, FloatVariable, CategoricalVariable


class DummyEvaluator(TestEvaluator):
    def _create_opt_problem(self) -> OptProblem:
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(num_independent=3, num_dependent=4)

        var_names = ["x1", "x2", "x3"]

        # Define default values
        defaults = [2, 0.5, "one"]
        var_bounds = ([-4, 7], [0, 50], ["neg", "half", "one", "two"])

        # Assigning float class type to variables 0, 1, and 2
        for idx in range(2):
            local_default = defaults[idx]
            local_bound = var_bounds[idx]
            name = var_names[idx]
            new_prob.variables[idx] = FloatVariable(
                name=name, default=local_default, bounds=local_bound
            )

        # Assigning Categorical class type to variables x3
        idx = 2
        local_default = defaults[idx]
        local_bound = var_bounds[idx]
        name = var_names[idx]
        new_prob.variables[idx] = CategoricalVariable(
            name=name,
            default=local_default,
            bounds=local_bound,
            units=None,
            description="",
            options={},
            class_type="cat",
        )

        resp_names = ["f", "c1", "c2", "res"]
        resp_bounds = (
            [-np.inf, np.inf],
            [-np.inf, 0.0],
            [-1, 1],
            ["bad", "neutral", "good"],
        )

        # Assigning float class type to responses 0, 1, and 2
        for idx in range(3):
            local_bound = resp_bounds[idx]
            name = resp_names[idx]
            new_prob.responses[idx] = FloatVariable(
                name=name,
                bounds=local_bound,
                units=None,
                description="",
                options={},
                class_type="float",
            )

        # Assigning Categorical class type to response res
        idx = 3
        local_bound = resp_bounds[idx]
        name = resp_names[idx]
        new_prob.responses[idx] = CategoricalVariable(
            name=name,
            bounds=local_bound,
            units=None,
            description="",
            options={},
            class_type="cat",
        )

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["c1", "c2"]

        return new_prob

    def _def_initial_guess(self) -> list:
        return [2, 0.5]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        #   f = x3 * (x1*x2/4 + x1^2 - x2)
        #  c1 = (x1 + 1)^2 + (x2 - 7)^2 - x2
        #  c2 = 2*cos(x1 + 1) + 5*sin(x2)
        #      'bad'      if c2 < -2
        # res = 'neutral'  if  c2 > 2
        #      'good'     otherwise
        sites["f"] = (
            sites["x1"] * sites["x2"] / 4 + sites["x1"] * sites["x1"] - sites["x2"]
        )

        sites.loc[sites["x3"] == "neg", "f"] *= -1
        sites.loc[sites["x3"] == "half", "f"] *= 0.5
        sites.loc[sites["x3"] == "two", "f"] *= 2

        a = sites["x1"] + 1
        b = sites["x2"] - 7
        sites["c1"] = a * a + b * b - sites["x2"]

        sites["c2"] = 2 * np.cos(sites["x1"] + 1) + 5 * np.sin(sites["x2"])

        sites.loc[sites["c2"] < -2, "res"] = "bad"
        sites.loc[sites["c2"] > 2, "res"] = "neutral"
        sites.loc[(sites["c2"] >= -2) & (sites["c2"] <= 2), "res"] = "good"


@pytest.fixture
def expected_solution() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "x1": [-1.25],
            "x2": [10],
            "x3": ["two"],
            "f": [-23.125],
            "c1": [-0.9375],
            "c2": [-0.782281],
            "res": ["good"],
        }
    )


def test_abstract():
    # Base class can't be instantiated
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        TestEvaluator()

    # Class doesn't define an _evaluate method
    class Fail1(TestEvaluator):
        def _create_opt_problem(self, **kwargs) -> None:
            pass

    expected_message = ""

    # Check Python version
    if sys.version_info.minor < 12:
        expected_message = "with abstract methods? _evaluate"
    else:
        expected_message = "Can't instantiate abstract class Fail1 without an implementation for abstract method '_evaluate'"
    with pytest.raises(TypeError, match=expected_message):
        Fail1()

    # Class doesn't define a _create_opt_problem method
    class Fail2(TestEvaluator):
        def _evaluate(self):
            pass

    # Check Python version
    if sys.version_info.minor < 12:
        expected_message = "with abstract methods? _create_opt_problem"
    else:
        expected_message = "Can't instantiate abstract class Fail2 without an implementation for abstract method '_create_opt_problem'"

    with pytest.raises(TypeError, match=expected_message):
        Fail2()


def test_known_solution(expected_solution: pd.DataFrame):
    def check_solution(solution):
        assert isinstance(solution, pd.DataFrame)
        assert (
            solution[["x1", "x2", "x3", "res"]]
            == expected_solution[["x1", "x2", "x3", "res"]]
        ).all(axis=None)
        assert solution.loc[0, "f"] == pytest.approx(expected_solution.loc[0, "f"])
        assert solution.loc[0, "c1"] == pytest.approx(expected_solution.loc[0, "c1"])
        assert solution.loc[0, "c2"] == pytest.approx(expected_solution.loc[0, "c2"])

    eval = DummyEvaluator()

    # Defaults to None
    assert eval.known_solution is None

    # Should raise an error for non-list like returns
    class DummyEvaluatorFail(DummyEvaluator):
        def _def_known_solution(self):
            return 15

    with pytest.raises(TypeError):
        eval = DummyEvaluatorFail()

    # Raise an error when variables are missing
    class DummyEvaluatorFail(DummyEvaluator):
        def _def_known_solution(self):
            return pd.DataFrame(data={"x1": [-1.25], "x2": 10})

    with pytest.raises(ValueError):
        eval = DummyEvaluatorFail()

    # DataFrame with variables and reponses is given
    class DummyEvaluatorVarAndResp(DummyEvaluator):
        def _def_known_solution(self):
            return pd.DataFrame(
                data={
                    "x1": [-1.25],
                    "x2": [10],
                    "x3": ["two"],
                    "f": [-23.125],
                    "c1": [-0.9375],
                    "c2": [-0.782281],
                    "res": ["good"],
                }
            )

    eval = DummyEvaluatorVarAndResp()
    check_solution(eval.known_solution)

    # DataFrame with only variables given
    class DummyEvaluatorVar(DummyEvaluator):
        def _def_known_solution(self):
            return pd.DataFrame(data={"x1": [-1.25], "x2": [10], "x3": ["two"]})

    eval = DummyEvaluatorVar()
    check_solution(eval.known_solution)

    # Series with only variables given
    class DummyEvaluatorVarAndRespSeries(DummyEvaluator):
        def _def_known_solution(self):
            return pd.Series(
                data={
                    "x1": -1.25,
                    "x2": 10,
                    "x3": "two",
                    "f": -23.125,
                    "c1": -0.9375,
                    "c2": -0.782281,
                    "res": "good",
                }
            )

    eval = DummyEvaluatorVarAndRespSeries()
    check_solution(eval.known_solution)

    # List passed
    class DummyEvaluatorVarList(DummyEvaluator):
        def _def_known_solution(self):
            return [-1.25, 10, "two"]

    eval = DummyEvaluatorVarList()
    check_solution(eval.known_solution)

    # Numpy array passed
    class DummyEvaluatorVarNumPy(DummyEvaluator):
        def _def_known_solution(self):
            return np.array([-1.25, 10, "two"])

    eval = DummyEvaluatorVarNumPy()
    check_solution(eval.known_solution)
