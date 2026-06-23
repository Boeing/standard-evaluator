"""Utility functions for creating OptProblem and EvaluatorInfo instances."""

from typing import Dict, List, Union

from standard_evaluator.problem import OptProblem, Variable, FloatVariable
from standard_evaluator.evaluator import EvaluatorInfo


def collect_names(input_list: List[Variable]) -> List[str]:
    """Extract names from a list of Variable objects.

    Args:
        input_list: A list of Variable objects.

    Returns:
        A list of variable names.
    """
    return [x.name for x in input_list]


def _legacy_dict_to_opt_problem(
    problem_dict: Dict[str, Union[List, Dict]], name: str = "Basic"
) -> OptProblem:
    """Convert a legacy problem dictionary to an OptProblem instance.

    This is an internal helper used only for deserializing old saved models.
    It is not part of the public API.

    Args:
        problem_dict: A legacy problem definition dictionary with 'variables'
            and 'responses' as nested dicts keyed by name.
        name: Name for the resulting OptProblem.

    Returns:
        An OptProblem instance.
    """
    type_mapping = {
        "float": "float",
        "int": "int",
        float: "float",
        int: "int",
        "categorical": "cat",
    }
    fields = ["default", "shift", "scale"]
    full_dict = {"name": f"{name}_problem"}
    for local_type in ["variables", "responses"]:
        local_info = []
        for local_name, info in problem_dict[local_type].items():
            local_dict = {"name": local_name}
            for field in fields:
                if field in info:
                    local_value = info[field]
                    if isinstance(local_value, dict):
                        if local_value["use"]:
                            local_dict[field] = local_value["value"]
                    else:
                        local_dict[field] = info[field]
            if "type" in info:
                local_dict["class_type"] = type_mapping[info["type"]]
            else:
                local_dict["class_type"] = "float"
            if local_dict["class_type"] == "cat":
                local_dict["bounds"] = info["bounds"]
            else:
                if "bounds" in info:
                    local_dict["bounds"] = tuple(info["bounds"])
            local_info.append(local_dict)
        full_dict[local_type] = local_info
    if "objectives" in problem_dict:
        full_dict["objectives"] = problem_dict["objectives"]
    if "constraints" in problem_dict:
        full_dict["constraints"] = problem_dict["constraints"]
    return OptProblem.model_validate(full_dict)


def create_opt_problem(
    num_independent: int, num_dependent: int, name: str = "opt"
) -> OptProblem:
    """Creates an optimization problem with specified independent and dependent variables.

    Args:
        num_independent: The number of independent variables to create. Must be > 0.
        num_dependent: The number of dependent responses to create. Must be > 0.
        name: The name of the optimization problem.

    Returns:
        An OptProblem containing the specified independent variables and responses.
        Variables are named "x0", "x1", etc. Responses are named "f0", "f1", etc.

    Raises:
        ValueError: If num_independent or num_dependent is not an integer > 0.
    """
    if not (isinstance(num_independent, int) and num_independent > 0):
        raise ValueError("num_independent must be an integer greater than 0.")
    if not (isinstance(num_dependent, int) and num_dependent > 0):
        raise ValueError("num_dependent must be an integer greater than 0.")

    variables = [FloatVariable(name=f"x{i}") for i in range(num_independent)]
    responses = [FloatVariable(name=f"f{i}") for i in range(num_dependent)]
    return OptProblem(name=name, variables=variables, responses=responses, objectives=["f0"])


def create_evaluator_info(
    num_inputs: int, num_outputs: int, name: str = "interface"
) -> EvaluatorInfo:
    """Creates an evaluator information object with specified inputs and outputs.

    Args:
        num_inputs: The number of input variables to create. Must be > 0.
        num_outputs: The number of output variables to create. Must be > 0.
        name: The name of the evaluator information object.

    Returns:
        An EvaluatorInfo containing the specified input and output variables.
        Inputs are named "x0", "x1", etc. Outputs are named "f0", "f1", etc.

    Raises:
        ValueError: If num_inputs or num_outputs is not an integer > 0.
    """
    if not (isinstance(num_inputs, int) and num_inputs > 0):
        raise ValueError("num_inputs must be an integer greater than 0.")
    if not (isinstance(num_outputs, int) and num_outputs > 0):
        raise ValueError("num_outputs must be an integer greater than 0.")

    inputs = [FloatVariable(name=f"x{i}") for i in range(num_inputs)]
    outputs = [FloatVariable(name=f"f{i}") for i in range(num_outputs)]
    return EvaluatorInfo(name=name, inputs=inputs, outputs=outputs)
