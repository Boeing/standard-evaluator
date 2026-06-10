# Utilities to help convert a problem into a fully defined state
from copy import deepcopy
from typing import Tuple, Union, Dict, List
import numpy as np

from standard_evaluator.problem import OptProblem, Variable, FloatVariable
from standard_evaluator.evaluator import EvaluatorInfo


def legacy_to_opt_problem(
    problem_dict: Dict[str, Union[List, Dict]], name: str = "Basic"
) -> OptProblem:
    TYPE_MAPPING = {
        "float": "float",
        "int": "int",
        float: "float",
        int: "int",
        "categorical": "cat",
    }
    FIELDS = ["default", "shift", "scale"]
    full_dict = {"name": f"{name}_problem"}
    for local_type in ["variables", "responses"]:
        local_info = []
        for local_name, info in problem_dict[local_type].items():
            local_dict = {
                "name": local_name,
            }
            for field in FIELDS:
                if field in info:
                    # We sometimes store a dictionary instead of a value.
                    local_value = info[field]
                    if isinstance(local_value, dict):
                        if local_value["use"]:
                            local_dict[field] = local_value["value"]
                    else:
                        local_dict[field] = info[field]
            if "type" in info:
                local_dict["class_type"] = TYPE_MAPPING[info["type"]]
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

def collect_names(input_list: List[Variable]) -> List[str]:
    return [x.name for x in input_list]


def opt_problem_to_legacy(problem: OptProblem) -> Dict[str, Union[List, Dict]]:
    full_dict = {}
    problem_dict = problem.model_dump()
    
    

    # Right now we have a 1:1 mapping between types in Standard Evaluator and DE.
    # We expect to add more types to Standard Evaluator, hence the Type Mapping.
    TYPE_MAPPING = {"floatarray":'float',
        "float": "float", "int": "int", "cat": "categorical"}
    FIELDS = ["shift", "scale", "bounds"]
    for local_type in ["variables", "responses"]:
        local_info = {}
        for info in problem_dict[local_type]:
            local_dict = {}
            for field in ["shift", "scale"]:
                local_dict[field] = info[field]
            # We need to convert the set for the bounds to a list:
            if info["bounds"]:
                local_dict["bounds"] = list(info["bounds"])
            else:
                local_dict["bounds"] = [-np.inf, np.inf]
            if local_type == "variables":
                # For variables we also store the default value            
                local_dict["default"] = info["default"]
            if info["class_type"] == "floatarray":
                local_dict["shape"] = info["shape"]
            local_dict["type"] = TYPE_MAPPING[info["class_type"]]
            local_info[info["name"]] = local_dict
        full_dict[local_type] = local_info
    full_dict["objectives"] = problem_dict["objectives"]
    full_dict["constraints"] = problem_dict["constraints"]
    return full_dict


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
    # Check that num_independent and num_dependent are integers greater than 0
    if not (isinstance(num_independent, int) and num_independent > 0):
        raise ValueError("num_independent must be an integer greater than 0.")
    if not (isinstance(num_dependent, int) and num_dependent > 0):
        raise ValueError("num_dependent must be an integer greater than 0.")

    # Create independent variables
    variables = [FloatVariable(name=f"x{i}") for i in range(num_independent)]

    # Create responses
    responses = [FloatVariable(name=f"f{i}") for i in range(num_dependent)]
    # Create and return the optimization problem
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
    # Check that num_independent and num_dependent are integers greater than 0
    if not (isinstance(num_inputs, int) and num_inputs > 0):
        raise ValueError("num_inputs must be an integer greater than 0.")
    if not (isinstance(num_outputs, int) and num_outputs > 0):
        raise ValueError("num_outputs must be an integer greater than 0.")

    # Create independent variables
    inputs = [FloatVariable(name=f"x{i}") for i in range(num_inputs)]

    # Create responses
    outputs = [FloatVariable(name=f"f{i}") for i in range(num_outputs)]

    # Create and return the optimization problem
    return EvaluatorInfo(name=name, inputs=inputs, outputs=outputs)


def problem_calculate_fields(base_problem: dict, expand_inf: bool = False) -> dict:
    """Creates a new problem dict with defaults and shift/scale set if not existing.

    Args:
        base_problem: The optimization problem to be expanded.
        expand_inf: Whether or not to expand the bounds.

    Returns:
        The updated problem dictionary.
    """
    new_problem = deepcopy(base_problem)

    # Save the name and whether an element is a variable or response in the dictionary for the element
    problem_expand_name_info(new_problem)
    # Calculate the problem default value, if not already set. Otherwise just check that they are valid.
    problem_calculate_default(new_problem["variables"])
    # Calculate the shift values for the variables if not already set.
    problem_calculate_shift(new_problem["variables"])
    # Calculate the shift values for the responses if not already set.
    problem_calculate_shift(new_problem["responses"])
    # Calculate the scale values for the variables if not already set.
    problem_calculate_scale(new_problem["variables"])
    # Calculate the scale values for the responses if not already set.
    problem_calculate_scale(new_problem["responses"])
    # Set all responses to active if they are not already
    problem_set_response_active(new_problem)

    # If asked for we expand the bounds and create four new fields, low, high, use_low, and use_high
    if expand_inf:
        problem_convert_bounds(new_problem["variables"])
        problem_convert_bounds(new_problem["responses"])

    return new_problem


def problem_expand_name_info(total_info: dict) -> None:
    """Saves names and variable/response type info into the problem dict.

    Args:
        total_info: The dictionary with the problem information. Will be updated.
    """
    for element, element_name in zip(
        ["variables", "responses"], ["Variable", "Response"]
    ):
        for name, info in total_info[element].items():
            info["name"] = name
            info["info"] = element_name


def problem_calculate_default(total_info: dict) -> None:
    """Checks the default value if set; if not, sets it in the dictionary.

    Args:
        total_info: The dictionary with the variable information. Will be updated.
    """
    for var_name, var_info in total_info.items():
        # If no type is defined for the response we set it to float
        if "type" not in var_info:
            var_info["type"] = "float"
        # Get the lower and upper bounds from the problem
        low, high = problem_get_bounds(var_name, var_info)

        # Save the default value in the problem dictionary
        var_info["default"] = check_default(var_name, var_info, low, high)

        # If the active field is set we use its value, otherwise we assume the variable is active.
        if "active" in var_info:
            active = var_info["active"]
        else:
            active = True
        # If the lower and upper bounds are identical we automatically set the variable as inactive
        if low == high:
            active = False
        # Store whether or not the variable is active
        var_info["active"] = active


def problem_calculate_shift(total_info: dict) -> None:
    """Checks the shift value if set; if not, sets it in the dictionary.

    Args:
        total_info: The dictionary with the variable or response information.
            Will be updated.
    """
    for entry_name, entry_info in total_info.items():
        # By default the shift will not be used
        shift_set = False
        # by default shift is 0 unless bounds are defined.
        shift = 0.0
        # Overwrite the default if the shift value is defined in the problem
        if "shift" in entry_info:
            (shift, shift_set) = _expand_entry(entry_info["shift"])
        else:
            # If bounds are defined for the problem use them
            if "bounds" in entry_info:
                # Get the lower and upper bounds from the problem
                (low, high) = problem_get_bounds(entry_name, entry_info)
                shift = _calculate_shift_including_inf(low, high)
        entry_info["shift"] = {"value": shift, "use": shift_set}


def _calculate_shift_including_inf(low: float, high: float) -> float:
    """Calculates the shift factor accounting for infinite bounds.

    Args:
        low: Lower bound.
        high: Upper bound.

    Returns:
        The shift factor.
    """
    if low == -np.inf:
        if high == np.inf:
            # If no upper bound is defined as well we set shift to 0
            shift = 0.0
        else:
            # When no lower bound is defined we set the shift factor
            # to upper bound - 10.0
            shift = high - 10.0
    else:
        # The shift factor is equal to the lower bound unless the lower
        #  bound is -inf
        shift = low
    return shift


def problem_calculate_scale(total_info: dict) -> None:
    """Checks the scale value if set; if not, sets it in the dictionary.

    Args:
        total_info: The dictionary with the variable or response information.
            Will be updated.
    """
    for entry_name, entry_info in total_info.items():
        # By default the scale will not be used, and be set to 1.0
        scale_set = False
        scale = 1.0
        # Check if bounds are defined for the problem. if not, set low to 0 and high to 1, which
        # will set the scale to 1 if not already set.
        if "bounds" in entry_info:
            # Get the lower and upper bounds from the problem
            (low, high) = problem_get_bounds(entry_name, entry_info)
        else:
            (low, high) = (-np.inf, np.inf)
        # Set the scale value if defined.
        if "scale" in entry_info:
            (scale, scale_set) = _expand_entry(entry_info["scale"])
        else:
            # If no scale is defined we pre-set it
            if not ((low == -np.inf) or (high == np.inf) or (low == high)):
                # We set the scale to 1.0 / (high - low) unless either the upper or lower
                # bound are +- infinity, or they are identical
                scale = 1.0 / (high - low)
        entry_info["scale"] = {"value": scale, "use": scale_set}


def _expand_entry(info: Union[float, dict]) -> Tuple[float, float]:
    """Expands an entry that can be either a value or a value-and-flag dict.

    Args:
        info: The entry to expand (a float/int or a dict with 'value' and 'use').

    Returns:
        A tuple of (value, flag) indicating the value and whether to use it.
    """
    if isinstance(info, (float, int)):
        # If we have shift saved as just a float we save the value and
        # store it as set
        value = info
        value_set = True
    else:
        # if the problem definition is a dictionary we use the information
        value = info["value"]
        value_set = info["use"]
    return (value, value_set)


def problem_set_response_active(total_info: dict) -> None:
    """Sets all responses to active if not already defined.

    Args:
        total_info: The dictionary with the problem information. Will be updated.
    """
    # Check if the objectives are defined seperately
    if "objectives" in total_info:
        objectives = total_info["objectives"]
    else:
        objectives = []

    for res_name, res_info in total_info["responses"].items():
        if "active" not in res_info:
            # We need to set the response to active if no active field is set
            res_info["active"] = True
        # We also check if bounds are set on this response. If not we set them to +- Inf
        if "bounds" not in res_info:
            res_info["bounds"] = [-np.inf, np.inf]
        # If no type is defined for the response we set it to float
        if "type" not in res_info:
            res_info["type"] = "float"
        # By default the response is not an objective
        is_objective = False
        # Check if the "objective" flag is set
        if "objective" in res_info:
            is_objective = res_info["objective"]
        # See if the response is listed in the objectives. This will
        # always set objective to true, even if the flag in the
        # response is set to false
        if res_name in objectives:
            is_objective = True
        res_info["objective"] = is_objective


def problem_convert_bounds(total_info: dict) -> None:
    """Calculates and sets the low, high, use_low, and use_high fields.

    Args:
        total_info: The dictionary with the variable information. Will be updated.
    """
    for var_info in total_info.values():

        # Get the lower and upper bounds from the problem
        low = var_info["bounds"][0]
        high = var_info["bounds"][1]
        # We need to see if the lower bound is -inf, and safe that in
        # the low_inf variable
        low_inf = low == -np.inf
        # We want to pre-set a numerical value for the lower bound if the user
        # decides to un-check the lower bound -inf checkbox
        if low_inf:
            low = np.min([-1000000.0, high - 1000.0])
        # We need to see if the upper bound is inf, and safe that in
        # the high_inf variable
        high_inf = high == np.inf
        # We want to pre-set a numerical value for the upper bound if the user
        # decides to un-check the upper bound inf checkbox
        if high_inf:
            high = np.max([1000000.0, low + 1000.0])
        # We will use the lower bound if it is not -infinity
        var_info["low"] = {"value": low, "use": not low_inf}
        # We will use the upper bound if it is not infinity
        var_info["high"] = {"value": high, "use": not high_inf}


def problem_get_bounds(var_name: str, var_info: dict) -> Tuple[float, float]:
    """Checks if bounds are defined and returns them.

    Args:
        var_name: Name of the variable / response.
        var_info: Information dict for the variable / response.

    Returns:
        A tuple of (lower_bound, upper_bound).

    Raises:
        ValueError: If bounds are not defined, not a list, not length 2,
            or upper < lower.
    """
    if "bounds" not in var_info:
        raise ValueError(f"No bounds defined for {var_name}")
    if not isinstance(var_info["bounds"], list):
        raise ValueError(f"Bounds info defined for {var_name} is not a list")
    if len(var_info["bounds"]) != 2:
        raise ValueError(
            f"Bounds info defined for {var_name} is not a list of length 2"
        )
    low = var_info["bounds"][0]
    high = var_info["bounds"][1]
    if high < low:
        raise ValueError(
            f"Upper bound ({high}) smaller than lower bound ({low}) for variable {var_name}"
        )
    return (low, high)


def check_default(var_name, var_info, low, high) -> float:
    """Checks if a default value is set; if not, computes it from bounds.

    Args:
        var_name: Name of the variable.
        var_info: Information dict for the variable.
        low: Lower bound.
        high: Upper bound.

    Returns:
        The default value.

    Raises:
        ValueError: If the default is outside the bounds.
    """

    # If no default values are set calculate them as the average
    # between upper and lower bound
    if "default" in var_info:
        default = var_info["default"]
        if default < low:
            raise ValueError(
                f"Default value ({default}) smaller than lower bound ({low}) for variable {var_name}"
            )
        if default > high:
            raise ValueError(
                f"Default value ({default}) larger than upper bound ({high}) for variable {var_name}"
            )
    else:
        default = calculate_default_value(low, high)
    return default


def calculate_default_value(low: float, high: float) -> float:
    """Calculates the default value based on lower and upper bounds.

    Args:
        low: Lower bound.
        high: Upper bound.

    Returns:
        The calculated default value.
    """

    # The lower bound is unbounded (-inf)
    if low == -np.inf:
        if high == np.inf:
            # This variable is unbounded, so we set default to 0.
            default = 0.0
        else:
            # This variable has no lower bound, so we set default to upper bound
            default = high
    # We have a lower bound, but the upper bound is +inf
    elif high == np.inf:
        # Set the default value to the lower bound
        default = low
    # Both upper and lower bound are actual values. Set default to be centered.
    else:
        default = (low + high) / 2.0
    return default
