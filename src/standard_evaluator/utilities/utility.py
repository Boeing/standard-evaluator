"""Utility routines for Pandas DataFrames and optimization test problems."""

from copy import deepcopy
from numbers import Number
from typing import Tuple, Union, List, Dict, Optional
import pandas as pd
import numpy as np


from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator.problem import Variable, ArrayVariable, CategoricalVariable, FloatVariable, IntVariable
from standard_evaluator.problem import OptProblem

def check_prob(prob: dict) -> None:
    """Validates a problem definition dictionary.

    Sets type to 'float' if not defined for a variable or response.
    Sets bounds to [-inf, inf] for float/int variables without bounds.

    Args:
        prob: Problem definition to validate. Will be updated if needed.

    Raises:
        TypeError: If prob is not a dict or has invalid format.
        KeyError: If required keys are missing.
        ValueError: If bounds or types are invalid.
    """

    # Helper function for checking bounds on responses and variables

    def check_bounds(val_type: str, name: str, value: dict):

        # Make sure it is correct type
        if not isinstance(value["bounds"], (list, tuple)):
            raise TypeError(
                f'Design Explorer: Bounds for {val_type} "{name}" '
                "are in an incorrect format! They should be defined as a two "
                "item list or tuple. Received a "
                f'{type(value["bounds"]).__name__} instead.'
            )

        # Check number of categories
        if value["type"] == "categorical":
            if len(value["bounds"]) < 2:
                raise ValueError(
                    f"Design Explorer: Categorical {val_type} "
                    f'"{name}" must have at least two categories! Received '
                    f'{len(value["bounds"])} categories.'
                )
            return

        # Make sure numerical bounds are  defined
        if len(value["bounds"]) != 2:
            raise ValueError(
                f'Design Explorer: Bounds for {val_type} "{name}" '
                f'must have two elements! Received {len(value["bounds"])} elements.'
            )

        # Lower bound is wrong type
        if not isinstance(value["bounds"][0], (int, float, np.number)):
            raise TypeError(
                "Design Explorer: Lower bound is not a numeric "
                f'value! Received a {type(value["bounds"][0]).__name__} instead.'
            )

        # Upper bound is wrong type
        if not isinstance(value["bounds"][1], (int, float, np.number)):
            raise TypeError(
                "Design Explorer: Upper bound is not a numeric "
                f'value! Received a {type(value["bounds"][1]).__name__} instead.'
            )

        # Lower bound is greater than upper bound
        if value["bounds"][0] > value["bounds"][1]:
            raise ValueError(
                f"Design Explorer: Invalid bounds for {val_type} "
                f'"{name}"! Lower bound ({value["bounds"][0]}) is greater '
                f'than upper bound ({value["bounds"][1]}).'
            )

    # Make sure optimization problem is a dictionary
    if not isinstance(prob, dict):
        raise TypeError(
            f"Design Explorer: Problem definition should be in "
            f"dictionary format! Received a {type(prob).__name__} instead."
        )

    # Validate variables are in correct format
    if "variables" not in prob:
        raise KeyError(
            'Design Explorer: Problem definition is missing the "variables" key!'
        )
    if not isinstance(prob["variables"], dict):
        if not isinstance(prob["variables"], list):
            raise TypeError(
                f"Design Explorer: Variable definition in the problem "
                "definition must be a dictionary or list! Received a "
                f'{type(prob["variables"]).__name__}.'
            )
        # We expand the information for each variable into a dictionary with defaults.
        helper_dict = {}
        for var in prob["variables"]:
            helper_dict[var] = {"type": "float", "bounds": [-np.inf, np.inf]}
        prob["variables"] = helper_dict
    else:
        for var, info in prob["variables"].items():
            if "type" not in info:
                # We set float as a default type
                info["type"] = "float"
            elif not isinstance(info["type"], str):
                # if type is not string lines below may be skipped unintentionally
                raise ValueError(
                    f'Design Explorer: The type for variable "{var}" must be'
                    f" given as a string."
                )
            elif info["type"] not in ["float", "int", "categorical"]:
                raise ValueError(
                    f'Design Explorer: The type for variable "{var}" is '
                    f'{info["type"]}, but it must be either "float", "int", or '
                    f'"categorical".'
                )
            if "bounds" not in info:
                # If the variable is float or int we set the default bounds to [-inf, inf]
                if info["type"] in ["float", "int"]:
                    info["bounds"] = [-np.inf, np.inf]
                else:
                    raise KeyError(
                        "Design Explorer: No bounds defined for variable "
                        f'"{var}" in problem definition and the variable is not an '
                        "integer or float."
                    )

            if var.startswith("__"):
                raise ValueError(
                    "Design Explorer: To avoid confusion, variable names "
                    "should not start with a double underscore! Rename "
                    f"variable {var}"
                )

            check_bounds("variable", var, info)

    # Validate responses are in correct format
    if "responses" not in prob:
        raise KeyError(
            "Design Explorer: No responses have been defined in "
            'problem definition! The "responses" key is missing.'
        )
    if not isinstance(prob["responses"], dict):
        if not isinstance(prob["responses"], list):
            raise TypeError(
                f"Design Explorer: Response definition in the problem "
                "definition must be a dictionary or list! Received a "
                f'{type(prob["responses"]).__name__}.'
            )
        # We expand the information for each response into a dictionary with defaults.
        helper_dict = {}
        for var in prob["responses"]:
            helper_dict[var] = {"type": "float"}
        prob["responses"] = helper_dict

    else:
        for resp, info in prob["responses"].items():
            if "type" not in info:
                # We set float as a default type
                info["type"] = "float"
            if resp.startswith("__"):
                raise ValueError(
                    "SeqOpt: To avoid confusion, response names "
                    "should not start with a double underscore! Rename "
                    f"response {resp}"
                )

            if "bounds" in info:
                check_bounds("response", resp, info)


def apply_types(data_frame: pd.DataFrame, type_list: pd.Series) -> None:
    """Enforces column types in the DataFrame.

    Args:
        data_frame: The DataFrame whose column types will be set.
        type_list: Series with column names as index and dtypes as values.
    """
    pd.options.mode.copy_on_write = True
    for var in data_frame.columns.intersection(type_list.index):
        first_element = data_frame[var].iloc[0]
        if not isinstance(first_element, np.ndarray):
            data_frame[var] = data_frame[var].astype(type_list[var])

def apply_types_from_evaluator_info(data_frame: pd.DataFrame, my_info: EvaluatorInfo) -> None:
    """Enforces column types in the DataFrame using the given EvaluatorInfo.

    Args:
        data_frame: The DataFrame whose column types will be set.
        my_info: The EvaluatorInfo used to determine column dtypes.

    Raises:
        TypeError: If data_frame is not a DataFrame, my_info is not an
            EvaluatorInfo, or it contains non-Variable inputs/outputs.
    """
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("The given data_frame is not a pd.DataFrame")

    pd.options.mode.copy_on_write = True
    my_info_types = get_types_from_evaluator_info(my_info)

    for var in data_frame.columns.intersection(my_info_types.index):
        data_frame[var] = data_frame[var].astype(my_info_types[var])

def get_types(opt_prob: dict, variables_only: bool = False) -> pd.Series:
    """Determines dtypes to use for a given problem definition.

    Args:
        opt_prob: Problem definition dictionary.
        variables_only: If True, only return dtypes for variables.

    Returns:
        Series with variable/response names as index and dtypes as values.
    """
    type_info = {}
    # Loop through all variables and responses
    if variables_only is True:
        type_targets = ["variables"]
    else:
        type_targets = ["variables", "responses"]

    for typ in type_targets:
        for name, info in opt_prob[typ].items():
            # Build categorical variable type
            if info["type"] == "categorical":
                type_info[name] = pd.CategoricalDtype(info["bounds"], ordered=True)
            elif info["type"] == "int":
                type_info[name] = "Int64"
            elif info["type"] == "float":
                if 'shape' in info:
                    type_info[name] = 'object'
                else:
                    type_info[name] = "float64"
            else:
                raise ValueError(
                    f"Design Explorer: The type for variable {name} is "
                    f'{info["type"]}, but it must be either "float", "int", or '
                    f'"categorical".'
                )
    # Return series with index as variable (and response) name and dtype as value.
    # This is the same format returned when getting the dtypes from pandas
    return pd.Series(data=type_info)


def get_types_from_evaluator_info(my_info: EvaluatorInfo, variables_only: bool = False) -> pd.Series:
    """Determines dtypes to use for a given EvaluatorInfo.

    Args:
        my_info: The EvaluatorInfo to extract dtypes from.
        variables_only: If True, only return dtypes for inputs.

    Returns:
        Series with variable/response names as index and dtypes as values.

    Raises:
        TypeError: If my_info is not an EvaluatorInfo, variables_only is not
            a bool, or EvaluatorInfo contains non-Variable inputs/outputs.
    """

    if not isinstance(my_info, EvaluatorInfo):
        raise TypeError("Not given an EvaluatorInfo object.")
    if not isinstance(variables_only, bool):
        raise TypeError("variables_only flags is not a bool, as expected.")
    type_info = {}    
    type_targets = my_info.inputs.copy()
    if variables_only is False:
        type_targets += my_info.outputs.copy()
        
    # Loop through all Variable objects
    for var in type_targets:        
        
        # Must start with restrictive types first as they are all Float variables
        if isinstance(var, CategoricalVariable):
            type_info[var.name] = pd.CategoricalDtype(var.bounds, ordered=True)
        elif isinstance(var, IntVariable):
            type_info[var.name] = "Int64"
        elif isinstance(var, ArrayVariable):
            if isinstance(var, CategoricalVariable):
                raise TypeError("The underlying data type is expected to be NumPy float64 arrays.")
            type_info[var.name] = 'object'
        elif isinstance(var, FloatVariable):
            type_info[var.name] = "float64"       
        else:
            raise TypeError("The EvaluatorInfo given contains bad input/output data.")


    # Return series with index as variable (and response) name and dtype as value.
    # This is the same format returned when getting the dtypes from pandas
    return pd.Series(data=type_info)

def create_df_from_problem(
    problem: dict, data: Union[list, np.ndarray] = None, names: list = None
) -> pd.DataFrame:
    """Creates a DataFrame from a problem definition.

    Args:
        problem: An optimization problem in dictionary form.
        data: A list or 2D array of values to store in the DataFrame.
        names: The names of the columns to set with the data.

    Returns:
        DataFrame with columns as defined by the problem.

    Raises:
        TypeError: If data is not a list or numpy array.
    """
    # We have data defined to be added
    if data is not None:
        # Create the list of names to be used.
        names = _create_name_list(problem, names)
        if isinstance(data, np.ndarray):
            # check that the dimensions are right
            if len(data.shape) == 1:
                # Need to reshape the numpy array
                data = data.reshape(1, data.shape[0])
        elif isinstance(data, list):
            data = _check_data_list(data, len(names))
        else:
            raise TypeError("Data is supposed to be list or numpy array")
        # my_types = data_frame.dtypes
        data_frame = pd.DataFrame(data, columns=names)
        apply_types(data_frame, get_types(problem))
    return data_frame


def create_df_from_evaluator_info(
    my_info: EvaluatorInfo, data: Union[List, np.ndarray] = None, 
    names: Optional[List[Union[str, Variable]]] = None
) -> pd.DataFrame:
    """Creates a DataFrame from an EvaluatorInfo and rolled data.

    Args:
        my_info: The EvaluatorInfo used to form the DataFrame.
        data: A list or 2D array of values to store in the DataFrame.
        names: The names or Variables representing columns to set.

    Returns:
        DataFrame with columns as defined by the EvaluatorInfo.

    Raises:
        TypeError: If data is not a list or numpy array.
    """
    # Create the list of names to be used.
    names = _create_name_list_from_evaluator_info(my_info, names)

    # Create the empty dataframe
    data_frame = pd.DataFrame(columns=names)

    # We have data defined to be added
    if data is not None:              
        if isinstance(data, np.ndarray):
            # check that the dimensions are right
            if len(data.shape) == 1:
                # Need to reshape the numpy array
                data = data.reshape(1, data.shape[0])
        elif isinstance(data, list):
            data = _check_data_list(data, len(names))
        else:
            raise TypeError("Data is supposed to be list or numpy array")
        
        # Create dataframe from data
        data_frame = pd.DataFrame(data, columns=names)
    # Apply types to the dataframe - even if empty
    apply_types_from_evaluator_info(data_frame, my_info)
    return data_frame

def _check_data_list(data: list, num_names: int) -> list:
    """Checks that the data list is of the right size and shape.

    Args:
        data: A list (1D or 2D) of values.
        num_names: Expected number of columns.

    Returns:
        Updated data list (list of lists).

    Raises:
        ValueError: If an element has a different number of entries than expected.
    """
    if len(data) > 0:
        if not isinstance(data[0], list):
            # This is as most a one-dimensional array. Convert it to a 2D array
            data = [data]
        for ele in data:
            if len(ele) != num_names:
                raise ValueError(
                    f"Element {ele} in data parameter has a different number of "
                    + "entries than the name parameter"
                )
    return data


def _create_name_list(problem: dict, names: list) -> list:
    """Creates the list of names for the data.

    If no names are defined, uses all variables and responses.

    Args:
        problem: An optimization problem in dictionary form.
        names: A list of names to use. If None, uses all names.

    Returns:
        List of names to be used.

    Raises:
        NameError: If a name is not in the problem's variables or responses.
    """
    all_names = list(problem["variables"].keys()) + list(problem["responses"].keys())
    if names is None:
        # If no names are defined we expect values for all elements
        names = all_names
    else:
        for my_name in names:
            if my_name not in all_names:
                raise NameError(
                    f"Entry {my_name} is not in the list of variables and responses"
                )
    return names

def _create_name_list_from_evaluator_info(my_info: EvaluatorInfo, names: Optional[List[Union[str, Variable]]] = None) -> List[str]:
    """Creates the list of names from an EvaluatorInfo.

    If no names or Variables are given, uses all inputs and outputs.

    Args:
        my_info: An EvaluatorInfo object.
        names: A list of names/Variables to use. If None, uses all.

    Returns:
        List of name strings to be used.

    Raises:
        NameError: If a name is not in the EvaluatorInfo's inputs or outputs.
    """
    all_names = [v.name for v in my_info.inputs] + [v.name for v in my_info.outputs]
    if names is None:
        # If no names are defined we expect values for all elements
        names_as_strings = all_names
    else:
        names_as_strings = []
        for element in names:
            my_name = element.name if isinstance(element, Variable) else element
            if my_name not in all_names:
                raise NameError(
                    f"Entry {my_name} is not in the list of variables and responses"
                )
            names_as_strings.append(my_name)            
    return names_as_strings

def concat_w_empty(inputs: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenates a list of DataFrames, ignoring empty ones.

    Args:
        inputs: List of DataFrames to concatenate.

    Returns:
        The concatenated DataFrame, or an empty DataFrame if all are empty.
    """
    # Filter out all the empty DataFrames
    inputs_wo_empty_dfs = [df for df in inputs if df is not None and not df.empty]
    if len(inputs_wo_empty_dfs) == 0:
        # There are no points, so we return an empty DataFrame
        results = pd.DataFrame()
    else:
        # Build dataframe with inputs
        results = pd.concat(inputs_wo_empty_dfs, ignore_index=True)
    return results


def remove_duplicates(
    sites_df: pd.DataFrame, variables: List[str], responses: List[str]
) -> pd.DataFrame:
    """Removes duplicated sites from a DataFrame.

    Checks that duplicate rows in variables also have duplicate responses.

    Args:
        sites_df: Sites DataFrame.
        variables: A list of variable names.
        responses: A list of response names.

    Returns:
        DataFrame with duplicates removed.

    Raises:
        ValueError: If duplicate variable rows have different response values.
    """
    # Checks for duplicate values in variables and the passed sites and accordingly raise ValueError
    duplicated_variable_sites = sites_df.duplicated(subset=variables)
    duplicated_sites = sites_df.duplicated(subset=variables + responses)

    # If there are duplicates in the variables that are not there in the responses then we need to throw error message
    if duplicated_variable_sites.equals(duplicated_sites) != True:
        raise ValueError(
            "There are sites that have the same variable values but different responses."
        )

    # drop the duplicated rows
    return sites_df.drop_duplicates(subset=variables + responses)


def get_constant_vars(problem_dict: Dict) -> Dict[int, Tuple[str, float]]:
    """Determines the constant variables of a given problem dictionary.

    Args:
        problem_dict: The problem dictionary with a 'variables' key.

    Returns:
        A dict with constant variable indices as keys and (name, value)
        tuples as values.
    """
    constant_var_dict = {}
    input_dicts = problem_dict["variables"]
    vars_ = list(input_dicts.keys())

    for var_idx, var_ in enumerate(vars_):
        var_dict = input_dicts[var_]
        # No bounds, then not constant
        if "bounds" not in var_dict:
            continue
        elif "type" not in var_dict or var_dict["type"] != "categorical":
            # floats and ints
            [lb, ub] = var_dict["bounds"]
            if abs(ub - lb) <= 1.0e-12:
                constant_var_dict[var_idx] = (var_, lb)
        else:
            # if var is categorical then remove repeated entries
            if len(set(var_dict["bounds"])) == 1:
                constant_var_dict[var_idx] = (var_, var_dict["bounds"][0])
    return constant_var_dict

def restrict_problem(
    full_problem: OptProblem,
    variables: Optional[List[str]] = None,
    responses: Optional[List[str]] = None,
) -> OptProblem:
    """Restricts an OptProblem to a subset of variables and responses.

    Args:
        full_problem: The full OptProblem instance.
        variables: Target variable names. If None, all variables are kept.
        responses: Target response names. If None, all responses are kept.

    Returns:
        A new OptProblem representing the restricted problem.
    """

    # Create a deep copy to avoid mutating the original
    restricted_problem = deepcopy(full_problem)

    # Filter variables
    if variables is not None:
        restricted_problem.variables = [
            var for var in full_problem.variables if var.name in variables
        ]

    # Filter responses
    if responses is not None:
        restricted_problem.responses = [
            resp for resp in full_problem.responses if resp.name in responses
        ]

    # Filter objectives if present
    if hasattr(full_problem, "objectives") and full_problem.objectives is not None:
        if responses is not None:
            restricted_problem.objectives = [
                obj for obj in full_problem.objectives if obj in responses
            ]
        else:
            restricted_problem.objectives = deepcopy(full_problem.objectives)

    # Filter constraints if present
    if hasattr(full_problem, "constraints") and full_problem.constraints is not None:
        if responses is not None:
            restricted_problem.constraints = [
                con for con in full_problem.constraints if con in responses
            ]
        else:
            restricted_problem.constraints = None

    return restricted_problem


def get_shift_scale_value(var: Variable) -> Tuple[Union[float, int, np.ndarray], Union[float, int, np.ndarray]]:
    """Returns the (shift, scale) for a single Variable, handling all types.

    For CategoricalVariable (shift/scale are None), returns (0.0, 1.0).
    For ArrayVariable, returns the array-valued shift and scale directly.
    For FloatVariable/IntVariable, returns the scalar shift and scale.

    Args:
        var: A Variable instance.

    Returns:
        A tuple of (shift_value, scale_value). Scalars for float/int/categorical,
        numpy arrays for array variables.

    Raises:
        TypeError: If shift or scale is not numeric.
        ValueError: If scale is zero.
    """
    shift_val = var.shift if var.shift is not None else 0.0
    scale_val = var.scale if var.scale is not None else 1.0

    # Validate numeric types (scalars)
    if not isinstance(shift_val, (int, float, Number, np.ndarray)):
        raise TypeError(
            f"Shift value for '{var.name}' must be numeric! "
            f"Received a {type(shift_val).__name__} instead."
        )
    if not isinstance(scale_val, (int, float, Number, np.ndarray)):
        raise TypeError(
            f"Scale value for '{var.name}' must be numeric! "
            f"Received a {type(scale_val).__name__} instead."
        )
    if np.any(np.asarray(scale_val) == 0):
        raise ValueError(
            f"Scale value for '{var.name}' must be non-zero!"
        )
    return shift_val, scale_val



def _swap_bounds_where_negative_scale(
    lower: Union[float, int, np.ndarray],
    upper: Union[float, int, np.ndarray],
    scale_val: Union[float, int, np.ndarray],
) -> Tuple[Union[float, int, np.ndarray], Union[float, int, np.ndarray]]:
    """Swaps lower and upper bounds element-wise where scale is negative.

    For scalar scale (float/int), swaps the entire lower/upper pair if
    scale < 0. For array scale (np.ndarray), performs element-wise swapping
    using np.where — only indices where scale < 0 are swapped.

    Args:
        lower: Lower bound value. Scalar for FloatVariable/IntVariable,
            numpy array for ArrayVariable.
        upper: Upper bound value. Same type as lower.
        scale_val: Scale value. Scalar for FloatVariable/IntVariable,
            numpy array for ArrayVariable.

    Returns:
        A tuple of (lower, upper) with swaps applied where scale is negative.
    """
    if isinstance(scale_val, np.ndarray):
        neg_mask = scale_val < 0
        if np.any(neg_mask):
            lower, upper = np.where(neg_mask, upper, lower), np.where(neg_mask, lower, upper)
    elif scale_val < 0:
        lower, upper = upper, lower
    return lower, upper


def update_bounds_to_optimizer_space(element: Variable, shift_val, scale_val) -> None:
    """Updates a Variable's bounds and default to optimizer space in place.

    Applies the transformation: new_value = (old_value + shift) * scale.
    Swaps bounds element-wise where scale is negative. Resets shift to 0 and scale to 1.
    Categorical variables are left unchanged (shift/scale reset to None).

    Args:
        element: The Variable to update.
        shift_val: The shift value (scalar or array).
        scale_val: The scale value (scalar or array).
    """
    if isinstance(element, CategoricalVariable):
        element.shift = None
        element.scale = None
        return

    if element.bounds is not None:
        lower, upper = _swap_bounds_where_negative_scale(*element.bounds, scale_val)
        lower = (lower + shift_val) * scale_val
        upper = (upper + shift_val) * scale_val
        element.bounds = (lower, upper)

    if hasattr(element, "default") and element.default is not None:
        element.default = (element.default + shift_val) * scale_val

    if isinstance(element, ArrayVariable):
        element.shift = np.zeros_like(np.asarray(shift_val, dtype=np.float64))
        element.scale = np.ones_like(np.asarray(scale_val, dtype=np.float64))
    elif isinstance(element, IntVariable):
        element.shift = 0
        element.scale = 1
    else:
        element.shift = 0.0
        element.scale = 1.0


def update_bounds_to_design_space(element: Variable, shift_val, scale_val) -> None:
    """Updates a Variable's bounds and default back to design space in place.

    Reverses the transformation: old_value = new_value / scale - shift.
    Swaps bounds element-wise where scale is negative. Restores original shift and scale.
    Categorical variables only have shift/scale restored to None.

    Args:
        element: The Variable to update.
        shift_val: The original shift value (scalar or array).
        scale_val: The original scale value (scalar or array).
    """
    if isinstance(element, CategoricalVariable):
        element.shift = None
        element.scale = None
        return

    if element.bounds is not None:
        lower, upper = _swap_bounds_where_negative_scale(*element.bounds, scale_val)
        lower = lower / scale_val - shift_val
        upper = upper / scale_val - shift_val
        element.bounds = (lower, upper)

    if hasattr(element, "default") and element.default is not None:
        element.default = element.default / scale_val - shift_val

    element.shift = shift_val
    element.scale = scale_val
