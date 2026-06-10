# Methods to deal with unrolling and rolling of array variables or responses within a Pandas DataFrame
"""Methods for unrolling and rolling array variables/responses within Pandas DataFrames."""
import typing
import pandas as pd
import numpy as np


from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator.problem import ArrayVariable, CategoricalVariable, FloatVariable, IntVariable, Variable


def generate_names(name: str, shape: tuple) -> typing.List[str]:
    """Generates names for array variables/responses using NumPy array syntax.

    For example, a variable "input" with shape (2,) generates:
    ['input[0]', 'input[1]']

    Args:
        name: Name of the variable or response.
        shape: Shape of the variable or response (a NumPy shape tuple).

    Returns:
        List of all element names represented by the array.

    Raises:
        TypeError: If name is not a string or shape is not a tuple.
    """
    if not isinstance(name, str):
        raise TypeError("Name is not a string")
    if not isinstance(shape, tuple):
        raise TypeError("Shape is not a tuple")

    new_names = []
    for idx in np.ndindex(shape):
        new_names.append(f"{name}[" + ",".join([str(value) for value in idx]) + "]")
    return new_names


def unroll_names(sub_problem: typing.Mapping[str, typing.Mapping]) -> typing.List[str]:
    """Unrolls names in a variable or response section of a SE problem.

    If a variable or response has a shape defined it is unrolled, otherwise
    the name is used directly.

    Args:
        sub_problem: Variable or response section of a SE problem dictionary.

    Returns:
        Names of variables or responses, unrolled if needed.
    """
    unrolled_names = []
    for name, info in sub_problem.items():
        # If the variable or response define a shape we need to unroll it to generate all unique names
        if "shape" in info:
            unrolled_names += generate_names(name, info["shape"])
        else:
            # if no shape is defined we just use the variable or response name
            unrolled_names.append(name)
    return unrolled_names

def unroll_names_using_variables(variables: typing.List[Variable]) -> typing.List[str]:
    """Unrolls names from a list of Variables.

    Args:
        variables: A list of Variable objects.

    Returns:
        Names of variables, unrolled if needed.

    Raises:
        TypeError: If variables is not a list or not a list of Variables.
    """
    if not isinstance(variables, list):
        raise TypeError('Not given a list')
    elif np.any([not isinstance(var, (IntVariable, FloatVariable, ArrayVariable,  CategoricalVariable)) for var in variables]):
        raise TypeError('The list is not a list of Variables')
    
    unrolled_names = []
    for var in variables:
        name = var.name
        if isinstance(var, ArrayVariable):
            # If the variable defines a shape we need to unroll it to generate all unique names
            unrolled_names += generate_names(name, var.shape)
        else:
            # if no shape is defined we just use the variable or response name
            unrolled_names.append(name)
    return unrolled_names

# Unrolling the arrays inside the DataFrame
def unroll_data_frame_numpy(rolled_df: pd.DataFrame) -> np.array:
    """Unrolls all arrays inside a DataFrame into a normal 2-D NumPy array.

    Note that if a site has an array of a different size than the other sites
    this method will fail.

    Args:
        rolled_df: The DataFrame to unroll.

    Returns:
        Unrolled 2-D array of all the sites.

    Raises:
        TypeError: If rolled_df is not a Pandas DataFrame.
    """
    if not isinstance(rolled_df, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")

    all_rows = rolled_df.to_numpy()
    np_list = []
    for row_index in range(all_rows.shape[0]):
        # Heterogeneous input needs to be set to object to ensure that there is no improper conversion.
        # Homogeneous input should remain as is. 
        # So use the type casting from .to_numpy

        single_row = np.concatenate(
            [np.array(ele).flatten() for ele in all_rows[row_index]],
            dtype = all_rows.dtype
            )
        
        np_list.append(single_row)
    unrolled = np.stack(np_list, axis=0)
    return unrolled


# Unrolling a DataFrame to a new DataFrame including variable naming
def unroll_data_frame(
    input_sites: pd.DataFrame, problem: typing.Mapping[str, typing.Mapping]
) -> pd.DataFrame:
    """Unrolls all arrays inside a DataFrame and returns a new DataFrame with variable naming.

    Args:
        input_sites: The DataFrame to unroll.
        problem: The problem dictionary.

    Returns:
        Unrolled DataFrame with variable naming.

    Raises:
        TypeError: If input_sites is not a DataFrame or problem is not a dict.
        KeyError: If 'variables' is not a key in the problem.
    """
    if not isinstance(input_sites, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")
    if not isinstance(problem, dict):
        raise TypeError("problem is not a dict")
    # Check that the problem has a key called variables. If not, raise a TypeError. This is a required key.
    # The problem dictionary should have a key called variables that is a dict that contains all the variable / response names.
    if "variables" not in problem:
        raise KeyError(f"variables is not a key in the problem")
    # Create a NumPy array that unrolles all arrays in the DataFrame. Only use the columns for variables
    unrolled = unroll_data_frame_numpy(input_sites[list(problem["variables"])])
    # Generate the unrolled names for all variables
    var_names = unroll_names(problem["variables"])
    unrolled_sites = pd.DataFrame(unrolled, columns=var_names)
    return unrolled_sites

def unroll_data_frame_using_variables(rolled_df: pd.DataFrame, variables: typing.List[Variable]) -> pd.DataFrame:
    """Unrolls arrays for specified variable columns and returns a new DataFrame.

    Args:
        rolled_df: The DataFrame to unroll.
        variables: A list of Variable objects.

    Returns:
        Unrolled DataFrame with columns given by variables using unrolled naming.

    Raises:
        TypeError: If rolled_df is not a DataFrame, variables is not a list,
            or variables is not a list of Variables.
    """


    if not isinstance(rolled_df, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")
    if not isinstance(variables, list):
        raise TypeError('Not given a list')
    elif np.any([not isinstance(var, (IntVariable, FloatVariable, ArrayVariable,  CategoricalVariable)) for var in variables]):
        raise TypeError('The list is not a list of Variables')
    
    # Create a NumPy array that unrolls arrays in the sub DataFrame.
    unrolled_np = unroll_data_frame_numpy(rolled_df[[var.name for var in variables]])

    # Generate the unrolled names for all variables in the list
    unrolled_var_names = unroll_names_using_variables(variables)
    unrolled_df = pd.DataFrame(unrolled_np, columns=unrolled_var_names)

    # Need to ensure dtypes are still preserved. ArrayVariables are assumed to be float arrays
    for var in variables:
        if isinstance(var, CategoricalVariable):
            unrolled_df[var.name] = unrolled_df[var.name].astype(pd.CategoricalDtype(var.bounds, ordered=True))
        elif isinstance(var, IntVariable):
            unrolled_df[var.name] = unrolled_df[var.name].astype('int')
        elif isinstance(var, ArrayVariable):
            unrolled_var_names = generate_names(var.name, var.shape)
            for nm in unrolled_var_names:
                unrolled_df[nm]= unrolled_df[nm].astype('float')
        else:
            unrolled_df[var.name] = unrolled_df[var.name].astype('float')
    
    return unrolled_df


# Write a function that takes an unrolled DataFrame and a problem dictionary and returns a new DataFrame 
# that stores responses that are NumPy arrays in their columns.
def roll_data_frame(unrolled_df: pd.DataFrame, problem: typing.Mapping[str, typing.Mapping]
) -> pd.DataFrame:
    """Rolls arrays back into a DataFrame with array responses in single columns.

    Args:
        unrolled_df: The DataFrame to roll back.
        problem: The problem dictionary.

    Returns:
        Rolled DataFrame with array responses stored in single columns.

    Raises:
        TypeError: If unrolled_df is not a DataFrame, problem is not a dict,
            problem does not contain 'responses', or the DataFrame is missing
            required columns.
    """
    if not isinstance(unrolled_df, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")
    if not isinstance(problem, dict):
        raise TypeError("problem is not a dict")
    # Check that the problem has a key called responses. If not, raise a TypeError. This is a required key.
    # The problem dictionary should have a key called responses  that is a dict that contains all the response names.
    if "responses" not in problem:
        raise TypeError(f"responses is not a key in the problem")
    
    names_to_roll = []
    shapes_to_roll = []
    for name in problem['responses']:
        # If the response define a shape we need to roll it
        if "shape" in problem['responses'][name]:
            names_to_roll += [name]
            shapes_to_roll += [problem['responses'][name]['shape']]

    rolled_df = pd.DataFrame([])
    unrollable_names = [name for name in problem['responses'] if name not in names_to_roll ]
    if unrollable_names:
        # get the names of unrollable variables for later use        
        if np.any([unrollable_name not in unrolled_df for unrollable_name in unrollable_names]):
            raise TypeError(f"The given DataFrame does not contain the responses: {[unrollable_name for unrollable_name in unrollable_names if unrollable_name not in unrolled_df]}")
        rolled_df = unrolled_df[unrollable_names].copy()

    for name, roll_shape in zip(names_to_roll, shapes_to_roll):
        unrolled_name = generate_names(name, roll_shape)
        # the data needs to have all of the unrolled data
        if any([var_name not in unrolled_df for var_name in unrolled_name]):
            raise TypeError(f"The given DataFrame does not contain the unrolled var names for {name}")
        rolled_df[name] =[row_data.reshape(roll_shape) for row_data in unrolled_df[unrolled_name].to_numpy()]

    return rolled_df[list(problem['responses'].keys())]


def roll_data_frame_using_variables(unrolled_df: pd.DataFrame, variables: typing.List[Variable]) -> pd.DataFrame:
    """Rolls arrays back into a DataFrame from a list of variables.

    Args:
        unrolled_df: The DataFrame to roll back.
        variables: A list of Variable objects.

    Returns:
        Rolled DataFrame with array variables stored in single columns.

    Raises:
        TypeError: If unrolled_df is not a DataFrame, variables is not a list,
            variables is not a list of Variables, or the DataFrame is missing
            required columns.
    """
    if not isinstance(unrolled_df, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")
    if not isinstance(variables, list):
        raise TypeError('Not given a list')
    elif np.any([not isinstance(var, (IntVariable, FloatVariable, ArrayVariable,  CategoricalVariable)) for var in variables]):
        raise TypeError('The list is not a list of Variables')
        
    vars_to_roll = [var for var in variables if isinstance(var, ArrayVariable)]
    unrollable_names = [var.name for var in variables if not isinstance(var, ArrayVariable)]
    
    rolled_df = pd.DataFrame([])
    if unrollable_names:
        # get the names of unrollable variables for later use        
        if np.any([unrollable_name not in unrolled_df for unrollable_name in unrollable_names]):
            raise TypeError(f"The given DataFrame does not contain the columns: {[unrollable_name for unrollable_name in unrollable_names if unrollable_name not in unrolled_df]}")
        rolled_df = unrolled_df[unrollable_names].copy()

    for var in vars_to_roll:
        unrolled_name = generate_names(var.name, var.shape)
        # the data needs to have all of the unrolled data
        if any([var_name not in unrolled_df for var_name in unrolled_name]):
            raise TypeError(f"The given DataFrame does not contain the unrolled var names for {var.name}")
        rolled_df[var.name] = [row_data.reshape(var.shape) for row_data in unrolled_df[unrolled_name].to_numpy()]

    response_names = [var.name for var in variables]
    return rolled_df[response_names]
    

def check_rolled_data_frame_against_variable_shapes(input_df: pd.DataFrame,  variables: typing.List[Variable]) -> pd.DataFrame:
    """Checks DataFrame shapes against expected variable shapes.

    Args:
        input_df: The DataFrame to check.
        variables: A list of Variable objects.

    Raises:
        TypeError: If input_df is not a DataFrame, variables is not a list,
            or variables is not a list of Variables.
        AssertionError: If the shapes in the DataFrame do not match expected shapes.
    """   
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")
    if not isinstance(variables, list):
        raise TypeError('Not given a list')
    elif np.any([not isinstance(var, (IntVariable, FloatVariable, ArrayVariable,  CategoricalVariable)) for var in variables]):
        raise TypeError('The list is not a list of Variables')   
    

    shapes = get_variable_shape_in_data_frame(input_df, variables)
    for var, var_shape in zip(variables, shapes):
        if isinstance(var, ArrayVariable):
            if var.shape != var_shape:
                raise AssertionError(f"Column {var.name} has shape {var_shape} but it should have shape {var.shape}.")
        else:
            print(isinstance(var, ArrayVariable))
            if not len(var_shape)==0:
                raise AssertionError(f"Column {var.name} has shape {var_shape} but it should have shape ().")
    
def get_variable_shape_in_data_frame(input_df:pd.DataFrame, variables: typing.List[Variable]) -> typing.List[typing.Tuple]:    
    """Gets the shape(s) of the variables from the given DataFrame.

    Args:
        input_df: The DataFrame to check.
        variables: A list of Variable objects.

    Returns:
        A list of shapes, one for each variable in variables.

    Raises:
        TypeError: If input_df is not a DataFrame, variables is not a list,
            or variables is not a list of Variables.
        ValueError: If a variable is not in the DataFrame.
    """
    
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("Input is not a DataFrame")
    if not isinstance(variables, list):
        raise TypeError('Not given a list')
    elif np.any([not isinstance(var, (IntVariable, FloatVariable, ArrayVariable,  CategoricalVariable)) for var in variables]):
        raise TypeError('The list is not a list of Variables')   
    vars_names = [var.name for var in variables]
    col_names = input_df.columns
    for var_name in vars_names:
        if var_name not in col_names:
            raise ValueError(f"{var_name} is not in the given DataFrame")        
    return [input_df[var.name].iloc[0].shape if hasattr(input_df[var.name].iloc[0],'shape') else () for var in variables]
