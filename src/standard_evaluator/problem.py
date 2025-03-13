"""
Definition of optimization problems information Pydantic classes.

Created Nov. 14, 2024

@author Joerg Gablonsky
"""

import re
from typing import List, Optional, Dict, Tuple, Union, Literal
import typing
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, field_serializer
from numpydantic import NDArray, Shape
from standard_evaluator import unique_names

MAXINT = 2**63

# Generating names of array variables
# We want to generate variable / response names that reflect that we have an array.
def generate_names(name: str, shape: tuple) -> typing.List[str]:
    """Generate names for variables / responses that reflect that they are arrays

    The names generated use NumPy array syntax. For example, a variable input that
    is a 1-d array of length 2 (shape (2,)) will generate this list:

    ['input[0]', 'input[1]']

    Parameters
    ----------
    name : str
        Name of the variable or response
    shape : tuple
        Shape of the variable or response. This is a NumPy shape

    Returns
    -------
    list[str]
        List of all the variables / responses represented by the array

    Raises
    ------
    TypeError
        If the name is not a string, or the shape is not a tuple
    """
    if not isinstance(name, str):
        raise TypeError("Name is not a string")
    if not isinstance(shape, tuple):
        raise TypeError("Shape is not a tuple")

    new_names = []
    for idx in np.ndindex(shape):
        new_names.append(f"{name}[" + ",".join([str(value) for value in idx]) + "]")
    return new_names


class FloatVariable(BaseModel):
    name: str = Field(description="Name of the variable.")
    default: Optional[float] = Field(
        default=None, description="Default value for this variable"
    )
    bounds: Optional[Tuple[float, float]] = Field(
        default=(-np.inf, np.inf), description="Lower and upper bounds of the variable"
    )
    shift: Optional[float] = Field(
        default=0.0,
        description="Shift value to be used for this variable.",
    )
    scale: Optional[float] = Field(
        default=1.0,
        description="Scale value to be used for this variable." + "Cannot be zero.",
    )
    units: Optional[str] = None
    description: Optional[str] = Field(
        default="",
        description="A description of the variable.",
    )
    options: Dict = Field(
        default_factory=dict, description="Options for the variable."
    )
    class_type: Literal["float"] = Field(default="float", description="Class marker")

    @field_validator("bounds")
    def validate_bounds(cls, var):
        if var is None:
            return (-np.inf, np.inf)
        if var[0] > var[1]:
            raise ValueError(
                f"Lower bounds are larger than upper bounds: {var[0]} > {var[1]}"
            )
        return var

    @field_validator("scale")
    def validate_scale(cls, scale):
        if scale == 0:
            raise ValueError(f"Scale cannot be 0.")
        return scale

    @field_validator("name")
    def check_name(cls, name: str) -> str:
        # We want to make sure the following checks are only done when
        # the name is not an empty string
        if len(name) > 0:
            if re.search(r"\s", name):
                raise ValueError(f"Name cannot contain white spaces.")
        return name


class IntVariable(FloatVariable):
    default: Optional[int] = Field(
        default=None, description="Default value for this variable"
    )
    bounds: Optional[Tuple[int, int]] = Field(
        default=[-MAXINT, MAXINT], description="Lower and upper bounds of the variable"
    )
    shift: Optional[int] = Field(
        default=0,
        description="NumPy array with the shift values to be used for the array variable.",
    )
    scale: Optional[int] = Field(
        default=1,
        description="NumPy array with the scale values to be used for the array variable. "
        + "Cannot be zero.",
    )
    class_type: Literal["int"] = Field(default="int", description="Class marker")

    @field_validator("bounds")
    def validate_bounds(cls, var):
        if var is None:
            return (-MAXINT, MAXINT)
        if var[0] > var[1]:
            raise ValueError(
                f"Lower bounds are larger than upper bounds: {var[0]} > {var[1]}"
            )
        return var


class ArrayVariable(FloatVariable):
    """Class defining array variables. The underlying data type are NumPy float64 arrays

    Raises:
        ValueError: Lower bounds greater than upper bounds for at least one element
        ValueError: Neither shape or default are set
        ValueError: Scale for at least one element is set to 0.
        ValueError: Inconsistent shapes for elements in the class
    """

    shape: Optional[Tuple[int, ...]] = Field(
        default=None, description="Shape of the arrays"
    )
    default: Optional[NDArray[Shape["*,..."], np.float64]] = Field(
        default=None,
        description="NumPy array with the default values to be used for the array variable",
    )
    bounds: Optional[
        Tuple[
            Union[float, int, NDArray[Shape["*,..."], np.float64]],
            Union[float, int, NDArray[Shape["*,..."], np.float64]],
        ]
    ] = Field(default=(None, None), description="Lower and upper bounds of the arrays")
    shift: Optional[Union[float, int, NDArray[Shape["*,..."], np.float64]]] = Field(
        default=None,
        description="NumPy array with the shift values to be used for the array variable.",
    )
    scale: Optional[Union[float, int, NDArray[Shape["*,..."], np.float64]]] = Field(
        default=None,
        description="NumPy array with the scale values to be used for the array variable. "
        + "Cannot be zero.",
    )
    class_type: Literal["floatarray"] = "floatarray"

    @field_validator("bounds")
    def validate_bounds(cls, var):
        if var is None:
            return (None, None)
        if np.any(var[0] > var[1]):
            raise ValueError(
                f"Lower bounds are larger than upper bounds: {var[0]} > {var[1]}"
            )
        return var

    @field_validator("scale")
    def validate_scale(cls, scale):
        # Because we only want to do this validation once we have a shape defined it
        # happens in the check_shape method
        return scale

    def _expand(self, value, scaler=1.0):
        if value is not None:
            if isinstance(value, (float, int)):
                value = np.ones(self.shape) * value
        else:
            value = np.ones(self.shape) * scaler
        return value

    @model_validator(mode="after")
    def check_shape(self):
        shape = self.shape
        default = self.default
        shift = self.shift
        scale = self.scale

        if shape is None:
            # If no shape is defined  default must be set. If default is
            # set it's shape will define the shape. If neither is set we throw an error
            if default is not None:
                shape = default.shape
            else:
                raise ValueError(
                    f"Shape for this variable is not set, and neither are default or value"
                )
            self.shape = shape

        # If shift is set we want to check if only a single value is set. If that is the
        # case we expand it to the full shape.
        shift = self._expand(shift, 0.0)
        self.shift = shift

        # If scale is set we want to check if only a single value is set. If that is the
        # case we expand it to the full shape.
        scale = self._expand(scale)

        # We have to make sure none of the values of scale are 0.
        if np.any(scale == 0.0):
            raise ValueError(f"Scale for one element in the matrix is set to 0.0")

        self.scale = scale

        # Check that the bounds are set correctly
        (low_bound, up_bound) = self.bounds
        low_bound = self._expand(low_bound, -np.inf)
        up_bound = self._expand(up_bound, np.inf)
        self.bounds = (low_bound, up_bound)

        fields = [default, shift, scale, low_bound, up_bound]
        field_names = ["default", "shift", "scale", "lower bound", "upper bound"]
        # Check if default is not None. If default is set we check that the shape is
        # set correctly.
        for local_field, field_name in zip(fields, field_names):
            if local_field is not None:
                if local_field.shape != shape:
                    raise ValueError(
                        f"Shape for this variable is {shape}, but {field_name} has shape {local_field.shape}"
                    )

        return self
    
    # If the bounds are set to (-inf, inf) we don't need to store them.
    @field_serializer('bounds')
    def serialize_bounds(self, bounds: Tuple[
            NDArray[Shape["*,..."], np.float64],
            NDArray[Shape["*,..."], np.float64],
        ]):
        if np.all(np.isinf(bounds[0])) & np.all(np.isinf(bounds[0])):
            return(None)
        else:
            lower = bounds[0]
            upper = bounds[1]
            if np.all(upper == upper.ravel()[0]):
                upper = upper.ravel()[0]
            if np.all(lower == lower.ravel()[0]):
                lower = lower.ravel()[0]
            bounds = (lower, upper)
            return(bounds)

    # If shift is set to all zeros we don't need to save them
    @field_serializer('shift')
    def serialize_shift(self, shift: NDArray[Shape["*,..."], np.float64]):
        if np.all(shift == 0.0):
            return None
        else:
            return shift

    # If scale is set to all ones we don't need to save them
    @field_serializer('scale')
    def serialize_scale(self, scale: NDArray[Shape["*,..."], np.float64]):
        if np.all(scale == 1.0):
            return None
        else:
            return scale

# Define the Union of the different variable types. Note that we use that for responses as well
Variable = Union[FloatVariable, IntVariable, ArrayVariable]

class OptProblem(BaseModel):
    """
    Represents information about an optimization problem.

    Args:
        name: The name of the problem. Defaults to 'opt_problem'
        variables: A list of input variables.
        responses: A list of output variables.
        objectives: Names of the objective(s) for the optimization problem. Must be either variables or responses defined in the problem..
        constraints: Names of the constraints of the optimization problem. Must be responses defined in the problem. To define bounds on variables use the variable bounds.
        description: A description of the optimization problem. To define mathematical symbols use markdown syntax.
        cite: Listing of relevant citations that should be referenced when publishing work that uses this class.
        options: Additional options for the problem.
    """

    name: str = Field(
        default="opt_problem",
        description='The name of the problem. Defaults to "opt_problem"',
    )
    class_type: Literal["OptProblem"] = "OptProblem"
    variables: List[Variable] = Field(description="Input variables")
    responses: List[Variable] = Field(description="Output variables")
    objectives: List[str] = Field(
        default_factory=list,
        description="Names of the objective(s) for the optimization problem. Must be either variables or responses defined in the problem.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Names of the constraints of the optimization problem. Must be responses defined in the problem. To define bounds on variables use the variable bounds.",
    )
    description: Optional[str] = Field(
        default=None,
        description="A description of the optimization problem. To define mathematical symbols use markdown syntax.",
    )
    cite: Optional[str] = Field(
        default=None,
        description="Listing of relevant citations that should be referenced when publishing work that uses this class.",
    )
    options: Dict = Field(
        default_factory=dict, description="Additional options for the problem."
    )

    @field_validator("variables", "responses")
    def validate_outputs(cls, var):
        unique_names(var)
        return var

    def unroll_names(self, elements: List[Variable]) -> List[str]:
        all_names = []
        for local_element in elements:
            name = local_element.name
            if isinstance(local_element, ArrayVariable):
                array_names = generate_names(name, local_element.shape)
                all_names += array_names
            else:
                all_names.append(name)
        return all_names

    @model_validator(mode="after")
    def check_problem(self):

        # This method needs to be updated to allow using an element from a FloatArray
        variable_names = self.unroll_names(self.variables)
        response_names = self.unroll_names(self.responses)
        elements = set(variable_names + response_names)
        if self.objectives is not None:
            # Check if all the objectives are either a variable or response
            for name in self.objectives:
                if name not in elements:
                    raise ValueError(
                        f"{name} is defined as an objective, but not defined as a variable or response."
                    )

        if self.constraints is not None:
            # Check if all the objectives are either a variable or response
            for name in self.constraints:
                if name not in response_names:
                    if name in variable_names:
                        raise ValueError(
                            f"{name} is defined as a constraint, but defined as a variable. Please define bounds on the variable itself. Constraints should only be responses."
                        )
                    raise ValueError(
                        f"{name} is defined as a constraint, but not defined as a response."
                    )

        return self