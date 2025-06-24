"""
Definition of optimization problems information Pydantic classes.

Created Nov. 14, 2024

@author Joerg Gablonsky
"""

import re
from typing import List, Optional, Dict, Tuple, Union, Literal, Any
import typing
import numpy as np
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    field_serializer,
)
from numpydantic import NDArray, Shape
from standard_evaluator import unique_names

MAXINT = 2**63


def generate_names(name: str, shape: tuple) -> typing.List[str]:
    """Generate names for variables / responses that reflect that they are arrays.

    The names generated use NumPy array syntax. For example, a variable input that
    is a 1-d array of length 2 (shape (2,)) will generate this list:

    ['input[0]', 'input[1]']

    Parameters
    ----------
    name : str
        Name of the variable or response.
    shape : tuple
        Shape of the variable or response. This is a NumPy shape.

    Returns
    -------
    list[str]
        List of all the variables / responses represented by the array.

    Raises
    ------
    TypeError
        If the name is not a string, or the shape is not a tuple.
    """
    if not isinstance(name, str):
        raise TypeError("Name is not a string")
    if not isinstance(shape, tuple):
        raise TypeError("Shape is not a tuple")

    new_names = []
    for idx in np.ndindex(shape):
        new_names.append(f"{name}[" + ",".join([str(value) for value in idx]) + "]")
    return new_names


class FloatVariable(BaseModel, validate_assignment=True):
    """Class representing a floating-point variable with bounds and scaling.

    Attributes:
        name: Name of the variable.
        default: Default value for this variable.
        bounds: Lower and upper bounds of the variable.
        shift: Shift value to be used for this variable.
        scale: Scale value to be used for this variable. Cannot be zero.
        units: Units of the variable.
        description: A description of the variable.
        options: Additional options for the variable.
        class_type: Class marker for identifying the variable type.
    """

    name: str = Field(description="Name of the variable.")
    default: Optional[float] = Field(
        default=None, description="Default value for this variable", 
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
        description="Scale value to be used for this variable. Cannot be zero.",
    )
    units: Optional[str] = None
    description: Optional[str] = Field(
        default="",
        description="A description of the variable.",
    )
    options: Dict = Field(default_factory=dict, description="Options for the variable.")
    class_type: Literal["float"] = Field(default="float", description="Class marker")

#    class Config:
#        validate_assignment = True

    def calculate_default(self, overwrite: bool = True) -> None:
        """Calculate the default value based on the bounds.

        If the overwrite flag is not true and default is set, the method exits.
        The default value is set based on the bounds of the variable.

        Parameters
        ----------
        overwrite : bool
            Flag indicating whether to overwrite the default value if it is already set.
        """
        low, high = self.bounds
        if not overwrite and self.default is not None:
            return
        if low == -np.inf:
            if high == np.inf:
                default_value = 0
            else:
                default_value = high
        elif high == np.inf:
            default_value = low
        else:
            default_value = (low + high) / 2
        self.default = default_value

    @field_validator("bounds")
    def validate_bounds(cls, var):
        """Validate the bounds of the variable.

        If the bounds are not set, they default to (-inf, inf). Raises a ValueError
        if the lower bound is greater than the upper bound.

        Parameters
        ----------
        var : Tuple[float, float]
            The bounds to validate.

        Returns
        -------
        Tuple[float, float]
            The validated bounds.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """
        if var is None:
            return (-np.inf, np.inf)
        if var[0] > var[1]:
            raise ValueError(
                f"Lower bounds are larger than upper bounds: {var[0]} > {var[1]}"
            )
        return var

    @field_validator("scale")
    def validate_scale(cls, scale):
        """Validate the scale of the variable.

        Raises a ValueError if the scale is set to zero.

        Parameters
        ----------
        scale : float
            The scale to validate.

        Returns
        -------
        float
            The validated scale.

        Raises
        ------
        ValueError
            If the scale is zero.
        """
        if scale == 0:
            raise ValueError(f"Scale cannot be 0.")
        return scale

    @field_validator("name")
    def check_name(cls, name: str) -> str:
        """Check the validity of the variable name.

        Raises a ValueError if the name contains whitespace or is an empty string.

        Parameters
        ----------
        name : str
            The name to validate.

        Returns
        -------
        str
            The validated name.

        Raises
        ------
        ValueError
            If the name contains whitespace or is empty.
        """
        if len(name) > 0:
            if re.search(r"\s", name):
                raise ValueError(f"Name cannot contain white spaces.")
        return name


class IntVariable(FloatVariable, validate_assignment=True):
    """Class representing an integer variable with bounds and scaling.

    Inherits from FloatVariable and overrides specific attributes and methods
    for integer handling.
    """

    default: Optional[int] = Field(
        default=None, description="Default value for this variable"
    )
    bounds: Optional[Tuple[int, int]] = Field(
        default=[-MAXINT, MAXINT], description="Lower and upper bounds of the variable"
    )
    shift: Optional[int] = Field(
        default=0,
        description="Shift value to be used for this variable.",
    )
    scale: Optional[int] = Field(
        default=1,
        description="Scale value to be used for this variable. Cannot be zero.",
    )
    class_type: Literal["int"] = Field(default="int", description="Class marker")

    def calculate_default(self, overwrite: bool = True) -> None:
        """Calculate the default value based on the bounds for integer variables.

        If the overwrite flag is not true and default is set, the method exits.
        The default value is set based on the bounds of the variable.

        Parameters
        ----------
        overwrite : bool
            Flag indicating whether to overwrite the default value if it is already set.
        """
        if not overwrite and self.default is not None:
            return
        low, high = self.bounds
        if low == -MAXINT:
            if high == MAXINT:
                default_value = 0
            else:
                default_value = high
        elif high == MAXINT:
            default_value = low
        else:
            default_value = int(np.floor((low + high) / 2))
        self.default = default_value

    @field_validator("bounds")
    def validate_bounds(cls, var):
        """Validate the bounds of the integer variable.

        If the bounds are not set, they default to (-MAXINT, MAXINT). Raises a ValueError
        if the lower bound is greater than the upper bound.

        Parameters
        ----------
        var : Tuple[int, int]
            The bounds to validate.

        Returns
        -------
        Tuple[int, int]
            The validated bounds.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """
        if var is None:
            return (-MAXINT, MAXINT)
        if var[0] > var[1]:
            raise ValueError(
                f"Lower bounds are larger than upper bounds: {var[0]} > {var[1]}"
            )
        return var


class CategoricalVariable(FloatVariable, validate_assignment=True):
    """Class representing a categorical variable.

    Attributes:
        default: Default value for this variable.
        bounds: Values this variable can take on. Must be defined.
        shift: Shift value to be used for this variable. Not applicable for categorical variables.
        scale: Scale value to be used for this variable. Not applicable for categorical variables.
        class_type: Class marker for identifying the variable type.
    """

    default: Optional[Any] = Field(
        default=None, description="Default value for this variable"
    )
    bounds: List[Any] = Field(
        description="Values this variable or response can take on. Must be defined."
    )
    shift: Literal[None] = Field(
        default=None,
        description="Shift value to be used for this variable. Does not make sense for categorical variables.",
    )
    scale: Literal[None] = Field(
        default=None,
        description="Scale value to be used for this variable. Does not make sense for categorical variables.",
    )
    units: Optional[str] = None
    class_type: Literal["cat"] = Field(default="cat", description="Class marker")

    def calculate_default(self, overwrite: bool = True) -> None:
        """Calculate the default value for categorical variables.

        If the overwrite flag is not true and default is set, the method exits.
        For categorical variables, the first value in bounds is used as the default.

        Parameters
        ----------
        overwrite : bool
            Flag indicating whether to overwrite the default value if it is already set.
        """
        if not overwrite and self.default is not None:
            return
        self.default = self.bounds[0]

    @field_validator("bounds")
    def validate_bounds(cls, var):
        """Validate the bounds of the categorical variable.

        Raises a ValueError if no valid values are defined.

        Parameters
        ----------
        var : List[Any]
            The bounds to validate.

        Returns
        -------
        List[Any]
            The validated bounds.

        Raises
        ------
        ValueError
            If no valid values are defined.
        """
        if len(var) == 0:
            raise ValueError(
                f"Need to define at least one valid value for this variable"
            )
        return var

    @model_validator(mode="after")
    def check_default(self):
        """Check that the default value, if defined, is valid.

        Raises a ValueError if the default value is not in the bounds.

        Returns
        -------
        CategoricalVariable
            The instance of the class after validation.
        """
        bounds = self.bounds
        default = self.default
        if default is not None:
            if default not in bounds:
                raise ValueError(f"Default not in bounds: {default}")
        return self

class ArrayVariable(FloatVariable, validate_assignment=False):
    """Class defining array variables. The underlying data type are NumPy float64 arrays.

    Raises:
        ValueError: Lower bounds greater than upper bounds for at least one element.
        ValueError: Neither shape nor default are set.
        ValueError: Scale for at least one element is set to 0.
        ValueError: Inconsistent shapes for elements in the class.
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

    def calculate_default(self, overwrite: bool = True) -> None:
        """Calculate the default value based on the bounds for array variables.

        If the overwrite flag is not true and default is set, the method exits.
        The default value is calculated based on the bounds of the variable.

        Parameters
        ----------
        overwrite : bool
            Flag indicating whether to overwrite the default value if it is already set.
        """
        if not overwrite and self.default is not None:
            return
        low, high = self.bounds
        # Create a new array for the average
        default_value = np.where(
            (low == -np.inf) & (high == np.inf), 0,
            np.where(
                low == -np.inf, high,
                np.where(
                    high == np.inf, low,
                    (low + high) / 2
                )
            )
        )
        self.default = default_value
        
    @field_validator("bounds")
    def validate_bounds(cls, var):
        """Validate the bounds of the array variable.

        If the bounds are not set, they default to (None, None). Raises a ValueError
        if any lower bound is greater than the corresponding upper bound.

        Parameters
        ----------
        var : Tuple[Union[float, int, NDArray[Shape["*,..."], np.float64]], Union[float, int, NDArray[Shape["*,..."], np.float64]]]
            The bounds to validate.

        Returns
        -------
        Tuple[Union[float, int, NDArray[Shape["*,..."], np.float64]], Union[float, int, NDArray[Shape["*,..."], np.float64]]]
            The validated bounds.

        Raises
        ------
        ValueError
            If any lower bound is greater than the corresponding upper bound.
        """
        if var is None:
            return (None, None)
        if np.any(var[0] > var[1]):
            raise ValueError(
                f"Lower bounds are larger than upper bounds: {var[0]} > {var[1]}"
            )
        return var

    @field_validator("scale")
    def validate_scale(cls, scale):
        """Validate the scale of the array variable.

        This method does not perform any checks as validation occurs in the check_shape method.

        Parameters
        ----------
        scale : Union[float, int, NDArray[Shape["*,..."], np.float64]]
            The scale to validate.

        Returns
        -------
        Union[float, int, NDArray[Shape["*,..."], np.float64]]
            The validated scale.
        """
        return scale

    def _expand(self, value, scaler=1.0):
        """Expand a single value to match the shape of the array.

        If the value is None, it is replaced with an array of the specified shape filled with the scaler value.

        Parameters
        ----------
        value : Union[float, int, None]
            The value to expand.
        scaler : float, optional
            The value to fill the array if value is None. Defaults to 1.0.

        Returns
        -------
        NDArray[Shape["*,..."], np.float64]
            The expanded array.
        """
        if value is not None:
            if isinstance(value, (float, int)):
                value = np.ones(self.shape) * value
        else:
            value = np.ones(self.shape) * scaler
        return value

    @model_validator(mode="after")
    def check_shape(self):
        """Check and validate the shape of the array variable.

        This method ensures that the shape is defined and consistent across default, shift, scale, and bounds.

        Returns
        -------
        ArrayVariable
            The instance of the class after validation.

        Raises
        ------
        ValueError
            If the shape is not set and neither default nor value is defined.
            If any of the shapes for default, shift, scale, or bounds are inconsistent.
        """
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

    @field_serializer("bounds")
    def serialize_bounds(
        self,
        bounds: Tuple[
            NDArray[Shape["*,..."], np.float64],
            NDArray[Shape["*,..."], np.float64],
        ],
    ):
        """Serialize the bounds for storage.

        If the bounds are set to (-inf, inf), they are not stored. Otherwise, the bounds are
        serialized to a more compact form if possible.

        Parameters
        ----------
        bounds : Tuple[NDArray[Shape["*,..."], np.float64], NDArray[Shape["*,..."], np.float64]]
            The bounds to serialize.

        Returns
        -------
        Optional[Tuple[Union[float, int], Union[float, int]]]
            The serialized bounds, or None if they are set to (-inf, inf).
        """
        if np.all(np.isinf(bounds[0])) & np.all(np.isinf(bounds[0])):
            return None
        else:
            lower = bounds[0]
            upper = bounds[1]
            if np.all(upper == upper.ravel()[0]):
                upper = upper.ravel()[0]
            if np.all(lower == lower.ravel()[0]):
                lower = lower.ravel()[0]
            bounds = (lower, upper)
            return bounds

    @field_serializer("shift")
    def serialize_shift(self, shift: NDArray[Shape["*,..."], np.float64]):
        """Serialize the shift values for storage.

        If the shift is set to all zeros, it is not stored. Otherwise, the shift values are returned.

        Parameters
        ----------
        shift : NDArray[Shape["*,..."], np.float64]
            The shift values to serialize.

        Returns
        -------
        Optional[NDArray[Shape["*,..."], np.float64]]
            The serialized shift values, or None if they are all zeros.
        """
        if np.all(shift == 0.0):
            return None
        else:
            return shift

    @field_serializer("scale")
    def serialize_scale(self, scale: NDArray[Shape["*,..."], np.float64]):
        """Serialize the scale values for storage.

        If the scale is set to all ones, it is not stored. Otherwise, the scale values are returned.

        Parameters
        ----------
        scale : NDArray[Shape["*,..."], np.float64]
            The scale values to serialize.

        Returns
        -------
        Optional[NDArray[Shape["*,..."], np.float64]]
            The serialized scale values, or None if they are all ones.
        """
        if np.all(scale == 1.0):
            return None
        else:
            return scale

# Define the Union of the different variable types. Note that we use that for responses as well
Variable = Union[FloatVariable, IntVariable, ArrayVariable, CategoricalVariable]


from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

class OptProblem(BaseModel):
    """
    Represents information about an optimization problem.

    Args:
        name: The name of the problem. Defaults to 'opt_problem'.
        variables: A list of input variables.
        responses: A list of output variables.
        objectives: Names of the objective(s) for the optimization problem. Must be either variables or responses defined in the problem.
        constraints: Names of the constraints of the optimization problem. Must be responses defined in the problem. To define bounds on variables use the variable bounds.
        description: A description of the optimization problem. To define mathematical symbols use markdown syntax.
        cite: Listing of relevant citations that should be referenced when publishing work that uses this class.
        options: Additional options for the problem.
    """

    name: str = Field(
        default="opt_problem",
        description='The name of the problem. Defaults to "opt_problem".',
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

    def calculate_default(self, overwrite: bool = True) -> None:
        """Calculate and set default values for all input variables.

        This method iterates through the list of variables and calls the
        `calculate_default` method on each variable to compute its default value.

        Parameters
        ----------
        overwrite : bool
            A flag indicating whether to overwrite existing default values. Defaults to True.
        """
        for variable in self.variables:
            variable.calculate_default(overwrite=overwrite)

    def variable_names(self) -> List[str]:
        """Retrieve the names of all input variables.

        This method returns a list of names for all variables defined in the
        `variables` attribute.

        Returns
        -------
        List[str]
            A list of names of the input variables.
        """
        return [element.name for element in self.variables]
    
    def response_names(self) -> List[str]:
        """Retrieve the names of all output variables.

        This method returns a list of names for all variables defined in the
        `responses` attribute.

        Returns
        -------
        List[str]
            A list of names of the output variables.
        """
        return [element.name for element in self.responses]
    
    def set_defaults(self, new_defaults: dict) -> None:
        """Set new default values for input variables based on a provided dictionary.

        This method updates the default values of the variables in the `variables`
        list if their names match the keys in the `new_defaults` dictionary.

        Parameters
        ----------
        new_defaults : dict
            A dictionary where keys are variable names and values are the new default values to set.
        """
        for index in range(len(self.variables)):
            element = self.variables[index]
            if element.name in new_defaults:
                self.variables[index].default = new_defaults[element.name]


    @field_validator("variables", "responses")
    def validate_outputs(cls, var):
        """Validate the names of the variables and responses.

        This method checks that the names of the variables and responses are unique
        using the `unique_names` function.

        Parameters
        ----------
        var : List[Variable]
            The list of variables or responses to validate.

        Returns
        -------
        List[Variable]
            The validated list of variables or responses.
        """
        unique_names(var)
        return var

    def unroll_names(self, elements: List[Variable]) -> List[str]:
        """Unroll the names of the variables and responses into a flat list.

        This method generates a list of names for all elements, including
        expanding array variables into their individual components.

        Parameters
        ----------
        elements : List[Variable]
            The list of variables or responses to unroll.

        Returns
        -------
        List[str]
            A flat list of names for the variables and responses.
        """
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
        """Check the validity of the optimization problem setup.

        This method verifies that all objectives and constraints are defined correctly.
        It checks that objectives are either variables or responses and that constraints
        are defined as responses.

        Returns
        -------
        OptProblem
            The instance of the class after validation.

        Raises
        ------
        ValueError
            If any objective is not defined as a variable or response.
            If any constraint is defined as a variable instead of a response.
        """
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