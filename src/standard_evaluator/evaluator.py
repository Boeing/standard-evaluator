"""
Definition of evaluator information Pydantic classes.

Created Dec. 10, 2024

@author Joerg Gablonsky
"""

import re
from typing import List, Optional, Dict, Tuple, Union, Literal
import typing
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from numpydantic import NDArray, Shape

from standard_evaluator import Variable
from standard_evaluator import unique_names


class EvaluatorInfo(BaseModel):
    """
    Represents information about an evaluator.

    Args:
        name: The name of the evaluator.
        class_type: Identifier
        inputs: Input elements
        outputs: Output elements
        description: A description of the component. To define mathematical symbols use markdown syntax.
        cite: Listing of relevant citations that should be referenced when publishing work that uses this class.
        tool: Name of the tool wrapped
        evaluator_identifier: Unique identifier for the evaluator.
        version: Version of the evaluator.
        component_type: Component type (ExplicitComponent, ImplicitComponent, Group, etc.).
        options: Additional options for the component.
    """

    name: str = Field(description="The name of the evaluator")
    class_type: Literal["EvaluatorInfo"] = "EvaluatorInfo"
    inputs: List[Variable] = Field(description="Input elements")
    outputs: List[Variable] = Field(description="Output elements")
    description: Optional[str] = Field(
        default=None,
        description="A description of the component. To define mathematical symbols use markdown syntax.",
    )
    cite: Optional[str] = Field(
        default=None,
        description="Listing of relevant citations that should be referenced when publishing work that uses this class.",
    )
    tool: Optional[str] = Field(default=None, description="Name of the tool wrapped")
    evaluator_identifier: Optional[str] = Field(
        default=None, description="Unique identifier for the evaluator."
    )
    version: Optional[str] = Field(
        default=None, description="Version of the evaluator."
    )
    component_type: Optional[str] = Field(
        default=None,
        description="Component type (ExplicitComponent, ImplicitComponent, Group, etc.).",
    )
    options: Dict = Field(
        default_factory=dict, description="Additional options for the problem."
    )

    @field_validator("inputs", "outputs")
    def validate_outputs(cls, var):
        unique_names(var)
        return var

class EquationInfo(EvaluatorInfo):
    class_type: typing.Literal["EquationInfo"] = "EquationInfo"
    equations : typing.Union[str, typing.List[str]] = Field(description="String or list of strings containing the equation or equations that should be used to calculate the outputs.")


class GroupInfo(EvaluatorInfo):
    class_type: typing.Literal["GroupInfo"] = "GroupInfo"
    component_order: typing.List[str] = Field(description="List of the components in this group in the order they should be displayed")
    components: typing.Dict[str, "JoinedInfo"]
    promotions: typing.Dict[str, typing.List[typing.Tuple[str, str]]] = Field(default = {}, description="Dictionary of lists that map the name of an input / output in a component to an input / output of this group")
    linkage: typing.List[typing.Tuple[str, str]] = Field(default=[], description="Map that allows linkage between inputs and outputs between components inside this group. Needs to use component_name.element_name")

JoinedInfo = typing.Union[ EquationInfo, EvaluatorInfo, GroupInfo]
