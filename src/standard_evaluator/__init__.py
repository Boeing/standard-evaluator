"""A Python module for the standard evaluator"""

from .version import __version__
from .standard_base import StandardBase
from .standard_base import OptionsDictionaryUnit
from .standard_base import opt_dict_unit_merge
from .standard_evaluator import StandardEval
from .standard_group import StandardGroup
from .utilities import unique_names
from .problem import ArrayVariable, FloatVariable, IntVariable, Variable, MAXINT
from .problem import OptProblem, CategoricalVariable
from .evaluator import EvaluatorInfo, GroupInfo, EquationInfo, JoinedInfo
from .converters import evaluator_info_to_opt_problem, opt_problem_to_evaluator_info
from .aviary_encoder import AviaryEncoder

# from .om_converter import get_input_output_info, convert_om_variable, convert_om_variable_list
# from .om_converter import get_interface_component_standard, get_interface
from .om_converter import get_state, set_state, save_state, load_state
from .om_converter import get_interface, show_structure, convert_om_var, convert_dict
from .om_converter import (
    create_problem,
    create_explicit_component,
    load_assembly,
    save_assembly,
)
from .om_converter import set_opt_problem, get_opt_problem

__all__ = [
    "__version__",
    "StandardBase",
    "OptionsDictionaryUnit",
    "opt_dict_unit_merge",
    "StandardEval",
    "StandardGroup",
    "unique_names",
    "ArrayVariable",
    "FloatVariable",
    "CategoricalVariable",
    "IntVariable",
    "Variable",
    "MAXINT",
    "OptProblem",
    "EvaluatorInfo",
    "GroupInfo",
    "EquationInfo",
    "JoinedInfo",
    "evaluator_info_to_opt_problem",
    "opt_problem_to_evaluator_info",
    "AviaryEncoder",
    # "get_input_output_info",
    # "convert_om_variable",
    # "convert_om_variable_list",
    # "get_interface_component_standard",
    "convert_dict",
    "get_interface",
    "get_state",
    "set_state",
    "save_state",
    "load_state",
    "show_structure",
    "create_problem",
    "create_explicit_component",
    "load_assembly",
    "save_assembly",
    "convert_om_var",
    "get_opt_problem",
    "set_opt_problem",
]
