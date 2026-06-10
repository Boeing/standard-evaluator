# standard_evaluator.evaluators package
"""Evaluator classes for standard_evaluator.

This module re-exports all evaluator classes, excel utilities, and test
evaluator benchmark functions for convenient access.
"""

import importlib.util

from .abstract_evaluator import Evaluator
from .evaluator import PyEvaluator
from .numpy_evaluator import NumpyEvaluator
from .executable_evaluator import ExecutableEvaluator
from .shift_scale_evaluator import ShiftScaleEvaluator
from .open_mdao_evaluator import OpenMDAOEvaluator
from .excel_evaluator import ExcelEvaluator
from .test_evaluator import TestEvaluator

from .excel_utilities import (
    SpreadsheetModel,
    MacroDefinition,
    VarType,
    get_interface_from_excel_named_ranges,
    create_variable_from_excel_address,
)

# Optional MatlabEvaluator import - requires matlab.engine package
can_use_matlab = False
try:
    import matlab.engine

    from .matlab_evaluator import MatlabEvaluator

    can_use_matlab = True
except Exception:
    try:
        # try to determine if the problem is the matlab package or a version mismatch
        if importlib.util.find_spec("matlab.engine") is not None:
            print("matlabengine is installed! Check that matlab versions match.")
        else:
            print("matlabengine is NOT installed.")
    except Exception:
        print("matlabengine not available.")

# Import all test evaluator classes via test subpackage auto-discovery
from .test import __all__ as _test_all
from . import test as _test_module

# Build __all__ with all exported names
__all__ = [
    # Core evaluator classes
    "Evaluator",
    "PyEvaluator",
    "NumpyEvaluator",
    "ExecutableEvaluator",
    "ShiftScaleEvaluator",
    "OpenMDAOEvaluator",
    "ExcelEvaluator",
    "TestEvaluator",
    # Excel utilities
    "SpreadsheetModel",
    "MacroDefinition",
    "VarType",
    "get_interface_from_excel_named_ranges",
    "create_variable_from_excel_address",
    # Module-level flag
    "can_use_matlab",
]

# Add MatlabEvaluator if available
if can_use_matlab:
    __all__.append("MatlabEvaluator")

# Re-export all test evaluator classes into this namespace
for _name in _test_all:
    globals()[_name] = getattr(_test_module, _name)
    __all__.append(_name)
