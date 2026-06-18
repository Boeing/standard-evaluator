"""A set of evaluators that evaluate functions with known solutions. These
evaluators can be useful for testing optimization algorithms or as a standin
while testing ``standard_evaluator`` functionality. All evaluators defined here
have a ``known_solution`` property which returns a DataFrame containing the
site(s) and response values at the known solution.
"""

import importlib
import inspect
import os
import sys

from standard_evaluator.evaluators.test_evaluator import TestEvaluator as _TestEvaluatorBase

__all__ = []

# Get the test evaluator directory
_base_dir = os.path.dirname(__file__)
# Loop through all files/directories in this folder
for _fname in sorted(os.listdir(_base_dir)):
    # Get absolute path to file/directory
    _abs_path = os.path.join(_base_dir, _fname)

    # Skip directories
    if not os.path.isfile(_abs_path):
        continue

    # Get name and file extension
    _name, _ext = os.path.splitext(_fname)
    # Skip non-python files and this file
    if _ext != ".py" or _name == "__init__":
        continue

    # Import file
    _mod = importlib.import_module("." + _name, __package__)

    # Loop through all members in this module
    for _mem_name, _obj in inspect.getmembers(_mod):
        # Skip non-classes and the abstract TestEvaluator base class
        if not inspect.isclass(_obj) or _mem_name == "TestEvaluator":
            continue

        # Only include classes that inherit from TestEvaluator
        if not issubclass(_obj, _TestEvaluatorBase):
            continue

        # Only include classes defined in this module (not imported base classes)
        if _obj.__module__ != _mod.__name__:
            continue

        # Add class to the test module
        setattr(sys.modules[__name__], _mem_name, _obj)
        # Add class to __all__
        __all__.append(_mem_name)

# Verify that all 43 expected classes were discovered
_EXPECTED_COUNT = 43
if len(__all__) < _EXPECTED_COUNT:
    raise ImportError(
        f"Expected at least {_EXPECTED_COUNT} test evaluator classes but only "
        f"discovered {len(__all__)}. Missing classes may indicate failed imports "
        f"in the test evaluator modules."
    )
