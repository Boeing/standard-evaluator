import re
import importlib
import pprint
import json
import typing
from typing import List, Optional, Dict, Tuple, Union, Literal
import copy
import inspect
from enum import Enum
from collections.abc import Callable
from base64 import b64encode
import inspect
import pathlib

import json_numpy
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
from numpy import generic, ndarray
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from numpydantic import NDArray, Shape

import openmdao
import openmdao.api as om
from openmdao.core.constants import _ReprClass

try:
    import dymos
    import aviary
    from aviary.variable_info.enums import ProblemType, EquationsOfMotion, LegacyCode
    from aviary.utils.aviary_values import AviaryValues
    from aviary.subsystems.propulsion.engine_deck import EngineDeck
    _AVIARY_AVAILABLE = True
except ImportError:
    _AVIARY_AVAILABLE = False
    dymos = None
    aviary = None
    ProblemType = None
    EquationsOfMotion = None
    LegacyCode = None
    AviaryValues = None
    EngineDeck = None


def _require_aviary():
    """Raise ImportError if aviary is not available."""
    if not _AVIARY_AVAILABLE:
        raise ImportError(
            "aviary and dymos are required for this functionality. "
            "Install them with: pip install standard-evaluator[aviary]"
        )


class AviaryEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        _require_aviary()
        super().__init__(*args, **kwargs)

    def default(self, obj):
        if isinstance(obj, type):
            return {'__type__': str(obj)}
        elif isinstance(obj, set):
            return {'__set__': list(obj)}
        elif isinstance(obj, Enum):
            return {'__enum__': {'type': str(type(obj)),
                                 'value': repr(obj),
                                 'module': str(inspect.getmodule(obj))}}
        elif isinstance(obj, (ndarray, generic)):
            data = obj.data if obj.flags["C_CONTIGUOUS"] else obj.tobytes()
            return {
                "__numpy__": b64encode(data).decode(),
                "dtype": dtype_to_descr(obj.dtype),
                "shape": obj.shape,
            }
        elif isinstance(obj, tuple):
            return {'__tuple__': list(obj)}
        elif isinstance(obj, _ReprClass):
            return {'__repr_class__': str(obj)}
        elif isinstance(obj, Callable):
            # For a callable (function) we need to get the name and module so we can
            # load the function.
            func_name = str(obj).split(' ')[1]
            module_name = str(inspect.getmodule(obj)).split("'")[1]
            return {'__Callable__': {
                'name': func_name,
                'module': module_name,
            }}
        elif isinstance(obj, type(...)):
            return {'__Ellipsis__': 'Use an ellipsis'}
        elif isinstance(obj, pathlib.WindowsPath):
            # A Windows path can easily be translated into a string
            return {'__pathlib.WindowsPath__': str(obj)}
        elif isinstance(obj, AviaryValues):
            # An AviaryValues instance can be converted into a dictionary which
            # then gets unrolled recursively
            return {'__aviary_values__': dict(obj)}
        elif isinstance(obj, om.OptionsDictionary):
            # We want to make sure we can also handle descendants of the OptionsDictionary.
            # Therefore we are storing the actual type of the class
            return {'__openmdao.api.OptionsDictionary__':
                    {'type': str(type(obj)),
                     'value': obj.__getstate__()}
                }
        elif isinstance(obj, aviary.subsystems.aerodynamics.aerodynamics_builder.CoreAerodynamicsBuilder):
            return {'__CoreAerodynamicsBuilder__': 'Not implemented'}
        elif isinstance(obj, aviary.subsystems.propulsion.engine_deck.EngineDeck):
            return {'__EngineDeck__': {'type': str(type(obj)),
                                 'module': str(inspect.getmodule(obj)),
                                 'options': obj.options}}
        elif isinstance(obj, aviary.subsystems.propulsion.propulsion_builder.CorePropulsionBuilder):
            return {'__CorePropulsionBuilder__': 'Not implemented'}
        elif isinstance(obj, dymos.Radau):
            return {'__dymos.Radau__': 'Not implemented'}
        elif isinstance(obj, dymos.transcriptions.grid_data.GridData):
            return {'__dymos.transcriptions.grid_data.GridData__': 'Not implemented'}
        print(f'class type {type(obj)} from module {inspect.getmodule(obj)}')
        # Let the base class default method raise the TypeError
        return super().default(obj)