""" Implementing the Standard Evaluator for components    """

import inspect
import typing

from standard_evaluator import StandardBase
try:
    from aviary.variable_info.variable_meta_data import _MetaData
    _AVIARY_AVAILABLE = True
except ImportError:
    _AVIARY_AVAILABLE = False
    _MetaData = None


def _require_aviary():
    """Raise ImportError if aviary is not available."""
    if not _AVIARY_AVAILABLE:
        raise ImportError(
            "aviary is required for StandardEval. "
            "Install it with: pip install standard-evaluator[aviary]"
        )


class StandardEval(StandardBase):
    """A class to be used in conjunction with OpenMDAO components to have a common way of dealing with options.

    This class implements the ability of the system to keep track of options,
    and also pre-defines that this component depends on the 'class_options'.
    """

    def initialize(self):
        """Initialize the instance by defining that this will use `class_options`, 
        and 'metadata', and set the 'metadata' options to `_MetaData` by default."""
        _require_aviary()
        super().initialize()
        # Define an option to set the meta data. By default it will be set to `_MetaData` from
        # Aviary
        self.options.declare(
            "metadata",
            types=dict,
            desc="A dictionary containing metadata and default values to use for variables or responses",
        )
        self.options["metadata"] = _MetaData

    def _lookup_info(
        self,
        name: str,
        val: typing.Optional[typing.Any] = None,
        units: typing.Optional[str] = None,
        desc: typing.Optional[str] = None,
    ) -> typing.Tuple[typing.Any, str, str]:
        """Helper method to look up information based on the meta data.

        Arguments:
            name {str} -- Name of the element to look up information for

        Keyword Arguments:
            val {typing.Optional[typing.Any]} -- Value to use. If None value will be looked up (default: {None})
            units {typing.Optional[str]} --Units to use. If None units will be looked up (default: {None})
            desc {typing.Optional[str]} -- Description to use. If None description will be looked up (default: {None})

        Returns:
            typing.Tuple[typing.Any, str, str] -- Updated values, units, and description. 
        """
        meta = self.options["metadata"][name]
        if units is None:
            units = meta["units"]
        if desc is None:
            desc = meta["desc"]
        if val is None:
            val = meta["default_value"]
        return (val, units, desc)
