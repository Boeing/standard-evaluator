""" Implementing the Standard Evaluator for groups

Groups are different in that they have components within them. When we find out what all the
options for the group we have to take into account the options for all the components in the
group.
    """

import inspect
import typing

from standard_evaluator import StandardBase, OptionsDictionaryUnit, opt_dict_unit_merge

class StandardGroup(StandardBase):
    """A class to be used in conjunction with OpenMDAO groups to have a common way of dealing with options.

    This class implements the ability of the system to keep track of options,
    and also pre-defines that this component depends on the 'class_options'.
    """

    def initialize(self):
        """Initialize the instance by defining that this will use `class_options`, 
        and 'metadata', and set the 'metadata' options to `_MetaData` by default."""
        super().initialize()

    def full_options(self) -> OptionsDictionaryUnit:
        """An instance method to return all options for this class.
        
        Often this will be just a call to the `required_options` class method. But for example
        in group the full set of options that are valid are depending on the components in the
        instance, so this has to be calculated after an instance is created, and the `setup()`
        method has been called.

        Returns:
            OptionsDictionaryUnit -- All valid options for this class.
        """
        
        options_list = []
        for subsys in self._subsystems_myproc:
            if isinstance(subsys, StandardBase):
                options_list.append(subsys.full_options())
        
        return opt_dict_unit_merge(options_list)

    def current_options(self) -> OptionsDictionaryUnit:
        """Return all the current options. 
        This is a combination between the original options (defaults) and values that
        were overwritten by the end user.

        Returns:
            OptionsDictionaryUnit -- The combination of all options
        """
        options_list = []
        for subsys in self._subsystems_myproc:
            if isinstance(subsys, StandardBase):
                options_list.append(subsys.current_options())
        
        return opt_dict_unit_merge(options_list)

    def get_interface(self) -> dict:
        """Generate a dictionary with all the information known about this component

        Returns:
            dict: Dictionary with all information known about the component
        """
        info = super().get_interface()

        info["components"] = []
        # Since this is a group we need to collect the information for all subsystems
        for subsys in self._subsystems_myproc:
            if isinstance(subsys, StandardBase):
                info["components"].append(subsys.get_interface())
        info["general"]["group"] = True

        return info
