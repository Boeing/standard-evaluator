""" Implementing the Standard Evaluator for components    """
import inspect
import typing
from typing import List

from openmdao.core.constants import _UNDEFINED
from openmdao.visualization.tables.table_builder import generate_table
from openmdao.utils.options_dictionary import OptionsDictionary

TAG_KEY = "Options:"

class OptionsDictionaryUnit(OptionsDictionary):
    """Expand the OptionsDictionary to allow the definition of Units

Arguments:
    OptionsDictionary {[type]} -- [description]

Raises:
    KeyError: [description]
    TypeError: [description]
    TypeError: [description]
    ValueError: [description]
    ValueError: [description]

Returns:
    [type] -- [description]
    """
    def declare(self, name, default=_UNDEFINED, values=None, types=None, desc='',
                units="unitless", 
                upper=None, lower=None, allow_none=False, recordable=True,
                deprecation=None):
        r"""
        Declare an option.

        The value of the option must satisfy the following:
        1. If values only was given when declaring, value must be in values.
        2. If types only was given when declaring, value must satisfy isinstance(value, types).
        3. It is an error if both values and types are given.

        Parameters
        ----------
        name : str
            Name of the option.
        default : object or Null
            Optional default value that must be valid under the above 3 conditions.
        values : set or list or tuple or None
            Optional list of acceptable option values.
        types : type or tuple of types or None
            Optional type or list of acceptable option types.
        desc : str
            Optional description of the option.
        units : str
            Optional definition of units. Defaults to `unitless`
        upper : float or None
            Maximum allowable value.
        lower : float or None
            Minimum allowable value.
        check_valid : function or None
            User-supplied function with arguments (name, value) that raises an exception
            if the value is not valid.
        allow_none : bool
            If True, allow None as a value regardless of values or types.
        recordable : bool
            If True, add to recorder.
        set_function : None or function
            User-supplied function with arguments (Options metadata, value) that pre-processes
            value and returns a new value.
        deprecation : str or None
            If None, it is not deprecated. If a str, use as a DeprecationWarning
            during __setitem__ and __getitem__.  If a tuple of the form (msg, new_name),
            display msg as with str, and forward any __setitem__/__getitem__ to new_name.
        """
        if values is not None and not isinstance(values, (set, list, tuple)):
            self._raise(f"In declaration of option '{name}', the 'values' arg must be of type None,"
                        f" list, or tuple - not {values}.", exc_type=TypeError)

        if types is not None and not isinstance(types, (type, set, list, tuple)):
            self._raise(f"In declaration of option '{name}', the 'types' arg must be None, a type "
                        f"or a tuple - not {types}.", exc_type=TypeError)

        if types is not None and types is not list and values is not None:
            self._raise(f"'types' and 'values' were both specified for option '{name}'.")

        if types is bool:
            values = (True, False)

        if not recordable:
            self._all_recordable = False

        default_provided = default is not _UNDEFINED

        if default_provided and default is None:
            # specifying default=None implies allow_none
            allow_none = True

        alias = None
        if deprecation is not None:
            if isinstance(deprecation, (list, tuple)):
                if len(deprecation) != 2:
                    self._raise("deprecation must be None, str, or a tuple or list containing "
                                "(str, str).", RuntimeError)
                dep, alias = deprecation
                # [message, alias, display warning (becomes False after first display)]
                deprecation = [dep, alias, True]
            else:
                deprecation = [deprecation, None, True]

        self._dict[name] = {
            'val': default,
            'values': values,
            'types': types,
            'desc': desc,
            'upper': upper,
            'lower': lower,
            'check_valid': None,
            'has_been_set': default_provided,
            'allow_none': allow_none,
            'recordable': recordable,
            'set_function': None,
            'deprecation': deprecation,
            'units': units
        }

        # If a default is given, check for validity
        if default_provided:
            self._assert_valid(name, default)

    def get_item(self, name):
        value = self[name]
        units = self._dict[name]["units"]
        return(value, units)

    def to_table(self, fmt='github', missingval='N/A', max_width=None, display=True):
        """
        Get a table representation of this OptionsDictionary as a table in the requested format.

        Parameters
        ----------
        fmt : str
            The formatting of the requested table.  Options are
            ['github', 'rst', 'text', 'html', 'tabulator'] and several 'grid' and 'outline'
            formats that mimic those found in the python 'tabulate' library.
            Default value of 'github' produces a table in GitHub-flavored markdown.
            'html' and 'tabulator' produce output viewable in a browser.
        missingval : str
            The value to be displayed in place of None.
        max_width : int or None
            If not None, try to limit the total width of the table to this value.
        display : bool
            If True, display the table, typically by writing it to stdout or opening a
            browser.

        Returns
        -------
        str
            A string representation of the table in the requested format.
        """
        hdrs = ['Option', 'Default', 'Acceptable Values', 'Acceptable Types', 'Description', 'Units']
        rows = []

        deprecations = False
        for meta in self._dict.values():
            if meta['deprecation'] is not None:
                deprecations = True
                hdrs.append('Deprecation')
                break

        for key in sorted(self._dict.keys()):
            option = self._dict[key]
            default = option['val'] if option['val'] is not _UNDEFINED else '**Required**'
            default_str = str(default)

            # if the default is an object instance, replace with the (unqualified) object type
            idx = default_str.find(' object at ')
            if idx >= 0 and default_str[0] == '<':
                parts = default_str[:idx].split('.')
                default = parts[-1]

            acceptable_values = option['values']
            if acceptable_values is not None:
                if not isinstance(acceptable_values, (set, tuple, list)):
                    acceptable_values = (acceptable_values,)
                acceptable_values = [value for value in acceptable_values]

            acceptable_types = option['types']
            if acceptable_types is not None:
                if not isinstance(acceptable_types, (set, tuple, list)):
                    acceptable_types = (acceptable_types,)
                acceptable_types = [type_.__name__ for type_ in acceptable_types]

            desc = option['desc']

            deprecation = option['deprecation']
            if deprecation is not None:
                deprecation = deprecation[0]
            units = option['units']

            if deprecations:
                rows.append([key, default, acceptable_values, acceptable_types, desc, units,
                             deprecation])
            else:
                rows.append([key, default, acceptable_values, acceptable_types, desc, units])
        
        if len(rows) == 0:
            # This is an empty class
            rows.append(['']*len(hdrs))

        kwargs = {
            'tablefmt': fmt,
            'headers': hdrs,
            'missing_val': missingval,
            'max_width': max_width,
        }
        if fmt == 'tabulator':
            kwargs['filter'] = False
            kwargs['sort'] = False

        tab = generate_table(rows, **kwargs)

        if display:
            tab.display()

        return str(tab)


def opt_dict_unit_merge(options : List[OptionsDictionaryUnit], 
    base_options : typing.Optional[OptionsDictionaryUnit] = None,
    ) -> OptionsDictionaryUnit:
    """[summary]

    Arguments:
        options {List[OptionsDictionaryUnit]} -- A list of options to merge. Note that if options
                            are defined more than once the final values are used.

    Keyword Arguments:
        base_options {typing.Optional[OptionsDictionaryUnit]} -- If used these options are expanded
                            with the passed in options. (default: {None})
    Returns:
        OptionsDictionaryUnit -- Merge of all the options passed in.
    """
    if base_options is None:
        # We create an empty OptionsDictionaryUnit object to collect all options
        base_options = OptionsDictionaryUnit()
    for single_option in options:
        # Iterating over all options passed in
        for (name, info) in single_option._dict.items():
            # Declaring all options that have been defined in this single_option
            values = info["values"]
            if info['types']  is not None:
                if issubclass(info['types'], bool):
                    values = None
            base_options.declare(name=name, default=info["val"], values=values,
            types=info["types"], desc=info["desc"], upper=info["upper"],
            lower=info["lower"], allow_none=info["allow_none"], recordable=info["recordable"],
            deprecation=info["deprecation"])
    return(base_options)



class StandardBase(object):
    """A class to be used in conjunction with OpenMDAO elements to have a common way of dealing with options.

    This class implements the ability of the system to keep track of options,
    and also pre-defines that this component depends on the 'class_options'.
    """

    def initialize(self):
        """Initialize the instance by defining that this will use `class_options`"""
        self.options.declare(
            "class_options",
            types=OptionsDictionaryUnit,
            desc="collection of Aircraft/Mission specific options",
        )
        # Set the options based on the class method  to define the options.
        self.options["class_options"] = self.required_options()

    @classmethod
    def _define_options(cls) -> OptionsDictionaryUnit:
        """Abstract method that allows a developer to define information about the parameters this class uses.

        This method is called by a class method that is adding options to the `class_options` option.

        Returns:
            OptionsDictionaryUnit -- The required options for this class, their default values, and, if required, their units.
        """
        return OptionsDictionaryUnit()

    @classmethod
    def required_options(cls) -> OptionsDictionaryUnit:
        """A class method that will return the required options and their default values for
        this class.

        Returns:
            OptionsDictionaryUnit: Information about the required options and their default values
        """
        return cls._define_options()

    @classmethod
    def required_options_names(cls) -> set[str]:
        """A class method returning a set of strings with the names of the required options.

        Returns:
            set(str): Names of the required options.
        """
        return set(cls.required_options()._dict.keys())


    def full_options(self) -> OptionsDictionaryUnit:
        """An instance method to return all options for this class.
        
        Often this will be just a call to the `required_options` class method. But for example
        in group the full set of options that are valid are depending on the components in the
        instance, so this has to be calculated after an instance is created, and the `setup()`
        method has been called.

        Returns:
            OptionsDictionaryUnit -- All valid options for this class.
        """
        return self.required_options()

    @property
    def full_options_names(self) -> set[str]:
        """A class method returning a set of strings with the names of the full options.

        Returns:
            set(str): Names of the required options.
        """
        return set(self.full_options()._dict.keys())

    def lookup_option_value(self, name) -> typing.Any:
        class_options: OptionsDictionaryUnit = self.options["class_options"]
        if name in class_options:
            value = class_options[name]
        else:
            # Let's see if the value is set in the default full options
            required_opts = self.full_options()
            if name in required_opts:
                print(f"Looking up value for {name} from the full options method")
                value = required_opts[name]
            else:
                raise KeyError(f'KeyError: key not found: {name}')
        return value

    
    def _check_options(
        self, options: typing.Optional[typing.Union[str, list[str]]]
    ) -> list[str]:
        """Check that the options are correct, and part of the required options.

        Arguments:
            options {typing.Optional[typing.Union[str, list[str]]]} -- Name or list of
                        names for options. Defaults to None.

        Raises:
            TypeError: Options need to be a string or list of strings
            TypeError: Entry in the options list is not a string
            ValueError: Trying to define an option for a variable or response that is not part of the
                required options list. Options need to be defined.

        Returns:
            list[str] -- List of options.
        """
        # Get the names of required options given.
        req_options = self.required_options_names()
        if isinstance(options, str):
            # If only a single string is passed in we convert it into a list of strings
            options = [options]
        elif not isinstance(options, list):
            raise TypeError("Options is not a string or list of strings")
        # Check if the option(s) defined for this variable or response are are part of the
        # named required options. If not throw an error. This ensures that the required
        # options set contains all required options.
        print(req_options)
        for entry in options:
            if not isinstance(entry, str):
                raise TypeError(
                    f"Entry {entry} in options list is not a string, instead a {type(entry)}"
                )
            if not entry in req_options:
                raise ValueError(
                    f"Trying to set {entry} as an option, but it is not defined as an option for this class"
                )
        return options

    def _lookup_info(
        self,
        name: str,
        val: typing.Optional[typing.Any] = None,
        units: typing.Optional[str] = None,
        desc: typing.Optional[str] = None,
    ) -> typing.Tuple[typing.Any, str, str]:
        """Helper method to allow implementation of a way to look up information.

        Arguments:
            name {str} -- Name of the element to look up information for

        Keyword Arguments:
            val {typing.Optional[typing.Any]} -- Value to use. If None value will be looked up (default: {None})
            units {typing.Optional[str]} --Units to use. If None units will be looked up (default: {None})
            desc {typing.Optional[str]} -- Description to use. If None description will be looked up (default: {None})

        Returns:
            typing.Tuple[typing.Any, str, str] -- Updated values, units, and description. 
        """
        # In this base class implementation we are not doing any lookup
        return (val, units, desc)

    def _add_info(
        self,
        name: str,
        element_type: typing.Optional[str] = None,
        val: typing.Optional[typing.Any] = None,
        units: typing.Optional[str] = None,
        desc: typing.Optional[str] = None,
        shape_by_conn: bool = False,
        options: typing.Optional[typing.Union[str, list[str]]] = None,
        look_up: bool = True,
    ):
        """Provide a clean way to add variables or responses from the  hierarchy into
        components as inputs, and allow the definition of options this
        variable or response depends on. It takes
        the standard OpenMDAO inputs of the variable's or response's name, initial
        value, units, and description, as well as the component which
        the variable or response is being added to.

        Args:
            name (str): OpenMDAO variable or response name
            param_type (typing.Optional[str], optional): Description if information is for a
                    variable or response. Defaults to None.
            val (typing.Optional[typing.Any], optional): Initial value. If none the initial value might
                        be provided by the _lookup_info method. Defaults to None.
            units (typing.Optional[str], optional): Units. If none the initial value might
                        be provided by the _lookup_info method. Defaults to None.
            desc (typing.Optional[str], optional): Description. If none the initial value might
                        be provided by the _lookup_info method. Defaults to None.
            shape_by_conn (bool, optional): Decide whether or not the shape of the variable or
                        response is defined by its connections. Defaults to False.
            options (typing.Optional[typing.Union[str, list[str]]], optional): Name or list of
                        names for options this variable or response depends on. Defaults to
                        None.
            look_up (bool, optional): Flag whether to look up values, units, and description. Defaults
                        to true
        """
        if look_up:
            # Use either the passed in values, or the values from the look-up method
            (val, units, desc) = self._lookup_info(
                name=name,
                val=val,
                units=units,
                desc=desc,
            )
        if element_type not in ["variables", "responses"]:
            raise ValueError(
                f"Option type is {element_type}, only `variables` or `responses is allowed."
            )
        if element_type == "variables":
            func_name = self.add_input
        else:
            func_name = self.add_output

        tags = None
        if options is not None:
            # Check that the options are valid list of required options
            options = self._check_options(options)
            # Save all the options in tags, using the TAG_KEY for uniqueness.
            tags = [f"{TAG_KEY}{opt}" for opt in options]
        # Call either the add_input or add_output routine that is defined by the OpenMDAO
        # base object
        func_name(
            name,
            val=val,
            units=units,
            desc=desc,
            shape_by_conn=shape_by_conn,
            tags=tags,
        )

    def add_standard_input(
        self,
        name: str,
        val: typing.Optional[typing.Any] = None,
        units: typing.Optional[str] = None,
        desc: typing.Optional[str] = None,
        shape_by_conn: bool = False,
        options: typing.Optional[typing.Union[str, list[str]]] = None,
        look_up: bool = True,
    ):
        """Provide a clean way to add variables from the variable hierarchy into
        components as Aviary inputs, and allow the definition of options this
        variable depends on. It takes
        the standard OpenMDAO inputs of the variable's name, initial
        value, units, and description, as well as the component which
        the variable is being added to.

        Args:
            name (str): OpenMDAO variable name
            val (typing.Optional[typing.Any], optional): Initial value of the variable. If none
                        the initial value will be taken from the MetaData. Defaults to None.
            units (typing.Optional[str], optional): Units for the variable. If none the units will
                        be taken from the MetaData. Defaults to None.
            desc (typing.Optional[str], optional): Description of the variable. If none the
                        description will be taken from the MetaData. Defaults to None.
            shape_by_conn (bool, optional): Decide whether or not the shape of the variable is
                        defined by its connections. Defaults to False.
            options (typing.Optional[typing.Union[str, list[str]]], optional): Name or list of
                        names for options this variable depends on. Defaults to None.
            look_up (bool, optional): Flag whether to look up values, units, and description. Defaults
                        to true
        """
        # Use either the passed in values, or the values from the metadata
        self._add_info(
            name=name,
            element_type="variables",
            val=val,
            units=units,
            desc=desc,
            shape_by_conn=shape_by_conn,
            options=options,
            look_up=look_up,
        )

    def add_standard_output(
        self,
        name: str,
        val: typing.Optional[typing.Any] = None,
        units: typing.Optional[str] = None,
        desc: typing.Optional[str] = None,
        shape_by_conn: bool = False,
        options: typing.Optional[typing.Union[str, list[str]]] = None,
        look_up: bool = True,
    ):
        """Provide a clean way to add responses from the response hierarchy into
        components as Aviary outputs, and allow the definition of options this
        response depends on. It takes
        the standard OpenMDAO inputs of the response's name, initial
        value, units, and description, as well as the component which
        the response is being added to.

        Args:
            name (str): OpenMDAO response name
            val (typing.Optional[typing.Any], optional): Initial value of the response. If none
                        the initial value will be taken from the MetaData. Defaults to None.
            units (typing.Optional[str], optional): Units for the response. If none the units will
                        be taken from the MetaData. Defaults to None.
            desc (typing.Optional[str], optional): Description of the response. If none the
                        description will be taken from the MetaData. Defaults to None.
            shape_by_conn (bool, optional): Decide whether or not the shape of the response is
                        defined by its connections. Defaults to False.
            options (typing.Optional[typing.Union[str, list[str]]], optional): Name or list of
                        names for options this response depends on. Defaults to None.
            look_up (bool, optional): Flag whether to look up values, units, and description. Defaults
                        to true
        """
        # Use either the passed in values, or the values from the metadata
        self._add_info(
            name=name,
            element_type="responses",
            val=val,
            units=units,
            desc=desc,
            shape_by_conn=shape_by_conn,
            options=options,
            look_up=look_up,
        )

    def _get_options_info(self) -> dict:
        # Create a dictionary that collects the information for all options
        option_names = self.full_options_names
        defined_names = [info for (info,_) in self.options["class_options"].items()]
        option_info = {}
        for name in option_names:
            # Get the values for the options as they are at this point. This will ensure wer get
            # the actual values of the options as they were when the class is fully set.
            if name in defined_names:
                # This option is still set in the "class_options". This means it probably was
                # overwritten by the user, or the user created the initial options from the
                # required_options class method
                (val, units) = self.options["class_options"].get_item(name)
            else:
                # This option was not set, and is not actually used as an option.
                # We go back to the values and units defined in the required_options class method.
                (val, units) = self.full_options().get_item(name)
            option_info[name] = {
                "input": set(),
                "output": set(),
                "value": val,
                "units": units,
            }
        # Iterate over all elements that are in the object. Do this for inputs and outputs
        for element_type in ["input", "output"]:
            for name, local_info in self.get_io_metadata(iotypes=element_type).items():
                # If there are actual entries in the tags we check them
                if len(local_info["tags"]) > 0:
                    # This element has tags associated with it
                    for option in option_names:
                        # Check if this option is used for this element
                        if f"{TAG_KEY}{option}" in local_info["tags"]:
                            option_info[option][element_type].add(name)

        # Need to convert any sets to a list to be "jsonifiable". Also remove empty sets
        for option in option_names:
            for element_type in ["input", "output"]:
                if len(option_info[option][element_type]) > 0:
                    option_info[option][element_type] = list(
                        option_info[option][element_type]
                    )
                else:
                    del option_info[option][element_type]
        return option_info
    
    def current_options(self) -> OptionsDictionaryUnit:
        """Return all the current options. 
        This is a combination between the original options (defaults) and values that
        were overwritten by the end user.

        Returns:
            OptionsDictionaryUnit -- The combination of all options
        """
        return(opt_dict_unit_merge([self.full_options(), self.options["class_options"]]))
