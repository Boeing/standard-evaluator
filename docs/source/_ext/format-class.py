"""
Created Nov. 29, 2022

@author Mikel Woo
"""
from typing import Optional

from autodocsumm import AutoSummClassDocumenter
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.ext.autodoc import bool_option


class FormatClass(AutoSummClassDocumenter):
    """Adds the :print-options: option to the autoclass directive. Using this
    option will print the unique identifier of the class as well as format the
    available options in an easy to read manner. It can be used as follows:

        .. autoclass:: sample.Class
            :print-options:

    Also adds the :known-solution: option which is used by test evaluators
    to print out the known solution(s). Use the :known-solution: option.

    This feature is implemented similarly to how the autodocsumm extension
    creates the attribute and method summary tables. That extension simply
    overrides the autodoc formatters and adds in the summary table capability.
    To ensure compatability with that extension, this extension does the same
    thing but overriding the autodocsumm class formatter.
    """
    # Set this to have highest priority
    priority = 10 + AutoSummClassDocumenter.priority

    # Copy options
    option_spec = dict(AutoSummClassDocumenter.option_spec)
    # Add print-options, known-solution, and print-test-info options
    option_spec['print-options'] = bool_option
    option_spec['known-solution'] = bool_option
    option_spec['print-test-info'] = bool_option

    def add_content(self, more_content: Optional[StringList]) -> None:
        """This method generates the RST content that will be rendered. When
        Sphinx encounters an autoclass directive it will call this method to
        fill in what content should be in place of the directive.

        Parameters
        ----------
        more_content : Optional[StringList]
            List of strings containing the content of anything placed within the
            directive that is not an option. For example:

                .. autoclass:: sample.Test
                    :print-options:

                    This line will show up
                    As will this one

                    .. math::
                        y = mx + b

                    This line and the equation above will show up

            All the lines (even blank ones) after the blank line below
            :print-options: will be in more_content
        """
        # Call add_content of the autodoc ClassDocumenter which the autosummary
        # AutoSummClassDocumenter derives from. This is to allow us to control
        # when the summary tables are printed
        super(AutoSummClassDocumenter, self).add_content(more_content)

        # Print options if print-options is set and no-print-options isn't
        if 'print-options' in self.options and 'no-print-options' not in self.options:
            self._print_options()

        # Print test info if print-test-info is set
        if 'print-test-info' in self.options:
            self._print_test_info()

        # Print known solution(s) if known-solution is set and no-known-solution isn't
        if 'known-solution' in self.options and 'no-known-solution' not in self.options:
            self._print_known_solution()

        # Add table of attributes and methods below unique_name and options
        # This method handles the :autosummary: option from autodocsumm. It's
        # placed here to have options printed before the method summary.
        self.add_autosummary(True)

    def _print_options(self):
        """Prints out and formats option information for a given class. If the
        class has a ``unique_name`` attribute then that will be printed as well.
        If the class has an ``options`` attribute with options defined in a
        dictionary, then each option will be printed in a list with its
        explanation, bounds, and default value.
        """
        # Source name tells sphinx which document is being writtne to
        source_name = self.get_sourcename()
        # A reference to the class object
        obj = self.object

        was_printed = False

        # Print unique name
        if hasattr(obj, 'unique_name'):
            # Add space between class description and this
            self.add_line('|', source_name)
            self.add_line('', source_name)
            # Give the title a CSS class for styling
            self.add_line('.. rst-class:: title', source_name)
            self.add_line('', source_name)
            self.add_line(f'**Identifier:** *{obj.unique_name}*', source_name)
            self.add_line('', source_name)

            was_printed = True

        # Print options list
        if hasattr(obj, 'options'):
            self.add_line('|', source_name)
            self.add_line('', source_name)
            self.add_line('.. rst-class:: title', source_name)
            self.add_line('', source_name)
            self.add_line('**Options**', source_name)
            self.add_line('', source_name)

            # Add options-list class to style options list
            self.add_line('.. rst-class:: options-list', source_name)
            self.add_line('', source_name)

            tab = '  '
            # Print option info
            for name, info in obj.options.items():
                # Name
                self.add_line(f'- **{name}**', source_name)
                self.add_line('', source_name)
                # Explanation
                self.add_line('  ' + info['expl'], source_name)
                self.add_line('', source_name)
                # Bounds
                self.add_line(tab + f'- Bounds: {info["bounds"]}', source_name)
                # Default value
                if isinstance(info['value'], (int, float)):
                    self.add_line(tab + f'- Default: {info["value"]:,}', source_name)
                else:
                    self.add_line(tab + f'- Default: {info["value"]}', source_name)

            was_printed = True

        # Add a gap between content printed above and method summary table
        if was_printed:
            self.add_line('', source_name)
            self.add_line('|', source_name)

        # Make sure there is a blank line at the end so nothing breaks
        self.add_line('', source_name)

    def _print_test_info(self):
        """Renders test evaluator metadata (description, citation, test_info)
        as RST content when the :print-test-info: option is set.
        """
        source_name = self.get_sourcename()
        obj = self.object

        # Attempt instantiation — skip silently for abstract classes
        try:
            inst = obj()
        except TypeError:
            return

        # Skip if the instance doesn't have opt_problem
        if not hasattr(inst, 'opt_problem'):
            return

        # Render description if available
        description = inst.opt_problem.description
        if description and description.strip():
            self._render_description(description.strip(), source_name)

        # Render citation if available
        cite = inst.opt_problem.cite
        if cite and cite.strip():
            self._render_citation(cite.strip(), source_name)

        # Render test_info metadata (always available for valid instances)
        test_info = inst.test_info
        self._render_test_info_table(test_info, source_name)

    def _render_description(self, description, source_name):
        """Render the opt_problem.description as a formatted RST block.

        If the description is wrapped in $$ delimiters (starts AND ends with $$),
        it is rendered inside a .. math:: directive. Otherwise it is rendered as
        plain paragraph text.
        """
        self.add_line('|', source_name)
        self.add_line('', source_name)
        self.add_line('.. rst-class:: title', source_name)
        self.add_line('', source_name)
        self.add_line('**Problem Description**', source_name)
        self.add_line('', source_name)

        # Check if content is wrapped in $$ delimiters (LaTeX math block)
        if description.startswith('$$') and description.endswith('$$'):
            # Strip $$ and render as math directive
            math_content = description[2:-2].strip()
            self.add_line('.. math::', source_name)
            self.add_line('', source_name)
            for line in math_content.split('\n'):
                self.add_line('   ' + line, source_name)
        else:
            # Plain text description — render as paragraph
            for line in description.split('\n'):
                self.add_line(line, source_name)

        self.add_line('', source_name)

    def _render_citation(self, cite, source_name):
        """Render the opt_problem.cite as a styled admonition block."""
        self.add_line('.. admonition:: Reference', source_name)
        self.add_line('', source_name)
        for line in cite.split('\n'):
            self.add_line('   ' + line.strip(), source_name)
        self.add_line('', source_name)

    def _render_test_info_table(self, test_info, source_name):
        """Render the test_info dictionary as an RST field list."""
        self.add_line('.. rst-class:: title', source_name)
        self.add_line('', source_name)
        self.add_line('**Problem Metadata**', source_name)
        self.add_line('', source_name)

        labels = {
            'test_goal': 'Test Goal',
            'problem_type': 'Problem Type',
            'n_vars': 'Variables',
            'n_continuous': 'Continuous Variables',
            'n_discrete': 'Discrete Variables',
            'n_constraints': 'Constraints',
            'n_equality_constraints': 'Equality Constraints',
            'n_inequality_constraints': 'Inequality Constraints',
            'bounded_variables': 'Bounded Variables',
        }

        for key, label in labels.items():
            value = test_info[key]
            if isinstance(value, bool):
                display_value = 'Yes' if value else 'No'
            elif key in ('test_goal', 'problem_type'):
                display_value = str(value).replace('_', ' ').title()
            else:
                display_value = str(value)
            self.add_line(f':{label}: {display_value}', source_name)

        self.add_line('', source_name)
        self.add_line('|', source_name)
        self.add_line('', source_name)

    def _print_known_solution(self):
        # Source name tells sphinx which document is being writtne to
        source_name = self.get_sourcename()
        # A reference to the class object
        obj = self.object

        # Don't print anything if there is no known_solution property
        if not hasattr(obj, 'known_solution'):
            return

        # Instantiate the class so we can access it's properties
        # Skip classes that aren't fully defined (ie. missing abstract methods)
        try:
            inst = obj()
        except TypeError:
            return

        # Get known solution, variable names, and response names
        sol = inst.known_solution

        # Skip if the instance doesn't have a problem attribute
        if not hasattr(inst, 'problem') or inst.problem is None:
            return

        variables = list(inst.problem['variables'].keys())
        responses = list(inst.problem['responses'].keys())

        # Add space between docstring and solution table
        self.add_line('|', source_name)
        self.add_line('', source_name)
        # Add CSS title class to format Known Solution title
        self.add_line('.. rst-class:: title', source_name)
        self.add_line('', source_name)

        # No known solution
        if sol is None:
            self.add_line('**Known Solution**', source_name)
            self.add_line('', source_name)
            self.add_line('No known solution!', source_name)
        # Print known solution(s)
        else:
            # Print title
            if len(sol) == 1:
                self.add_line('**Known Solution**', source_name)
            else:
                self.add_line('**Known Solutions**', source_name)

            self.add_line('', source_name)

            tab = ' '*4

            # Create table with a row for each solution
            self.add_line('.. list-table::', source_name)
            # Table will span whole width of rendered area
            self.add_line(tab + ':width: 100', source_name)
            # Only one header row
            self.add_line(tab + ':header-rows: 1', source_name)
            self.add_line('', source_name)

            # Fill in header row
            for i, name in enumerate(variables + responses):
                if i == 0:
                    self.add_line(tab + f'* - {name}', source_name)
                else:
                    self.add_line(tab + f'  - {name}', source_name)

            # Fill in table
            for i in range(len(sol)):
                row = sol.iloc[i]

                for j, name in enumerate(variables + responses):
                    if j == 0:
                        self.add_line(tab + f'* - {row[name]}', source_name)
                    else:
                        self.add_line(tab + f'  - {row[name]}', source_name)

        # Add gap between this and content below
        self.add_line('', source_name)
        self.add_line('|', source_name)
        self.add_line('', source_name)


def _register_format_class(app, config):
    """Register FormatClass after autodocsumm has registered its documenter.

    autodocsumm registers AutoSummClassDocumenter at config-inited priority 600.
    We register at priority 700 so that FormatClass (which subclasses
    AutoSummClassDocumenter) takes final precedence in the documenter registry.
    """
    app.add_autodocumenter(FormatClass, override=True)


def setup(app: Sphinx) -> None:
    """Loads in required extensions and registers actions, documenters, etc.
    with Sphinx to allow them to be used.

    Parameters
    ----------
    app : Sphinx
        The current Sphinx instance.
    """
    # Load the autodoc and autodocsumm extentsions since both are required
    app.setup_extension('sphinx.ext.autodoc')
    app.setup_extension('autodocsumm')

    # Register FormatClass after autodocsumm's _after_config_inited (priority 600)
    # to ensure our custom documenter with :print-test-info: takes precedence.
    app.connect("config-inited", _register_format_class, priority=700)
