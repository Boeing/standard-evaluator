"""Defining the OpenMDAO Evaluator"""

import copy
from typing import Tuple
import numpy as np
import pandas as pd
import openmdao.api as om
import openmdao.utils.general_utils as om_utils
import openmdao.core as om_core

from standard_evaluator.utilities import problem_calculate_fields, legacy_to_opt_problem
from standard_evaluator.evaluators.abstract_evaluator import Evaluator


class OpenMDAOEvaluator(Evaluator):
    """A class to expose an OpenMDAO problem as an evaluator"""

    def __init__(
        self,
        om_prob: om.Problem,
        comp_cost: float = 100,
        scan_model: bool = True,
        use_defined_problem: bool = True,
    ) -> None:
        """Initialize the evaluator, saving relevant information

        Args:
            om_prob (om.Problem): The OpenMDAO problem
            comp_cost (float, optional): Computational costs of the evaluator.
                Defaults to 100.
            scan_model (bool, optional): Flag whether or not to scan the whole
                model for inputs and outputs. Defaults to True.
            use_defined_problem (bool, optional): Flag to decide whether or not
                to use the optimization problem defined in the OpenMDAO model.
                Defaults to True.

        Raises:
            ValueError: Scan_model and use_defined_problem cannot both be false
                at the same time.
        """
        # Save the OpenMDAO model
        self.om_problem = om_prob
        name = om_prob._name
        self.scan_model = scan_model

        if not scan_model and not use_defined_problem:
            raise ValueError(
                "scan_model and use_defined_problem cannot both "
                "be False at the same time."
            )

        if scan_model:
            # Capture all variables and responses that are defined in the model if we scan the model
            self._scan_model()
        else:
            self._main_problem = None
        if use_defined_problem:
            problem = self.get_om_opt_problem()
        else:
            problem = copy.deepcopy(self._main_problem)
        opt_problem = legacy_to_opt_problem(problem)
        super().__init__(name=name, comp_cost=comp_cost, opt_problem=opt_problem)

    def _evaluate(self, sites: pd.DataFrame):
        """Evaluate the OpenMDAO model on all the sites defined in the DataFrame

        Args:
            sites (pd.DataFrame): Sites to be evaluated. Will be updated in this method.
        """
        # Iterate over all sites in the data frame
        for ind in range(len(sites)):
            # Set the values of all the variables
            for ele in self.inputs:
                self.om_problem.set_val(ele, sites.iloc[ind][ele])
            # Execute the OpenMDAO model
            self.om_problem.run_model()
            # Extract all responses
            for ele in self.outputs:
                sites.at[ind, ele] = self.om_problem.get_val(ele)

    def get_om_opt_problem(self) -> dict:
        """Convert an OpenMDAO model into a DE optimization problem

        Returns:
            dict: Return the OpenMDAO model as a De optimization dictionary.
        """
        problem = {}
        # Create the variable information
        problem["variables"] = self._map_elements(
            self.om_problem.model.get_design_vars()
        )
        # Create the response information, which includes constraints
        problem["responses"] = self._map_elements(self.om_problem.model.get_responses())
        problem["objectives"] = list(self.om_problem.model.get_objectives().keys())
        # If the model has been scanned we fix any variables that have not been
        # defined, and expose all responses
        if self.scan_model:
            for my_type in ["variables", "responses"]:
                for variable in self._main_problem[my_type]:
                    if variable in problem[my_type]:
                        continue

                    problem[my_type][variable] = copy.deepcopy(
                        self._main_problem[my_type][variable]
                    )
                    if my_type == "variables":
                        problem[my_type][variable]["active"] = False
                        problem[my_type][variable]["bounds"] = [
                            problem[my_type][variable]["default"],
                            problem[my_type][variable]["default"],
                        ]

        return problem_calculate_fields(problem)

    def _scan_model(self):
        """Collect all valid variables and responses in the OpenMDAO model, and
        save them in the internal variable _main_problem
        """
        # Get all the inputs for all components in the model
        variables = self.om_problem.model.list_inputs(
            out_stream=None, prom_name=True, shape=True
        )
        outer_dict = self.om_problem.model.get_io_metadata(
            iotypes="input", metadata_keys=["tags"], return_rel_names=False
        )
        # Check whether the model is a group. If we have a group we can use the _auto_ivc
        # automated component to get the overall inputs.
        is_group = isinstance(self.om_problem.model, om_core.group.Group)
        # Store all of the variables (promoted or local only)
        variable_dict = {}
        for values in variables:
            if values[0] in outer_dict:
                if "tags" in outer_dict[values[0]]:
                    values[1]["tags"] = outer_dict[values[0]]["tags"]

            if self._check_internal(values[1]):
                print(f"Internal variable {values[0]}")
                continue
            name, local_dict = self._expand_info(values[1])
            if is_group:
                if "_auto_ivc" in self.om_problem.model.get_source(name):
                    # We want to make sure we are only using variables that are input to the
                    # overall model, not local variables that are linked to responses from
                    # another component.
                    variable_dict[name] = local_dict
            else:
                variable_dict[name] = local_dict
        # Get all outputs / responses
        res = self.om_problem.model.list_outputs(
            out_stream=None,
            prom_name=True,
            bounds=False,
            scaling=True,
            shape=True,
            val=False,
        )
        outer_dict = self.om_problem.model.get_io_metadata(
            iotypes="output", metadata_keys=["tags"], return_rel_names=False
        )
        response_dict = {}
        for values in res:
            if values[0] in outer_dict:
                if "tags" in outer_dict[values[0]]:
                    values[1]["tags"] = outer_dict[values[0]]["tags"]

            if self._check_internal(values[1]):
                print(f"Internal response {values[0]}")
                continue
            name, local_dict = self._expand_info(values[1])
            response_dict[name] = local_dict
        self._main_problem = problem_calculate_fields(
            {"variables": variable_dict, "responses": response_dict}
        )

    def _check_internal(self, info: dict) -> bool:
        """Check if this is a variable or response that is marked as internal

        Arguments:
            info {dict} -- The dictionary from OpenMDAO containing information
                    about a variable or response

        Returns:
            bool -- True if this is an internal variable or response, False else
        """
        internal_ele = False
        if "tags" in info:
            if "internal" in info["tags"]:
                internal_ele = True
        return internal_ele

    def _expand_info(self, info: dict) -> Tuple[str, dict]:
        """Expand the information for a variable or response to save in a DE
        problem dictionary

        Args:
            info (dict): The dictionary from OpenMDAO containing information
                about a variable or response

        Raises:
            TypeError: Currently we can only use variables or responses that
                are not arrays.

        Returns:
            dict: The DE style dictionary containing all information about the
                variable or response.
        """

        def helper(name: str) -> float:
            """Helper routine to return the info for a name, or none if not defined

            Parameters
            ----------
            name : str
                Name of the element to look up

            Returns
            -------
            float
                Value of the dictionary or None if not defined
            """
            return_value = None
            if name in info:
                return_value = info[name]

            return return_value

        local_dict = {}
        name = info["prom_name"]
        shape = info["shape"]
        if shape != (1,):
            raise TypeError(f"Element {name} is of shape {shape}, must be (1, )")
        ref = helper("ref")
        ref0 = helper("ref0")
        adder = helper("adder")
        scaler = helper("scaler")
        adder, scaler = om_utils.determine_adder_scaler(ref0, ref, adder, scaler)
        local_dict["shift"] = adder
        local_dict["scale"] = scaler
        if "val" in info:
            local_dict["default"] = info["val"][0]
        local_dict["bounds"] = [-np.inf, np.inf]
        return name, local_dict

    def _map_elements(self, info: dict) -> dict:
        """Convert information for variables and responses
        This method converts a dictionary from OpenMDAO to a DE dictionary

        Args:
            info (dict): The dictionary from OpenMDAO containing information
                about a variable or response

        Returns:
            dict: The DE style dictionary containing all information about the
                variable or response.
        """
        full_dict = {}
        for ele in info:
            ele_dict = {}
            if "lower" in info[ele]:
                ele_dict["bounds"] = [info[ele]["lower"], info[ele]["upper"]]
            if "scaler" in info[ele] and info[ele]["scaler"] is not None:
                ele_dict["scale"] = info[ele]["scaler"]
            full_dict[ele] = ele_dict
        return full_dict
