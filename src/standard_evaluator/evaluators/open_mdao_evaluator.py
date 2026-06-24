"""Defining the OpenMDAO Evaluator"""

import copy

import numpy as np
import pandas as pd
import openmdao.api as om
import openmdao.utils.general_utils as om_utils
import openmdao.core as om_core

from standard_evaluator.problem import OptProblem, FloatVariable
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
            # Capture all variables and responses that are defined in the model
            self._scan_model()
        else:
            self._scanned_problem = None

        if use_defined_problem:
            opt_problem = self._get_om_opt_problem()
        else:
            opt_problem = copy.deepcopy(self._scanned_problem)

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

    def _get_om_opt_problem(self) -> OptProblem:
        """Convert an OpenMDAO model into an OptProblem.

        Returns:
            OptProblem: The optimization problem derived from the OpenMDAO model.
        """
        # Gather design variables
        design_vars = self._map_elements(
            self.om_problem.model.get_design_vars()
        )
        # Gather responses (includes constraints)
        responses = self._map_elements(self.om_problem.model.get_responses())
        objectives = list(self.om_problem.model.get_objectives().keys())

        # If the model has been scanned we fix any variables that have not been
        # defined, and expose all responses
        if self.scan_model and self._scanned_problem is not None:
            scanned_vars = {v.name: v for v in self._scanned_problem.variables}
            scanned_resps = {r.name: r for r in self._scanned_problem.responses}
            existing_var_names = {v.name for v in design_vars}
            existing_resp_names = {r.name for r in responses}

            for var_name, var_obj in scanned_vars.items():
                if var_name not in existing_var_names:
                    # Fixed variable: bounds set to [default, default]
                    fixed_var = copy.deepcopy(var_obj)
                    fixed_var.bounds = [fixed_var.default, fixed_var.default]
                    design_vars.append(fixed_var)

            for resp_name, resp_obj in scanned_resps.items():
                if resp_name not in existing_resp_names:
                    responses.append(copy.deepcopy(resp_obj))

        return OptProblem(
            name="openmdao_problem",
            variables=design_vars,
            responses=responses,
            objectives=objectives,
        )

    def _scan_model(self):
        """Collect all valid variables and responses in the OpenMDAO model,
        and save them as an OptProblem in self._scanned_problem.
        """
        # Get all the inputs for all components in the model
        variables = self.om_problem.model.list_inputs(
            out_stream=None, prom_name=True, shape=True
        )
        outer_dict = self.om_problem.model.get_io_metadata(
            iotypes="input", metadata_keys=["tags"], return_rel_names=False
        )
        # Check whether the model is a group
        is_group = isinstance(self.om_problem.model, om_core.group.Group)

        # Store all of the variables
        variable_dict = {}  # Use dict to deduplicate by name
        for values in variables:
            if values[0] in outer_dict:
                if "tags" in outer_dict[values[0]]:
                    values[1]["tags"] = outer_dict[values[0]]["tags"]

            if self._check_internal(values[1]):
                print(f"Internal variable {values[0]}")
                continue
            var_obj = self._expand_info_to_variable(values[1])
            if is_group:
                if "_auto_ivc" in self.om_problem.model.get_source(var_obj.name):
                    variable_dict[var_obj.name] = var_obj
            else:
                variable_dict[var_obj.name] = var_obj

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
        response_dict = {}  # Use dict to deduplicate by name
        for values in res:
            if values[0] in outer_dict:
                if "tags" in outer_dict[values[0]]:
                    values[1]["tags"] = outer_dict[values[0]]["tags"]

            if self._check_internal(values[1]):
                print(f"Internal response {values[0]}")
                continue
            resp_obj = self._expand_info_to_variable(values[1])
            response_dict[resp_obj.name] = resp_obj

        self._scanned_problem = OptProblem(
            name="scanned_problem",
            variables=list(variable_dict.values()),
            responses=list(response_dict.values()),
            objectives=[],
        )

    def _check_internal(self, info: dict) -> bool:
        """Check if this is a variable or response that is marked as internal.

        Args:
            info: The dictionary from OpenMDAO containing information
                about a variable or response.

        Returns:
            True if this is an internal variable or response.
        """
        internal_ele = False
        if "tags" in info:
            if "internal" in info["tags"]:
                internal_ele = True
        return internal_ele

    def _expand_info_to_variable(self, info: dict) -> FloatVariable:
        """Expand the information for a variable or response to a FloatVariable.

        Args:
            info: The dictionary from OpenMDAO containing information
                about a variable or response.

        Raises:
            TypeError: Currently we can only use variables or responses that
                are not arrays.

        Returns:
            A FloatVariable with the extracted information.
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

        var_name = info["prom_name"]
        shape = info["shape"]
        if shape != (1,):
            raise TypeError(f"Element {var_name} is of shape {shape}, must be (1, )")

        ref = helper("ref")
        ref0 = helper("ref0")
        adder = helper("adder")
        scaler = helper("scaler")
        adder, scaler = om_utils.determine_adder_scaler(ref0, ref, adder, scaler)

        default = info["val"][0] if "val" in info else 0.0

        return FloatVariable(
            name=var_name,
            bounds=[-np.inf, np.inf],
            shift=adder if adder is not None else 0.0,
            scale=scaler if scaler is not None else 1.0,
            default=default,
        )

    def _map_elements(self, info: dict) -> list:
        """Convert information for variables and responses from OpenMDAO format.

        Args:
            info: The dictionary from OpenMDAO containing information
                about variables or responses.

        Returns:
            A list of FloatVariable objects.
        """
        result = []
        for ele in info:
            bounds = [-np.inf, np.inf]
            scale = 1.0
            if "lower" in info[ele]:
                bounds = [info[ele]["lower"], info[ele]["upper"]]
            if "scaler" in info[ele] and info[ele]["scaler"] is not None:
                scale = info[ele]["scaler"]
            result.append(FloatVariable(name=ele, bounds=bounds, scale=scale))
        return result
