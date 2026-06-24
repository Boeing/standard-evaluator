"""
Created Aug. 31, 2022

@author Mikel Woo
"""

from abc import abstractmethod
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from importlib.metadata import version
from numpy.typing import NDArray
from pydantic import BaseModel

from standard_evaluator.evaluators import NumpyEvaluator
from standard_evaluator.utilities import (
    apply_types_from_evaluator_info,
    remove_duplicates,
    get_opt_problem_constant_vars,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo


def _legacy_problem_dict_to_opt_problem(problem_dict: dict) -> OptProblem:
    """Convert a legacy problem dictionary to an OptProblem.

    This is used only for deserializing old saved models. It is not part of
    the public API and should not be used for new code.

    Args:
        problem_dict: A legacy problem dict with structure like:
            {"variables": {"x": {"type": "float", "bounds": [...], ...}},
             "responses": {"y": {"type": "float", ...}}, ...}

    Returns:
        An OptProblem instance.
    """
    TYPE_MAPPING = {
        "float": "float",
        "int": "int",
        float: "float",
        int: "int",
        "categorical": "cat",
    }
    FIELDS = ["default", "shift", "scale"]
    full_dict = {"name": "legacy_problem"}
    for local_type in ["variables", "responses"]:
        local_info = []
        for local_name, info in problem_dict[local_type].items():
            local_dict = {"name": local_name}
            for field in FIELDS:
                if field in info:
                    local_value = info[field]
                    if isinstance(local_value, dict):
                        if local_value["use"]:
                            local_dict[field] = local_value["value"]
                    else:
                        local_dict[field] = info[field]
            if "type" in info:
                local_dict["class_type"] = TYPE_MAPPING[info["type"]]
            else:
                local_dict["class_type"] = "float"
            if local_dict["class_type"] == "cat":
                local_dict["bounds"] = info["bounds"]
            else:
                if "bounds" in info:
                    local_dict["bounds"] = tuple(info["bounds"])
            local_info.append(local_dict)
        full_dict[local_type] = local_info
    if "objectives" in problem_dict:
        full_dict["objectives"] = problem_dict["objectives"]
    if "constraints" in problem_dict:
        full_dict["constraints"] = problem_dict["constraints"]
    return OptProblem.model_validate(full_dict)


class SurrogateModel(NumpyEvaluator):
    """Defines behavior common to all surrogate models and what behaviors
    should be defined by the models themselves."""

    # ==================
    # |   Properties   |
    # ==================

    @property
    def constant_variables(self) -> Dict[int, Tuple[str, float]]:
        """The constant variables and the values they take on.

        Returns:
            Dict[int, Tuple[str, float]]: A dict with constant variable indices
                as keys and (name, value) tuples as values.
        """
        return get_opt_problem_constant_vars(self._opt_problem)

    @property
    def nonconstant_variables(self) -> List[str]:
        """The non-constant variables.

        Returns:
            List[str]: A list of the non-constant variables, in order.
        """
        return [
            var
            for var_idx, var in enumerate(self.inputs)
            if var_idx not in self.constant_variables
        ]

    @property
    def xlb(self) -> NDArray[np.float64]:
        """The lower left corner of the box that bounds the input sites.

        Returns:
            NDArray[np.float64]: An array of length nind containing the lower
                left corner of the bounding box.
        """
        return np.min(self.sites_input, axis=0)

    @property
    def xub(self) -> NDArray[np.float64]:
        """The upper right corner of the box that bounds the input sites.

        Returns:
            NDArray[np.float64]: An array of length nind containing the upper
                right corner of the bounding box.
        """
        return np.max(self.sites_input, axis=0)

    @property
    def nind(self) -> int:
        """The number of nonconstant independent variables.

        Returns:
            int: Count of nonconstant independent variables.
        """
        return len(self.nonconstant_variables)

    @property
    def sites(self) -> pd.DataFrame:
        """Sites currently used to calibrate the model.

        Returns:
            pd.DataFrame: The calibration sites.
        """
        return self._sites

    @property
    def sites_as_np(self) -> NDArray[np.float64]:
        """Sites currently used to calibrate the model as a float array.

        Returns:
            NDArray[np.float64]: The calibration sites as a NumPy array.
        """
        return self.remove_constants(self._sites)

    @property
    def sites_input(self) -> NDArray[np.float64]:
        """Input sites used to calibrate the model.

        Returns:
            NDArray[np.float64]: The input portion of the calibration sites.
        """
        return np.atleast_2d(
            self.dataframe_to_float_ndarray(self._sites[self.nonconstant_variables])
        )

    @property
    def sites_output(self) -> NDArray[np.float64]:
        """Output values used to calibrate the model.

        Returns:
            NDArray[np.float64]: The output portion of the calibration sites.
        """
        return np.atleast_2d(self.dataframe_to_float_ndarray(self._sites[self.outputs]))

    @property
    def nsites(self) -> int:
        """Number of sites in the model.

        Returns:
            int: The site count.
        """
        return len(self.sites)

    # ==============
    # |   Public   |
    # ==============

    def __init__(
        self,
        sites: pd.DataFrame,
        name: Optional[str] = None,
        comp_cost: float = 100,
        cache: str = None,
        cache_options: dict = None,
        logging: bool = False,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
        num_independent: int = None,
        num_dependent: int = None,
        options: Optional[BaseModel] = None,
        **kwargs,
    ) -> None:
        """Initialize the surrogate model.

        Args:
            sites: Sites to initialize the model with. Should include both
                variable and true response values.
            name: Name to give model for easier identification. Defaults to
                the model type name.
            comp_cost: Computational cost estimate. Defaults to 100.
            cache: Path to SQLite database for cached sites.
            cache_options: Options to modify caching behavior.
            logging: When True the evaluator keeps track of every site.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
            num_independent: Number of independent variables.
            num_dependent: Number of dependent variables.
            options: Pydantic model containing options for this surrogate model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name=name,
            comp_cost=comp_cost,
            cache=cache,
            cache_options=cache_options,
            logging=logging,
            interface=interface,
            opt_problem=opt_problem,
            num_independent=num_independent,
            num_dependent=num_dependent,
            options=options,
            **kwargs
        )

        # ensure data is valid w.r.t. problem specification
        self.check_consistency_of_sites(sites)

        # We want to make a copy of the sites passed in, and not modify them.
        # We also want to make sure there are no duplicates in the sites
        self._sites = remove_duplicates(sites, self.inputs, self.outputs)

    def __call__(self, sites: pd.DataFrame, names: List[str] = None) -> None:
        """Predict the desired response values for the given sites.

        Note:
            The passed dataframe will be modified in place.

        Args:
            sites: Sites to get predicted response values at.
            names: Which responses are computed. Defaults to None (all).
        """
        return super().__call__(sites, names=names)

    def _evaluate(self, sites: pd.DataFrame, **kwargs):
        """Evaluate the surrogate model. Modifies the dataframe in place.

        Args:
            sites: The dataframe containing non-constant input values,
                updated with the responses.
            **kwargs: Additional keyword arguments.
        """
        results = self.eval_np(self.remove_constants(sites)[:, : self.nind], **kwargs)
        # Store the results in the dataframe
        if "names" in kwargs:
            if kwargs["names"] is not None:
                sites.loc[:, kwargs["names"]] = results
            else:
                sites[self.outputs] = results
        else:
            sites[self.outputs] = results

    def update(self, add_sites: Union[pd.DataFrame, NDArray[np.float64]]) -> None:
        """Update the model with additional sites.

        Args:
            add_sites: Sites to add into the model. Should include variable
                and true response values.

        Raises:
            TypeError: If add_sites is not a DataFrame or NumPy array.
            ValueError: If the input array has the wrong number of columns or
                the DataFrame is missing required columns.
        """
        if not isinstance(add_sites, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                "SurrogateModel: add_sites must be a DataFrame or NumPy array!"
                f" Received a {type(add_sites).__name__} instead."
            )

        # If we get a numpy array we convert it to the full DataFrame, adding in constant
        # variables and ensure the dimensions are right
        if isinstance(add_sites, np.ndarray):
            # Check that the dimensions are right
            num_names = self.nind + len(self.outputs)
            if add_sites.shape[1] != num_names:
                raise ValueError(
                    type(self).__name__
                    + ": Input array has "
                    + "wrong number of columns! Expected array with "
                    + f"{num_names} columns but received array with "
                    + f"{add_sites.shape[1]}"
                )
            # Convert the passed in NumPy array to a DF
            add_sites_df = self._get_df_from_np(add_sites)
        else:
            name_set = set(self.inputs + self.outputs)
            if not name_set.issubset(add_sites.columns):
                raise ValueError(
                    "Input DataFrame does not contain all variables and responses!"
                )
            # We use the passed in DataFrame
            add_sites_df = add_sites

        # Remove duplicates from the additions sites
        add_sites_df = remove_duplicates(
            add_sites_df, variables=self.inputs, responses=self.outputs
        )

        # Combine the current model sites and the new model sites into a single DataFrame
        all_sites = pd.concat([self.sites, add_sites_df])
        apply_types_from_evaluator_info(all_sites, self.interface)

        # Remove duplicates
        all_sites = remove_duplicates(
            all_sites, variables=self.inputs, responses=self.outputs
        )

        # Remove constant variable values
        all_sites_np = self.remove_constants(all_sites)

        nind = self.nind

        # The number of new sites
        new_sites_number = len(all_sites) - self.nsites

        # We now call the actual training method. Note that the training method
        # needs to update the sites via the _update_sites method.
        # We pass in the combined sites from the current model and the new sites
        self._def_update(
            sites_input=all_sites_np[:, :nind],
            sites_output=all_sites_np[:, nind:],
            new_sites_number=new_sites_number,
        )

    def _update_sites(
        self, sites_input: NDArray[np.float64], sites_output: NDArray[np.float64]
    ) -> None:
        """Store the sites used for training the model.

        Expands the information to include constant variables.
        ToDo: In the future we will also make sure any rolling / unrolling happens here.

        Args:
            sites_input: The input variable values for training.
            sites_output: The response values for training.
        """
        sites_np = np.hstack([sites_input, sites_output])
        self._sites = self._get_df_from_np(sites_np)

    def _get_df_from_np(self, sites_np: NDArray[np.float64]) -> pd.DataFrame:
        """Convert a NumPy array of sites (without constants) to a DataFrame.

        Args:
            sites_np: The NumPy array with floating point values for all sites.

        Returns:
            pd.DataFrame: A DataFrame with the fully expanded sites including
                constant variables.
        """
        # add constant vars after dataframe is formed
        columns = self.nonconstant_variables + self.outputs
        add_sites_df = pd.DataFrame(sites_np, columns=columns)

        # add constants to dataframe
        constant_var_dict = self.constant_variables
        for const_var_idx, (constant_var, constant_value) in constant_var_dict.items():
            add_sites_df[constant_var] = constant_value
            columns.insert(const_var_idx, constant_var)

        # updated columns sort dataframe
        return add_sites_df[columns]

    def to_dict(self) -> dict:
        """Save all information necessary to rebuild the model into a dictionary.

        Returns:
            dict: The dictionary containing all the information needed to rebuild
                this model.
        """
        return {
            "type": type(self).__name__,
            "info": self._def_to_dict(),
            "opt_problem": self._opt_problem.model_dump(),
            "version": version("standard_evaluator"),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, model_info: dict) -> "SurrogateModel":
        """Build a model from dictionary information.

        Parameters
        ----------
        model_info : dict
            Information needed to build the model. Should have the following:

            .. code-block:: python

                {
                    'type': 'TypeOfModel',
                    'info': {
                        # Parameters needed to build model
                    },
                    'opt_problem': {
                        # OptProblem serialized via model_dump()
                    },
                    'name': 'name_of_model',
                    'version': '6.0.0'
                }

        Returns
        -------
        SurrogateModel
            An instance of the model built using the provided information.

        Raises
        ------
        TypeError
            model_info is not a dictionary.
        KeyError
            model_info is missing necessary keys.
        NameError
            model_info['type'] is not a valid model type.
        ValueError
            Trying to instantiate a model with information from a different
            model.
        """
        # Make sure a dictionary was passed in
        if not isinstance(model_info, dict):
            raise TypeError(
                "SurrogateModel: model_info must be a dictionary! "
                f"Received a {type(model_info).__name__} instead."
            )

        # Make sure it has the required keys
        has_opt_problem = "opt_problem" in model_info
        has_legacy_problem = "problem" in model_info

        if has_opt_problem:
            required_keys = {"type", "info", "opt_problem", "name", "version"}
        elif has_legacy_problem:
            required_keys = {
                "type", "info", "problem", "name", "version",
                "design explorer version",
            }
        else:
            required_keys = {"type", "info", "opt_problem", "name", "version"}

        if not required_keys.issubset(model_info):
            raise KeyError(
                "SurrogateModel: model_info is missing the following "
                f'keys: {", ".join(required_keys.difference(model_info))}'
            )

        # Function was called using the abstract class
        if cls.__name__ == "SurrogateModel":
            # Get all available surrogate models
            children = cls.__subclasses__()

            # Check if type matches any of the model names
            for child in children:
                if model_info["type"].lower() == child.__name__.lower():
                    instance = child._def_from_dict(model_info)
                    break
            else:
                raise NameError(
                    f'SurrogateModel: {model_info["type"]} is not a '
                    "valid surrogate model! The valid models are:\n\n"
                    + "\n".join([child.__name__ for child in children])
                )

        # Function was called using a specific model
        else:
            # Make sure the type matches the class name
            if cls.__name__.lower() != model_info["type"].lower():
                raise ValueError(
                    f"{cls.__name__}: Cannot instantiate a "
                    f'{model_info["type"]} as a {cls.__name__}! Try using the '
                    "SurrogateModel.from_dict method instead."
                )

            # Instantiate the class with provided info
            instance = cls._def_from_dict(model_info)

        return instance

    def check_consistency_of_sites(self, sites: pd.DataFrame):
        """Validate that sites are consistent with the problem definition.

        Args:
            sites: Sites to be checked.

        Raises:
            TypeError: If sites is not a DataFrame.
            ValueError: If sites are missing input variables or fixed variable
                values do not match their bounds.
        """
        if not isinstance(sites, pd.DataFrame):
            raise TypeError(
                "SurrogateModel: sites not given as DataFrame!"
                f" Received a {type(sites).__name__} instead."
            )

        for var in self.inputs:
            if var not in sites:
                raise ValueError(
                    "SurrogateModel: sites given do not contain data for "
                    f" the input variable {var}."
                )

        # Check fixed variables using opt_problem bounds
        for var_idx, (var_name, fixed_value) in self.constant_variables.items():
            if var_name in sites and not all(sites[var_name] == fixed_value):
                raise ValueError(
                    f"SurrogateModel: site data for {var_name} not in the fixed value {fixed_value}"
                )

    def remove_constants(self, sites: pd.DataFrame) -> NDArray[np.float64]:
        """Strip constant variables from sites and return as a float array.

        Args:
            sites: Sites to be reduced to an array.

        Returns:
            NDArray[np.float64]: Array of nonconstant input and output values.
        """
        self.check_consistency_of_sites(sites)

        cols = [
            var for var in self.nonconstant_variables + self.outputs if var in sites
        ]
        return self.dataframe_to_float_ndarray(sites[cols])

    def check_input_array(self, site_inputs: np.ndarray) -> None:
        """Validate that site_inputs is an (m x nind) array.

        Args:
            site_inputs: Array whose rows are sites and columns are
                nonconstant variables.

        Raises:
            TypeError: If site_inputs is not an ndarray or has the wrong shape.
        """
        if not isinstance(site_inputs, np.ndarray):
            raise TypeError("SurrogateModel: Input is not an array as expected.")
        if np.shape(site_inputs)[1] != self.nind and len(np.shape(site_inputs)) == 2:
            raise TypeError(
                f"SurrogateModel: Input array does not have the "
                f"shape ( _," + str(self.nind) + ")."
            )

    def get_response_indices(self, names: List[str] = None) -> List[int]:
        """Get the indices of the responses in the names list.

        Args:
            names: List of response names to get indices for. If None, all
                responses are used.

        Returns:
            List[int]: List of indices for the requested responses.
        """
        if names is None:
            target_responses = self.outputs
        else:
            target_responses = names

        response_indices = [self.outputs.index(r) for r in target_responses]
        return response_indices

    # ================
    # |   Abstract   |
    # ================

    @abstractmethod
    def eval_np(self, sites: np.ndarray, names: list = None) -> NDArray[np.float64]:
        """Predict the desired response values for the given sites.

        Args:
            sites: Sites to compute predicted response values.
            names: Which responses are computed. Defaults to None (all).

        Returns:
            NDArray[np.float64]: Predicted response values for the given sites.
        """

    @abstractmethod
    def _def_update(
        self,
        sites_input: NDArray[np.float64],
        sites_output: NDArray[np.float64],
        new_sites_number: int,
    ):
        """Define how the model should be updated when additional sites are added.

        Args:
            sites_input: The variable values including existing and new sites.
            sites_output: The response values including existing and new sites.
            new_sites_number: The number of new sites.
        """

    @abstractmethod
    def _def_to_dict(self) -> dict:
        """Define the information needed to rebuild the model.

        The dictionary that is returned will be assigned to the 'info' key
        of the model dictionary.

        Returns:
            dict: Dictionary containing the parameters, site data, etc.
                needed to rebuild the model.
        """

    @classmethod
    @abstractmethod
    def _def_from_dict(cls, model_info: dict) -> "SurrogateModel":
        """Define how the model gets rebuilt from the provided model information.

        Args:
            model_info: Dictionary containing model information.

        Returns:
            SurrogateModel: Instantiated model built from the given information.
        """
