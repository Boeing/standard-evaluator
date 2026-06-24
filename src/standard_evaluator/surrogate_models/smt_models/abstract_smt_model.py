"""
Created Apr. 09, 2024

@author Anjali Prasad
"""

from pathlib import Path
from typing import Optional, Type

import numpy as np
from numpy.typing import NDArray

import pandas as pd
import dask
import smt.surrogate_models
from pydantic import BaseModel, ConfigDict, Field

from standard_evaluator.surrogate_models.abstract_model import SurrogateModel
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo


class AbstractSmtModelParameters(BaseModel):
    """Trained SMT model objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: list  # List of trained SMT surrogate model objects


class AbstractSmtModelOptions(BaseModel):
    """Base options for all SMT-based surrogate models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    print_global: bool = Field(default=True, description="Global print toggle")
    print_training: bool = Field(
        default=True, description="Whether to print training information"
    )
    print_prediction: bool = Field(
        default=True, description="Whether to print prediction information"
    )
    print_problem: bool = Field(
        default=True, description="Whether to print problem information"
    )
    print_solver: bool = Field(
        default=True, description="Whether to print solver information"
    )
    use_xlimits: bool = Field(
        default=False, description="Whether to use variable bounds as xlimits"
    )
    data_dir: Optional[Path] = Field(
        default=None, description="Directory for cached data"
    )
    parameters: Optional[AbstractSmtModelParameters] = None


def _options_to_smt_dict(
    options: AbstractSmtModelOptions, opt_problem=None, nonconstant_variables=None
) -> dict:
    """Convert Pydantic options to dict for SMT model constructor.

    Excludes 'parameters', 'use_xlimits', and handles 'data_dir' conversion.
    Computes 'xlimits' from opt_problem when use_xlimits is True.

    :param options: Pydantic options instance for the SMT model
    :type options: AbstractSmtModelOptions
    :param opt_problem: OptProblem defining variable bounds (needed when use_xlimits=True)
    :type opt_problem: OptProblem, optional
    :param nonconstant_variables: List of non-constant variable names to include in xlimits
    :type nonconstant_variables: list, optional
    :return: Dictionary suitable for passing as kwargs to SMT model constructor
    :rtype: dict
    """
    exclude_fields = {"parameters", "use_xlimits", "data_dir"}
    result = {}

    for field_name, value in options.model_dump().items():
        if field_name in exclude_fields:
            continue
        result[field_name] = value

    # Handle xlimits
    if options.use_xlimits and opt_problem is not None:
        xlimits = []
        for var in opt_problem.variables:
            if nonconstant_variables is None or var.name in nonconstant_variables:
                xlimits.append(list(var.bounds))
        result["xlimits"] = np.array(xlimits)

    # Handle data_dir
    if options.data_dir is not None:
        result["data_dir"] = str(options.data_dir)

    return result


class AbstractSmtModel(SurrogateModel):
    """Defines behavior common to all smt toolbox specific surrogate models and what behaviors should
    be defined by the models themselves.
    """

    # ==============
    # |   Public   |
    # ==============

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The AbstractSmtModelOptions class defining SMT model options.
        """
        return AbstractSmtModelOptions

    def full_options(self) -> BaseModel:
        """Return the live Pydantic options instance for this model.

        Overrides the base implementation which returns a fresh default
        instance. This returns the actual runtime options, which may include
        trained model parameters set during or after construction.

        Returns
        -------
        BaseModel
            The active Pydantic options instance.
        """
        return self._options

    # --- Fitted parameter property (read-only, single source of truth) ---

    @property
    def model(self) -> list:
        """Trained SMT model objects (read-only).

        :return: List of trained SMT surrogate model objects.
        :rtype: list
        :raises AttributeError: If the model has not been trained or
            initialized with parameters.
        """
        if self._options.parameters is None:
            raise AttributeError(
                f"{type(self).__name__} has not been trained or "
                "initialized with parameters"
            )
        return self._options.parameters.model

    @model.setter
    def model(self, value):
        raise AttributeError(
            f"Cannot set 'model' directly on {type(self).__name__}. "
            "Use options with parameters=AbstractSmtModelParameters(...) instead."
        )

    @dask.delayed
    def _build_single_model(
        self, sites: np.ndarray, site_vals: np.ndarray
    ) -> "SurrogateModel":
        """
        :return: The SurrogateModel containing the data of the model file
        :rtype: SurrogateModel
        """
        # Convert Pydantic options to dict for SMT model constructor
        local_options = _options_to_smt_dict(
            self._options,
            opt_problem=self.opt_problem,
            nonconstant_variables=self.nonconstant_variables,
        )

        # Builds smt model
        sm = self.model_type(**local_options)
        sm.set_training_values(sites, site_vals)
        sm.train()
        return sm

    def _train(
        self, sites_input: NDArray[np.float64], sites_output: NDArray[np.float64]
    ) -> None:
        """
        Constructs the model for each responses & save the model for each response, then
        builds RBF Mathematical Model
        """

        sm_builds = []

        # Get the indices of all responses. We use this since there might be responses
        # that are themselves arrays, and hence they will map to multiple columns
        response_indices = self.get_response_indices()
        for indx in response_indices:
            # Call parallel build utility to get dask.Delayed object
            sm_builds.append(
                self._build_single_model(
                    sites=sites_input,
                    site_vals=sites_output[:, indx],
                )
            )

        # Build models in parallel
        trained_models = list(dask.compute(*sm_builds))

        # Rebuild self._options.parameters with trained models
        self._options.parameters = AbstractSmtModelParameters(model=trained_models)

    def __init__(
        self,
        sites: pd.DataFrame,
        model_type: smt.surrogate_models.surrogate_model.SurrogateModel,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Constructor

        Args:
            sites: Sites to initialize the model with. Should include both
                variable and true response values.
            model_type: SMT surrogate model type.
            options: Pydantic BaseModel instance with model options. If
                ``options.parameters`` is provided, the model is initialized
                directly from the trained state; otherwise training occurs.
            name: Name to give model for easier identification. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        # Coerce parent-class options to the correct subclass so the base
        # Evaluator type check passes. This allows callers to pass e.g.
        # AbstractSmtModelOptions to any concrete SMT model.
        if options is not None and isinstance(options, BaseModel):
            expected_cls = type(self)._define_options()
            if not isinstance(options, expected_cls):
                options = expected_cls.model_validate(options.model_dump())

        super().__init__(
            sites=sites,
            name=name,
            interface=interface,
            opt_problem=opt_problem,
            options=options,
        )
        self.model_type = model_type

        # Branch on options.parameters for direct init vs training
        if self._options.parameters is not None:
            # Direct initialization path: use pre-trained model objects
            pass
        else:
            # Training path: train from sites
            sites_np = self.sites_as_np
            nind = self.nind
            self._train(
                sites_input=sites_np[:, :nind],
                sites_output=sites_np[:, nind:],
            )

    def eval_np(self, sites: np.ndarray, names: list = None) -> np.ndarray:
        """Predict the desired response values (given by ``names``) for the given sites.

        :param sites: Sites to compute predicted response values.
        :type sites: np.ndarray
        :param names: Which responses are computed, defaults to None
        :type names: list[str], optional
        :return: Predicted response values for the given sites.
        :rtype: np.ndarray
        """
        # Gathering predicted results over each response
        result = []
        # Get the indices of all responses. We use this since there might be responses
        # that are themselves arrays, and hence they will map to multiple columns
        response_indices = self.get_response_indices(names)
        for indx in response_indices:
            sm = self.model[indx]
            predicted_vals = sm.predict_values(sites)
            result.append(np.squeeze(predicted_vals))

        # Reshape output
        result_eval = np.vstack(result).T.reshape(
            (sites.shape[0], len(response_indices))
        )

        return result_eval

    def _def_update(
        self,
        sites_input: NDArray[np.float64],
        sites_output: NDArray[np.float64],
        new_sites_number: int,
    ):
        """Define how the model should be updated when additional sites are added.

        Arguments:
            sites_input {NDArray[np.float64]} -- The NumPy variable values to be use when updating the models. Also includes existing sites.
            sites_output {NDArray[np.float64]} -- The NumPy response values to be use when updating the models. Also includes existing sites.
            new_sites_number {int} -- The number of new sites
        """
        # Adjust site data
        self._update_sites(sites_input=sites_input, sites_output=sites_output)

        # Store the updated sites and train the SMT model with new sites
        self._train(sites_input=sites_input, sites_output=sites_output)

    def remove_duplicates(self, sites_df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicated sites
        The method checks that if there are duplicate rows in the dataframe for variables that they are also duplicates for responses,
        throws an error if that is not the case, and returns the DataFrame with all duplicates removed.

        :param sites_df: Sites to add to model
        :type sites_df: pd.Dataframe

        :return: no duplicated sites
        :rtype: pd.Dataframe
        """
        # Checks for duplicate values in variables and the passed sites and accordingly raise ValueError
        duplicated_variable_sites = sites_df.duplicated(subset=self.inputs)
        duplicated_sites = sites_df.duplicated(subset=self.inputs + self.outputs)

        # If there are duplicates in the variables that are not there in the responses then we need to throw error message
        if duplicated_variable_sites.equals(duplicated_sites) != True:
            raise ValueError(
                "There are sites that have the same variable values but different responses."
            )

        # drop the duplicated rows
        sites_df.drop_duplicates(subset=self.inputs + self.outputs, inplace=True)

        return sites_df

    def _variance(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the variance of the predictions at a number of points.

        :param x: An array (n_pts x nind) of points at which to compute the variance, Input values for the prediction points.
        :type x: np.ndarray

        :return: An array (n_pts x ndep) of estimated variances.
        :rtype: np.nd
        """
        variance_estimates = []
        response_indices = self.get_response_indices()
        # TODO: This should be nice to parallelize!!
        for indx in response_indices:
            sm = self.model[indx]

            # Predicts variance
            variance_predicted = sm.predict_variances(x)
            variance_estimates.append(variance_predicted)
        result_variance = np.stack(
            variance_estimates, axis=1
        )  # convert result to numpy matrix that contains a column for each response
        return result_variance

    def jacobian(self, x: np.array) -> np.array:
        """
        Computes the jacobian of the model at a set of input sites.

        :param x: An array (npts x nind) of input vectors at which to compute the derivative information.
        :type x: np.array

        :return: An array (npts x nind x ndep) of jacobian matrices indexed by the input sites.
        :rtype: np.array
        """
        gradients = []
        response_indices = self.get_response_indices()
        for indx in response_indices:
            sm = self.model[indx]
            grad = []
            for ind in range(self.nind):
                # computes the gradient and apppends to the list
                grad.append(sm.predict_derivatives(x, ind))
            result_grad = np.hstack(grad).reshape((x.shape[0], self.nind))

            # assemble the gradients into a matrix
            gradients.append(result_grad)

        # Stack the gradients to return jacobian

        jac = np.stack(gradients, axis=-1)
        jac = np.swapaxes(jac, 1, 2)

        return jac

    def _def_to_dict(self) -> dict:
        """Define the information needed to rebuild the model. The dictionary
        that is returned will be assigned to the 'info' key of the model dictionary.

        Serializes the Pydantic options (excluding non-serializable trained
        model objects) along with site data and model metadata.

        Returns
        -------
        dict
            Dictionary containing the options, site data, etc. needed to
            rebuild the model.
        """
        # Serialize options excluding non-serializable trained model objects
        options_data = self._options.model_dump(exclude={"parameters"})

        model_dict = {
            "name": self.name,
            "variables": self.nonconstant_variables,
            "responses": self.outputs,
            "sites_input": self.sites_input.tolist(),
            "sites_output": self.sites_output.tolist(),
            "options": options_data,
            "model_type": type(self).__name__,
        }

        return model_dict

    @classmethod
    def _def_from_dict(cls, model_info: dict) -> "SurrogateModel":
        """Define how the model gets rebuilt given the provided model
        information. The info key contains a dictionary with all of the data
        needed for rebuilding the model like parameters, site information, etc.
        Also of interest might be the version key which specifies which version
        of Design Explorer the model information was generated with. Use this
        for any necessary backward compatibility.

        Handles both:
        - New Pydantic format: ``info`` dict has an ``"options"`` key containing
          serialized Pydantic options (produced by ``model_dump``).
        - Legacy format: ``info`` dict has individual option key-value pairs
          without an ``"options"`` wrapper key (for models saved before the
          Pydantic migration).

        Parameters
        ----------
        model_info : dict
            Information about the model

        Returns
        -------
        SurrogateModel
            Instantiated model built from the given information.
        """
        info = model_info["info"]

        # Support both new format (opt_problem) and legacy format (problem key)
        if "opt_problem" in model_info:
            opt_prob = OptProblem.model_validate(model_info["opt_problem"])
        elif "problem" in model_info:
            from standard_evaluator.surrogate_models.abstract_model import (
                _legacy_problem_dict_to_opt_problem,
            )
            opt_prob = _legacy_problem_dict_to_opt_problem(model_info["problem"])
        else:
            raise KeyError("model_info must contain 'opt_problem' or 'problem' key")

        # Extract site data
        model_name = info.get("name")
        variables = info.get("variables", [])
        responses = info.get("responses", [])
        sites_input = np.array(info["sites_input"])
        sites_output = np.array(info["sites_output"])

        # Reconstruct DataFrame from sites
        sites_array = np.hstack([sites_input, sites_output])
        sites_df = pd.DataFrame(
            data=sites_array, columns=variables + responses
        )

        # Determine the options class for this model
        options_cls = cls._define_options()

        # Detect format: new Pydantic format has "options" key
        if "options" in info:
            # New Pydantic format — use model_validate
            options = options_cls.model_validate(info["options"])
        else:
            # Legacy format — individual keys stored at top level
            # Collect any keys that match options fields
            options_fields = set(options_cls.model_fields.keys())
            old_options_data = {}
            for key in info:
                if key in options_fields:
                    old_options_data[key] = info[key]
            options = options_cls.model_validate(old_options_data)

        return cls(
            sites=sites_df,
            name=model_name,
            opt_problem=opt_prob,
            options=options,
        )
