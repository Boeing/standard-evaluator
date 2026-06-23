"""Polynomial response surface surrogate model."""

from enum import Enum
from typing import List, Optional, Type

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpydantic import NDArray as NpNDArray, Shape
from pydantic import BaseModel

from standard_evaluator.surrogate_models.abstract_model import SurrogateModel
from standard_evaluator.surrogate_models.polynomial_model_utils.monomial_ordering import (
    grlex_ordering_to_deg,
    grrevlex_ordering_to_deg,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator.utilities.utility import restrict_problem


class CoefficientOrdering(str, Enum):
    """Enumeration of valid coefficient ordering methods for polynomial models."""

    DEC_GRLEX = "dec_grlex"
    ASC_GRLEX = "asc_grlex"
    DEC_GRREVLEX = "dec_grrevlex"
    ASC_GRREVLEX = "asc_grrevlex"


class PolynomialModelParameters(BaseModel):
    """Pydantic model storing fitted/computed parameters for a PolynomialModel."""

    coefficients: NpNDArray[Shape["*,..."], np.float64]
    degree_exponents: NpNDArray[Shape["*,..."], np.float64]


class PolynomialModelOptions(BaseModel):
    """Pydantic model defining options for PolynomialModel."""

    degree: int = 1
    coefficient_ordering: CoefficientOrdering = CoefficientOrdering.DEC_GRLEX
    parameters: Optional[PolynomialModelParameters] = None


class PolynomialModel(SurrogateModel):
    """
    Class representing a polynomial response surface model.

    Attributes
    ----------
    degree : int
        The degree of the polynomial model.
    coefs : numpy.ndarray
        A numpy array (nterms x ndep) containing the coefficients of the polynomial responses.
    nterms : int
        The number of coefficients for an output of the polynomial model.
    deg_exp : numpy.ndarray
        A numpy array (nterms x nind) containing the degree exponents of the polynomial responses.
    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The PolynomialModelOptions class defining polynomial model parameters.
        """
        return PolynomialModelOptions

    # ==================
    # |   Properties   |
    # ==================

    @property
    def degree(self) -> int:
        """The degree of the polynomial model."""
        return self.lookup_option_value("degree")

    @property
    def coefs(self) -> np.ndarray:
        """Coefficients array (nterms x ndep) of the polynomial model."""
        return self.lookup_option_value("parameters").coefficients

    @property
    def nterms(self) -> int:
        """The number of terms in the polynomial model."""
        return self.lookup_option_value("parameters").coefficients.shape[1]

    @property
    def deg_exp(self) -> np.ndarray:
        """Degree exponents array (nterms x nind) of the polynomial model."""
        return self.lookup_option_value("parameters").degree_exponents

    def __init__(
        self,
        sites: pd.DataFrame,
        name: Optional[str] = None,
        options: Optional[PolynomialModelOptions] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
        num_independent: Optional[int] = None,
        num_dependent: Optional[int] = None,
        **kwargs
    ):
        """Constructor for PolynomialModel.

        Parameters
        ----------
        sites : pd.DataFrame
            Sites to initialize the model with. Should include both
            variable and true response values.
        name : str, optional
            Name to give model for easier identification, by default None.
        options : PolynomialModelOptions, optional
            Pydantic model containing options for this polynomial model, by default None.
        interface : EvaluatorInfo, optional
            Evaluator interface information, by default None.
        opt_problem : OptProblem, optional
            Optimization problem definition, by default None.
        num_independent : int, optional
            Number of independent variables, by default None.
        num_dependent : int, optional
            Number of dependent variables, by default None.
        **kwargs
            Additional keyword arguments.
        """
        # Initialize parent class
        super().__init__(
            sites=sites,
            name=name,
            options=options,
            interface=interface,
            opt_problem=opt_problem,
            num_independent=num_independent,
            num_dependent=num_dependent,
            **kwargs
        )

        # Get options values
        degree_val = self.lookup_option_value("degree")
        params = self.lookup_option_value("parameters")

        # Build the model
        if params is None:
            # Need to build coefficients from data
            self._model_building(sites, degree_val)
        else:
            coefficients_val = params.coefficients
            degree_exponents_val = params.degree_exponents

            # Validate compatibility if degree_exponents were provided (non-empty)
            if (degree_exponents_val.size > 0
                    and coefficients_val.shape[1] != degree_exponents_val.shape[0]):
                raise ValueError(
                    f"Incompatible shapes: coefficients has {coefficients_val.shape[1]} terms "
                    f"but degree_exponents has {degree_exponents_val.shape[0]} terms"
                )

            # Generate degree exponents if not provided
            if degree_exponents_val.size == 0:
                ordering = self.lookup_option_value("coefficient_ordering")
                degree_exponents_val = self._generate_degree_exponents(
                    self.nind, degree_val, ordering
                )

            # Store final parameters
            self._options.parameters = PolynomialModelParameters(
                coefficients=coefficients_val.astype(float),
                degree_exponents=degree_exponents_val.astype(float)
            )

    def eval_np(self, x: np.array, names: Optional[List[str]] = None) -> np.ndarray:
        """Evaluate the polynomial model at given points.

        Parameters
        ----------
        x : np.array
            An array (npts x nind) containing the desired evaluation locations.
        names : List[str], optional
            Which responses are computed, by default None.

        Returns
        -------
        np.ndarray
            An array (npts x ndep) of model output values.
        """
        # Check input
        self.check_input_array(x)

        # Form all required monomials of inputs
        x = x[:, np.newaxis, :]
        x_monos = np.prod(x**self.deg_exp, axis=-1)

        # Multiply against the coefficients
        evals = x_monos.dot(self.coefs.T).astype(float)

        # Filter response values
        if names is not None:
            evals = evals[:, self.get_response_indices(names)]

        return evals

    def jacobian(self, x: np.array) -> np.ndarray:
        """Compute the jacobian of the polynomial model at a set of points.

        Args:
            x: Array (npts x nind) of evaluation locations.

        Returns:
            np.ndarray: Array (npts x nind x ndep) of jacobian values.
        """
        # Check input
        self.check_input_array(x)

        # Decrement powers and grab coefficients
        deg_exp_rep = np.repeat(self.deg_exp[:, :, np.newaxis], self.nind, axis=2)
        power_coefs = np.diagonal(deg_exp_rep, axis1=1, axis2=2)
        deg_dec = deg_exp_rep - np.eye(self.nind)[np.newaxis, :, :]
        deg_dec = np.clip(deg_dec, 0, self.degree)

        # Form all required monomials of inputs
        x = x[:, np.newaxis, :, np.newaxis]
        sites_output = (np.prod(x**deg_dec, axis=2) * power_coefs).astype(float)

        # Construct the jacobian matrices
        jac = np.einsum("ijk,lj->ilk", sites_output, self.coefs)

        return jac

    def get_response_models(self, target_responses: List[str]) -> "PolynomialModel":
        """Restrict the polynomial model to the target set of responses.

        Args:
            target_responses: A list of desired response names.

        Returns:
            PolynomialModel: A new model restricted to the target responses.

        Raises:
            ValueError: If target responses are not in the model.
        """
        # Check responses are in model
        if not set(target_responses).issubset(set(self.outputs)):
            raise ValueError("Target responses must be contained in model responses.")

        # Get response indices
        target_indices = self.get_response_indices(target_responses)

        # Project coefficients to subspace
        ss_coefs = self.coefs[target_indices, :]

        # Get necessary data
        ss_sites = self._get_df_from_np(self.sites_as_np)
        ss_deg = self.degree
        ss_deg_exp = self.deg_exp

        # Build restricted opt problem
        filtered_responses = [
            resp.name for resp in self.opt_problem.responses
            if resp.name in target_responses
        ]
        ss_opt_problem = restrict_problem(
            full_problem=self.opt_problem, responses=filtered_responses
        )

        # Create options for the restricted model
        restricted_options = PolynomialModelOptions(
            degree=ss_deg,
            coefficient_ordering=self.lookup_option_value("coefficient_ordering"),
            parameters=PolynomialModelParameters(
                coefficients=ss_coefs,
                degree_exponents=ss_deg_exp
            )
        )

        # Build projected polynomial model
        ss_poly_model = PolynomialModel(
            sites=ss_sites,
            options=restricted_options,
            name=self.name,
            opt_problem=ss_opt_problem,
        )

        return ss_poly_model

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
        pass

    def _def_to_dict(self) -> dict:
        """Return dictionary describing the polynomial surrogate model.

        Returns:
            dict: Dictionary containing model type, problem, and info needed
                to reproduce the model.
        """
        return {
            "sites": self._sites.to_dict(),
            "sites_dtypes": convert_df_datatypes_to_list(self._sites.dtypes),
            "options": {
                "degree": self.degree,
                "coefficient_ordering": self.lookup_option_value(
                    "coefficient_ordering"
                ).value,
                "parameters": {
                    "coefficients": self.coefs.tolist(),
                    "degree_exponents": self.deg_exp.tolist(),
                },
            },
        }

    @classmethod
    def _def_from_dict(cls, model_dict: dict) -> "PolynomialModel":
        """Initialize a PolynomialModel from a dictionary containing model data.

        Args:
            model_dict: Dictionary containing model information.

        Returns:
            PolynomialModel: Model object generated from the input dictionary.

        Raises:
            KeyError: If required keys are missing from model_dict['info'].
        """
        info = model_dict["info"]

        # Detect old vs new format and handle accordingly
        if "options" in info:
            # New format with options
            required_keys = ["sites", "sites_dtypes"]
            for key in required_keys:
                if key not in info:
                    raise KeyError(
                        f"required key {key} not found in model_dict['info']"
                    )

            # Extract options data
            options_data = info["options"]
            params_data = options_data.get("parameters", {})
            params = PolynomialModelParameters(
                coefficients=np.asarray(
                    params_data["coefficients"], dtype=np.float64
                ),
                degree_exponents=np.asarray(
                    params_data["degree_exponents"], dtype=np.float64
                )
            )
            options = PolynomialModelOptions(
                degree=options_data.get("degree", 1),
                coefficient_ordering=CoefficientOrdering(
                    options_data.get("coefficient_ordering", "dec_grlex")
                ),
                parameters=params
            )
        else:
            # Old format without options
            required_keys = ["sites", "sites_dtypes", "degree", "coefs", "deg_exp"]
            for key in required_keys:
                if key not in info:
                    raise KeyError(
                        f"required key {key} not found in model_dict['info']"
                    )

            # Handle coefficient ordering (backward compatibility)
            coefficient_ordering = info.get("coefficient_ordering", "dec_grlex")
            if isinstance(coefficient_ordering, str):
                coefficient_ordering = CoefficientOrdering(coefficient_ordering)

            # Create options from individual fields
            params = PolynomialModelParameters(
                coefficients=np.asarray(info["coefs"], dtype=np.float64),
                degree_exponents=np.asarray(info["deg_exp"], dtype=np.float64)
            )
            options = PolynomialModelOptions(
                degree=info["degree"],
                coefficient_ordering=coefficient_ordering,
                parameters=params
            )

        # Reconstruct sites DataFrame
        sites_df = pd.DataFrame.from_dict(info["sites"])
        for column, column_type in zip(sites_df.columns, info["sites_dtypes"]):
            sites_df[column] = sites_df[column].astype(column_type)

        # Support both new format (opt_problem) and legacy format (problem key)
        if "opt_problem" in model_dict:
            model_opt_prob = OptProblem.model_validate(model_dict["opt_problem"])
        elif "problem" in model_dict:
            from standard_evaluator.surrogate_models.abstract_model import (
                _legacy_problem_dict_to_opt_problem,
            )
            model_opt_prob = _legacy_problem_dict_to_opt_problem(model_dict["problem"])
        else:
            raise KeyError("model_dict must contain 'opt_problem' or 'problem' key")

        poly_model = PolynomialModel(
            sites=sites_df,
            options=options,
            opt_problem=model_opt_prob,
            name=model_dict.get("name"),
        )

        return poly_model

    def _generate_degree_exponents(
        self, nind: int, degree: int, ordering: CoefficientOrdering
    ) -> np.ndarray:
        """Generate degree exponents based on the specified ordering.

        Parameters
        ----------
        nind : int
            Number of independent variables.
        degree : int
            Maximum degree of the polynomial.
        ordering : CoefficientOrdering
            The coefficient ordering method to use.

        Returns
        -------
        np.ndarray
            Array of degree exponents.

        Raises
        ------
        ValueError
            If an invalid ordering is specified.
        """
        if ordering == CoefficientOrdering.DEC_GRLEX:
            return np.array(grlex_ordering_to_deg(nind, degree)[::-1])
        if ordering == CoefficientOrdering.ASC_GRLEX:
            return np.array(grlex_ordering_to_deg(nind, degree))
        if ordering == CoefficientOrdering.DEC_GRREVLEX:
            return np.array(grrevlex_ordering_to_deg(nind, degree)[::-1])
        if ordering == CoefficientOrdering.ASC_GRREVLEX:
            return np.array(grrevlex_ordering_to_deg(nind, degree))
        raise ValueError(f"Invalid coefficient ordering: {ordering}")

    def _model_building(self, sites: pd.DataFrame, degree: int) -> None:
        """Build the polynomial model from data.

        Parameters
        ----------
        sites : pd.DataFrame
            Sites containing both variable and response values.
        degree : int
            Degree of the polynomial model.
        """
        # Get coefficient ordering
        ordering = self.lookup_option_value("coefficient_ordering")

        # Generate degree exponents
        deg_exp = self._generate_degree_exponents(
            self.nind, degree, ordering
        ).astype(float)

        # Get sites input and output as numpy arrays
        sites_input = self.sites_input
        sites_output = self.sites_output

        # Generate input monomials
        input_monos = np.prod(
            sites_input[:, np.newaxis, :] ** deg_exp, axis=-1
        ).astype(float)

        # Least squares solve for coefficients
        lstsq_res = np.linalg.lstsq(input_monos, sites_output, rcond=None)
        coefs = lstsq_res[0].T

        # Store parameters in options
        self._options.parameters = PolynomialModelParameters(
            coefficients=coefs,
            degree_exponents=deg_exp
        )


def convert_df_datatypes_to_list(df_datatypes: pd.Series) -> List[str]:
    """Convert DataFrame dtypes to a list of strings.

    Parameters
    ----------
    df_datatypes : pd.Series
        Series containing DataFrame column dtypes.

    Returns
    -------
    List[str]
        List of dtype strings.
    """
    dtype_list = []
    for i in range(0, len(df_datatypes)):
        dtype_list.append(str(df_datatypes.iloc[i]))
    return dtype_list
