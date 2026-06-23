"""
Abstract Evaluator base class.

Defines behavior common to all evaluators and what behaviors should be
defined by the evaluators themselves.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type, Union, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, conint, field_validator
from typing import Optional as _Optional

import standard_evaluator as se

import standard_evaluator.utilities as utils

from standard_evaluator.evaluators.decorators.cache import cacher
from standard_evaluator.evaluators.decorators.site_logger import site_logger
from standard_evaluator.problem import OptProblem
from standard_evaluator.converters import (
    opt_problem_to_evaluator_info,
    evaluator_info_to_opt_problem,
)
from standard_evaluator.evaluator import EvaluatorInfo


from standard_evaluator.utilities.mapping import (
    res_element_to_string,
    flatten_items_to_arrays,
    compress_whitespace,
)
from standard_evaluator.utilities.option_merge import combine_instances


class _EmptyOptions(BaseModel):
    """Default empty options model."""
    pass


PositiveInt = conint(strict=True, gt=0)


class ValidInputs(BaseModel):
    """Pydantic model to validate num_independent/num_dependent for TestEvaluator subclasses.

    When None is passed, the value defaults to 1.
    """
    num_independent: _Optional[PositiveInt] = None
    num_dependent: _Optional[PositiveInt] = None

    @field_validator('num_independent', 'num_dependent', mode='before')
    @classmethod
    def none_to_one(cls, v):
        if v is None:
            return 1
        return v


class Evaluator(ABC):
    """Defines behavior common to all evaluators and what behaviors should be
    defined by the evaluators themselves.
    """

    # ==================
    # |   Properties   |
    # ==================
    @property
    def opt_problem(self) -> OptProblem:
        """**(readonly)** The OptProblem defining the pre-defined optimization problem.

        Note that for some evaluators you will only get information for variables and
        responses since no optimization problem is defined.

        :type: OptProblem
        """
        return self._opt_problem

    @property
    def interface(self) -> EvaluatorInfo:
        """**(readonly)** The EvaluatorInfo defining the inputs and outputs of the evaluator.

        :type: EvaluatorInfo
        """
        return self._interface

    @property
    def inputs(self) -> List[str]:
        """**(readonly)** The inputs of the evaluator.

        :type: list[str]
        """
        return self._inputs

    @property
    def outputs(self) -> List[str]:
        """**(readonly)** The outputs of the evaluator.

        :type: list[str]
        """
        return self._outputs

    @property
    def nind(self) -> int:
        """Number of independent variables

        :type: int
        """
        return len(self.inputs)

    @property
    def ndep(self) -> int:
        """Number of dependent variables

        :type: int
        """
        return len(self.outputs)

    @property
    def name(self) -> str:
        """**(readonly)** Name given to evaluator for easier identification.
        Defaults to the name of the class. A user provided name will be appended.

        :type: str
        """
        return self._name

    @property
    def comp_cost(self) -> float:
        """How costly this evaluator is to calculate responses. Defaults to 100.

        :type: float
        """
        return self._comp_cost

    @comp_cost.setter
    def comp_cost(self, value: float) -> None:
        # Value is not numeric
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{type(self).__name__}: comp_cost must be a "
                + f"numeric value! Received a {type(value).__name__} instead."
            )

        # Value is negative
        if value < 0:
            raise ValueError(
                f"{type(self).__name__}: comp_cost must be "
                + f"non-negative! {value} is less than zero."
            )

        self._comp_cost = value

    # ======================
    # |   Public Methods   |
    # ======================

    def __init__(
        self,
        name: str = None,
        comp_cost: float = 100,
        cache: str = None,
        cache_options: dict = None,
        logging: bool = False,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
        options: Optional[BaseModel] = None,
        **kwargs,
    ) -> None:
        """Initialize the problem and set the variables and responses.

        Parameters
        ----------
        name : str, optional
            Name to give the evaluator. Defaults to evaluator type name. If a
            value is provided, this will be appended to the default.
        comp_cost : float, optional
            Computational cost estimate. Defaults to 100.
        cache : str, optional
            Path to SQLite database where cached sites are stored, by default None.
            Enabling this option will also make the
            ``get_cache`` function available. This funtion will allow you access
            the sites that are currently defined in the cache.
        cache_options : dict, optional
            Options to modify caching behavior, by default None. The below
            options are available:

            - ``table_name`` (str): Name of table in database where sites are stored,
              by default 'Design_Data'
            - ``delta`` (float): Unscaled distance used to determine if two
              sites are duplicate, by default 1e-8
            - ``auto_update`` (bool): When ``True`` the database will be re-read
              before evaluating sites, by default False. This can be useful if
              another process is modifying the database concurrently.
        logging : bool, optional
            When set to True the evaluator will keep track of every site that is
            passed to it, by default False. The logged sites can be access
            through the ``get_log`` function.
        """
        self._name = type(self).__name__
        if name is not None and name != self._name:
            self._name += "_" + name

        # TODO: We should let users know they must pass in something that resembles a problem even though it is optional
        self.comp_cost = comp_cost
        if opt_problem is not None:
            # We defined the interface via an OptProblem instance. This takes precendence.
            # First we need to check that the passed in information is an OptProblem
            if not isinstance(opt_problem, se.OptProblem):
                raise TypeError("opt_problem argument must be of type OptProblem if set.")
            self._opt_problem = opt_problem

        elif interface is not None:
            # We defined the interface via an EvaluatorInfo instance
            self._opt_problem = evaluator_info_to_opt_problem(interface)
        else:
            raise ValueError(
                f"{type(self).__name__}: Either 'opt_problem' or 'interface' must be provided."
            )

        # Calculate the defaults if they are not already given
        self._opt_problem.calculate_default(overwrite=False)
        # Store the default values in the problem if they are not already
        # stored there
        self._store_default_in_problem()

        # Convert the OptProblem to the interface problem
        self._interface = opt_problem_to_evaluator_info(self._opt_problem)

        self._inputs = list(utils.collect_names(self._interface.inputs))
        self._outputs = list(utils.collect_names(self._interface.outputs))

        # Create the data type information
        self._dtypes = utils.get_types_from_evaluator_info(self._interface)

        # Store the number of independents and dependents
        self._num_independent = len(self.inputs)
        self._num_dependent = len(self.outputs)

        # Initialize counter for finite difference evals
        self.num_finite_difference_evals = 0

        # Initialize options from the class-level definition
        self._options: BaseModel = self.required_options()

        # Override with user-provided options if given
        if options is not None:
            expected_cls = type(self._options)
            if not isinstance(options, expected_cls):
                raise TypeError(
                    f"{type(self).__name__}: options must be an instance of "
                    f"{expected_cls.__name__}, got {type(options).__name__} instead."
                )
            # Re-validate all field values through the model to catch
            # mutations that bypassed Pydantic's validators.
            try:
                self._options = expected_cls(**options.model_dump())
            except ValidationError as exc:
                raise ValueError(
                    f"{type(self).__name__}: provided options failed validation: {exc}"
                ) from exc


        # Wrap _evaluate with caching if cache is provided
        if cache not in (None, ""):
            options = cache_options if cache_options is not None else {}
            org_evaluate = self._evaluate
            self._evaluate = cacher(
                org_evaluate,
                opt_problem=self._opt_problem,
                db_name=cache,
                **options,
            )
            self.get_cache = self._evaluate.get_cache

        # Wrap _evaluate with logging if logging is True
        if logging:
            org_evaluate = self._evaluate
            self._evaluate = site_logger(org_evaluate)
            self.get_log = self._evaluate.get_log     
        


    def __call__(self, sites: pd.DataFrame, **kwargs) -> None:
        """Calling method to evaluate the actual function.

        .. note::
            The passed dataframe will be modified in place.

        :param sites: The dataframe that contains the input values, and is
            updated with the responses
        :type sites: DataFrame
        """
        # Make sure we got a dataframe
        if not isinstance(sites, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__}: sites must be a DataFrame! "
                + f"Received a {type(sites).__name__} instead."
            )
        
        # Make sure input is rolled data
        rolled_input = [(var.name, var.shape) for var in self._interface.inputs if hasattr(var, 'shape')]
        if rolled_input:
            bad_input = []
            for name, shape in rolled_input:
                unrolled_names = set(utils.generate_names(name, shape))
                bad_input.extend(list(unrolled_names.intersection(sites.columns)))
            if bad_input:
                raise ValueError(
                    type(self).__name__
                    + ": Input DataFrame contains unrolled variables! "
                    + f"Bad columns: {bad_input}")
        # Make sure all columns were given
        if not set(self.inputs).issubset(sites.columns):
            raise ValueError(
                type(self).__name__
                + ": Input DataFrame does "
                + "not contain all variables! Missing: "
                + ", ".join(set(self.inputs).difference(sites.columns))
                + " Check that input is not unrolled."
            )

        # Check fixed variables
        fixed_vars = [
            var
            for var in self._opt_problem.variables
            if np.array_equal(var.bounds[0], var.bounds[1])
        ]
        # mask is a NumPy boolean array of length equal to the number of rows in the sites DataFrame, initialized with all True values.
        mask = np.ones(len(sites), dtype=bool)

        if fixed_vars:
            for var in fixed_vars:
                if var.name not in sites.columns:
                    raise ValueError(
                        f"{type(self).__name__}: Input DataFrame missing fixed variable column '{var.name}'"
                    )
                mask &= sites[var.name] == var.bounds[0]
                # mask is a boolean array where:
                # True  = row satisfies all fixed variable constraints (valid row)
                # False = row does not satisfy at least one fixed variable constraint (invalid row)

            # Rows with invalid fixed values get NaN responses
            invalid_rows = (
                ~mask
            )  # this is inverse of mask which is True for invalid rows
            if invalid_rows.any():
                for output in self.outputs:
                    sites.loc[invalid_rows, output] = np.nan

            # Evaluate only valid rows
            valid_sites = sites.loc[mask].copy()
            # Make sure the data types for all columns are set correctly
            utils.apply_types_from_evaluator_info(valid_sites, self.interface)
            self._evaluate(valid_sites, **kwargs)

            # Update original sites DataFrame with evaluated results
            # For each output column, update only the rows where mask is True
            for output in self.outputs:
                sites.loc[mask, output] = valid_sites[output]
        else:
            # Make sure the data types for all columns are set correctly
            utils.apply_types_from_evaluator_info(sites, self.interface)
            self._evaluate(sites, **kwargs)

        # Make sure responses were returned
        if "names" in kwargs:
            if kwargs["names"] is not None:
                if not set(kwargs["names"]).issubset(sites.columns):
                    raise ValueError(
                        f"{type(self).__name__}: Evaluation function did not include "
                        + "some responses! Missing: "
                        + ", ".join(set(kwargs["names"]).difference(sites.columns))
                    )
        else:
            if not set(self.outputs).issubset(sites.columns):
                raise ValueError(
                    f"{type(self).__name__}: Evaluation function did not include "
                    + "all responses! Missing: "
                    + ", ".join(set(self.outputs).difference(sites.columns))
                )

        # Again ensure that all data types are set correctly
        utils.apply_types_from_evaluator_info(sites, self.interface)

    # =======================
    # |   Virtual Methods   |
    # =======================

    def eval_np(self, sites: np.ndarray, names: Optional[List] = None, **kwargs) -> np.ndarray:
        """Method to allow calling the function with a numpy array

        :param sites: The sites to be evaluated (as an np.ndarray)
        :type sites: Numpy array
        :param names: Optional list of names of responses to use
        :type: list

        :return: Numpy array with the response values for all sites
        :rtype: Numpy array
        """

        # Convert sites to list
        converted_sites = sites.tolist()
        evaluated_sites = self.eval_list(converted_sites, names, **kwargs)

        return np.array(evaluated_sites)

    def eval_list(
        self, sites: List, names: Optional[List] = None, **kwargs
    ) -> List:
        # pylint: disable=W0613
        """Method to allow calling the function with the DE5 evaluator class

        :param sites: The list sites to be evaluated (implemented as a list of list of unrolled data)
        :type sites: List of lists
        :param names: Optional list of names of responses to use
        :type: list

        :return: List of list of unrolled response values for each site
        :rtype: List of lists
        """
        if isinstance(names, str):
            raise TypeError(
                f"The names field was given the string {names}. Instead, use [{names}]"
            )

        input_vars = self._interface.inputs

        # Need flag for if rolled or not
        input_is_rolled = np.any(
            [hasattr(ele, "shape") for row in sites for ele in row]
        )

        if input_is_rolled:
            input_as_df = pd.DataFrame(sites, columns=[var.name for var in input_vars])
            utils.apply_types_from_evaluator_info(input_as_df, self._interface)
        else:
            # Create a rolled dataframe from the input list by making unrolled data a dataframe
            unrolled_names = utils.unroll_names_using_variables(input_vars)
            unrolled_df = pd.DataFrame(sites, columns=unrolled_names)
            input_as_df = utils.roll_data_frame_using_variables(unrolled_df, input_vars)

        # Call the evaluator, should produce rolled data
        self._evaluate(input_as_df)

        # Need the vars associated to names
        if names is None:
            names = self.outputs
        named_vars = [v for v in self._interface.outputs if v.name in names]

        # results of input_unrolled is an unrolled df. Need to roll back, get sub df, and turn it into np array
        rolled_results = input_as_df[names]
        if input_is_rolled:
            results = rolled_results.to_numpy()
        else:
            unrolled_results = utils.unroll_data_frame_using_variables(
                rolled_results, named_vars
            )
            results = unrolled_results.to_numpy()

        if isinstance(sites, list):
            # Convert the responses into a list of list
            results = results.tolist()

        # Return the responses
        return results

    def default_site(self) -> pd.DataFrame:
        """Provide a DataFrame that contains all inputs and their default values

        A developer can overwrite the default values, but in general these values
        are based on the default values of the interface inputs.

        :return: DataFrame containing default values of all inputs
        :rtype: pd.DataFrame

        Note: This is just providing a different name for the `initial_guess`
        method.
        """
        return self.initial_guess()

    def initial_guess(self) -> pd.DataFrame:
        """Provide an initial guess for an optimizer

        :return: DataFrame providing the initial guess to use with an optimizer
        :rtype: pd.DataFrame
        """
        data = self._def_initial_guess()
        site = pd.DataFrame([data], columns=self.inputs)
        utils.apply_types_from_evaluator_info(site, self.interface)
        return site

    def _store_default_in_problem(self) -> None:
        """Store the default values in the problem dictionary if not already stored there"""
        # Get the default values for all variables
        defaults = self._def_initial_guess()
        if len(defaults) > 0:
            # Store  default values in the problem dictionary if provided.
            for var, default in zip(self._opt_problem.variables, defaults):
                # Set the default values
                var.default = default

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        This method can be overwritten by a descendant. Right now it just returns
        an empty list since the OptProblem is already setting defaults.
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [var.default for var in self._opt_problem.variables]
    
    def _get_partials_by_central_difference(self, sites: pd.DataFrame, opt_problem: OptProblem) -> np.ndarray:
        """
        Compute partial derivatives by central finite differences for the selected responses.

        The method constructs +delta and -delta perturbation batches for each free flattened
        variable across all provided sites and evaluates the model once to compute central
        differences for the responses selected for partial calculation.

        :param sites: DataFrame of input sites (one row per site). Column names must match
                    variable names defined in ``opt_problem.var_map``.
        :type sites: pandas.DataFrame

        :param opt_problem: OptProblem instance that provides ``var_map``, free-variable
                            mappings, and response-selection indices.
        :type opt_problem: OptProblem

        :returns: 3-D numpy array of shape ``(num_sites, num_selected_responses, n_free)`` where:
                - ``num_sites`` = number of rows in ``sites``
                - ``num_selected_responses`` = ``opt_problem.num_partials_responses``
                - ``n_free`` = number of free flattened variables (``opt_problem.num_flat_vars``)
        :rtype: numpy.ndarray

        :notes:
            - If there are no free variables, an empty-last-dimension array is returned.
            - ``sites`` is not modified by this method.
        """
        # Finite-difference step size
        # TODO: upgrade this to a custom delta for each var
        delta = 1e-8

        # Number of rows / evaluation points
        num_sites = len(sites)

        # Pull required mappings and counts from the OptProblem
        var_map = opt_problem.var_map
        free_flat_positions = opt_problem.free_flat_var_positions
        num_flat_vars = opt_problem.num_flat_vars
        selected_response_flat_indices = opt_problem.flat_partials_res_indices
        num_selected_responses = opt_problem.num_partials_responses

        # -------------------------------------------------------------------------
        # Build the "base" flattened input matrix:
        # shape = (num_sites, n_full_flat_vars)
        #
        # Each column corresponds to one flattened variable element in var_map.
        # Each row corresponds to one site from the input DataFrame.
        # -------------------------------------------------------------------------
        n_full_flat = len(var_map)
        X_base_full = np.empty((num_sites, n_full_flat), dtype=float)

        # Fill each flattened variable column from the corresponding column in `sites`
        for col_idx, (_, vr) in enumerate(var_map.iterrows()):
            name = vr["name"]
            multi_idx = vr["multi"]

            if multi_idx is None:
                # Scalar variable: just copy the column directly
                X_base_full[:, col_idx] = sites[name].to_numpy(dtype=float)
            else:
                # Array variable: extract the specific element at multi-index `multi_idx`
                X_base_full[:, col_idx] = np.asarray(
                    [np.asarray(cell, dtype=float)[tuple(multi_idx)] for cell in sites[name].to_numpy()],
                    dtype=float,
                )

        # If there are no free variables, there is nothing to differentiate with respect to
        if num_flat_vars == 0:
            return np.zeros((num_sites, num_selected_responses, 0), dtype=float)

        # -------------------------------------------------------------------------
        # Build perturbation batches for each free variable.
        #
        # We tile the base input once per free variable, then perturb one free
        # variable element at a time for every site.
        # -------------------------------------------------------------------------
        X_base_tiled = np.tile(X_base_full, (num_flat_vars, 1))

        # Row indices for the tiled batch
        rows = np.arange(num_flat_vars * num_sites)

        # For each free variable, repeat its full flattened index once per site
        var_indices_full = np.repeat(free_flat_positions, num_sites)

        # Create +delta and -delta copies of the tiled base
        X_plus = X_base_tiled.copy()
        X_minus = X_base_tiled.copy()

        # Apply perturbation to the appropriate element in each row
        X_plus[rows, var_indices_full] += delta
        X_minus[rows, var_indices_full] -= delta

        # Stack the plus/minus batches so evaluation can be done in one call
        # Final shape = (2 * num_flat_vars * num_sites, n_full_flat_vars)
        X_batch_full = np.vstack((X_plus, X_minus))

        # -------------------------------------------------------------------------
        # Evaluate the full batch once
        # -------------------------------------------------------------------------
        F_batch = np.asarray(self.eval_np(X_batch_full))

        # If there is only one output column, make sure response array is 2D
        if F_batch.ndim == 1:
            F_batch = F_batch.reshape(-1, 1)

        # Keep only the responses needed for partial derivatives
        # Shape = (2 * num_flat_vars * num_sites, num_selected_responses)
        F_selected = F_batch[:, selected_response_flat_indices]

        # Split the evaluated batch into +delta and -delta halves
        half = F_selected.shape[0] // 2
        F_plus_sel = F_selected[:half, :]
        F_minus_sel = F_selected[half:, :]

        # Reshape so that:
        # - first axis = free variable index
        # - second axis = site index
        # - third axis = selected response index
        F_plus_blocks = F_plus_sel.reshape(num_flat_vars, num_sites, num_selected_responses)
        F_minus_blocks = F_minus_sel.reshape(num_flat_vars, num_sites, num_selected_responses)

        # Central finite difference approximation:
        # derivative ~= (f(x+h) - f(x-h)) / (2h)
        deriv_var_site_resp = (F_plus_blocks - F_minus_blocks) / (2.0 * delta)

        # Reorder axes to:
        # shape = (num_sites, num_selected_responses, num_flat_vars)
        all_partials = deriv_var_site_resp.transpose(1, 2, 0)

        return all_partials
    
    def _compute_signature(self, sites: pd.DataFrame) -> bytes:
        """
        Create a compact deterministic signature representing the contents of ``sites``.

        The signature is derived from ``pd.util.hash_pandas_object(sites, index=True)`` and
        converted to bytes. It can be used for quick equality checks to decide whether cached
        partials are still valid for the same DataFrame.

        :param sites: DataFrame to be fingerprinted. The row index is included in the hash.
        :type sites: pandas.DataFrame

        :returns: Deterministic byte string representing the content of ``sites``.
        :rtype: bytes

        :notes:
            - This signature is not cryptographically secure. It is intended for quick
            equivalence detection (cache invalidation), not for security.
            - Two DataFrames with identical values and identical indices will produce the
            same signature.
        """

        # hash each row deterministically, include index by setting index=True
        # convert array of int64 hash values to bytes for cheap equality checks
        hash_values = pd.util.hash_pandas_object(sites, index=True).values
        return hash_values.tobytes()

    def _compute_or_get_partials(self, sites: pd.DataFrame, opt_problem: OptProblem) -> np.ndarray:
        """
        Return cached partials if valid; otherwise compute and cache them.

        The method computes a deterministic signature for ``sites`` using ``_compute_signature``.
        If the stored cache signature matches and ``_partials_cache_value`` exists, that cached
        value is returned. Otherwise, partials are computed via
        ``_get_partials_by_central_difference``, cached, and returned.

        :param sites: Sites for which partials are requested (one row per site).
        :type sites: pandas.DataFrame

        :param opt_problem: OptProblem instance providing mapping and selection data.
        :type opt_problem: OptProblem

        :returns: Partials array with shape ``(num_sites, num_selected_responses, num_flat_vars)``.
        :rtype: numpy.ndarray

        :side effect:
            - On cache miss, stores the computed partials in ``self._partials_cache_value`` and
            the signature in ``self._partials_cache_signature``. Also increments
            ``self.num_finite_difference_evals`` by 1.
        :precondition:
            - ``opt_problem`` must have had partial mappings set up (``_setup_maps_and_partials``).
        """

        sig = self._compute_signature(sites)
        if (
            getattr(self, "_partials_cache_signature", None) == sig
            and getattr(self, "_partials_cache_value", None) is not None
        ):
            return self._partials_cache_value

        all_partials = self._get_partials_by_central_difference(sites, opt_problem)
        self.num_finite_difference_evals += 1
        self._partials_cache_signature = sig
        self._partials_cache_value = all_partials
        return all_partials

    def _evaluate_partials(self, sites: pd.DataFrame, opt_problem: OptProblem) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and return the gradient and constraint Jacobian arrays for ``sites``.

        The method retrieves (or computes) the full partials array and then uses the
        precomputed selection mappings (``grad_cols`` and ``jac_cols``) to place rows from
        the partials into the appropriate gradient and Jacobian output arrays.

        :param sites: DataFrame of sites (one site per row).
        :type sites: pandas.DataFrame

        :param opt_problem: OptProblem instance providing ``num_objs_total``, ``num_cons_total``,
                            ``num_flat_vars``, ``grad_cols``, and ``jac_cols``.
        :type opt_problem: OptProblem

        :returns: Tuple ``(gradient, jacobian)`` where:
                - ``gradient`` is a numpy array of shape ``(num_sites, num_objs_total, num_flat_vars)``.
                - ``jacobian`` is a numpy array of shape ``(num_sites, num_cons_total, num_flat_vars)``.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]

        :notes:
            - The method expects ``all_partials`` to have shape
            ``(num_sites, num_selected_responses, num_flat_vars)`` where selected responses
            are ordered as given by ``opt_problem.flat_partials_res_indices``.
            - ``grad_cols`` and ``jac_cols`` contain sentinel values (e.g. ``-1``) for selected
            responses that are not objectives/constraints; those are skipped.
        :raises ValueError: If required opt_problem attributes (e.g., ``grad_cols``) are missing.
        """
        
        num_sites = len(sites)

        all_partials = self._compute_or_get_partials(sites, opt_problem)
        num_objs_total = opt_problem.num_objs_total
        num_cons_total = opt_problem.num_cons_total
        num_flat_vars = opt_problem.num_flat_vars
        grad_cols = opt_problem.grad_cols
        jac_cols = opt_problem.jac_cols

        gradient = np.zeros((num_sites, num_objs_total, num_flat_vars), dtype=float)
        for sel_local_idx, grad_col in enumerate(grad_cols):
            if grad_col >= 0:
                gradient[:, grad_col, :] = all_partials[:, sel_local_idx, :]

        jacobian = np.zeros((num_sites, num_cons_total, num_flat_vars), dtype=float)
        for sel_local_idx, jac_col in enumerate(jac_cols):
            if jac_col >= 0:
                jacobian[:, jac_col, :] = all_partials[:, sel_local_idx, :]

        return gradient, jacobian


    # =========================
    # |   Options Methods    |
    # =========================

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this evaluator.

        Subclasses should override this to return their own BaseModel subclass
        with fields representing the available options.

        Returns
        -------
        Type[BaseModel]
            A Pydantic BaseModel subclass defining the options schema.
        """
        return _EmptyOptions

    @classmethod
    def required_options(cls) -> BaseModel:
        """Return an instance of the options model with default values.

        Returns
        -------
        BaseModel
            An instance of the model returned by _define_options().
        """
        return cls._define_options()()

    @classmethod
    def required_options_names(cls) -> set:
        """Return the set of option names defined by this evaluator class.

        Returns
        -------
        set[str]
            Names of all fields in the options model.
        """
        return set(cls._define_options().model_fields.keys())

    def full_options(self) -> BaseModel:
        """Return all valid options for this evaluator instance.

        By default this delegates to required_options(). Subclasses may override
        this when the full set of options depends on runtime state.

        Returns
        -------
        BaseModel
            An instance of the options model.
        """
        return self.required_options()

    @property
    def full_options_names(self) -> set:
        """Names of all options returned by full_options().

        Returns
        -------
        set[str]
        """
        return set(type(self.full_options()).model_fields.keys())

    def lookup_option_value(self, name: str) -> Any:
        """Look up the current value of an option by name.

        Checks self._options first, then falls back to full_options() defaults.

        Parameters
        ----------
        name : str
            The option name to look up.

        Returns
        -------
        Any
            The current value of the option.

        Raises
        ------
        KeyError
            If the name is not a known option.
        """
        if name in type(self._options).model_fields:
            return getattr(self._options, name)
        full = self.full_options()
        if name in type(full).model_fields:
            return getattr(full, name)
        raise KeyError(f"KeyError: key not found: {name}")

    def _check_options(
        self,
        options: Union[str, List[str]],
        values: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Validate that option names exist and optionally validate values.

        Parameters
        ----------
        options : str or list[str]
            Name or list of names of options to check.
        values : dict, optional
            If provided, a mapping of option name to value. The values will be
            validated against the Pydantic options model.

        Returns
        -------
        list[str]
            The validated list of option names.

        Raises
        ------
        TypeError
            If options is not a string or list of strings, or if an entry is
            not a string.
        ValueError
            If an option name is not in the required options set, or if
            provided values fail Pydantic validation.
        """
        req_names = self.required_options_names()
        if isinstance(options, str):
            options = [options]
        elif not isinstance(options, list):
            raise TypeError("Options is not a string or list of strings")
        for entry in options:
            if not isinstance(entry, str):
                raise TypeError(
                    f"Entry {entry} in options list is not a string, instead a {type(entry)}"
                )
            if entry not in req_names:
                raise ValueError(
                    f"Trying to set {entry} as an option, but it is not defined as an option for this class"
                )
        if values is not None:
            model_cls = type(self.required_options())
            defaults = self.required_options().model_dump()
            defaults.update(values)
            try:
                model_cls(**defaults)
            except ValidationError as exc:
                raise ValueError(
                    f"Option value validation failed: {exc}"
                ) from exc
        return options

    def current_options(self) -> BaseModel:
        """Return the current options as a combined model instance.

        Merges full_options() defaults with the user-overridden values in
        self._options using combine_instances.

        Returns
        -------
        BaseModel
            Combined options instance.
        """
        _, combined = combine_instances([self._options, self.full_options()])
        return combined

    # ========================
    # |   Abstract Methods   |
    # ========================

    @abstractmethod
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Perform the true response calculation. Modifies the dataframe in place.

        :param df: The dataframe that contains the input values. Will be modified
        :type df: DataFrame
        """

    def evaluate_gradient(self, sites: pd.DataFrame, opt_problem: OptProblem) -> np.ndarray:
        """
        Public method: return the numeric gradient array for ``sites``.

        This is a convenience wrapper that calls ``_evaluate_partials`` and returns only the
        gradient portion.

        :param sites: DataFrame of sites (one row per site).
        :type sites: pandas.DataFrame

        :param opt_problem: OptProblem instance containing gradient mapping information.
        :type opt_problem: OptProblem

        :returns: Gradient array shaped ``(num_sites, num_objs_total, num_flat_vars)``.
        :rtype: numpy.ndarray

        :notes:
            - Uses cached partials when available to avoid redundant finite-difference work.
            - Preconditions and exceptions mirror those of ``_evaluate_partials``.
        """
        
        gradient, _ = self._evaluate_partials(sites, opt_problem)
        return gradient

    def evaluate_jacobian(self, sites: pd.DataFrame, opt_problem: OptProblem) -> np.ndarray:
        """
        Public method: return the constraint Jacobian array for ``sites``.

        This convenience wrapper calls ``_evaluate_partials`` and returns only the Jacobian
        portion.

        :param sites: DataFrame of sites (one row per site).
        :type sites: pandas.DataFrame

        :param opt_problem: OptProblem instance containing Jacobian mapping information.
        :type opt_problem: OptProblem

        :returns: Jacobian array shaped ``(num_sites, num_cons_total, num_flat_vars)``.
        :rtype: numpy.ndarray

        :notes:
            - Uses cached partials when available to avoid redundant finite-difference work.
            - Preconditions and exceptions mirror those of ``_evaluate_partials``.
        """
        
        _, jacobian = self._evaluate_partials(sites, opt_problem)
        return jacobian
