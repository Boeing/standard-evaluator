import pandas as pd

from typing import Optional, Type

from pydantic import BaseModel, Field

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo

from smt.surrogate_models import RMTB


class RegularizedMinimalEnergyTensorProductBSplinesOptions(AbstractSmtModelOptions):
    """Options for the Regularized Minimal-Energy Tensor-Product B-Splines model."""

    # RMTB requires xlimits from the SMT library, so default to True
    use_xlimits: bool = Field(
        default=True, description="Whether to use variable bounds as xlimits"
    )
    smoothness: float = Field(default=1.0)
    regularization_weight: float = Field(default=1e-14)
    energy_weight: float = Field(default=0.0001)
    extrapolate: bool = False
    min_energy: bool = True
    approx_order: int = Field(default=4, ge=1)
    solver: str = Field(default="krylov")
    derivative_solver: str = Field(default="krylov")
    grad_weight: float = Field(default=0.5)
    solver_tolerance: float = Field(default=1e-12)
    nonlinear_maxiter: int = Field(default=10)
    line_search: str = Field(default="backtracking")
    save_energy_terms: bool = False
    order: int = Field(default=3)
    num_ctrl_pts: int = Field(default=15)


class RegularizedMinimalEnergyTensorProductBSplines(AbstractSmtModel):
    """
    Class representing a Regularized minimal-energy tensor-product splines model
    RMTS is implemented in SMT with two choices of splines:

    1. B-splines (RMTB): RMTB uses B-splines with a uniform knot vector in each dimension. The number of B-spline control points and the B-spline order in each dimension are options that trade off efficiency and precision of the interpolant.

    2. Cubic Hermite splines (RMTC): RMTC divides the domain into tensor-product cubic elements. For adjacent elements, the values and derivatives are continuous. The number of elements in each dimension is an option that trades off efficiency and precision.

    #Reference link: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/rmts.html

    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The RegularizedMinimalEnergyTensorProductBSplinesOptions class.
        """
        return RegularizedMinimalEnergyTensorProductBSplinesOptions

    def __init__(
        self,
        sites: pd.DataFrame,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Initializes a RMTB model from an OptProblem and site data.

        Args:
            sites: The site data.
            options: Pydantic BaseModel instance with RMTB model options.
            name: Name of the RMTB model. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        super().__init__(
            sites=sites, model_type=RMTB, options=options, name=name,
            interface=interface, opt_problem=opt_problem,
        )
