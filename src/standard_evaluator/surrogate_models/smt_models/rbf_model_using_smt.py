import pandas as pd

from typing import Optional, Type

from pydantic import BaseModel, Field, field_validator

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo

from smt.surrogate_models import RBF


class RadialBasisFunctionModelOptions(AbstractSmtModelOptions):
    """Options for the Radial Basis Function surrogate model."""

    d0: float = Field(default=1.0, description="Basis function scaling parameter")
    poly_degree: int = Field(
        default=-1, description="Polynomial degree: -1, 0, or 1"
    )
    reg: float = Field(default=1e-10, description="Regularization coefficient")
    max_print_depth: int = Field(default=5, ge=0, description="Max print depth")

    @field_validator("poly_degree")
    @classmethod
    def validate_poly_degree(cls, v: int) -> int:
        if v not in {-1, 0, 1}:
            raise ValueError(f"poly_degree must be -1, 0, or 1, got {v}")
        return v


class RadialBasisFunctionModel(AbstractSmtModel):
    """
    Class representing a RadialBasisFunction model.

    #Reference link: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/rbf.html

    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines options for this model.

        Returns
        -------
        Type[BaseModel]
            The RadialBasisFunctionModelOptions class.
        """
        return RadialBasisFunctionModelOptions

    def __init__(
        self,
        sites: pd.DataFrame,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Initializes a RBF model from an OptProblem and site data.

        Args:
            sites: The site data.
            options: Pydantic BaseModel instance with RBF model options.
            name: Name of the RBF model. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        super().__init__(
            sites=sites, model_type=RBF, options=options, name=name,
            interface=interface, opt_problem=opt_problem,
        )
