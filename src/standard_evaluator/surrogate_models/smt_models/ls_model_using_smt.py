import pandas as pd

from typing import Optional, Type

from pydantic import BaseModel

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo

from smt.surrogate_models import LS


class LeastSquaresApproximationModelOptions(AbstractSmtModelOptions):
    """Options for the Least Squares Approximation surrogate model.

    No additional fields beyond those in AbstractSmtModelOptions.
    """

    pass


class LeastSquaresApproximationModel(AbstractSmtModel):
    """
    Class representing a LeastSquaredApproximation model.

    #Reference link: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/ls.html

    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The LeastSquaresApproximationModelOptions class.
        """
        return LeastSquaresApproximationModelOptions

    def __init__(
        self,
        sites: pd.DataFrame,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Initializes a LeastSquares Approximation model from an OptProblem and site data.

        Args:
            sites: The site data.
            options: Pydantic BaseModel instance with LS model options.
            name: Name of the LS model. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        super().__init__(
            sites=sites, model_type=LS, options=options, name=name,
            interface=interface, opt_problem=opt_problem,
        )
