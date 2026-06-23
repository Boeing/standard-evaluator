from typing import Optional, Type

import pandas as pd
from pydantic import BaseModel

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo

from smt.surrogate_models import QP


class SecondOrderPolynomialApproximationModelOptions(AbstractSmtModelOptions):
    """Options for the Second Order Polynomial Approximation surrogate model.

    No additional fields beyond those in AbstractSmtModelOptions.
    """

    pass


class SecondOrderPolynomialApproximationModel(AbstractSmtModel):
    """
    Class representing a SecondOrderPolynomialApproximationModel.
    #No.of sites in the dataframe should be greater than equal to (self.nx + 1) * (self.nx + 2) / 2.0:, where nx is the number of variables
    i.e X.shape[0] < (self.nx + 1) * (self.nx + 2) / 2.0:, where nx is the number of variables
    Reference link: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/qp.html

    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The SecondOrderPolynomialApproximationModelOptions class.
        """
        return SecondOrderPolynomialApproximationModelOptions

    def __init__(
        self,
        sites: pd.DataFrame,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Initializes a SecondOrder Polynomial approximation model from an OptProblem and site data.

        Args:
            sites: The site data.
            options: Pydantic BaseModel instance with QP model options.
            name: Name of the QP model. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        super().__init__(
            sites=sites,
            model_type=QP,
            options=options,
            name=name,
            interface=interface,
            opt_problem=opt_problem,
        )
