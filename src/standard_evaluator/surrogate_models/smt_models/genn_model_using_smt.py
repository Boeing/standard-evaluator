import pandas as pd

from typing import List, Optional, Type

from pydantic import BaseModel, Field

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo

# from smt.surrogate_models.surrogate_model import SurrogateModel  # Note that smt also defines a SurrogateModel class that we do not want to import since it then overwrites the DE class
from smt.surrogate_models import GENN

# from smt.surrogate_models.genn import GENN, load_smt_data # older version of GENN SMT file rquires load_smt_data method before training


class GradientEnhancedNeuralNetworksModelOptions(AbstractSmtModelOptions):
    """Options for the Gradient-Enhanced Neural Networks surrogate model."""

    alpha: float = Field(default=0.05, description="Optimizer learning rate")
    beta1: float = Field(default=0.9, description="Adam optimizer tuning parameter")
    beta2: float = Field(default=0.99, description="Adam optimizer tuning parameter")
    lambd: float = Field(default=0.01, description="Regularization coefficient")
    gamma: float = Field(
        default=1.0, description="Gradient-enhancement coefficient"
    )
    hidden_layer_sizes: List[int] = Field(default=[12, 12])
    mini_batch_size: int = Field(default=-1)
    num_epochs: int = Field(default=1)
    num_iterations: int = Field(default=1000)
    seed: int = Field(default=-1)
    is_print: bool = False
    is_normalize: bool = False
    is_backtracking: bool = False


class GradientEnhancedNeuralNetworksModel(AbstractSmtModel):
    """
    Class representing a GradientEnhancedNeuralNetworks model.

    Reference link: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/genn.html
    ToDo: Implement the test cases for GENN model after implementing the derivative method for it

    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The GradientEnhancedNeuralNetworksModelOptions class.
        """
        return GradientEnhancedNeuralNetworksModelOptions

    def __init__(
        self,
        sites: pd.DataFrame,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Initializes a GradientEnhancedNeuralNetworks model from an OptProblem and site data.

        Args:
            sites: The site data.
            options: Pydantic BaseModel instance with GENN model options.
            name: Name of the GENN model. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        super().__init__(
            sites=sites, model_type=GENN, options=options, name=name,
            interface=interface, opt_problem=opt_problem,
        )
        raise NotImplementedError(
            "Not yet Implemented! Need to provide ability to define derivative for this training"
        )
