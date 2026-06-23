import numpy as np
import scipy as sp
import pandas as pd
import numdifftools as nd
import pytest
import os
import dask

from typing import List, Optional, Type
from functools import partial

from pydantic import BaseModel, Field

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
)
from standard_evaluator.problem import OptProblem
from standard_evaluator import EvaluatorInfo

from smt.surrogate_models import IDW


class InverseDistanceWeightingModelOptions(AbstractSmtModelOptions):
    """Options for the Inverse Distance Weighting surrogate model."""

    p: float = Field(default=2.5, gt=0, description="Order of distance norm")


class InverseDistanceWeightingModel(AbstractSmtModel):
    """
    Class representing a InverseDistanceWeighting model.

    #Reference link: https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/idw.html
    """

    @classmethod
    def _define_options(cls) -> Type[BaseModel]:
        """Return the Pydantic model class that defines the options for this model.

        Returns
        -------
        Type[BaseModel]
            The InverseDistanceWeightingModelOptions class.
        """
        return InverseDistanceWeightingModelOptions

    def __init__(
        self,
        sites: pd.DataFrame,
        options: Optional[BaseModel] = None,
        name: Optional[str] = None,
        interface: EvaluatorInfo = None,
        opt_problem: OptProblem = None,
    ) -> None:
        """Initializes an InverseDistanceWeighting model from an OptProblem and site data.

        Args:
            sites: The site data.
            options: Pydantic BaseModel instance with IDW model options.
            name: Name of the IDW model. Defaults to None.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining the optimization problem.
        """
        super().__init__(
            sites=sites, model_type=IDW, options=options, name=name,
            interface=interface, opt_problem=opt_problem,
        )
