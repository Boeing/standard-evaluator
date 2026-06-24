"""A set of models that can be used to estimate true response values of
arbitrary/unknown functions. Each model defines a ``__call__`` method that can
be used to predict response values at given sites. Models can also be
updated/recalibrated to include additional sites with known true response
values. This is done through the ``update`` method.

.. note::
    Dataframes passed to ``__call__`` will be modified in place with response values.
"""

from standard_evaluator.surrogate_models.abstract_model import SurrogateModel
from standard_evaluator.surrogate_models.polynomial_model import (
    CoefficientOrdering,
    PolynomialModel,
    PolynomialModelOptions,
)

__all__ = [
    "SurrogateModel",
    "PolynomialModel",
    "PolynomialModelOptions",
    "CoefficientOrdering",
]

# Conditionally import SMT model classes if the smt package is available.
# The smt package is an optional dependency; importing surrogate_models should
# succeed even when smt is not installed.
try:
    from standard_evaluator.surrogate_models.smt_models import (
        AbstractSmtModel,
        GradientEnhancedNeuralNetworksModel,
        GradientEnhancedNeuralNetworksModelOptions,
        InverseDistanceWeightingModel,
        InverseDistanceWeightingModelOptions,
        LeastSquaresApproximationModel,
        LeastSquaresApproximationModelOptions,
        RadialBasisFunctionModel,
        RadialBasisFunctionModelOptions,
        RegularizedMinimalEnergyTensorProductBSplines,
        RegularizedMinimalEnergyTensorProductBSplinesOptions,
        SecondOrderPolynomialApproximationModel,
        SecondOrderPolynomialApproximationModelOptions,
    )

    __all__.extend([
        "AbstractSmtModel",
        "GradientEnhancedNeuralNetworksModel",
        "GradientEnhancedNeuralNetworksModelOptions",
        "InverseDistanceWeightingModel",
        "InverseDistanceWeightingModelOptions",
        "LeastSquaresApproximationModel",
        "LeastSquaresApproximationModelOptions",
        "RadialBasisFunctionModel",
        "RadialBasisFunctionModelOptions",
        "RegularizedMinimalEnergyTensorProductBSplines",
        "RegularizedMinimalEnergyTensorProductBSplinesOptions",
        "SecondOrderPolynomialApproximationModel",
        "SecondOrderPolynomialApproximationModelOptions",
    ])
except ImportError:
    # SMT models are not available if the smt package is not installed.
    # Users can install them with: pip install standard_evaluator[surrogate]
    pass
