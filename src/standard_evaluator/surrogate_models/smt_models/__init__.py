"""Wrappers of the Surrogate Modeling Toolbox surrogate models."""

try:
    import smt  # noqa: F401 — check that the smt package itself is available
except ImportError as e:
    raise ImportError(
        "The 'smt' package is required to use SMT surrogate models. "
        "Install it with: pip install standard_evaluator[surrogate]"
    ) from e

from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModel,
    AbstractSmtModelOptions,
    AbstractSmtModelParameters,
)

# Concrete model imports — these are added as models are migrated.
# Missing modules will raise ModuleNotFoundError naturally.
try:
    from standard_evaluator.surrogate_models.smt_models.genn_model_using_smt import (
        GradientEnhancedNeuralNetworksModel,
        GradientEnhancedNeuralNetworksModelOptions,
    )
except ModuleNotFoundError:
    # Optional model module may be absent in partial/migrating installations.
    pass

try:
    from standard_evaluator.surrogate_models.smt_models.idw_model_using_smt import (
        InverseDistanceWeightingModel,
        InverseDistanceWeightingModelOptions,
    )
except ModuleNotFoundError:
    # Optional model module may be absent in partial/migrating installations.
    pass

try:
    from standard_evaluator.surrogate_models.smt_models.ls_model_using_smt import (
        LeastSquaresApproximationModel,
        LeastSquaresApproximationModelOptions,
    )
except ModuleNotFoundError:
    # Optional model module may be absent in partial/migrating installations.
    pass

try:
    from standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt import (
        RadialBasisFunctionModel,
        RadialBasisFunctionModelOptions,
    )
except ModuleNotFoundError:
    # Optional model module may be absent in partial/migrating installations.
    pass

try:
    from standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt import (
        RegularizedMinimalEnergyTensorProductBSplines,
        RegularizedMinimalEnergyTensorProductBSplinesOptions,
    )
except ModuleNotFoundError:
    # Optional model module may be absent in partial/migrating installations.
    pass

try:
    from standard_evaluator.surrogate_models.smt_models.sopa_model_using_smt import (
        SecondOrderPolynomialApproximationModel,
        SecondOrderPolynomialApproximationModelOptions,
    )
except ModuleNotFoundError:
    # Optional model module may be absent in partial/migrating installations.
    pass

__all__ = [
    "AbstractSmtModel",
    "AbstractSmtModelOptions",
    "AbstractSmtModelParameters",
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
]
