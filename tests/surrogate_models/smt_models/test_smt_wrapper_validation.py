"""Property-based tests for SMT wrapper validation enforcement.

Tests that invalid option values are rejected at construction time with
Pydantic ValidationError for RadialBasisFunctionModelOptions,
InverseDistanceWeightingModelOptions, and
RegularizedMinimalEnergyTensorProductBSplinesOptions.

Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement
"""

import pytest
from pydantic import ValidationError

from hypothesis import given, settings
from hypothesis import strategies as st

from standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt import (
    RadialBasisFunctionModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.idw_model_using_smt import (
    InverseDistanceWeightingModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt import (
    RegularizedMinimalEnergyTensorProductBSplinesOptions,
)


class TestRadialBasisFunctionModelOptionsValidation:
    """Property 1: Validation Enforcement (SMT subset) - RBF poly_degree.

    For RadialBasisFunctionModelOptions.poly_degree not in {-1, 0, 1},
    construction SHALL raise a Pydantic ValidationError.

    Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement

    **Validates: Requirements 16.1**
    """

    @settings(max_examples=100)
    @given(
        poly_degree=st.integers().filter(lambda x: x not in {-1, 0, 1})
    )
    def test_invalid_poly_degree_raises_validation_error(self, poly_degree):
        """Any poly_degree not in {-1, 0, 1} raises ValidationError.

        Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement

        **Validates: Requirements 16.1**
        """
        with pytest.raises(ValidationError):
            RadialBasisFunctionModelOptions(poly_degree=poly_degree)


class TestInverseDistanceWeightingModelOptionsValidation:
    """Property 1: Validation Enforcement (SMT subset) - IDW p.

    For InverseDistanceWeightingModelOptions.p set to a value <= 0,
    construction SHALL raise a Pydantic ValidationError.

    Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement

    **Validates: Requirements 16.3**
    """

    @settings(max_examples=100)
    @given(
        p=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False)
    )
    def test_invalid_p_raises_validation_error(self, p):
        """Any p value <= 0 raises ValidationError.

        Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement

        **Validates: Requirements 16.3**
        """
        with pytest.raises(ValidationError):
            InverseDistanceWeightingModelOptions(p=p)


class TestRegularizedMinimalEnergyTensorProductBSplinesOptionsValidation:
    """Property 1: Validation Enforcement (SMT subset) - RMTS approx_order.

    For RegularizedMinimalEnergyTensorProductBSplinesOptions.approx_order < 1,
    construction SHALL raise a Pydantic ValidationError.

    Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement

    **Validates: Requirements 16.2**
    """

    @settings(max_examples=100)
    @given(
        approx_order=st.integers(max_value=0)
    )
    def test_invalid_approx_order_raises_validation_error(self, approx_order):
        """Any approx_order < 1 raises ValidationError.

        Feature: surrogate-model-pydantic-options, Property 1: Validation Enforcement

        **Validates: Requirements 16.2**
        """
        with pytest.raises(ValidationError):
            RegularizedMinimalEnergyTensorProductBSplinesOptions(
                approx_order=approx_order
            )
