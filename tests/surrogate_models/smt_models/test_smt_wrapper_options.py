"""Unit tests for all SMT wrapper options classes.

Tests cover:
1. Default construction of each options class
2. _define_options() returns the correct class for each model
3. lookup_option_value returns correct values for model-specific fields
4. Options inherit base AbstractSmtModelOptions fields
5. lookup_option_value raises KeyError for unknown names

Requirements: 10.1-10.7, 12.1, 13.1-13.3
"""

import numpy as np
import pandas as pd
import pytest

import standard_evaluator as se
from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt import (
    RadialBasisFunctionModel,
    RadialBasisFunctionModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.idw_model_using_smt import (
    InverseDistanceWeightingModel,
    InverseDistanceWeightingModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.ls_model_using_smt import (
    LeastSquaresApproximationModel,
    LeastSquaresApproximationModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.genn_model_using_smt import (
    GradientEnhancedNeuralNetworksModelOptions,
)
from standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt import (
    RegularizedMinimalEnergyTensorProductBSplines,
    RegularizedMinimalEnergyTensorProductBSplinesOptions,
)
from standard_evaluator.surrogate_models.smt_models.sopa_model_using_smt import (
    SecondOrderPolynomialApproximationModel,
    SecondOrderPolynomialApproximationModelOptions,
)


# --- Shared Fixtures ---


@pytest.fixture
def opt_prob():
    """Create a simple 2-variable, 1-response optimization problem."""
    new_prob = se.utilities.create_opt_problem(num_independent=2, num_dependent=1)
    new_prob.variables[0].bounds = (-2.0, 2.0)
    new_prob.variables[1].bounds = (-2.0, 2.0)
    new_prob.responses[0].name = "f"
    new_prob.objectives = []
    new_prob.constraints = []
    return new_prob


@pytest.fixture
def sites_df():
    """Create training sites DataFrame with 9 points."""
    sites = np.array(
        [
            [-2.0, -2.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
            [0.0, -2.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, -2.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ]
    )
    site_vals = np.array(
        [[8.0], [4.0], [8.0], [4.0], [0.0], [4.0], [8.0], [4.0], [8.0]]
    )
    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f"]
    )


# --- Base fields present in AbstractSmtModelOptions ---

BASE_FIELDS = {
    "print_global",
    "print_training",
    "print_prediction",
    "print_problem",
    "print_solver",
    "use_xlimits",
    "data_dir",
    "parameters",
}


# =============================================================================
# RadialBasisFunctionModelOptions Tests
# =============================================================================


class TestRadialBasisFunctionModelOptions:
    """Tests for RadialBasisFunctionModelOptions."""

    def test_default_construction(self):
        """RBF options can be constructed with all defaults."""
        opts = RadialBasisFunctionModelOptions()
        assert opts.d0 == 1.0
        assert opts.poly_degree == -1
        assert opts.reg == 1e-10
        assert opts.max_print_depth == 5

    def test_inherits_base_fields(self):
        """RBF options inherit all base AbstractSmtModelOptions fields."""
        opts = RadialBasisFunctionModelOptions()
        for field in BASE_FIELDS:
            assert field in RadialBasisFunctionModelOptions.model_fields

        # Check base defaults
        assert opts.print_global is True
        assert opts.print_training is True
        assert opts.use_xlimits is False
        assert opts.data_dir is None
        assert opts.parameters is None

    def test_define_options_returns_correct_class(self):
        """RadialBasisFunctionModel._define_options() returns RadialBasisFunctionModelOptions."""
        assert (
            RadialBasisFunctionModel._define_options()
            is RadialBasisFunctionModelOptions
        )

    def test_lookup_option_value_model_specific_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct values for RBF-specific fields."""
        opts = RadialBasisFunctionModelOptions(
            d0=2.5, poly_degree=1, reg=1e-8, max_print_depth=3
        )
        model = RadialBasisFunctionModel(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("d0") == 2.5
        assert model.lookup_option_value("poly_degree") == 1
        assert model.lookup_option_value("reg") == 1e-8
        assert model.lookup_option_value("max_print_depth") == 3

    def test_lookup_option_value_base_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct values for inherited base fields."""
        opts = RadialBasisFunctionModelOptions(
            print_global=False, print_training=False
        )
        model = RadialBasisFunctionModel(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("print_global") is False
        assert model.lookup_option_value("print_training") is False

    def test_lookup_option_value_unknown_name_raises_key_error(
        self, opt_prob, sites_df
    ):
        """lookup_option_value raises KeyError for unknown option names."""
        model = RadialBasisFunctionModel(
            sites=sites_df, opt_problem=opt_prob
        )
        with pytest.raises(KeyError):
            model.lookup_option_value("nonexistent_option")

    def test_lookup_option_value_defaults(self, opt_prob, sites_df):
        """lookup_option_value returns default values when no custom options set."""
        model = RadialBasisFunctionModel(
            sites=sites_df, opt_problem=opt_prob
        )
        assert model.lookup_option_value("d0") == 1.0
        assert model.lookup_option_value("poly_degree") == -1
        assert model.lookup_option_value("reg") == 1e-10
        assert model.lookup_option_value("max_print_depth") == 5


# =============================================================================
# InverseDistanceWeightingModelOptions Tests
# =============================================================================


class TestInverseDistanceWeightingModelOptions:
    """Tests for InverseDistanceWeightingModelOptions."""

    def test_default_construction(self):
        """IDW options can be constructed with all defaults."""
        opts = InverseDistanceWeightingModelOptions()
        assert opts.p == 2.5

    def test_inherits_base_fields(self):
        """IDW options inherit all base AbstractSmtModelOptions fields."""
        opts = InverseDistanceWeightingModelOptions()
        for field in BASE_FIELDS:
            assert field in InverseDistanceWeightingModelOptions.model_fields

        assert opts.print_global is True
        assert opts.use_xlimits is False
        assert opts.parameters is None

    def test_define_options_returns_correct_class(self):
        """InverseDistanceWeightingModel._define_options() returns InverseDistanceWeightingModelOptions."""
        assert (
            InverseDistanceWeightingModel._define_options()
            is InverseDistanceWeightingModelOptions
        )

    def test_lookup_option_value_model_specific_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct value for IDW-specific field."""
        opts = InverseDistanceWeightingModelOptions(p=3.0)
        model = InverseDistanceWeightingModel(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("p") == 3.0

    def test_lookup_option_value_base_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct values for inherited base fields."""
        opts = InverseDistanceWeightingModelOptions(print_solver=False)
        model = InverseDistanceWeightingModel(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("print_solver") is False

    def test_lookup_option_value_unknown_name_raises_key_error(
        self, opt_prob, sites_df
    ):
        """lookup_option_value raises KeyError for unknown option names."""
        model = InverseDistanceWeightingModel(
            sites=sites_df, opt_problem=opt_prob
        )
        with pytest.raises(KeyError):
            model.lookup_option_value("unknown_field")

    def test_lookup_option_value_defaults(self, opt_prob, sites_df):
        """lookup_option_value returns default values when no custom options set."""
        model = InverseDistanceWeightingModel(
            sites=sites_df, opt_problem=opt_prob
        )
        assert model.lookup_option_value("p") == 2.5


# =============================================================================
# LeastSquaresApproximationModelOptions Tests
# =============================================================================


class TestLeastSquaresApproximationModelOptions:
    """Tests for LeastSquaresApproximationModelOptions."""

    def test_default_construction(self):
        """LS options can be constructed with all defaults (no additional fields)."""
        opts = LeastSquaresApproximationModelOptions()
        # Should only have base fields
        assert opts.print_global is True
        assert opts.use_xlimits is False

    def test_inherits_base_fields(self):
        """LS options inherit all base AbstractSmtModelOptions fields."""
        for field in BASE_FIELDS:
            assert field in LeastSquaresApproximationModelOptions.model_fields

    def test_define_options_returns_correct_class(self):
        """LeastSquaresApproximationModel._define_options() returns LeastSquaresApproximationModelOptions."""
        assert (
            LeastSquaresApproximationModel._define_options()
            is LeastSquaresApproximationModelOptions
        )

    def test_lookup_option_value_base_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct values for inherited base fields."""
        opts = LeastSquaresApproximationModelOptions(
            print_global=False, print_prediction=False
        )
        model = LeastSquaresApproximationModel(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("print_global") is False
        assert model.lookup_option_value("print_prediction") is False

    def test_lookup_option_value_unknown_name_raises_key_error(
        self, opt_prob, sites_df
    ):
        """lookup_option_value raises KeyError for unknown option names."""
        model = LeastSquaresApproximationModel(
            sites=sites_df, opt_problem=opt_prob
        )
        with pytest.raises(KeyError):
            model.lookup_option_value("not_a_real_option")


# =============================================================================
# GradientEnhancedNeuralNetworksModelOptions Tests
# =============================================================================


class TestGradientEnhancedNeuralNetworksModelOptions:
    """Tests for GradientEnhancedNeuralNetworksModelOptions.

    Note: GradientEnhancedNeuralNetworksModel raises NotImplementedError on
    construction, so we test options in isolation without constructing a model.
    """

    def test_default_construction(self):
        """GENN options can be constructed with all defaults."""
        opts = GradientEnhancedNeuralNetworksModelOptions()
        assert opts.alpha == 0.05
        assert opts.beta1 == 0.9
        assert opts.beta2 == 0.99
        assert opts.lambd == 0.01
        assert opts.gamma == 1.0
        assert opts.hidden_layer_sizes == [12, 12]
        assert opts.mini_batch_size == -1
        assert opts.num_epochs == 1
        assert opts.num_iterations == 1000
        assert opts.seed == -1
        assert opts.is_print is False
        assert opts.is_normalize is False
        assert opts.is_backtracking is False

    def test_inherits_base_fields(self):
        """GENN options inherit all base AbstractSmtModelOptions fields."""
        for field in BASE_FIELDS:
            assert field in GradientEnhancedNeuralNetworksModelOptions.model_fields

        opts = GradientEnhancedNeuralNetworksModelOptions()
        assert opts.print_global is True
        assert opts.use_xlimits is False
        assert opts.parameters is None

    def test_define_options_returns_correct_class(self):
        """GradientEnhancedNeuralNetworksModel._define_options() returns correct class."""
        from standard_evaluator.surrogate_models.smt_models.genn_model_using_smt import (
            GradientEnhancedNeuralNetworksModel,
        )

        assert (
            GradientEnhancedNeuralNetworksModel._define_options()
            is GradientEnhancedNeuralNetworksModelOptions
        )

    def test_custom_values(self):
        """GENN options accept custom field values."""
        opts = GradientEnhancedNeuralNetworksModelOptions(
            alpha=0.1,
            beta1=0.95,
            beta2=0.999,
            lambd=0.05,
            gamma=2.0,
            hidden_layer_sizes=[24, 24, 12],
            mini_batch_size=32,
            num_epochs=10,
            num_iterations=500,
            seed=42,
            is_print=True,
            is_normalize=True,
            is_backtracking=True,
        )
        assert opts.alpha == 0.1
        assert opts.beta1 == 0.95
        assert opts.beta2 == 0.999
        assert opts.lambd == 0.05
        assert opts.gamma == 2.0
        assert opts.hidden_layer_sizes == [24, 24, 12]
        assert opts.mini_batch_size == 32
        assert opts.num_epochs == 10
        assert opts.num_iterations == 500
        assert opts.seed == 42
        assert opts.is_print is True
        assert opts.is_normalize is True
        assert opts.is_backtracking is True


# =============================================================================
# RegularizedMinimalEnergyTensorProductBSplinesOptions Tests
# =============================================================================


class TestRegularizedMinimalEnergyTensorProductBSplinesOptions:
    """Tests for RegularizedMinimalEnergyTensorProductBSplinesOptions."""

    def test_default_construction(self):
        """RMTS options can be constructed with all defaults."""
        opts = RegularizedMinimalEnergyTensorProductBSplinesOptions()
        assert opts.smoothness == 1.0
        assert opts.regularization_weight == 1e-14
        assert opts.energy_weight == 0.0001
        assert opts.extrapolate is False
        assert opts.min_energy is True
        assert opts.approx_order == 4
        assert opts.solver == "krylov"
        assert opts.derivative_solver == "krylov"
        assert opts.grad_weight == 0.5
        assert opts.solver_tolerance == 1e-12
        assert opts.nonlinear_maxiter == 10
        assert opts.line_search == "backtracking"
        assert opts.save_energy_terms is False
        assert opts.order == 3
        assert opts.num_ctrl_pts == 15

    def test_inherits_base_fields(self):
        """RMTS options inherit all base AbstractSmtModelOptions fields."""
        for field in BASE_FIELDS:
            assert (
                field
                in RegularizedMinimalEnergyTensorProductBSplinesOptions.model_fields
            )

        opts = RegularizedMinimalEnergyTensorProductBSplinesOptions()
        assert opts.print_global is True
        assert opts.use_xlimits is True
        assert opts.parameters is None

    def test_define_options_returns_correct_class(self):
        """RegularizedMinimalEnergyTensorProductBSplines._define_options() returns correct class."""
        assert (
            RegularizedMinimalEnergyTensorProductBSplines._define_options()
            is RegularizedMinimalEnergyTensorProductBSplinesOptions
        )

    @pytest.mark.rmtb_evaluator
    def test_lookup_option_value_model_specific_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct values for RMTS-specific fields."""
        opts = RegularizedMinimalEnergyTensorProductBSplinesOptions(
            smoothness=2.0,
            approx_order=3,
            solver="lu",
            use_xlimits=True,
        )
        model = RegularizedMinimalEnergyTensorProductBSplines(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("smoothness") == 2.0
        assert model.lookup_option_value("approx_order") == 3
        assert model.lookup_option_value("solver") == "lu"

    @pytest.mark.rmtb_evaluator
    def test_lookup_option_value_unknown_name_raises_key_error(
        self, opt_prob, sites_df
    ):
        """lookup_option_value raises KeyError for unknown option names."""
        opts = RegularizedMinimalEnergyTensorProductBSplinesOptions(
            use_xlimits=True,
        )
        model = RegularizedMinimalEnergyTensorProductBSplines(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        with pytest.raises(KeyError):
            model.lookup_option_value("fake_option")

    def test_custom_values(self):
        """RMTS options accept custom field values."""
        opts = RegularizedMinimalEnergyTensorProductBSplinesOptions(
            smoothness=2.5,
            regularization_weight=1e-10,
            energy_weight=0.01,
            extrapolate=True,
            min_energy=False,
            approx_order=2,
            solver="lu",
            derivative_solver="lu",
            grad_weight=0.8,
            solver_tolerance=1e-8,
            nonlinear_maxiter=20,
            line_search="null",
            save_energy_terms=True,
            order=4,
            num_ctrl_pts=20,
        )
        assert opts.smoothness == 2.5
        assert opts.regularization_weight == 1e-10
        assert opts.energy_weight == 0.01
        assert opts.extrapolate is True
        assert opts.min_energy is False
        assert opts.approx_order == 2
        assert opts.solver == "lu"
        assert opts.derivative_solver == "lu"
        assert opts.grad_weight == 0.8
        assert opts.solver_tolerance == 1e-8
        assert opts.nonlinear_maxiter == 20
        assert opts.line_search == "null"
        assert opts.save_energy_terms is True
        assert opts.order == 4
        assert opts.num_ctrl_pts == 20


# =============================================================================
# SecondOrderPolynomialApproximationModelOptions Tests
# =============================================================================


class TestSecondOrderPolynomialApproximationModelOptions:
    """Tests for SecondOrderPolynomialApproximationModelOptions."""

    def test_default_construction(self):
        """SOPA options can be constructed with all defaults (no additional fields)."""
        opts = SecondOrderPolynomialApproximationModelOptions()
        assert opts.print_global is True
        assert opts.use_xlimits is False

    def test_inherits_base_fields(self):
        """SOPA options inherit all base AbstractSmtModelOptions fields."""
        for field in BASE_FIELDS:
            assert (
                field
                in SecondOrderPolynomialApproximationModelOptions.model_fields
            )

    def test_define_options_returns_correct_class(self):
        """SecondOrderPolynomialApproximationModel._define_options() returns correct class."""
        assert (
            SecondOrderPolynomialApproximationModel._define_options()
            is SecondOrderPolynomialApproximationModelOptions
        )

    def test_lookup_option_value_base_fields(self, opt_prob, sites_df):
        """lookup_option_value returns correct values for inherited base fields."""
        opts = SecondOrderPolynomialApproximationModelOptions(
            print_global=False, print_problem=False
        )
        model = SecondOrderPolynomialApproximationModel(
            sites=sites_df, options=opts, opt_problem=opt_prob
        )
        assert model.lookup_option_value("print_global") is False
        assert model.lookup_option_value("print_problem") is False

    def test_lookup_option_value_unknown_name_raises_key_error(
        self, opt_prob, sites_df
    ):
        """lookup_option_value raises KeyError for unknown option names."""
        model = SecondOrderPolynomialApproximationModel(
            sites=sites_df, opt_problem=opt_prob
        )
        with pytest.raises(KeyError):
            model.lookup_option_value("nonexistent")
