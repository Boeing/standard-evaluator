"""Property-based tests for AbstractSmtModel options.

Tests the lookup_option_value behavior for unknown option names and
option accessor round-trip.
"""

import numpy as np
import pandas as pd
import pytest

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import standard_evaluator as se
from standard_evaluator.surrogate_models.smt_models import (
    RadialBasisFunctionModel,
)
from standard_evaluator.surrogate_models.smt_models.abstract_smt_model import (
    AbstractSmtModelOptions,
)


# Valid field names for AbstractSmtModelOptions
VALID_FIELD_NAMES = frozenset(AbstractSmtModelOptions.model_fields.keys())


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
    """Create training sites DataFrame."""
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


@pytest.fixture
def trained_model(opt_prob, sites_df):
    """Create a trained RadialBasisFunctionModel for testing lookup_option_value."""
    model = RadialBasisFunctionModel(
        sites=sites_df,
        opt_problem=opt_prob,
    )
    return model


class TestUnknownOptionNameRejection:
    """Property 7: Unknown Option Name Rejection.

    For any string that is not a field name in the model's options class,
    calling model.lookup_option_value(that_string) SHALL raise a KeyError.

    Feature: surrogate-model-pydantic-options, Property 7: Unknown Option Name Rejection

    **Validates: Requirements 13.3**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        name=st.text(min_size=1, max_size=50).filter(
            lambda s: s not in VALID_FIELD_NAMES
        )
    )
    def test_unknown_option_name_raises_key_error(self, trained_model, name):
        """Any string not matching a valid field name raises KeyError.

        Feature: surrogate-model-pydantic-options, Property 7: Unknown Option Name Rejection

        **Validates: Requirements 13.3**
        """
        with pytest.raises(KeyError):
            trained_model.lookup_option_value(name)


class TestOptionAccessorRoundTrip:
    """Property 6: Option Accessor Round-Trip.

    For any refactored model class, for any field name defined in its options
    class, and for any valid value for that field, constructing the model with
    options=ModelOptions(field_name=value) and then calling
    model.lookup_option_value("field_name") SHALL return the same value.

    Feature: surrogate-model-pydantic-options, Property 6: Option Accessor Round-Trip

    **Validates: Requirements 13.1, 13.2**
    """

    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        print_global=st.booleans(),
        print_training=st.booleans(),
        print_prediction=st.booleans(),
        print_problem=st.booleans(),
        print_solver=st.booleans(),
    )
    def test_print_fields_round_trip(
        self,
        opt_prob,
        sites_df,
        print_global,
        print_training,
        print_prediction,
        print_problem,
        print_solver,
    ):
        """Constructing RadialBasisFunctionModel with random print_* booleans
        and verifying lookup_option_value returns the same values.

        Feature: surrogate-model-pydantic-options, Property 6: Option Accessor Round-Trip

        **Validates: Requirements 13.1, 13.2**
        """
        options = AbstractSmtModelOptions(
            print_global=print_global,
            print_training=print_training,
            print_prediction=print_prediction,
            print_problem=print_problem,
            print_solver=print_solver,
        )

        model = RadialBasisFunctionModel(
            sites=sites_df, options=options, opt_problem=opt_prob
        )

        assert model.lookup_option_value("print_global") == print_global
        assert model.lookup_option_value("print_training") == print_training
        assert model.lookup_option_value("print_prediction") == print_prediction
        assert model.lookup_option_value("print_problem") == print_problem
        assert model.lookup_option_value("print_solver") == print_solver
