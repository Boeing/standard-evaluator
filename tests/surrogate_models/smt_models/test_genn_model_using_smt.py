import numpy as np
import pandas as pd
import pytest

from standard_evaluator.surrogate_models.smt_models import (
    GradientEnhancedNeuralNetworksModel,
)


class TestGENNModelPy:

    def test_default_options(self, opt_prob, sites):

        model_class = GradientEnhancedNeuralNetworksModel
        local_options = model_class.required_options()

        assert local_options.alpha == 0.05
        assert local_options.beta1 == 0.9
        assert local_options.beta2 == 0.99
        assert local_options.gamma == 1.0
        assert local_options.hidden_layer_sizes == [12, 12]
        assert local_options.is_backtracking is False
        assert local_options.is_normalize is False
        assert local_options.is_print is False
        assert local_options.lambd == 0.01
        assert local_options.mini_batch_size == -1
        assert local_options.num_epochs == 1
        assert local_options.num_iterations == 1000
        assert local_options.print_global is True
        assert local_options.print_prediction is True
        assert local_options.print_problem is True
        assert local_options.print_solver is True
        assert local_options.print_training is True
        assert local_options.seed == -1

    def test_check_use_xlimits(self, opt_prob, sites) -> None:
        my_options = GradientEnhancedNeuralNetworksModel.required_options()

        with pytest.raises(
            expected_exception=NotImplementedError,
        ):
            GradientEnhancedNeuralNetworksModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )
