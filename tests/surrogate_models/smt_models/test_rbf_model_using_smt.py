import numpy as np
import pandas as pd
import numdifftools as nd
import pytest

from standard_evaluator.surrogate_models.smt_models import (
    RadialBasisFunctionModel, RadialBasisFunctionModelOptions
)
from standard_evaluator.surrogate_models.smt_models import AbstractSmtModel


@pytest.fixture
def new_sites() -> pd.DataFrame:
    sites = np.array(
        [
            [0.66666667, -2.0],
            [0.66666667, -0.66666667],
            [0.66666667, 0.66666667],
            [0.66666667, 2.0],
            [2.0, -2.0],
            [2.0, -0.66666667],
            [2.0, 0.66666667],
            [-0.66666667, 2.0],
        ]
    )

    site_vals = np.array(
        [
            [6.08641975],
            [1.34567901],
            [0.16049383],
            [2.5308642],
            [37.0],
            [22.77777778],
            [12.11111111],
            [5.19753086],
        ]
    )
    return pd.DataFrame(data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f"])


@pytest.fixture
def multiresp_new_sites() -> pd.DataFrame:

    sites = np.array(
        [
            [-1.1801132, -1.23600234, -1.00750128, 0.95598218],
            [2.58823043, -0.59079174, -0.34969871, 0.84639297],
        ]
    )

    site_vals = np.array([[14.76794761, 5.94901524], [35.90873658, 0.98928299]])

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "x2", "x3", "f", "g"]
    )


class TestRBFModelPy:
    def test_initialization(self, opt_prob, sites):
        my_options = RadialBasisFunctionModelOptions(d0=1.0)

        rbf_m = RadialBasisFunctionModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        assert rbf_m.variables == ["x0", "x1"]
        assert rbf_m.responses == ["f"]
        assert rbf_m.name == "RadialBasisFunctionModel"

    def test_default_options(self, opt_prob, sites):

        local_options = RadialBasisFunctionModel.required_options()
        assert local_options.d0 == 1.0
        assert local_options.max_print_depth == 5
        assert local_options.poly_degree == -1
        assert local_options.print_global == True
        assert local_options.print_prediction == True
        assert local_options.print_problem == True
        assert local_options.print_solver == True
        assert local_options.print_training == True
        assert local_options.reg == 1e-10

    def test_required_options(self, opt_prob, sites) -> None:
        local_options = RadialBasisFunctionModel.required_options()
        abstract_options = AbstractSmtModel.required_options()

        # RBF options should be a subclass of the abstract options
        assert isinstance(local_options, type(abstract_options))

    def test_none_options(self, opt_prob, sites) -> None:
        rbf_eval = RadialBasisFunctionModel(sites=sites, opt_problem=opt_prob)

        assert rbf_eval.lookup_option_value("poly_degree") == -1

    def test_check_use_xlimits(self, opt_prob, sites) -> None:
        my_options = RadialBasisFunctionModel.required_options()

        # use_xlimits is a valid field in AbstractSmtModelOptions
        assert my_options.use_xlimits == False

    def test_default_options_error(self, opt_prob, sites):
        with pytest.raises(
            expected_exception=TypeError,
        ):
            RadialBasisFunctionModel(sites=sites, options=2, opt_problem=opt_prob)

    def test_target_responses(self, multiresp_opt_prob, multiresp_sites):
        my_options = RadialBasisFunctionModelOptions(d0=1.0)
        rbf_eval = RadialBasisFunctionModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )

        # Test target_responses
        test_sites = multiresp_sites[rbf_eval.variables]
        rbf_eval(test_sites, names=[rbf_eval.responses[1]])
        assert list(test_sites) == rbf_eval.variables + [rbf_eval.responses[1]]

        expected_sites = [
            [-1.43390995, -1.88266419, -0.41548999, -0.88564682, 1.90218082],
            [3.44145177, -1.80679821, -2.53763282, -0.86957506, 48.60879966],
            [-1.1801132, -1.23600234, -1.00750128, 0.95598218, 5.94901524],
            [2.58823043, -0.59079174, -0.34969871, 0.84639297, 0.98928299],
            [0.65145111, -1.78654187, -1.61057601, -0.42885959, 19.46252132],
        ]
        np.testing.assert_almost_equal(test_sites.to_numpy(), expected_sites)

    def test_eval_np(self, opt_prob, sites):
        my_options = RadialBasisFunctionModelOptions(d0=1.0)
        rbf_eval = RadialBasisFunctionModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        # Test evaluation
        rbf_test = rbf_eval.sites.copy()  # copies set of sites
        rbf_test[rbf_eval.responses] = 1.0

        # reference array of sites, compare that with the calculated values using eval_np
        rbf_eval(rbf_test)

        np.testing.assert_almost_equal(rbf_eval.sites.to_numpy(), rbf_test.to_numpy())

    def test_update(self, opt_prob, sites, new_sites):
        my_options = RadialBasisFunctionModelOptions(d0=6.0)

        rbf_update_model = RadialBasisFunctionModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        rbf_eval_test = rbf_update_model.sites.copy()  # copies set of sites
        rbf_eval_test[rbf_update_model.responses] = 2.0

        # reference array of sites, compare that with the calculated values using eval_np
        rbf_update_model(rbf_eval_test)
        rbf_update_model.update(new_sites)

        rbf_eval_test_new = rbf_update_model.sites.copy()  # copies set of sites
        # reference array of sites, compare that with the calculated values using eval_np
        rbf_eval_test_new[rbf_update_model.responses] = 2.0

        rbf_update_model(rbf_eval_test_new)

        np.testing.assert_approx_equal(
            rbf_update_model.sites.iat[1, 2], rbf_eval_test_new.iat[1, 2]
        )
        np.testing.assert_approx_equal(
            rbf_update_model.sites.iat[4, 1], rbf_eval_test_new.iat[4, 1]
        )
        np.testing.assert_array_equal(
            rbf_update_model.sites.shape, rbf_eval_test_new.shape
        )

    def test_update_randomsites(self, opt_prob, sites):
        my_options = RadialBasisFunctionModelOptions(d0=6.0)

        rbf_rand_model = RadialBasisFunctionModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        rbf_rand_eval = rbf_rand_model.sites.copy()  # copies set of sites
        rbf_rand_eval[rbf_rand_model.responses] = 1.1

        # reference array of sites, compare that with the calculated values using eval_np
        rbf_rand_model(rbf_rand_eval)
        random_sites = np.array(
            [[28.12, 23.56], [22.54, 14.77], [31.98, 23.65], [22.56, 39.66]]
        )

        calculated_sites = rbf_rand_model.eval_np(random_sites)

        expected_sites = [
            [-1.17219985e-10],
            [-3.42135592e-04],
            [-2.69484061e-13],
            [-6.80804912e-19],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    # Test-5: Test the model against 4 Variables & 2 responses, use sepearte test sites
    def test_multiresponse_sites(self, multiresp_opt_prob, multiresp_sites) -> None:
        my_options = RadialBasisFunctionModelOptions(d0=3.0)

        rbf_multiresp_eval = RadialBasisFunctionModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )
        rbf_multiresp_test = rbf_multiresp_eval.sites.copy()  # copied  set of sites
        # updating the response values of copied sites with dummy values, later check for actual values
        rbf_multiresp_test[rbf_multiresp_eval.responses] = 1.0
        # Test evaluation , reference array of sites, compare that with the calculated values using eval_np
        rbf_multiresp_eval(rbf_multiresp_test)

        np.testing.assert_almost_equal(
            rbf_multiresp_eval.sites.to_numpy(), rbf_multiresp_test.to_numpy()
        )

    # Test-6: Test the model against 4 Variables & 2 responses, use sepearte test sites with update method
    def test_multiresp_update(
        self, multiresp_opt_prob, multiresp_sites, multiresp_new_sites
    ):
        my_options = RadialBasisFunctionModelOptions(d0=3.0)

        rbf_multiresp_update_model = RadialBasisFunctionModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )
        rbf_multiresp_update_eval_test = (
            rbf_multiresp_update_model.sites.copy()
        )  # copied  set of
        # updating the response values of copied sites with dummy values, later check for actual values
        rbf_multiresp_update_eval_test[rbf_multiresp_update_model.responses] = 2.0

        rbf_multiresp_update_model(rbf_multiresp_update_eval_test)

        rbf_multiresp_update_model.update(multiresp_new_sites)

        rbf_multiresp_update_eval_test_new = (
            rbf_multiresp_update_model.sites.copy()
        )  # copied set of sites
        # updating the response values of copied sites with dummy values, later check for actual values
        rbf_multiresp_update_eval_test_new[rbf_multiresp_update_model.responses] = 2.0
        rbf_multiresp_update_model(rbf_multiresp_update_eval_test_new)

        # numpy assertion check to reference array of sites & compare that with the calculated values of eval_np
        np.testing.assert_approx_equal(
            rbf_multiresp_update_model.sites.iat[1, 2],
            rbf_multiresp_update_eval_test_new.iat[1, 2],
        )
        np.testing.assert_approx_equal(
            rbf_multiresp_update_model.sites.iat[4, 1],
            rbf_multiresp_update_eval_test_new.iat[4, 1],
        )
        np.testing.assert_array_equal(
            rbf_multiresp_update_model.sites.shape,
            rbf_multiresp_update_eval_test_new.shape,
        )

    # Test-7: Test for error message "NotImplementedError: RBF does not support variances"
    def test_prob_variance(self, opt_prob, sites):
        my_options = RadialBasisFunctionModelOptions(d0=6.0)

        rbf_var_model = RadialBasisFunctionModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        rbf_var_test = rbf_var_model.sites.copy()  # copies set of sites
        rbf_var_test[rbf_var_model.responses] = 2.0
        rbf_var_model(rbf_var_test)

        with pytest.raises(NotImplementedError, match="RBF does not support variances"):
            RadialBasisFunctionModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )._variance(sites)

    def test_jacobian(self, opt_prob, sites) -> None:
        my_options = RadialBasisFunctionModelOptions(d0=6.0)
        # Test gradient on rosenbrock model

        ros_kmpy = RadialBasisFunctionModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        np.random.seed(0)
        n_pts = 10
        random_x = np.random.uniform(ros_kmpy.xlb, ros_kmpy.xub, size=(n_pts, 2))

        def ros_kmpy_eval(x: np.ndarray) -> np.ndarray:
            return ros_kmpy.eval_np(x.reshape((-1, 2))).flatten()[0]

        approx_grads = np.array(
            [nd.Jacobian(ros_kmpy_eval, step=1e-6)(_x).flatten() for _x in random_x]
        )

        model_grads = ros_kmpy.jacobian(random_x)

        assert model_grads.shape[0] == n_pts
        assert model_grads.shape[1] == 1
        assert model_grads.shape[2] == 2

        tol = 1e-4

        assert (np.max(np.abs(approx_grads - model_grads[:, 0, :]))) < tol
