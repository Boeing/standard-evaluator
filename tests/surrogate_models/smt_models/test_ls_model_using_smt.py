import numpy as np
import pandas as pd
import numdifftools as nd
import pytest

from standard_evaluator.surrogate_models.smt_models import (
    LeastSquaresApproximationModel,
)


@pytest.fixture
def new_sites_unique() -> pd.DataFrame:
    sites = np.array(
        [
            [0.66666667, -2.0],
            [0.66666667, -0.66666667],
            [0.66666667, 0.66666667],
            [0.66666667, 2.0],
            [2.0, -2.0],
            [2.0, -0.66666667],
            [2.0, 1],
            [1.66666667, 0.4567],
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
            [901.00000000],
            [2.19753086],
        ]
    )
    return pd.DataFrame(data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f"])


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


@pytest.fixture
def multiresp_unique_new_sites() -> pd.DataFrame:

    sites = np.array(
        [
            [-2.17896943, -0.79293648, -0.34032885, 1.47197095],
            [1.79293648, -1.17896943, -0.47197095, 2.34032885],
        ]
    )

    site_vals = np.array([[0.85637809, 35.99282989], [5.16749767, 14.42510949]])

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "x2", "x3", "f", "g"]
    )


class TestLSModelPy:
    def test_initialization(self, opt_prob, sites):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_m = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        assert ls_m.variables == ["x0", "x1"]
        assert ls_m.responses == ["f"]
        assert ls_m.name == "LeastSquaresApproximationModel"

    def test_check_use_xlimits(self, opt_prob, sites) -> None:
        my_options = LeastSquaresApproximationModel.required_options()
        assert my_options.use_xlimits is False

    def test_default_options(self, opt_prob, sites):

        local_options = LeastSquaresApproximationModel.required_options()
        assert local_options.data_dir is None
        assert local_options.print_global is True
        assert local_options.print_prediction is True
        assert local_options.print_problem is True
        assert local_options.print_solver is True
        assert local_options.print_training is True

    def test_default_options_error(self, opt_prob, sites):
        with pytest.raises(
            expected_exception=TypeError,
            match="options must be an instance of",
        ):
            LeastSquaresApproximationModel(
                sites=sites, options=2, opt_problem=opt_prob
            )

    def test_eval_np(self, opt_prob, sites):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_eval = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        ls_test = ls_eval.sites.copy()
        ls_test[ls_eval.responses] = 1.0

        ls_eval(ls_test)
        calculated_sites = ls_test[ls_eval.responses].to_numpy()

        expected_sites = [
            [26.43209876],
            [20.50617285],
            [14.5802469],
            [8.65432099],
            [23.7654321],
            [17.83950619],
            [11.91358024],
            [5.98765433],
            [21.09876542],
            [15.17283951],
            [9.24691356],
            [3.32098765],
            [18.43209876],
            [12.50617285],
            [6.5802469],
            [0.65432099],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_update_no_new_sites(self, opt_prob, sites, new_sites):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_update_model = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        num_sites_in_model = len(ls_update_model.sites)

        ls_update_model.update(new_sites)

        num_sites_in_model_after_update = len(ls_update_model.sites)
        ls_eval_test_new = ls_update_model.sites.copy()

        assert num_sites_in_model == num_sites_in_model_after_update
        assert (ls_update_model.sites == ls_eval_test_new).all(axis=None)

    def test_update_new_sites(self, opt_prob, sites, new_sites_unique):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_update_model = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        num_sites_in_model = len(ls_update_model.sites)

        ls_update_model.update(new_sites_unique)
        num_sites_in_model_after_update = len(ls_update_model.sites)

        ls_eval_test_new = ls_update_model.sites.copy()
        ls_eval_test_new[ls_update_model.responses] = 2.0
        ls_update_model(ls_eval_test_new)

        np.testing.assert_approx_equal(ls_eval_test_new.iat[1, 2], -27.146086)
        np.testing.assert_approx_equal(ls_eval_test_new.iat[4, 1], -2.000000)
        np.testing.assert_approx_equal(ls_eval_test_new.iat[16, 2], 139.84341)
        assert num_sites_in_model + 2 == num_sites_in_model_after_update

    def test_randomsites(self, opt_prob, sites):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_rand_model = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        random_sites = np.array(
            [[28.12, 23.56], [22.54, 14.77], [31.98, 23.65], [22.56, 39.66]]
        )
        calculated_sites = ls_rand_model.eval_np(random_sites)

        expected_sites = [
            [-147.40790115],
            [-97.18123451],
            [-155.52790115],
            [-207.84345668],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_multiresponse_sites(self, multiresp_opt_prob, multiresp_sites) -> None:
        my_options = LeastSquaresApproximationModel.required_options()

        ls_multiresp_eval = LeastSquaresApproximationModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )
        ls_multiresp_test = ls_multiresp_eval.sites.copy()
        ls_multiresp_test[ls_multiresp_eval.responses] = 1.0
        ls_multiresp_eval(ls_multiresp_test)

        calculated_sites = ls_multiresp_test[ls_multiresp_eval.responses].to_numpy()

        expected_sites = [
            [25.41931167, 1.90218082],
            [83.54405058, 48.60879966],
            [14.76794761, 5.94901524],
            [35.90873658, 0.98928299],
            [4.98129029, 19.46252132],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_multiresp_update_no_new_sites(
        self, multiresp_opt_prob, multiresp_sites, multiresp_new_sites
    ):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_multiresp_update_model = LeastSquaresApproximationModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )

        num_sites_in_model = len(ls_multiresp_update_model.sites)

        ls_multiresp_update_model.update(multiresp_new_sites)
        num_sites_in_model_after_update = len(ls_multiresp_update_model.sites)
        ls_multiresp_update_eval_test_new = ls_multiresp_update_model.sites.copy()

        assert num_sites_in_model == num_sites_in_model_after_update
        assert (
            ls_multiresp_update_model.sites == ls_multiresp_update_eval_test_new
        ).all(axis=None)

    def test_multiresp_update_new_sites(
        self, multiresp_opt_prob, multiresp_sites, multiresp_unique_new_sites
    ):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_multiresp_update_model = LeastSquaresApproximationModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )

        num_sites_in_model = len(ls_multiresp_update_model.sites)
        ls_multiresp_update_model.update(multiresp_unique_new_sites)
        num_sites_in_model_after_update = len(ls_multiresp_update_model.sites)

        ls_multiresp_update_eval_test_new = ls_multiresp_update_model.sites.copy()
        ls_multiresp_update_eval_test_new[ls_multiresp_update_model.responses] = 2.0
        ls_multiresp_update_model(ls_multiresp_update_eval_test_new)

        np.testing.assert_approx_equal(
            ls_multiresp_update_eval_test_new.iat[0, 4], 15.643701
        )
        np.testing.assert_approx_equal(
            ls_multiresp_update_eval_test_new.iat[0, 5], -0.8552268
        )
        np.testing.assert_approx_equal(
            ls_multiresp_update_eval_test_new.iat[5, 4], -1.758510
        )
        np.testing.assert_approx_equal(
            ls_multiresp_update_eval_test_new.iat[6, 5], 11.371380
        )
        assert num_sites_in_model + 2 == num_sites_in_model_after_update

    def test_prob_variance(self, opt_prob, sites):
        my_options = LeastSquaresApproximationModel.required_options()

        ls_var_model = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        ls_var_test = ls_var_model.sites.copy()
        ls_var_test[ls_var_model.responses] = 2.0
        ls_var_model(ls_var_test)

        with pytest.raises(NotImplementedError, match="LS does not support variances"):
            LeastSquaresApproximationModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )._variance(sites)

    def test_jacobian(self, opt_prob, sites) -> None:
        my_options = LeastSquaresApproximationModel.required_options()

        ros_lspy = LeastSquaresApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        np.random.seed(0)
        n_pts = 10
        random_x = np.random.uniform(ros_lspy.xlb, ros_lspy.xub, size=(n_pts, 2))

        def ros_lspy_eval(x: np.ndarray) -> np.ndarray:
            return ros_lspy.eval_np(x.reshape((-1, 2))).flatten()[0]

        approx_grads = np.array(
            [nd.Jacobian(ros_lspy_eval, step=1e-6)(_x).flatten() for _x in random_x]
        )

        model_grads = ros_lspy.jacobian(random_x)

        assert model_grads.shape[0] == n_pts
        assert model_grads.shape[1] == 1
        assert model_grads.shape[2] == 2

        tol = 1e-4

        assert (np.max(np.abs(approx_grads - model_grads[:, 0, :]))) < tol
