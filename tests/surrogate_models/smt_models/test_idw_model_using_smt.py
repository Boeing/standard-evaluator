import numpy as np
import pandas as pd
import numdifftools as nd
import pytest

from standard_evaluator.surrogate_models.smt_models import (
    InverseDistanceWeightingModel,
)
from standard_evaluator.surrogate_models.smt_models.idw_model_using_smt import (
    InverseDistanceWeightingModelOptions,
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
def multiresp_new_sites_unique() -> pd.DataFrame:

    sites = np.array(
        [
            [-1.1801132, -1.23600234, -1.00750128, 0.95598218],
            [1.58823043, -2.59079174, -1.34969871, 2.84639297],
        ]
    )

    site_vals = np.array([[14.76794761, 5.94901524], [44.90873658, 2.98928299]])

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "x2", "x3", "f", "g"]
    )


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


class TestIDWModelPy:
    def test_initialization(self, opt_prob, sites):
        my_options = InverseDistanceWeightingModel.required_options()
        idw_m = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        assert idw_m.inputs == ["x0", "x1"]
        assert idw_m.outputs == ["f"]
        assert idw_m.name == "InverseDistanceWeightingModel"

    def test_default_options(self, opt_prob, sites):

        local_options = InverseDistanceWeightingModel.required_options()
        assert local_options.data_dir is None
        assert local_options.p == 2.5
        assert local_options.print_global is True
        assert local_options.print_prediction is True
        assert local_options.print_problem is True
        assert local_options.print_solver is True
        assert local_options.print_training is True

    def test_check_use_xlimits(self, opt_prob, sites) -> None:
        my_options = InverseDistanceWeightingModel.required_options()
        # use_xlimits is now a valid field in Pydantic options (default False)
        assert my_options.use_xlimits is False
        # Can set it to True
        opts_with_xlimits = InverseDistanceWeightingModelOptions(use_xlimits=True)
        assert opts_with_xlimits.use_xlimits is True

    def test_check_data_dir(self, opt_prob, sites) -> None:
        my_options = InverseDistanceWeightingModelOptions(data_dir="idw_smt_data")

        with pytest.raises(
            expected_exception=FileNotFoundError, match="No such file or directory:"
        ):
            InverseDistanceWeightingModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )

    def test_default_options_error(self, opt_prob, sites):
        with pytest.raises(
            expected_exception=TypeError,
            match="options must be an instance of",
        ):
            InverseDistanceWeightingModel(
                sites=sites, options=2, opt_problem=opt_prob
            )

    def test_eval_np(self, opt_prob, sites):
        my_options = InverseDistanceWeightingModelOptions(p=3.3)

        idw_eval = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        idw_test = idw_eval.sites.copy()
        idw_test[idw_eval.outputs] = 1.0

        idw_eval(idw_test)

        np.testing.assert_almost_equal(idw_eval.sites.to_numpy(), idw_test.to_numpy())

    def test_update_no_new_sites(self, opt_prob, sites, new_sites):

        my_options = InverseDistanceWeightingModelOptions(p=6)

        idw_update_model = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        idw_eval_test = idw_update_model.sites.copy()
        idw_eval_test[idw_update_model.outputs] = 2.0

        num_sites_in_model = len(idw_update_model.sites)

        idw_update_model(idw_eval_test)
        idw_update_model.update(new_sites)
        num_sites_in_model_after_update = len(idw_update_model.sites)

        idw_eval_test_new = idw_update_model.sites.copy()
        idw_eval_test_new[idw_update_model.outputs] = 2.0
        idw_update_model(idw_eval_test_new)

        np.testing.assert_approx_equal(
            idw_update_model.sites.iat[1, 2], idw_eval_test_new.iat[1, 2]
        )
        np.testing.assert_approx_equal(
            idw_update_model.sites.iat[4, 1], idw_eval_test_new.iat[4, 1]
        )
        assert num_sites_in_model == num_sites_in_model_after_update

    def test_update_new_sites(self, opt_prob, sites, new_sites_unique):
        my_options = InverseDistanceWeightingModelOptions(p=6)

        idw_update_model = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        idw_eval_test = idw_update_model.sites.copy()
        idw_eval_test[idw_update_model.outputs] = 2.0

        num_sites_in_model = len(idw_update_model.sites)

        idw_update_model(idw_eval_test)
        idw_update_model.update(new_sites_unique)
        num_sites_in_model_after_update = len(idw_update_model.sites)

        idw_eval_test_new = idw_update_model.sites.copy()
        idw_eval_test_new[idw_update_model.outputs] = 2.0
        idw_update_model(idw_eval_test_new)

        np.testing.assert_approx_equal(
            idw_update_model.sites.iat[1, 2], idw_eval_test_new.iat[1, 2]
        )
        np.testing.assert_approx_equal(
            idw_update_model.sites.iat[4, 1], idw_eval_test_new.iat[4, 1]
        )
        assert num_sites_in_model + 2 == num_sites_in_model_after_update

    def test_randomsites(self, opt_prob, sites):
        my_options = InverseDistanceWeightingModelOptions(p=6)

        idw_rand_model = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        random_sites = np.array(
            [[28.12, 23.56], [22.54, 14.77], [31.98, 23.65], [22.56, 39.66]]
        )
        calculated_sites = idw_rand_model.eval_np(random_sites)

        expected_sites = [[12.0801628], [11.78435988], [12.24321712], [12.16886354]]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_multiresponse_sites(self, multiresp_opt_prob, multiresp_sites) -> None:
        my_options = InverseDistanceWeightingModelOptions(p=3)

        idw_multiresp_eval = InverseDistanceWeightingModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )
        idw_multiresp_test = idw_multiresp_eval.sites.copy()
        idw_multiresp_test[idw_multiresp_eval.outputs] = 1.0
        idw_multiresp_eval(idw_multiresp_test)

        np.testing.assert_almost_equal(
            idw_multiresp_eval.sites.to_numpy(), idw_multiresp_test.to_numpy()
        )

    def test_multiresp_update_no_new_sites(
        self, multiresp_opt_prob, multiresp_sites, multiresp_new_sites
    ):
        my_options = InverseDistanceWeightingModelOptions(p=3)

        idw_multiresp_update_model = InverseDistanceWeightingModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )
        idw_multiresp_update_eval_test = idw_multiresp_update_model.sites.copy()
        idw_multiresp_update_eval_test[idw_multiresp_update_model.outputs] = 2.0

        num_sites_in_model = len(idw_multiresp_update_model.sites)

        idw_multiresp_update_model(idw_multiresp_update_eval_test)
        idw_multiresp_update_model.update(multiresp_new_sites)
        num_sites_in_model_after_update = len(idw_multiresp_update_model.sites)

        idw_multiresp_update_eval_test_new = idw_multiresp_update_model.sites.copy()
        idw_multiresp_update_eval_test_new[idw_multiresp_update_model.outputs] = 2.0
        idw_multiresp_update_model(idw_multiresp_update_eval_test_new)

        np.testing.assert_approx_equal(
            idw_multiresp_update_model.sites.iat[1, 2],
            idw_multiresp_update_eval_test_new.iat[1, 2],
        )
        np.testing.assert_approx_equal(
            idw_multiresp_update_model.sites.iat[4, 1],
            idw_multiresp_update_eval_test_new.iat[4, 1],
        )
        assert num_sites_in_model == num_sites_in_model_after_update

    def test_multiresp_update_new_sites(
        self, multiresp_opt_prob, multiresp_sites, multiresp_new_sites_unique
    ):
        my_options = InverseDistanceWeightingModelOptions(p=3)

        idw_multiresp_update_model = InverseDistanceWeightingModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )
        idw_multiresp_update_eval_test = idw_multiresp_update_model.sites.copy()
        idw_multiresp_update_eval_test[idw_multiresp_update_model.outputs] = 2.0

        num_sites_in_model = len(idw_multiresp_update_model.sites)

        idw_multiresp_update_model(idw_multiresp_update_eval_test)
        idw_multiresp_update_model.update(multiresp_new_sites_unique)
        num_sites_in_model_after_update = len(idw_multiresp_update_model.sites)

        idw_multiresp_update_eval_test_new = idw_multiresp_update_model.sites.copy()
        idw_multiresp_update_eval_test_new[idw_multiresp_update_model.outputs] = 2.0
        idw_multiresp_update_model(idw_multiresp_update_eval_test_new)

        np.testing.assert_approx_equal(
            idw_multiresp_update_model.sites.iat[1, 2],
            idw_multiresp_update_eval_test_new.iat[1, 2],
        )
        np.testing.assert_approx_equal(
            idw_multiresp_update_model.sites.iat[4, 1],
            idw_multiresp_update_eval_test_new.iat[4, 1],
        )
        assert num_sites_in_model + 1 == num_sites_in_model_after_update

    def test_prob_variance(self, opt_prob, sites):
        my_options = InverseDistanceWeightingModelOptions(p=6)

        idw_var_model = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        idw_var_test = idw_var_model.sites.copy()
        idw_var_test[idw_var_model.outputs] = 2.0
        idw_var_model(idw_var_test)

        with pytest.raises(NotImplementedError, match="IDW does not support variances"):
            InverseDistanceWeightingModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )._variance(sites)

    def test_jacobian(self, opt_prob, sites) -> None:
        # Test gradient on rosenbrock model
        my_options = InverseDistanceWeightingModelOptions(p=6)

        ros_idwpy = InverseDistanceWeightingModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        np.random.seed(0)
        n_pts = 10
        random_x = np.random.uniform(ros_idwpy.xlb, ros_idwpy.xub, size=(n_pts, 2))

        def ros_idwpy_eval(x: np.ndarray) -> np.ndarray:
            return ros_idwpy.eval_np(x.reshape((-1, 2))).flatten()[0]

        approx_grads = np.array(
            [nd.Jacobian(ros_idwpy_eval, step=1e-6)(_x).flatten() for _x in random_x]
        )

        model_grads = ros_idwpy.jacobian(random_x)

        assert model_grads.shape[0] == n_pts
        assert model_grads.shape[1] == 1
        assert model_grads.shape[2] == 2

        tol = 1e-4

        assert (np.max(np.abs(approx_grads - model_grads[:, 0, :]))) < tol
