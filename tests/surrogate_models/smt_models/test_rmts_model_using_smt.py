import numpy as np
import pandas as pd
import numdifftools as nd
import pytest

from standard_evaluator.surrogate_models.smt_models import (
    RegularizedMinimalEnergyTensorProductBSplines,
)
from tests.surrogate_models.smt_models.conftest import make_opt_prob


@pytest.fixture
def xlimits_opt_prob():
    return make_opt_prob(2, [(-2.0, 2.0), (-2.0, 2.0)], ["f"])


@pytest.fixture
def xlimits_sites() -> pd.DataFrame:
    sites = np.array(
        [
            [-0.66666667, -1.0],
            [-0.66666667, -0.66666667],
            [-0.66666667, 0.66666667],
            [-0.66666667, 1.0],
            [0.66666667, -1.0],
            [0.66666667, -0.66666667],
            [0.66666667, 0.66666667],
            [0.66666667, 1.0],
        ]
    )

    site_vals = np.array(
        [
            [45.0],
            [30.77777778],
            [20.11111111],
            [13.0],
            [8.75308642],
            [4.01234568],
            [2.82716049],
            [6.08641975],
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

    sites = np.array([[-2.17896943, -0.79293648], [1.79293648, -1.17896943]])

    site_vals = np.array([[0.85637809, 35.99282989], [5.16749767, 14.42510949]])

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f", "g"]
    )


@pytest.fixture
def rmts_multiresp_opt_prob():
    return make_opt_prob(2, [(-4.0, 4.0), (-2.0, 2.0)], ["f", "g"])


@pytest.fixture
def rmts_multiresp_sites_old() -> pd.DataFrame:
    sites = np.array(
        [
            [-1.43390995, -1.88266419],
            [3.44145177, -1.80679821],
            [-1.1801132, -1.23600234],
            [2.58823043, -0.59079174],
            [0.65145111, -1.78654187],
            [0.43200632, -1.2322081],
        ]
    )

    site_vals = np.array(
        [
            [25.41931167, 1.90218082],
            [83.54405058, 48.60879966],
            [14.76794761, 5.94901524],
            [35.90873658, 0.98928299],
            [4.98129029, 19.46252132],
            [4.42510949, 19.16749767],
        ]
    )

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f", "g"]
    )


@pytest.fixture
def basic_model(opt_prob, sites) -> RegularizedMinimalEnergyTensorProductBSplines:
    return RegularizedMinimalEnergyTensorProductBSplines(
        sites=sites, opt_problem=opt_prob
    )


@pytest.fixture
def multi_model(
    multiresp_opt_prob, multiresp_sites
) -> RegularizedMinimalEnergyTensorProductBSplines:
    return RegularizedMinimalEnergyTensorProductBSplines(
        sites=multiresp_sites, opt_problem=multiresp_opt_prob
    )


# @pytest.mark.rmtb_evaluator
class TestRMTBModelPy:
    def test_initialization(self, basic_model):
        assert basic_model.variables == ["x0", "x1"]
        assert basic_model.responses == ["f"]
        assert basic_model.name == "RegularizedMinimalEnergyTensorProductBSplines"

    def test_default_options(self):

        local_options = RegularizedMinimalEnergyTensorProductBSplines.required_options()
        assert local_options.approx_order == 4
        assert local_options.data_dir is None
        assert local_options.derivative_solver == "krylov"
        assert local_options.energy_weight == 0.0001
        assert local_options.extrapolate is False
        assert local_options.grad_weight == 0.5
        assert local_options.line_search == "backtracking"
        assert local_options.min_energy is True
        assert local_options.nonlinear_maxiter == 10
        assert local_options.num_ctrl_pts == 15
        assert local_options.order == 3
        assert local_options.print_global is True
        assert local_options.print_prediction is True
        assert local_options.print_problem is True
        assert local_options.print_solver is True
        assert local_options.print_training is True
        assert local_options.regularization_weight == 1e-14
        assert local_options.save_energy_terms is False
        assert local_options.smoothness == 1.0
        assert local_options.solver == "krylov"
        assert local_options.solver_tolerance == 1e-12
        assert local_options.use_xlimits is True

    def test_default_options_error(self, opt_prob, sites):
        with pytest.raises(
            expected_exception=TypeError,
            match="options must be an instance of",
        ):
            RegularizedMinimalEnergyTensorProductBSplines(
                sites=sites, options=2, opt_problem=opt_prob
            )

    @pytest.mark.rmtb_evaluator
    def test_target_responses(self, multiresp_sites, multi_model):
        test_sites = multiresp_sites[multi_model.variables]
        multi_model(test_sites, names=[multi_model.responses[1]])
        assert list(test_sites) == multi_model.variables + [multi_model.responses[1]]

    def test_eval_np_with_False_use_xlimits(self, xlimits_opt_prob, xlimits_sites):
        """
        Testing evaluation of the surrogate model after building the model with passed sites then
        asserting the desired response values with the predicted response
        """
        from standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt import (
            RegularizedMinimalEnergyTensorProductBSplinesOptions,
        )

        my_options = RegularizedMinimalEnergyTensorProductBSplinesOptions(
            use_xlimits=False
        )

        with pytest.raises(
            expected_exception=TypeError, match="'NoneType' object is not subscriptable"
        ):
            RegularizedMinimalEnergyTensorProductBSplines(
                sites=xlimits_sites, options=my_options, opt_problem=xlimits_opt_prob
            )

    def test_eval_np(self, basic_model):
        rmtb_test = basic_model.sites.copy()
        rmtb_test[basic_model.responses] = 1.0

        basic_model(rmtb_test)

        expected_sites = [
            [-2.0, -2.0, 44.96975642],
            [-2.0, -0.66666667, 30.7395948],
            [-2.0, 0.66666667, 20.08115782],
            [-2.0, 2.0, 12.97949267],
            [-0.66666667, -2.0, 8.77874145],
            [-0.66666667, -0.66666667, 4.05286613],
            [-0.66666667, 0.66666667, 2.86062665],
            [-0.66666667, 2.0, 5.18479035],
            [0.66666667, -2.0, 6.11207478],
            [0.66666667, -0.66666667, 1.38619946],
            [0.66666667, 0.66666667, 0.19395999],
            [0.66666667, 2.0, 2.51812369],
            [2.0, -2.0, 36.96975642],
            [2.0, -0.66666667, 22.7395948],
            [2.0, 0.66666667, 12.08115782],
            [2.0, 2.0, 4.97949267],
        ]

        np.testing.assert_almost_equal(rmtb_test.to_numpy(), expected_sites)

    def test_update_no_new_sites(self, basic_model, new_sites):
        num_sites_in_model = len(basic_model.sites)

        basic_model.update(new_sites)

        num_sites_in_model_after_update = len(basic_model.sites)
        rmtb_eval_test_new = basic_model.sites.copy()

        assert num_sites_in_model == num_sites_in_model_after_update
        assert (basic_model.sites == rmtb_eval_test_new).all(axis=None)

    def test_update_new_sites(self, basic_model, new_sites_unique):
        num_sites_in_model = len(basic_model.sites)

        basic_model.update(new_sites_unique)
        num_sites_in_model_after_update = len(basic_model.sites)

        rmtb_eval_test_new = basic_model.sites.copy()
        rmtb_eval_test_new[basic_model.responses] = 2.0
        basic_model(rmtb_eval_test_new)

        expected_sites = [
            [-2.00000000e00, -2.00000000e00, 4.49801837e01],
            [-2.00000000e00, -6.66666670e-01, 3.07285499e01],
            [-2.00000000e00, 6.66666670e-01, 2.01677636e01],
            [-2.00000000e00, 2.00000000e00, 1.30561881e01],
            [-6.66666670e-01, -2.00000000e00, 8.70194079e00],
            [-6.66666670e-01, -6.66666670e-01, 4.09387403e00],
            [-6.66666670e-01, 6.66666670e-01, 2.71553265e00],
            [-6.66666670e-01, 2.00000000e00, 5.09059289e00],
            [6.66666670e-01, -2.00000000e00, 6.13571337e00],
            [6.66666670e-01, -6.66666670e-01, 1.27922500e00],
            [6.66666670e-01, 6.66666670e-01, 2.98961698e-01],
            [6.66666670e-01, 2.00000000e00, 2.65470005e00],
            [2.00000000e00, -2.00000000e00, 3.70715921e01],
            [2.00000000e00, -6.66666670e-01, 2.26230393e01],
            [2.00000000e00, 6.66666670e-01, 1.24408437e01],
            [2.00000000e00, 2.00000000e00, 5.17334056e00],
            [2.00000000e00, 1.00000000e00, 9.00665385e02],
            [1.66666667e00, 4.56700000e-01, 2.05896387e00],
        ]

        np.testing.assert_almost_equal(
            rmtb_eval_test_new.to_numpy(), expected_sites, decimal=6
        )
        assert num_sites_in_model + 2 == num_sites_in_model_after_update

    def test_update_randomsites(self, basic_model):
        rmtb_rand_eval = basic_model.sites.copy()
        rmtb_rand_eval[basic_model.responses] = 1.1

        basic_model(rmtb_rand_eval)
        random_sites = np.array(
            [[28.12, 23.56], [22.54, 14.77], [31.98, 23.65], [22.56, 39.66]]
        )

        calculated_sites = basic_model.eval_np(random_sites)

        expected_sites = [[4.97949267], [4.97949267], [4.97949267], [4.97949267]]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    @pytest.mark.rmtb_evaluator
    def test_multiresp_update_no_new_sites(self, multi_model, multiresp_new_sites):

        num_sites_in_model = len(multi_model.sites)

        multi_model.update(multiresp_new_sites)
        num_sites_in_model_after_update = len(multi_model.sites)
        rmtb_multiresp_update_eval_test_new = multi_model.sites.copy()

        assert num_sites_in_model == num_sites_in_model_after_update
        assert (multi_model.sites == rmtb_multiresp_update_eval_test_new).all(axis=None)

    @pytest.mark.rmtb_evaluator
    def test_multiresp_update_new_sites(
        self,
        rmts_multiresp_opt_prob,
        rmts_multiresp_sites_old,
        multiresp_unique_new_sites,
    ):
        smt_unique_multiresp_update_model = (
            RegularizedMinimalEnergyTensorProductBSplines(
                sites=rmts_multiresp_sites_old, opt_problem=rmts_multiresp_opt_prob
            )
        )

        num_sites_in_model = len(smt_unique_multiresp_update_model.sites)
        smt_unique_multiresp_update_model.update(multiresp_unique_new_sites)
        num_sites_in_model_after_update = len(smt_unique_multiresp_update_model.sites)
        rmtb_multiresp_update_eval_test_new = (
            smt_unique_multiresp_update_model.sites.copy()
        )

        assert num_sites_in_model + 2 == num_sites_in_model_after_update
        assert (
            smt_unique_multiresp_update_model.sites
            == rmtb_multiresp_update_eval_test_new
        ).all(axis=None)

    def test_prob_variance(self, basic_model):
        rmtb_var_test = basic_model.sites.copy()
        rmtb_var_test[basic_model.responses] = 2.0
        basic_model(rmtb_var_test)

        with pytest.raises(
            NotImplementedError, match="RMTB does not support variances"
        ):
            basic_model._variance(basic_model.sites)


def test_jacobian(basic_model) -> None:
    # Test gradient on rosenbrock model
    np.random.seed(0)
    n_pts = 10
    random_x = np.random.uniform(basic_model.xlb, basic_model.xub, size=(n_pts, 2))

    def basic_model_eval(x: np.ndarray) -> np.ndarray:
        return basic_model.eval_np(x.reshape((-1, 2))).flatten()[0]

    approx_grads = np.array(
        [nd.Jacobian(basic_model_eval, step=1e-6)(_x).flatten() for _x in random_x]
    )

    model_grads = basic_model.jacobian(random_x)

    assert model_grads.shape[0] == n_pts
    assert model_grads.shape[1] == 1
    assert model_grads.shape[2] == 2

    tol = 1e-4
    assert (np.max(np.abs(approx_grads - model_grads[:, 0, :]))) < tol
