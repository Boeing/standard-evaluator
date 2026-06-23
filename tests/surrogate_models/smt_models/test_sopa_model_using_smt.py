import numpy as np
import pandas as pd
import numdifftools as nd
import pytest

from standard_evaluator.surrogate_models.smt_models import (
    SecondOrderPolynomialApproximationModel,
)
from tests.surrogate_models.smt_models.conftest import make_opt_prob


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
def sopa_multiresp_opt_prob():
    return make_opt_prob(2, [(-4.0, 4.0), (-2.0, 2.0)], ["f", "g"])


@pytest.fixture
def sopa_multiresp_sites_old() -> pd.DataFrame:
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
def multiresp_sites_old() -> pd.DataFrame:
    sites = np.array(
        [
            [-1.43390995, -1.88266419, -0.41548999, -0.88564682],
            [3.44145177, -1.80679821, -2.53763282, -0.86957506],
            [-1.1801132, -1.23600234, -1.00750128, 0.95598218],
            [2.58823043, -0.59079174, -0.34969871, 0.84639297],
            [0.65145111, -1.78654187, -1.61057601, -0.42885959],
        ]
    )

    site_vals = np.array(
        [
            [25.41931167, 1.90218082],
            [83.54405058, 48.60879966],
            [14.76794761, 5.94901524],
            [35.90873658, 0.98928299],
            [4.98129029, 19.46252132],
        ]
    )

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "x2", "x3", "f", "g"]
    )


@pytest.fixture
def multiresp_sites() -> pd.DataFrame:
    sites = [
        [-1.43391, -1.882664, -0.41549, -0.885647, 4.826612926, -5.969371964],
        [3.441452, -1.806798, -2.537633, -0.869575, -32.50561052, -41.95072587],
        [-1.180113, -1.236002, -1.007501, 0.955982, 2.193967525, -6.494930809],
        [2.58823, -0.590792, -0.349699, 0.846393, 5.226305201, 14.16880038],
        [2.53, -1.786542, -1.610576, -0.42886, 0.314526854, 6.20544283],
        [-1.43391, -0.590792, -0.41549, -0.885647, -1.564164173, -3.385627964],
        [-1.180113, -1.236002, -2.537633, -0.869575, -17.53914995, -106.2734526],
        [0.651451, -1.786542, -1.007501, 0.955982, 6.420994308, -4.743438525],
        [2.58823, -1.806798, -0.349699, 0.846393, 11.05727285, 11.73678838],
        [3.441452, 0.5, -1.610576, -0.42886, -8.416816317, 39.10273329],
        [3.441452, -1.882664, -0.41549, -0.885647, 6.100397634, 41.33961757],
        [2.58823, -1.806798, -2.537633, -0.869575, -22.00631156, -76.72401642],
        [-0.4323, -1.786542, -1.007501, 0.955982, 5.108489586, -4.871942701],
        [-1.180113, -1.236002, -0.349699, 0.846393, 2.244729778, -4.532943078],
        [-1.43391, -0.590792, -1.610576, -0.42886, -6.333772087, -13.54659881],
    ]

    return pd.DataFrame(data=sites, columns=["x0", "x1", "x2", "x3", "f", "g"])


@pytest.fixture
def multiresp_new_sites() -> pd.DataFrame:

    sites = [
        [2.58823, -0.590792, -0.349699, 0.846393, 5.226305201, 14.16880038],
        [-1.43391, -0.590792, -0.41549, -0.885647, -1.564164173, -3.385627964],
    ]

    return pd.DataFrame(data=sites, columns=["x0", "x1", "x2", "x3", "f", "g"])


@pytest.fixture
def multiresp_unique_new_sites() -> pd.DataFrame:

    sites = np.array([[-2.17896943, -0.79293648], [1.79293648, -1.17896943]])

    site_vals = np.array([[0.85637809, 35.99282989], [5.16749767, 14.42510949]])

    return pd.DataFrame(
        data=np.hstack([sites, site_vals]), columns=["x0", "x1", "f", "g"]
    )


class TestQPModelPy:
    def test_initialization(self, opt_prob, sites):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_m = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        assert smt_m.variables == ["x0", "x1"]
        assert smt_m.responses == ["f"]
        assert smt_m.name == "SecondOrderPolynomialApproximationModel"

    def test_default_options(self, opt_prob, sites):

        local_options = SecondOrderPolynomialApproximationModel.required_options()
        assert local_options.data_dir is None
        assert local_options.print_global is True
        assert local_options.print_prediction is True
        assert local_options.print_problem is True
        assert local_options.print_solver is True
        assert local_options.print_training is True

    def test_check_use_xlimits(self, opt_prob, sites) -> None:
        my_options = SecondOrderPolynomialApproximationModel.required_options()
        # use_xlimits is now a valid field inherited from AbstractSmtModelOptions
        assert my_options.use_xlimits is False

    def test_check_data_dir(self, opt_prob, sites) -> None:
        from standard_evaluator.surrogate_models.smt_models.sopa_model_using_smt import (
            SecondOrderPolynomialApproximationModelOptions,
        )
        from pathlib import Path

        my_options = SecondOrderPolynomialApproximationModelOptions(
            data_dir=Path("sopa_smt_data")
        )

        with pytest.raises(
            expected_exception=FileNotFoundError, match="No such file or directory:"
        ):
            SecondOrderPolynomialApproximationModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )

    def test_default_options_error(self, opt_prob, sites):
        with pytest.raises(
            expected_exception=TypeError,
            match="options must be an instance of",
        ):
            SecondOrderPolynomialApproximationModel(
                sites=sites, options=2, opt_problem=opt_prob
            )

    def test_target_responses(self, sopa_multiresp_opt_prob, sopa_multiresp_sites_old):
        my_options = SecondOrderPolynomialApproximationModel.required_options()
        smt_target = SecondOrderPolynomialApproximationModel(
            sites=sopa_multiresp_sites_old, options=my_options, opt_problem=sopa_multiresp_opt_prob
        )

        test_sites = sopa_multiresp_sites_old[smt_target.variables]
        smt_target(test_sites, names=[smt_target.responses[1]])
        assert list(test_sites) == smt_target.variables + [smt_target.responses[1]]

    def test_eval_np(self, opt_prob, sites):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_eval = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        smt_test = smt_eval.sites.copy()
        smt_test[smt_eval.responses] = 1.0

        smt_eval(smt_test)
        calculated_sites = smt_test[smt_eval.responses].to_numpy()

        expected_sites = [
            [37.88888888],
            [28.40740742],
            [22.48148147],
            [20.11111111],
            [15.86419753],
            [6.38271607],
            [0.45679012],
            [-1.91358024],
            [13.19753085],
            [3.71604939],
            [-2.20987656],
            [-4.58024692],
            [29.88888889],
            [20.40740742],
            [14.48148147],
            [12.11111112],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_update_no_new_sites(self, opt_prob, sites, new_sites):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_update_model = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        num_sites_in_model = len(smt_update_model.sites)

        smt_update_model.update(new_sites)

        num_sites_in_model_after_update = len(smt_update_model.sites)
        smt_eval_test_new = smt_update_model.sites.copy()

        assert num_sites_in_model == num_sites_in_model_after_update
        assert (smt_update_model.sites == smt_eval_test_new).all(axis=None)

    def test_update_new_sites(self, opt_prob, sites, new_sites_unique):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_update_model = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        num_sites_in_model = len(smt_update_model.sites)

        smt_update_model.update(new_sites_unique)
        num_sites_in_model_after_update = len(smt_update_model.sites)

        smt_eval_test_new = smt_update_model.sites.copy()
        smt_eval_test_new[smt_update_model.responses] = 2.0
        smt_update_model(smt_eval_test_new)

        np.testing.assert_approx_equal(smt_eval_test_new.iat[1, 2], 67.492844)
        np.testing.assert_approx_equal(smt_eval_test_new.iat[4, 1], -2.000000)
        np.testing.assert_approx_equal(smt_eval_test_new.iat[16, 2], 213.38234819)
        assert num_sites_in_model + 2 == num_sites_in_model_after_update

    def test_randomsites(self, opt_prob, sites):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_rand_model = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        random_sites = np.array(
            [[28.12, 23.56], [22.54, 14.77], [31.98, 23.65], [22.56, 39.66]]
        )
        calculated_sites = smt_rand_model.eval_np(random_sites)

        expected_sites = [
            [4698.45422866],
            [2872.70939285],
            [5957.61801924],
            [4121.72076118],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_multiresponse_sites_exception_error(
        self, multiresp_opt_prob, multiresp_sites_old
    ) -> None:
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        with pytest.raises(Exception) as excp:
            smt_model = SecondOrderPolynomialApproximationModel(
                sites=multiresp_sites_old, options=my_options, opt_problem=multiresp_opt_prob
            )
        assert excp.type == Exception
        assert (
            str(object=excp.value)
            == "Number of training points should be greater or equal to 15."
        )

    def test_multiresponse_sites(
        self, sopa_multiresp_opt_prob, sopa_multiresp_sites_old
    ) -> None:
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_multiresp_eval = SecondOrderPolynomialApproximationModel(
            sites=sopa_multiresp_sites_old, options=my_options, opt_problem=sopa_multiresp_opt_prob
        )
        smt_multiresp_test = smt_multiresp_eval.sites.copy()
        smt_multiresp_test[smt_multiresp_eval.responses] = 1.0
        smt_multiresp_eval(smt_multiresp_test)

        calculated_sites = smt_multiresp_test[smt_multiresp_eval.responses].to_numpy()

        expected_sites = [
            [25.41931167, 1.90218082],
            [83.54405058, 48.60879966],
            [14.76794761, 5.94901524],
            [35.90873658, 0.98928299],
            [4.98129029, 19.46252132],
            [4.42510949, 19.16749767],
        ]

        np.testing.assert_allclose(calculated_sites, expected_sites)

    def test_multiresp_update_no_new_sites(
        self, multiresp_opt_prob, multiresp_sites, multiresp_new_sites
    ):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_multiresp_update_model = SecondOrderPolynomialApproximationModel(
            sites=multiresp_sites, options=my_options, opt_problem=multiresp_opt_prob
        )

        num_sites_in_model = len(smt_multiresp_update_model.sites)
        smt_multiresp_update_model.update(multiresp_new_sites)
        num_sites_in_model_after_update = len(smt_multiresp_update_model.sites)
        smt_multiresp_update_eval_test_new = smt_multiresp_update_model.sites.copy()

        assert num_sites_in_model == num_sites_in_model_after_update
        assert (
            smt_multiresp_update_model.sites == smt_multiresp_update_eval_test_new
        ).all(axis=None)

    def test_multiresp_update_new_sites(
        self,
        sopa_multiresp_opt_prob,
        sopa_multiresp_sites_old,
        multiresp_unique_new_sites,
    ):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_unique_multiresp_update_model = SecondOrderPolynomialApproximationModel(
            sites=sopa_multiresp_sites_old, options=my_options, opt_problem=sopa_multiresp_opt_prob
        )

        num_sites_in_model = len(smt_unique_multiresp_update_model.sites)
        smt_unique_multiresp_update_model.update(multiresp_unique_new_sites)
        num_sites_in_model_after_update = len(smt_unique_multiresp_update_model.sites)

        smt_multiresp_update_eval_test_new = (
            smt_unique_multiresp_update_model.sites.copy()
        )
        smt_multiresp_update_eval_test_new[
            smt_unique_multiresp_update_model.responses
        ] = 2.0
        smt_unique_multiresp_update_model(smt_multiresp_update_eval_test_new)

        np.testing.assert_approx_equal(
            smt_multiresp_update_eval_test_new.iat[0, 2], 28.369088
        )
        np.testing.assert_approx_equal(
            smt_multiresp_update_eval_test_new.iat[0, 3], -0.4724564806
        )
        np.testing.assert_approx_equal(
            smt_multiresp_update_eval_test_new.iat[6, 2], 5.426514
        )
        np.testing.assert_approx_equal(
            smt_multiresp_update_eval_test_new.iat[7, 3], 13.473033
        )
        assert num_sites_in_model + 2 == num_sites_in_model_after_update

    def test_prob_variance(self, opt_prob, sites):
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        smt_var_model = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )
        smt_var_test = smt_var_model.sites.copy()
        smt_var_test[smt_var_model.responses] = 2.0
        smt_var_model(smt_var_test)

        with pytest.raises(NotImplementedError, match="QP does not support variances"):
            SecondOrderPolynomialApproximationModel(
                sites=sites, options=my_options, opt_problem=opt_prob
            )._variance(sites)

    def test_jacobian(self, opt_prob, sites) -> None:
        my_options = SecondOrderPolynomialApproximationModel.required_options()

        ros_sopapy = SecondOrderPolynomialApproximationModel(
            sites=sites, options=my_options, opt_problem=opt_prob
        )

        np.random.seed(0)
        n_pts = 10
        random_x = np.random.uniform(ros_sopapy.xlb, ros_sopapy.xub, size=(n_pts, 2))

        def ros_sopapy_eval(x: np.ndarray) -> np.ndarray:
            return ros_sopapy.eval_np(x.reshape((-1, 2))).flatten()[0]

        approx_grads = np.array(
            [nd.Jacobian(ros_sopapy_eval, step=1e-6)(_x).flatten() for _x in random_x]
        )

        model_grads = ros_sopapy.jacobian(random_x)

        assert model_grads.shape[0] == n_pts
        assert model_grads.shape[1] == 1
        assert model_grads.shape[2] == 2

        tol = 1e-4

        assert (np.max(np.abs(approx_grads - model_grads[:, 0, :]))) < tol
