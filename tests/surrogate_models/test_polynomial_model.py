import numpy as np
import pandas as pd
import pytest
import warnings

from standard_evaluator.surrogate_models.polynomial_model import (
    PolynomialModel, 
    PolynomialModelOptions, 
    PolynomialModelParameters,
    CoefficientOrdering
)
from standard_evaluator.evaluators.test.cantilevered_beam_with_fixed_variables import (
    CantileveredBeamFixedVariable,
)
from standard_evaluator.evaluators.test.cantilevered_beam import CantileveredBeam
import standard_evaluator as se
from standard_evaluator import (
    OptProblem,
    IntVariable,
    FloatVariable,
    CategoricalVariable,
)
from standard_evaluator.utilities.problem_dict_utility import (
    create_opt_problem,legacy_to_opt_problem, opt_problem_to_legacy
)


@pytest.fixture
def cantilever_beam_model():
    """Returns problem, sites"""

    # Define problem

    site_df = pd.DataFrame(
        data={
            "b1": [2, 4.5, 7, 10, 12],
            "b2": [0.1, 0.57, 0.70, 1.3, 1.983],
            "H": [3.1111, 3.6, 5.6, 6.8, 6.99],
            "x0": [2, 1, 4, 6, 7],
        }
    )
    cb = CantileveredBeam()
    cb(site_df)
    return opt_problem_to_legacy(cb.opt_problem), site_df


@pytest.fixture
def cantilever_beam_fixed_variables_model():
    """Returns problem, sites"""

    # Define problem

    site_df_fv = pd.DataFrame(
        data={
            "b1": [2, 4.5, 7, 10, 12],
            "b2": [0.1, 0.57, 0.70, 1.3, 1.983],
            "H": [3.1111, 3.6, 5.6, 6.8, 6.99],
            "x0": [2, 1, 4, 6, 7],
            "C0": [0.0, 0.0, 0.0, 0.0, 0.0],
            "C1": [1, 1, 1, 1, 1],
        }
    )
    cbfv = CantileveredBeamFixedVariable()
    cbfv(site_df_fv)
    return opt_problem_to_legacy(cbfv.opt_problem), site_df_fv


@pytest.fixture
def simple_bp_data_model():
    """Returns a bound, data set, and coefficients"""

    # Define response

    def f(x, y, z):

        return [x**2 - y**3 + 12.0 * y * z]

    # Define bound problem
    new_prob = se.utilities.create_opt_problem(num_independent=3, num_dependent=1)

    var_names = ["x", "y", "z"]
    var_bounds = ([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])

    for var, local_bound, name in zip(
        new_prob.variables, var_bounds, var_names
    ):
        var.bounds = local_bound
        var.name = name

    resp_names = ["f"]

    for resp, name in zip(
        new_prob.responses, resp_names
    ):
        resp.name = name

    # Define objectives and constraints
    new_prob.objectives = []
    new_prob.constraints = []

    # Define and evaluate sites

    sites = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
        ]
    )
    evals = np.array([f(p[0], p[1], p[2]) for p in sites]).flatten()

    total_sites = np.hstack((sites, evals.reshape(evals.size, 1)))
    column_list = ["x", "y", "z", "f"]

    sites_df = pd.DataFrame(data=total_sites, columns=column_list)

    # Defining coefficients

    degree = 3
    coefs = np.array([[1.0, -1.0, 12.0]])
    deg_exp = np.array([[2, 0, 0], [0, 3, 0], [0, 1, 1]])

    # Return data

    return new_prob, sites_df, evals, degree, coefs, deg_exp


@pytest.fixture
def simple_multiple_response_data():

    # Set random seed

    np.random.seed(0)

    # Sample two functions

    n_pts = 100

    sites = np.random.uniform(-1.0, 1.0, size=(n_pts, 2))

    def f(x, y):

        return x**4 + y**4

    def g(x, y):

        return x**2 * y**2 + x * y**3

    f_vals = np.array([f(_x, _y) for [_x, _y] in sites])
    g_vals = np.array([g(_x, _y) for [_x, _y] in sites])
    resp_vals = np.vstack([f_vals, g_vals]).T
    total_sites = np.hstack((sites, resp_vals))

    column_list = ["x", "y", "f", "g"]

    sites_df = pd.DataFrame(data=total_sites, columns=column_list)  # values

    new_prob = se.utilities.create_opt_problem(num_independent=2, num_dependent=2)

    var_names = ["x", "y"]
    var_bounds = ([-1.0, 1.0], [-1.0, 1.0])

    for var, local_bound, name in zip(
        new_prob.variables, var_bounds, var_names
    ):
        var.bounds = local_bound
        var.name = name

    resp_names = ["f","g"]

    for resp, name in zip(
        new_prob.responses, resp_names
    ):
        resp.name = name

    # Define objectives and constraints
    new_prob.objectives = []
    new_prob.constraints = []    

    # Degree

    degree = 4

    return new_prob, sites_df, resp_vals, degree


class TestPolynomialModel:
    def test_options_pattern_initialization(self, simple_bp_data_model):
        """Test initialization using the new Pydantic options pattern."""
        opt_prob, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model
        
        # Test with options object
        options = PolynomialModelOptions(
            degree=degree,
            coefficient_ordering=CoefficientOrdering.DEC_GRLEX,
            parameters=PolynomialModelParameters(
                coefficients=coefs,
                degree_exponents=deg_exp.astype(np.float64)
            )
        )
        
        poly_model = PolynomialModel(
            sites=sites_df,
            options=options,
            opt_problem=opt_prob,
        )
        
        # Test basics
        assert poly_model.nsites == 29
        assert poly_model.nind == 3
        assert poly_model.ndep == 1
        assert poly_model.degree == degree
        assert poly_model.lookup_option_value("degree") == degree
        assert poly_model.lookup_option_value("coefficient_ordering") == CoefficientOrdering.DEC_GRLEX
        assert poly_model.lookup_option_value("parameters") is not None
        assert np.allclose(poly_model.coefs, coefs)
        assert np.allclose(poly_model.deg_exp, deg_exp)
    
    def test_backward_compatibility_warnings(self, simple_bp_data_model):
        """Test that deprecated parameters trigger appropriate warnings."""
        opt_problem, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model
        bp = opt_problem_to_legacy(opt_problem)

        # Test deprecated parameters trigger warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            poly_model = PolynomialModel(
                sites=sites_df,
                degree=degree,
                coefficients=coefs,
                degree_exponents=deg_exp,
                problem=bp,
            )
            
            # Should have deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            
            # Check warning messages
            warning_messages = [str(warning.message) for warning in deprecation_warnings]
            assert any("deprecated" in msg for msg in warning_messages)
    
    def test_coefficient_ordering_options(self, simple_bp_data_model):
        """Test different coefficient ordering options."""
        opt_problem, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model
        
        for ordering in CoefficientOrdering:
            options = PolynomialModelOptions(
                degree=2,
                coefficient_ordering=ordering
            )
            
            poly_model = PolynomialModel(
                sites=sites_df,
                options=options,
                opt_problem=opt_problem,
            )
            
            assert poly_model.lookup_option_value("coefficient_ordering") == ordering
            assert poly_model.deg_exp is not None
            assert poly_model.coefs is not None
    
    def test_from_data_with_options(self, simple_bp_data_model):
        """Test from_data method with options parameter."""
        opt_problem, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model
        
        options = PolynomialModelOptions(
            degree=degree,
            coefficient_ordering=CoefficientOrdering.ASC_GRLEX
        )
        
        poly_model = PolynomialModel.from_data(
            sites=sites_df,
            degree=degree,
            options=options,
            opt_problem=opt_problem
        )
        
        assert poly_model.degree == degree
        assert poly_model.lookup_option_value("coefficient_ordering") == CoefficientOrdering.ASC_GRLEX
        
        # Test evaluation still works
        variable_list = opt_problem.variable_names()
        sites_np_array = sites_df[variable_list].values
        model_evals = poly_model.eval_np(sites_np_array)
        assert model_evals.shape[0] == 29
        assert model_evals.shape[1] == 1
    
    def test_serialization_backward_compatibility(self, simple_bp_data_model):
        """Test that both old and new serialization formats work."""
        opt_problem, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        bp = opt_problem_to_legacy(opt_problem)
        
        # Create model with new options
        options = PolynomialModelOptions(
            degree=degree,
            coefficient_ordering=CoefficientOrdering.DEC_GRLEX,
            parameters=PolynomialModelParameters(
                coefficients=coefs,
                degree_exponents=deg_exp.astype(np.float64)
            )
        )
        
        original_model = PolynomialModel(
            sites=sites_df,
            options=options,
            opt_problem=opt_problem,
        )
        
        # Test serialization
        model_dict = original_model.to_dict()
        
        # Verify new format uses "options" key with nested "parameters"
        assert "options" in model_dict["info"]
        assert "degree" in model_dict["info"]["options"]
        assert "parameters" in model_dict["info"]["options"]
        assert "coefficients" in model_dict["info"]["options"]["parameters"]
        assert "degree_exponents" in model_dict["info"]["options"]["parameters"]
        assert "coefficient_ordering" in model_dict["info"]["options"]
        
        # Test deserialization
        restored_model = PolynomialModel.from_dict(model_dict)
        
        # Verify models are equivalent
        assert restored_model.degree == original_model.degree
        assert np.allclose(restored_model.coefs, original_model.coefs)
        assert np.allclose(restored_model.deg_exp, original_model.deg_exp)
        
        # Test old format compatibility
        old_format_dict = {
            "type": "PolynomialModel",
            "version": "1.0",
            "design explorer version": "1.0",
            "info": {
                "sites": sites_df.to_dict(),
                "sites_dtypes": ["float64", "float64", "float64", "float64"],
                "degree": degree,
                "coefs": coefs.tolist(),
                "deg_exp": deg_exp.tolist(),
                "coefficient_ordering": "dec_grlex"
            },
            "problem": bp,
            "name": None
        }
        
        old_format_model = PolynomialModel.from_dict(old_format_dict)
        assert old_format_model.degree == degree
        assert np.allclose(old_format_model.coefs, coefs)
        assert np.allclose(old_format_model.deg_exp, deg_exp)

    def test_from_coefficients(self, simple_bp_data_model):
        """Test Initialization from coefficients using deprecated parameters."""
        opt_problem, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        # Test with deprecated parameters (should trigger warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            poly_model = PolynomialModel(
                sites=sites_df,
                degree=degree,
                coefficients=coefs,
                degree_exponents=deg_exp,
                opt_problem=opt_problem,
            )
            
            # Should have deprecation warnings
            assert len([warning for warning in w if issubclass(warning.category, DeprecationWarning)]) >= 1

        # Test basics
        assert poly_model.nsites == 29
        assert poly_model.nind == 3
        assert poly_model.ndep == 1

    def test_from_data_one_ndep(self, simple_bp_data_model):
        """Test Initialization from data using deprecated from_data method."""
        opt_prob, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        bp = opt_problem_to_legacy(opt_prob)

        # Test with deprecated from_data method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            poly_model = PolynomialModel.from_data(
                sites=sites_df, degree=degree, problem=bp
            )
            
            # Should have deprecation warnings for problem parameter
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

        # Test call
        tol = 1e-6


        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]
        sites_np_array = sites_df[variable_list].values
        max_res = np.max(np.abs(poly_model.eval_np(sites_np_array).flatten() - evals))
        assert max_res < tol

        # Test coefficients
        for coef, deg_exp in zip(poly_model.coefs[0], poly_model.deg_exp):
            target_coefs = [[2, 0, 0], [0, 3, 0], [0, 1, 1]]
            deg_exp_list = deg_exp.tolist()

            tol = 1e-6

            if deg_exp_list in target_coefs:
                if deg_exp_list == [2, 0, 0]:
                    assert np.abs(coef - 1.0) < tol
                elif deg_exp_list == [0, 3, 0]:
                    assert np.abs(coef + 1.0) < tol
                elif deg_exp_list == [0, 1, 1]:
                    assert np.abs(coef - 12.0) < tol
            else:
                assert np.abs(coef) < tol

    def test_from_data_multi_ndep(self, simple_multiple_response_data):
        """Test Initialization from data with multiple responses."""
        opt_prob, sites_df, evals, degree = simple_multiple_response_data
        bp = opt_problem_to_legacy(opt_prob)

        # Test with deprecated from_data method
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            poly_model = PolynomialModel.from_data(
                sites=sites_df, degree=degree, problem=bp
            )
            
            # Should have deprecation warnings for problem parameter
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

        # Test call
        tol = 1e-6
        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]        
        sites_np_array = sites_df[variable_list].values
        max_res = np.max(np.abs(poly_model.eval_np(sites_np_array) - evals))
        assert max_res < tol

        # Test coefficients
        for coef, deg_exp in zip(poly_model.coefs[0], poly_model.deg_exp):
            target_coefs = [[4, 0], [0, 4]]
            deg_exp_list = deg_exp.tolist()

            tol = 1e-6

            if deg_exp_list in target_coefs:
                if deg_exp_list == [4, 0]:
                    assert np.abs(coef - 1.0) < tol
                elif deg_exp_list == [0, 4]:
                    assert np.abs(coef - 1.0) < tol
            else:
                assert np.abs(coef) < tol

        for coef, deg_exp in zip(poly_model.coefs[1], poly_model.deg_exp):
            target_coefs = [[2, 2], [1, 3]]
            deg_exp_list = deg_exp.tolist()

            tol = 1e-6

            if deg_exp_list in target_coefs:
                if deg_exp_list == [2, 2]:
                    assert np.abs(coef - 1.0) < tol
                elif deg_exp_list == [1, 3]:
                    assert np.abs(coef - 1.0) < tol
            else:
                assert np.abs(coef) < tol

    def test_call_multi_ndep(self, simple_multiple_response_data):

        # Test Initialization from data

        opt_prob, sites_df, evals, degree = simple_multiple_response_data

        # Initialitize

        poly_model = PolynomialModel.from_data(
            sites=sites_df, degree=degree, opt_problem=opt_prob
        )

        # Test call

        tol = 1e-6
        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]        
        sites_np_array = sites_df[variable_list].values
        max_res = np.max(np.abs(poly_model.eval_np(sites_np_array) - evals))
        assert max_res < tol

    def test_call_one_ndep(self, simple_bp_data_model):

        # Initialization from coefficients

        opt_prob, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        # Initialize model

        poly_model = PolynomialModel(
            sites=sites_df,
            degree=degree,
            coefficients=coefs,
            degree_exponents=deg_exp,
            opt_problem=opt_prob,
        )

        # Test call

        tol = 1e-12

        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]
        sites_np_array = sites_df[variable_list].values
        model_evals = poly_model.eval_np(sites_np_array)
        assert model_evals.shape[0] == 29
        assert model_evals.shape[1] == 1

        max_res = np.max(np.abs(model_evals - evals.reshape((-1, 1))))
        assert max_res < tol

    def test_jacobian_one_ndep(self, simple_bp_data_model):

        # Initialization from coefficients

        opt_prob, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        # Initialize model

        poly_model = PolynomialModel(
            sites=sites_df,
            degree=degree,
            coefficients=coefs,
            degree_exponents=deg_exp,
            opt_problem=opt_prob,
        )

        # Define true jacobian

        def true_jac(x):

            res = np.zeros((x.shape[0], 3))
            res[:, 0] = 2.0 * x[:, 0]
            res[:, 1] = -3.0 * x[:, 1] ** 2 + 12.0 * x[:, 2]
            res[:, 2] = 12.0 * x[:, 1]

            return res


        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]
        sites_np_array = sites_df[variable_list].values
        jac_target = true_jac(sites_np_array)[:, np.newaxis, :]

        # Test jacobian

        model_jac = poly_model.jacobian(sites_np_array)

        assert model_jac.shape[0] == 29
        assert model_jac.shape[1] == 1
        assert model_jac.shape[2] == 3

        tol = 1e-12
        max_jac_diff = np.max(np.abs(model_jac - jac_target))
        assert max_jac_diff < tol

    def test_jacobian_multi_ndep(self, simple_multiple_response_data):

        # Test Initialization from data

        opt_prob, sites_df, evals, degree = simple_multiple_response_data

        # Initialitize

        poly_model = PolynomialModel.from_data(
            sites=sites_df, degree=degree, opt_problem=opt_prob
        )

        # Define true jacobian

        def true_jac(x):

            jacobians = []

            for _x in x:

                jac = np.zeros((2, 2))
                jac[0, 0] = 4.0 * _x[0] ** 3
                jac[0, 1] = 4.0 * _x[1] ** 3
                jac[1, 0] = 2.0 * _x[0] * _x[1] ** 2 + _x[1] ** 3
                jac[1, 1] = 2.0 * _x[0] ** 2 * _x[1] + 3.0 * _x[0] * _x[1] ** 2

                jacobians.append(jac)

            return np.array(jacobians)

        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]
        sites_np_array = sites_df[variable_list].values
        jac_target = true_jac(sites_np_array)

        # Evaluate jacobian

        model_jac = poly_model.jacobian(sites_np_array)

        # Test jacobian shape

        assert model_jac.shape[0] == len(sites_np_array)
        assert model_jac.shape[1] == 2
        assert model_jac.shape[2] == 2

        # Test jacobian values

        tol = 1e-6
        max_res = np.max(np.abs(model_jac - jac_target))
        assert max_res < tol

    def test_get_response_models(self, simple_multiple_response_data):

        # Test Initialization from data

        opt_prob, sites_df, evals, degree = simple_multiple_response_data

        # Initialitize

        poly_model = PolynomialModel.from_data(
            sites=sites_df, degree=degree, opt_problem=opt_prob
        )

        # Extract f

        f_poly_model = poly_model.get_response_models(["f"])

        # Test data

        assert f_poly_model.ndep == 1
        assert f_poly_model.outputs == ["f"]

        # Test call

        # Extract variable names from OptProblem object
        variable_list = [var.name for var in opt_prob.variables]
        sites_np_array = sites_df[variable_list].values

        tol = 1e-6
        f_model_evals = f_poly_model.eval_np(sites_np_array)

        assert f_model_evals.shape[0] == 100
        assert f_model_evals.shape[1] == 1

        max_res = np.max(np.abs(f_model_evals.flatten() - evals[:, 0]))
        assert max_res < tol

        # Extract g

        g_poly_model = poly_model.get_response_models(["g"])

        # Test data

        assert g_poly_model.ndep == 1
        assert g_poly_model.outputs == ["g"]

        # Test call

        tol = 1e-6
        g_model_evals = g_poly_model.eval_np(sites_np_array)

        assert g_model_evals.shape[0] == 100
        assert g_model_evals.shape[1] == 1

        max_res = np.max(np.abs(g_model_evals.flatten() - evals[:, 1]))
        assert max_res < tol

    def test_to_dict(self, simple_multiple_response_data):
        # tests the to_dict method for the PolynomialModel class

        opt_prob, sites_df, evals, degree = simple_multiple_response_data

        # Initialitize

        poly_model = PolynomialModel.from_data(
            sites=sites_df, degree=degree, opt_problem=opt_prob
        )

        model_dict = poly_model.to_dict()

        assert isinstance(model_dict, dict)

        key_list = ["type", "problem", "version", "info"]

        for key in key_list:
            assert key in model_dict.keys()

    def test_from_dict(self, simple_bp_data_model):
        # tests the from_dict method for the PolynomialModel class

        # Initialization from coefficients

        opt_prob, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        # Initialize model

        poly_model = PolynomialModel(
            sites=sites_df,
            degree=degree,
            coefficients=coefs,
            degree_exponents=deg_exp,
            opt_problem=opt_prob,
        )

        model_dict = poly_model.to_dict()

        new_poly_model = PolynomialModel.from_dict(model_dict)

        # Verify sites match
        pd.testing.assert_frame_equal(new_poly_model._sites, poly_model._sites)
        # Verify name matches
        assert new_poly_model._name == poly_model._name
        # Verify model parameters match (now accessed via properties)
        assert new_poly_model.degree == poly_model.degree
        assert np.allclose(new_poly_model.coefs, poly_model.coefs)
        assert np.allclose(new_poly_model.deg_exp, poly_model.deg_exp)

        # check for TypeError when non-dictionary object is used with method
        with pytest.raises(TypeError):
            poly_model_wrong_type = PolynomialModel.from_dict(3)

        # check for key error when required keys are not in model_dict
        with pytest.raises(KeyError):
            poly_model_wrong_key = PolynomialModel.from_dict(
                {"dummy_key": "dummy_value"}
            )

        # check for key error when required keys are not in model_dict['info']
        model_dict["info"] = {"dummy_key": "dummy_value"}
        with pytest.raises(KeyError):
            poly_model_wrong_nested_key = PolynomialModel.from_dict(model_dict)

    def test_init_with_problem_dict(self, simple_bp_data_model):
        """Test that __init__ works when given a legacy problem dict instead of opt_problem."""
        opt_prob, sites_df, evals, degree, coefs, deg_exp = simple_bp_data_model

        legacy_problem = opt_problem_to_legacy(opt_prob)

        poly_model = PolynomialModel(
            sites=sites_df,
            degree=degree,
            coefficients=coefs,
            degree_exponents=deg_exp,
            problem=legacy_problem,
        )

        assert poly_model.nsites == 29
        assert poly_model.nind == 3
        assert poly_model.ndep == 1

    def test_no_problem_error(self, simple_bp_data_model):
        """Test that from_data raises ValueError when neither opt_problem nor problem is given."""
        _, sites_df, _, degree, _, _ = simple_bp_data_model

        with pytest.raises(ValueError, match="Either 'opt_problem' or 'interface' must be provided"):
            PolynomialModel.from_data(sites=sites_df, degree=degree)

    def test_fixed_variables_and_integers(
        self, cantilever_beam_model, cantilever_beam_fixed_variables_model
    ):

        cb_problem, site_df = cantilever_beam_model
        cb_problem = legacy_to_opt_problem(cb_problem)

        cbfv_problem, site_df_fv = cantilever_beam_fixed_variables_model
        cbfv_problem = legacy_to_opt_problem(cbfv_problem)

        poly_model = PolynomialModel.from_data(sites=site_df, degree=3, opt_problem=cb_problem)
        poly_fv_model = PolynomialModel.from_data(
            sites=site_df_fv, degree=3, opt_problem=cbfv_problem
        )

        # The models should be the same model
        assert np.all(poly_model.deg_exp == poly_fv_model.deg_exp)
        assert np.all(poly_model.coefs == poly_fv_model.coefs)

        # The second model should have fixed vars
        assert ["C0", "C1"] == [
            val[0] for val in poly_fv_model.constant_variables.values()
        ]

        # We hard-code a Factorial experiment so that we do not need Design Explorer.
        fact_sites = pd.DataFrame(
            {
                "b1": {
                    0: 2.0,
                    1: 5.333333333333334,
                    2: 8.666666666666668,
                    3: 12.0,
                    4: 2.0,
                    5: 5.333333333333334,
                    6: 8.666666666666668,
                    7: 12.0,
                    8: 2.0,
                    9: 5.333333333333334,
                    10: 8.666666666666668,
                    11: 12.0,
                    12: 2.0,
                    13: 5.333333333333334,
                    14: 8.666666666666668,
                    15: 12.0,
                    16: 2.0,
                    17: 5.333333333333334,
                    18: 8.666666666666668,
                    19: 12.0,
                    20: 2.0,
                    21: 5.333333333333334,
                    22: 8.666666666666668,
                    23: 12.0,
                    24: 2.0,
                    25: 5.333333333333334,
                    26: 8.666666666666668,
                    27: 12.0,
                    28: 2.0,
                    29: 5.333333333333334,
                    30: 8.666666666666668,
                    31: 12.0,
                    32: 2.0,
                    33: 5.333333333333334,
                    34: 8.666666666666668,
                    35: 12.0,
                    36: 2.0,
                    37: 5.333333333333334,
                    38: 8.666666666666668,
                    39: 12.0,
                    40: 2.0,
                    41: 5.333333333333334,
                    42: 8.666666666666668,
                    43: 12.0,
                    44: 2.0,
                    45: 5.333333333333334,
                    46: 8.666666666666668,
                    47: 12.0,
                    48: 2.0,
                    49: 5.333333333333334,
                    50: 8.666666666666668,
                    51: 12.0,
                    52: 2.0,
                    53: 5.333333333333334,
                    54: 8.666666666666668,
                    55: 12.0,
                    56: 2.0,
                    57: 5.333333333333334,
                    58: 8.666666666666668,
                    59: 12.0,
                    60: 2.0,
                    61: 5.333333333333334,
                    62: 8.666666666666668,
                    63: 12.0,
                    64: 2.0,
                    65: 5.333333333333334,
                    66: 8.666666666666668,
                    67: 12.0,
                    68: 2.0,
                    69: 5.333333333333334,
                    70: 8.666666666666668,
                    71: 12.0,
                    72: 2.0,
                    73: 5.333333333333334,
                    74: 8.666666666666668,
                    75: 12.0,
                    76: 2.0,
                    77: 5.333333333333334,
                    78: 8.666666666666668,
                    79: 12.0,
                    80: 2.0,
                    81: 5.333333333333334,
                    82: 8.666666666666668,
                    83: 12.0,
                    84: 2.0,
                    85: 5.333333333333334,
                    86: 8.666666666666668,
                    87: 12.0,
                    88: 2.0,
                    89: 5.333333333333334,
                    90: 8.666666666666668,
                    91: 12.0,
                    92: 2.0,
                    93: 5.333333333333334,
                    94: 8.666666666666668,
                    95: 12.0,
                    96: 2.0,
                    97: 5.333333333333334,
                    98: 8.666666666666668,
                    99: 12.0,
                    100: 2.0,
                    101: 5.333333333333334,
                    102: 8.666666666666668,
                    103: 12.0,
                    104: 2.0,
                    105: 5.333333333333334,
                    106: 8.666666666666668,
                    107: 12.0,
                    108: 2.0,
                    109: 5.333333333333334,
                    110: 8.666666666666668,
                    111: 12.0,
                    112: 2.0,
                    113: 5.333333333333334,
                    114: 8.666666666666668,
                    115: 12.0,
                    116: 2.0,
                    117: 5.333333333333334,
                    118: 8.666666666666668,
                    119: 12.0,
                    120: 2.0,
                    121: 5.333333333333334,
                    122: 8.666666666666668,
                    123: 12.0,
                    124: 2.0,
                    125: 5.333333333333334,
                    126: 8.666666666666668,
                    127: 12.0,
                },
                "b2": {
                    0: 0.1,
                    1: 0.1,
                    2: 0.1,
                    3: 0.1,
                    4: 0.7333333333333333,
                    5: 0.7333333333333333,
                    6: 0.7333333333333333,
                    7: 0.7333333333333333,
                    8: 1.3666666666666667,
                    9: 1.3666666666666667,
                    10: 1.3666666666666667,
                    11: 1.3666666666666667,
                    12: 2.0,
                    13: 2.0,
                    14: 2.0,
                    15: 2.0,
                    16: 0.1,
                    17: 0.1,
                    18: 0.1,
                    19: 0.1,
                    20: 0.7333333333333333,
                    21: 0.7333333333333333,
                    22: 0.7333333333333333,
                    23: 0.7333333333333333,
                    24: 1.3666666666666667,
                    25: 1.3666666666666667,
                    26: 1.3666666666666667,
                    27: 1.3666666666666667,
                    28: 2.0,
                    29: 2.0,
                    30: 2.0,
                    31: 2.0,
                    32: 0.1,
                    33: 0.1,
                    34: 0.1,
                    35: 0.1,
                    36: 0.7333333333333333,
                    37: 0.7333333333333333,
                    38: 0.7333333333333333,
                    39: 0.7333333333333333,
                    40: 1.3666666666666667,
                    41: 1.3666666666666667,
                    42: 1.3666666666666667,
                    43: 1.3666666666666667,
                    44: 2.0,
                    45: 2.0,
                    46: 2.0,
                    47: 2.0,
                    48: 0.1,
                    49: 0.1,
                    50: 0.1,
                    51: 0.1,
                    52: 0.7333333333333333,
                    53: 0.7333333333333333,
                    54: 0.7333333333333333,
                    55: 0.7333333333333333,
                    56: 1.3666666666666667,
                    57: 1.3666666666666667,
                    58: 1.3666666666666667,
                    59: 1.3666666666666667,
                    60: 2.0,
                    61: 2.0,
                    62: 2.0,
                    63: 2.0,
                    64: 0.1,
                    65: 0.1,
                    66: 0.1,
                    67: 0.1,
                    68: 0.7333333333333333,
                    69: 0.7333333333333333,
                    70: 0.7333333333333333,
                    71: 0.7333333333333333,
                    72: 1.3666666666666667,
                    73: 1.3666666666666667,
                    74: 1.3666666666666667,
                    75: 1.3666666666666667,
                    76: 2.0,
                    77: 2.0,
                    78: 2.0,
                    79: 2.0,
                    80: 0.1,
                    81: 0.1,
                    82: 0.1,
                    83: 0.1,
                    84: 0.7333333333333333,
                    85: 0.7333333333333333,
                    86: 0.7333333333333333,
                    87: 0.7333333333333333,
                    88: 1.3666666666666667,
                    89: 1.3666666666666667,
                    90: 1.3666666666666667,
                    91: 1.3666666666666667,
                    92: 2.0,
                    93: 2.0,
                    94: 2.0,
                    95: 2.0,
                    96: 0.1,
                    97: 0.1,
                    98: 0.1,
                    99: 0.1,
                    100: 0.7333333333333333,
                    101: 0.7333333333333333,
                    102: 0.7333333333333333,
                    103: 0.7333333333333333,
                    104: 1.3666666666666667,
                    105: 1.3666666666666667,
                    106: 1.3666666666666667,
                    107: 1.3666666666666667,
                    108: 2.0,
                    109: 2.0,
                    110: 2.0,
                    111: 2.0,
                    112: 0.1,
                    113: 0.1,
                    114: 0.1,
                    115: 0.1,
                    116: 0.7333333333333333,
                    117: 0.7333333333333333,
                    118: 0.7333333333333333,
                    119: 0.7333333333333333,
                    120: 1.3666666666666667,
                    121: 1.3666666666666667,
                    122: 1.3666666666666667,
                    123: 1.3666666666666667,
                    124: 2.0,
                    125: 2.0,
                    126: 2.0,
                    127: 2.0,
                },
                "H": {
                    0: 3.0,
                    1: 3.0,
                    2: 3.0,
                    3: 3.0,
                    4: 3.0,
                    5: 3.0,
                    6: 3.0,
                    7: 3.0,
                    8: 3.0,
                    9: 3.0,
                    10: 3.0,
                    11: 3.0,
                    12: 3.0,
                    13: 3.0,
                    14: 3.0,
                    15: 3.0,
                    16: 4.333333333333333,
                    17: 4.333333333333333,
                    18: 4.333333333333333,
                    19: 4.333333333333333,
                    20: 4.333333333333333,
                    21: 4.333333333333333,
                    22: 4.333333333333333,
                    23: 4.333333333333333,
                    24: 4.333333333333333,
                    25: 4.333333333333333,
                    26: 4.333333333333333,
                    27: 4.333333333333333,
                    28: 4.333333333333333,
                    29: 4.333333333333333,
                    30: 4.333333333333333,
                    31: 4.333333333333333,
                    32: 5.666666666666666,
                    33: 5.666666666666666,
                    34: 5.666666666666666,
                    35: 5.666666666666666,
                    36: 5.666666666666666,
                    37: 5.666666666666666,
                    38: 5.666666666666666,
                    39: 5.666666666666666,
                    40: 5.666666666666666,
                    41: 5.666666666666666,
                    42: 5.666666666666666,
                    43: 5.666666666666666,
                    44: 5.666666666666666,
                    45: 5.666666666666666,
                    46: 5.666666666666666,
                    47: 5.666666666666666,
                    48: 7.0,
                    49: 7.0,
                    50: 7.0,
                    51: 7.0,
                    52: 7.0,
                    53: 7.0,
                    54: 7.0,
                    55: 7.0,
                    56: 7.0,
                    57: 7.0,
                    58: 7.0,
                    59: 7.0,
                    60: 7.0,
                    61: 7.0,
                    62: 7.0,
                    63: 7.0,
                    64: 3.0,
                    65: 3.0,
                    66: 3.0,
                    67: 3.0,
                    68: 3.0,
                    69: 3.0,
                    70: 3.0,
                    71: 3.0,
                    72: 3.0,
                    73: 3.0,
                    74: 3.0,
                    75: 3.0,
                    76: 3.0,
                    77: 3.0,
                    78: 3.0,
                    79: 3.0,
                    80: 4.333333333333333,
                    81: 4.333333333333333,
                    82: 4.333333333333333,
                    83: 4.333333333333333,
                    84: 4.333333333333333,
                    85: 4.333333333333333,
                    86: 4.333333333333333,
                    87: 4.333333333333333,
                    88: 4.333333333333333,
                    89: 4.333333333333333,
                    90: 4.333333333333333,
                    91: 4.333333333333333,
                    92: 4.333333333333333,
                    93: 4.333333333333333,
                    94: 4.333333333333333,
                    95: 4.333333333333333,
                    96: 5.666666666666666,
                    97: 5.666666666666666,
                    98: 5.666666666666666,
                    99: 5.666666666666666,
                    100: 5.666666666666666,
                    101: 5.666666666666666,
                    102: 5.666666666666666,
                    103: 5.666666666666666,
                    104: 5.666666666666666,
                    105: 5.666666666666666,
                    106: 5.666666666666666,
                    107: 5.666666666666666,
                    108: 5.666666666666666,
                    109: 5.666666666666666,
                    110: 5.666666666666666,
                    111: 5.666666666666666,
                    112: 7.0,
                    113: 7.0,
                    114: 7.0,
                    115: 7.0,
                    116: 7.0,
                    117: 7.0,
                    118: 7.0,
                    119: 7.0,
                    120: 7.0,
                    121: 7.0,
                    122: 7.0,
                    123: 7.0,
                    124: 7.0,
                    125: 7.0,
                    126: 7.0,
                    127: 7.0,
                },
            }
        )
        # Create the new column "x0"
        fact_sites["x0"] = 1  # Set all values to 1 initially
        fact_sites.loc[64:, "x0"] = 8  # Set values to 8 for rows 64 and onwards

        fact_fv_sites = fact_sites.copy()
        fact_fv_sites["C0"] = 0.0
        fact_fv_sites["C1"] = 1.0

        # fact_exp = FactorialExperiment.from_problem(problem=cb_problem,levels=[4,4,4,2])
        # fact_sites=fact_exp()
        poly_model(fact_sites)
        # fact_fv_exp = FactorialExperiment.from_problem(problem=cbfv_problem,levels=[4,4,4,2, 1,1])
        # fact_fv_sites=fact_fv_exp()
        poly_fv_model(fact_fv_sites)

        # evaluate & check that processed data is the same
        assert np.all(
            poly_fv_model.remove_constants(fact_fv_sites)
            == poly_model.remove_constants(fact_sites)
        )
