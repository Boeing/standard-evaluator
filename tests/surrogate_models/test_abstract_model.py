import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from numpy.typing import NDArray
from pydantic import BaseModel

from standard_evaluator.surrogate_models import SurrogateModel
import standard_evaluator as se
from standard_evaluator import (
    OptProblem,
    IntVariable,
    FloatVariable,
    CategoricalVariable,
)
from standard_evaluator.utilities import create_opt_problem


class Polynomial1DOptions(BaseModel):
    """Pydantic model defining options for Polynomial1D test model."""
    
    deg: int = 1  # Degree of the polynomial


class Polynomial1DModel(SurrogateModel):
    @classmethod
    def _define_options(cls):
        """Return the Pydantic model class that defines the options for this model."""
        return Polynomial1DOptions
        
    def __init__(
        self, sites: pd.DataFrame, 
        name: str = None, opt_problem: OptProblem = None,
        options: Polynomial1DOptions = None,
        **kwargs
    ) -> None:
        super().__init__(sites=sites, name=name, opt_problem=opt_problem, options=options, **kwargs)

        # Get degree from options
        self.deg = self.lookup_option_value("deg")
        
        sites_np = self.remove_constants(sites)
        # Get the number of non-constant variables
        nind = self.nind

        # Compute basis function values
        basis = self.calc_basis(sites_np[:, :nind])

        # Compute LHS matrix and RHS vector
        self._lhs = np.zeros((self.deg + 1, self.deg + 1))
        self._rhs = np.zeros((self.deg + 1, 1))
        for i in range(len(sites)):
            row = basis[[i], :]

            self._lhs += row.T * row
            self._rhs += self.sites.loc[i, "y"] * row.T

        # Invert LHS
        self._lhs = np.linalg.inv(self._lhs)

        # Solve for polynomial coefficients
        self.coefs = (self._lhs @ self._rhs).flatten()

    def calc_basis(self, sites: np.ndarray) -> np.ndarray:
        """Compute matrix of basis function values for each site.

        A = [1  x1  x1^2 ... x1^d]
            [1  x2  x2^2 ... x2^d]
            ...
            [1  xn  xn^2 ... xn^d]

        Where d is the degree of the polynomial and n is the number of sites.

        Parameters
        ----------
        sites : np.ndarray
            Sites to compute basis function values for.
            
        Returns
        -------
        np.ndarray
            Matrix of basis function values in increasing degree order.
        """
        A = np.zeros((len(sites), self.deg + 1))
        for i in range(self.deg + 1):
            A[:, i] = sites[:, 0] ** i

        return A

    def eval_np(self, sites: np.ndarray, names: list = None) -> np.ndarray:
        ret = np.zeros((len(sites), 1))
        for i in range(self.deg + 1):
            ret += self.coefs[i] * sites**i

        return ret

    def _def_update(
        self,
        sites_input: NDArray[np.float64],
        sites_output: NDArray[np.float64],
        new_sites_number: int,
    ):
        # Calculate basis function values for new sites
        basis = self.calc_basis(sites_input)
        # Update LHS and RHS to include new sites
        for i in range(len(sites_input)):
            row = basis[[i], :]

            # Update LHS using the Sherman-Morrison formula
            self._lhs -= (self._lhs @ row.T @ row @ self._lhs) / (
                1 + row @ self._lhs @ row.T
            )
            # Add new values to RHS
            self._rhs += sites_output[i, 0] * row.T

        # Compute new coefficients
        self.coefs = (self._lhs @ self._rhs).flatten()

        # Adjust site data
        self._update_sites(sites_input=sites_input, sites_output=sites_output)

    def _def_to_dict(self) -> dict:
        return {"deg": self.deg, "sites": self._sites}

    @classmethod
    def _def_from_dict(cls, model_info: dict) -> "Polynomial1DModel":
        # Need to make sure we pass the right information from the dictionary
        # to the model.
        options = Polynomial1DOptions(deg=model_info["info"]["deg"])

        # Support both new format (opt_problem) and legacy format (problem key)
        opt_problem = None
        if "opt_problem" in model_info and model_info["opt_problem"] is not None:
            opt_problem = OptProblem.model_validate(model_info["opt_problem"])
        elif "problem" in model_info and model_info["problem"] is not None:
            from standard_evaluator.surrogate_models.abstract_model import (
                _legacy_problem_dict_to_opt_problem,
            )
            opt_problem = _legacy_problem_dict_to_opt_problem(model_info["problem"])

        return cls(
            sites=model_info["info"]["sites"],
            options=options,
            opt_problem=opt_problem,
        )

@pytest.fixture
def problem():
    return {
        "variables": {"x": {"type": "float", "bounds": [-10, 10]}},
        "responses": {"y": {"type": "float"}},
        "objectives": [],
        "constraints": [],
    }


@pytest.fixture
def opt_problem() -> OptProblem:
    new_prob = se.utilities.create_opt_problem(num_independent=1, num_dependent=1)

    var_names = ["x"]
    # Define bound values
    var_bounds = ([-10, 10])
    for var, name in zip(new_prob.variables, var_names):
        var.bounds = var_bounds
        var.name = name

    resp_names = ["y"]
    for resp, name in zip(
        new_prob.responses, resp_names
    ):
        resp.name = name
    # Define objectives and constraints
    new_prob.objectives = []
    new_prob.constraints = []
    return new_prob

# The true function for the data below is:
#   y = 3x - 2
@pytest.fixture
def init_sites():
    # Init sites are skewed to lie above the true function
    return pd.DataFrame(
        data={
            "x": [-2.08, 3.07, 0.83, -5.37, 5.36, 1.39, -1.50, -9.92, 2.41, -6.34],
            "y": [-9.02, 2.95, 14.25, -8.92, 25.66, 6.67, -6.80, -26.44, 0.14, -26.48],
            "extra": ["test"] * 10,
        }
    )


@pytest.fixture
def add_sites():
    # Additional sites are slightly skewed below true function
    return pd.DataFrame(
        data={
            "x": [6.83, -2.12, -8.25, -4.61, -2.33, 1.31, -1.94, -1.15, -9.47, 6.79],
            "y": [
                15.11,
                -8.90,
                -31.51,
                -16.65,
                -12.87,
                3.69,
                -12.21,
                -8.58,
                -29.92,
                13.53,
            ],
            "extra": ["test"] * 10,
        }
    )


@pytest.fixture
def eval_sites():
    # Expected y-values at various points
    return pd.DataFrame(
        data={
            "x": [-10, -6.75, -3, -2, 0, 1, 3.4, 5, 8, 10],
            "y_init": [
                -30.158782,
                -20.037064,
                -8.358159,
                -5.243784,
                0.984965,
                4.099340,
                11.573839,
                16.556839,
                25.899963,
                32.128712,
            ],
            "y_update1": [
                -30.79787698,
                -20.7168456,
                -9.08488631,
                -5.9830305,
                0.22068112,
                3.32253693,
                10.76699087,
                15.72996017,
                25.0355276,
                31.23923921,
            ],
            "y_update2": [
                -31.27982154,
                -21.33709069,
                -9.86470894,
                -6.80540714,
                -0.68680354,
                2.37249826,
                9.71482258,
                14.60970546,
                23.78761086,
                29.90621446,
            ],
            "extra": ["test"] * 10,
        }
    )


def test_abstract():
    # Trying to instantiate abstract class
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SurrogateModel(None)

    # Derived class is missing eval_np
    class DummyModel1(SurrogateModel):
        def _def_update(self, x, y, append) -> None:
            pass

        def _def_to_dict(self) -> dict:
            pass

        @classmethod
        def _def_from_dict(cls, model_info: dict) -> "SurrogateModel":
            pass

    expected_message = ""

    # Check Python version
    if sys.version_info.minor < 12:
        expected_message = "Can't instantiate abstract class "
        "DummyModel1 with abstract methods? eval_np"
    else:
        expected_message = "Can't instantiate abstract class DummyModel1 without an implementation for abstract method 'eval_np'"

    with pytest.raises(TypeError, match=expected_message):
        DummyModel1()

    # Derived class is missing _def_update
    class DummyModel2(SurrogateModel):
        def eval_np(self, *args, **kwargs) -> None:
            pass

        def _def_to_dict(self) -> dict:
            pass

        @classmethod
        def _def_from_dict(cls, model_info: dict) -> "SurrogateModel":
            pass

    # Check Python version
    if sys.version_info.minor < 12:
        expected_message = "with abstract methods? _def_update"
    else:
        expected_message = "Can't instantiate abstract class DummyModel2 without an implementation for abstract method '_def_update'"

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        DummyModel2()

    # Derived class is missing _def_to_dict
    class DummyModel3(SurrogateModel):
        def eval_np(self, *args, **kwargs) -> None:
            pass

        def _def_update(self) -> None:
            pass

        @classmethod
        def _def_from_dict(cls, model_info: dict) -> "SurrogateModel":
            pass

    # Check Python version
    if sys.version_info.minor < 12:
        expected_message = "Can't instantiate abstract class "
        "DummyModel3 with abstract methods? _def_to_dict"
    else:
        expected_message = "Can't instantiate abstract class DummyModel3 without an implementation for abstract method '_def_to_dict'"

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        DummyModel3()

    # Derived class is missing _def_from_dict
    class DummyModel4(SurrogateModel):
        def eval_np(self, *args, **kwargs) -> None:
            pass

        def _def_update(self) -> None:
            pass

        def _def_to_dict(self) -> dict:
            pass

    # Check Python version
    if sys.version_info.minor < 12:
        expected_message = "Can't instantiate abstract class "
        "DummyModel4 with abstract methods? _def_from_dict",
    else:
        expected_message = "Can't instantiate abstract class DummyModel4 without an implementation for abstract method '_def_from_dict'"

    with pytest.raises(
        TypeError,
        match=expected_message,
    ):
        DummyModel4()

def test_init(init_sites: pd.DataFrame, opt_problem: OptProblem):
    # No name given
    model = Polynomial1DModel(sites=init_sites, opt_problem=opt_problem)

    # Check responses, objectives, constraints
    model_resp_names = [resp.name for resp in model.opt_problem.responses]
    opt_resp_names = [resp.name for resp in opt_problem.responses]
    assert model_resp_names == opt_resp_names
    assert model.opt_problem.objectives == opt_problem.objectives
    assert model.opt_problem.constraints == opt_problem.constraints

    # Check variables info
    model_resp = {resp.name: resp for resp in model.opt_problem.responses}
    opt_resp = {resp.name: resp for resp in opt_problem.responses}
    for resp_name, resp_obj in model_resp.items():
        opt_resp_obj = opt_resp[resp_name]
        assert resp_obj.class_type == opt_resp_obj.class_type
        assert resp_obj.bounds == opt_resp_obj.bounds
        assert resp_obj.default == opt_resp_obj.default    

    # Check variables info
    model_vars = {var.name: var for var in model.opt_problem.variables}
    opt_vars = {var.name: var for var in opt_problem.variables}
    for var_name, var_obj in model_vars.items():
        opt_var_obj = opt_vars[var_name]
        assert var_obj.class_type == opt_var_obj.class_type
        assert var_obj.bounds == opt_var_obj.bounds
        assert var_obj.default == opt_var_obj.default      

    assert model.opt_problem is opt_problem
    assert model.inputs == ["x"]
    assert model.outputs == ["y"]
    assert (model.sites[["x", "y"]] == init_sites[["x", "y"]]).all(axis=None)
    assert model.deg == 1
    assert model.nsites == len(init_sites)
    assert model.nind == 1
    assert model.ndep == 1
    assert model.name == "Polynomial1DModel"
    assert model.coefs == pytest.approx([0.98496523, 3.11437468])

    # Name given with custom options
    options = Polynomial1DOptions(deg=3)
    model = Polynomial1DModel(init_sites, name="TestName", opt_problem=opt_problem, options=options)
    assert model.deg == 3
    assert model.lookup_option_value("deg") == 3
    assert model.name == "Polynomial1DModel_TestName"

def test_update(opt_problem: OptProblem, init_sites: pd.DataFrame, add_sites: pd.DataFrame):
    model = Polynomial1DModel(opt_problem=opt_problem, sites=init_sites)

    # DataFrame or numpy array not passed
    with pytest.raises(TypeError):
        model.update([1, 2, 3])

    # DataFrame is missing columns
    with pytest.raises(
        ValueError,
        match="Input DataFrame does not contain " + "all variables and responses!",
    ):
        model.update(add_sites.drop(columns=["y"]))

    # numpy array has wrong number of columns
    # Too many
    with pytest.raises(ValueError, match="Input array has wrong number of columns!"):
        model.update(add_sites.to_numpy())
    # Too few
    with pytest.raises(ValueError, match="Input array has wrong number of columns!"):
        model.update(add_sites.drop(columns=["x", "extra"]).to_numpy())

    # Add some new sites to model
    add_sites1 = add_sites.iloc[:3].copy()
    expected_sites = pd.concat((init_sites, add_sites1), ignore_index=True)[["x", "y"]]
    model.update(add_sites1)
    assert model.coefs == pytest.approx([0.22068111904085663, 3.1018558095524327])
    assert len(model.sites) == len(expected_sites)
    assert (model.sites[["x", "y"]] == expected_sites[["x", "y"]]).all(axis=None)
    assert model.nsites == len(expected_sites)

    # Add more sites to model
    add_sites2 = add_sites.iloc[3:].copy()
    expected_sites = pd.concat((init_sites, add_sites), ignore_index=True)[["x", "y"]]
    model.update(add_sites2)
    assert model.coefs == pytest.approx([-0.68680354, 3.0593018])
    assert len(model.sites) == len(expected_sites)
    assert (model.sites[["x", "y"]] == expected_sites[["x", "y"]]).all(axis=None)
    assert model.nsites == len(expected_sites)

    
def test_call(
    init_sites: pd.DataFrame,
    add_sites: pd.DataFrame,
    eval_sites: pd.DataFrame,
    opt_problem: OptProblem,
):
    options = Polynomial1DOptions(deg=1)
    model = Polynomial1DModel(init_sites, name=None, opt_problem=opt_problem, options=options)

    sites = eval_sites[["x", "extra"]].copy()
    ret = model(sites)
    assert ret is None
    assert (sites.columns == ["x", "extra", "y"]).all()
    assert sites["y"].to_numpy() == pytest.approx(eval_sites["y_init"].to_numpy())

    model.update(add_sites.iloc[:3].copy())
    sites = eval_sites[["x", "extra"]].copy()
    model(sites)
    assert sites["y"].to_numpy() == pytest.approx(eval_sites["y_update1"].to_numpy())

    model.update(add_sites.iloc[3:].copy())
    sites = eval_sites[["x", "extra"]].copy()
    model(sites)
    assert sites["y"].to_numpy() == pytest.approx(eval_sites["y_update2"].to_numpy())


def test_to_from_dict(
    init_sites: pd.DataFrame,
    add_sites: pd.DataFrame,
    opt_problem: OptProblem
):
    # Create model with options
    options = Polynomial1DOptions(deg=1)
    model = Polynomial1DModel(sites=init_sites, opt_problem=opt_problem, options=options)

    # ===============
    # |   to_dict   |
    # ===============
    # Get model dictionary
    model_dict = model.to_dict()

    # Make sure dictionary is as expected
    expected_keys = {
        "type",
        "info",
        "opt_problem",
        "version",
        "name",
    }
    assert expected_keys == set(model_dict)
    assert model_dict["type"] == "Polynomial1DModel"
    assert (
        model_dict["version"]
        == version("standard_evaluator")
    )
    assert model_dict["name"] == "Polynomial1DModel"

    # Check opt_problem structure
    assert "variables" in model_dict["opt_problem"]
    assert "responses" in model_dict["opt_problem"]
    assert any(v["name"] == "x" for v in model_dict["opt_problem"]["variables"])
    assert any(r["name"] == "y" for r in model_dict["opt_problem"]["responses"])
    info_keys = {"deg", "sites"}
    model_info = model_dict["info"]
    assert info_keys == set(model_info)
    assert model_info["deg"] == 1
    assert (model_info["sites"] == init_sites).all(axis=None)

    # =================
    # |   from_dict   |
    # =================
    # Catch errors in from_dict
    # Dictionary not passed in
    with pytest.raises(TypeError):
        SurrogateModel.from_dict(["stuff", 53])

    # Dictionary missing keys
    with pytest.raises(KeyError):
        SurrogateModel.from_dict({"type": "", "info": {}, "version": "", "name": ""})

    # Type is not valid
    with pytest.raises(NameError):
        SurrogateModel.from_dict(
            {
                "type": "not a real model name",
                "info": {},
                "opt_problem": {},
                "version": "",
                "name": "",
            }
        )

    # Trying to instantiate wrong model type
    class DummyModel(SurrogateModel):
        def eval_np(self, sites, names=None):
            pass

        def _def_update(self, add_sites):
            pass

        def _def_to_dict(self):
            pass

        @classmethod
        def _def_from_dict(cls, model_info):
            pass

    with pytest.raises(ValueError):
        DummyModel.from_dict(model_dict)

    # Rebuild model
    # Using abstract class
    new_model = SurrogateModel.from_dict(model_dict)
    assert new_model is not model
    assert new_model.name == model_dict["name"]
    assert new_model.deg == model_dict["info"]["deg"]
    assert new_model.coefs == pytest.approx([0.98496523, 3.11437468], abs=1e-7)
    assert new_model._lhs[0] == pytest.approx([0.10717183, 0.00590274], abs=1e-7)
    assert new_model._lhs[1] == pytest.approx([0.00590274, 0.00485822], abs=1e-7)
    assert new_model._rhs[0] == pytest.approx([-27.99])
    assert new_model._rhs[1] == pytest.approx([675.0603])
    del new_model

    # Using class itself
    new_model = Polynomial1DModel.from_dict(model_dict)
    assert new_model is not model
    assert new_model.name == model_dict["name"]
    assert new_model.deg == model_dict["info"]["deg"]
    assert new_model.coefs == pytest.approx([0.98496523, 3.11437468], abs=1e-7)
    assert new_model._lhs[0] == pytest.approx([0.10717183, 0.00590274], abs=1e-7)
    assert new_model._lhs[1] == pytest.approx([0.00590274, 0.00485822], abs=1e-7)
    assert new_model._rhs[0] == pytest.approx([-27.99])
    assert new_model._rhs[1] == pytest.approx([675.0603])
    del new_model

    # Update model and rebuild
    model.update(add_sites.iloc[:3])
    model_dict = model.to_dict()

    new_model = SurrogateModel.from_dict(model_dict)
    assert new_model is not model
    assert new_model.name == model_dict["name"]
    assert new_model.deg == model_dict["info"]["deg"]
    assert new_model.coefs == pytest.approx([-0.36499445, 3.09401352], abs=1e-7)
    assert new_model._lhs[0] == pytest.approx([0.08146283, 0.00376143], abs=1e-7)
    assert new_model._lhs[1] == pytest.approx([0.00376143, 0.00311655], abs=1e-7)
    assert new_model._rhs[0] == pytest.approx([-53.29])
    assert new_model._rhs[1] == pytest.approx([1057.0871])
    del new_model

    # Update model again and rebuild
    model.update(add_sites.iloc[3:])
    model_dict = model.to_dict()

    new_model = SurrogateModel.from_dict(model_dict)
    assert new_model is not model
    assert new_model.name == model_dict["name"]
    assert new_model.deg == model_dict["info"]["deg"]
    assert new_model.coefs == pytest.approx([-1.74357662, 3.00584967], abs=1e-7)
    assert new_model._lhs[0] == pytest.approx([0.05388389, 0.0028674], abs=1e-7)
    assert new_model._lhs[1] == pytest.approx([0.0028674, 0.00211694], abs=1e-7)
    assert new_model._rhs[0] == pytest.approx([-116.3])
    assert new_model._rhs[1] == pytest.approx([1577.4301])


def test_options_functionality(init_sites: pd.DataFrame, opt_problem: OptProblem):
    """Test the options functionality for SurrogateModel"""
    
    # Test default options
    model = Polynomial1DModel(sites=init_sites, opt_problem=opt_problem)
    assert model.lookup_option_value("deg") == 1
    
    # Test required_options class method
    required_opts = Polynomial1DModel.required_options()
    assert required_opts.deg == 1
    
    # Test required_options_names
    option_names = Polynomial1DModel.required_options_names()
    assert option_names == {"deg"}
    
    # Test custom options
    custom_options = Polynomial1DOptions(deg=3)
    model_custom = Polynomial1DModel(sites=init_sites, opt_problem=opt_problem, options=custom_options)
    assert model_custom.lookup_option_value("deg") == 3
    assert model_custom.deg == 3


def test_options_validation(init_sites: pd.DataFrame, opt_problem: OptProblem):
    """Test that options validation works correctly"""
    
    # Test valid options
    valid_options = Polynomial1DOptions(deg=2)
    model = Polynomial1DModel(sites=init_sites, opt_problem=opt_problem, options=valid_options)
    assert model.lookup_option_value("deg") == 2
    
    # Test that invalid types raise validation errors
    with pytest.raises(Exception):  # Pydantic will raise ValidationError
        Polynomial1DOptions(deg="invalid")


def test_options_inheritance():
    """Test that the options system works with inheritance"""
    
    # Test that _define_options is properly implemented
    options_class = Polynomial1DModel._define_options()
    assert options_class == Polynomial1DOptions
    
    # Test that we can create an instance of the options
    options_instance = options_class()
    assert options_instance.deg == 1
    
    # Test that we can create with custom values
    custom_options = options_class(deg=5)
    assert custom_options.deg == 5
