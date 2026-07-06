import numpy as np
import pandas as pd
import pytest
from standard_evaluator.evaluators import OpenMDAOEvaluator
import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid


@pytest.fixture
def om_paraboloid():

    # Build Model
    prob = om.Problem()
    prob.model.add_subsystem("parab", Paraboloid(), promotes_inputs=["x", "y"])

    # define the component whose output will be constraint
    prob.model.add_subsystem(
        "const", om.ExecComp("g = x + y"), promotes_inputs=["x", "y"]
    )
    prob.model.add_subsystem(
        "const2", om.ExecComp("h = x * 3.8 / y"), promotes_inputs=["x", "y"]
    )

    prob.model.set_input_defaults("x", 3.0)
    prob.model.set_input_defaults("y", -3.0)

    prob.model.add_design_var("x", lower=-50, upper=50)
    prob.model.add_design_var("y", lower=-50, upper=50)

    prob.model.add_objective("parab.f_xy")

    # Add the constraint
    prob.model.add_constraint("const.g", lower=0.0, upper=10.0)

    # Run the setup
    prob.setup()

    # Execute the model at the current point
    prob.run_model()
    return prob


@pytest.fixture
def om_beam_group():
    from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup

    E = 1.0
    L = 1.0
    b = 0.1
    volume = 0.01
    num_elements = 50
    prob = om.Problem(
        model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements)
    )
    # Run the setup
    prob.setup()
    # Execute the model at the current point
    prob.run_model()
    return prob


@pytest.fixture
def paraboloid_lhs_sites():
    return pd.DataFrame(
        data={
            "x": {
                0: 49.27325521002058,
                1: -1.5269040132219018,
                2: 18.75174422525386,
                3: -17.94473247856712,
                4: -39.0237299214535,
            },
            "y": {
                0: 2.9178822613331263,
                1: 27.8354600156416,
                2: 37.66883037651556,
                3: -19.10233634006206,
                4: -35.69621267255161,
            },
            "parab.f_xy": {
                0: 2329.8448000488233,
                1: 988.4872987446788,
                2: 2687.7651435695366,
                3: 1006.548696865128,
                4: 4160.643136847308,
            },
            "const.g": {
                0: 52.191137471353706,
                1: 26.3085560024197,
                2: 56.42057460176942,
                3: -37.04706881862918,
                4: -74.71994259400512,
            },
        }
    )


@pytest.fixture
def paraboloid_lhs_sites_scan():
    return pd.DataFrame(
        data={
            "x": {
                0: 49.27325521002058,
                1: -1.5269040132219018,
                2: 18.75174422525386,
                3: -17.94473247856712,
                4: -39.0237299214535,
            },
            "y": {
                0: 2.9178822613331263,
                1: 27.8354600156416,
                2: 37.66883037651556,
                3: -19.10233634006206,
                4: -35.69621267255161,
            },
            "parab.f_xy": {
                0: 2329.8448000488233,
                1: 988.4872987446788,
                2: 2687.7651435695366,
                3: 1006.548696865128,
                4: 4160.643136847308,
            },
            "const.g": {
                0: 52.191137471353706,
                1: 26.3085560024197,
                2: 56.42057460176942,
                3: -37.04706881862918,
                4: -74.71994259400512,
            },
            "const2.h": {
                0: 64.1692683352249,
                1: -0.2084476149121578,
                2: 1.8916602226223949,
                3: 3.569719546584714,
                4: 4.154227090190723,
            },
        }
    )


class ActuatorDisc(om.ExplicitComponent):
    """Simple wind turbine model based on actuator disc theory"""

    def setup(self):

        # Inputs
        self.add_input("a", 0.5, desc="Induced Velocity Factor")
        self.add_input("Area", 10.0, units="m**2", desc="Rotor disc area")
        self.add_input("rho", 1.225, units="kg/m**3", desc="air density")
        self.add_input(
            "Vu", 10.0, units="m/s", desc="Freestream air velocity, upstream of rotor"
        )

        # Outputs
        self.add_output("Vr", 0.0, units="m/s", desc="Air velocity at rotor exit plane")
        self.add_output(
            "Vd", 0.0, units="m/s", desc="Slipstream air velocity, downstream of rotor"
        )
        self.add_output("Ct", 0.0, desc="Thrust Coefficient")
        self.add_output("thrust", 0.0, units="N", desc="Thrust produced by the rotor")
        self.add_output("Cp", 0.0, desc="Power Coefficient")
        self.add_output("power", 0.0, units="W", desc="Power produced by the rotor")

        # Every output depends on `a`
        self.declare_partials(of="*", wrt="a", method="cs")

        # Other dependencies
        self.declare_partials(of="Vr", wrt=["Vu"], method="cs")
        self.declare_partials(
            of=["thrust", "power"], wrt=["Area", "rho", "Vu"], method="cs"
        )

    def compute(self, inputs, outputs):
        """Considering the entire rotor as a single disc that extracts
        velocity uniformly from the incoming flow and converts it to
        power."""

        a = inputs["a"]
        Vu = inputs["Vu"]

        qA = 0.5 * inputs["rho"] * inputs["Area"] * Vu**2

        outputs["Vd"] = Vd = Vu * (1 - 2 * a)
        outputs["Vr"] = 0.5 * (Vu + Vd)

        outputs["Ct"] = Ct = 4 * a * (1 - a)
        outputs["thrust"] = Ct * qA

        outputs["Cp"] = Cp = Ct * (1 - a)
        outputs["power"] = Cp * qA * Vu


@pytest.fixture
def om_actuator_disc():
    prob = om.Problem()
    prob.model.add_subsystem(
        "a_disk", ActuatorDisc(), promotes_inputs=["a", "Area", "rho", "Vu"]
    )

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var("a", lower=0.0, upper=2.0)

    # negative one so we maximize the objective
    prob.model.add_objective("a_disk.Cp", scaler=-1)

    prob.setup()

    prob.set_val("a", 0.5)
    prob.set_val("Area", 10.0, units="m**2")
    prob.set_val("rho", 1.225, units="kg/m**3")
    prob.set_val("Vu", 10.0, units="m/s")
    prob.final_setup()
    return prob


def test_paraboloid(om_paraboloid: om.Problem):
    my_evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=False)
    # make sure the name is as expected
    assert "OpenMDAOEvaluator" in my_evaluator.name
    # make sure the right problem is stored in the evaluator via opt_problem
    opt = my_evaluator.opt_problem
    assert [v.name for v in opt.variables] == ["x", "y"]
    assert [r.name for r in opt.responses] == ["parab.f_xy", "const.g"]
    assert opt.objectives == ["parab.f_xy"]
    assert opt.constraints == []
    with pytest.raises(
        ValueError,
        match="scan_model and use_defined_problem cannot both be False at the same time.",
    ):
        OpenMDAOEvaluator(om_paraboloid, scan_model=False, use_defined_problem=False)


def test_paraboloid_evaluator(
    om_paraboloid: om.Problem, paraboloid_lhs_sites: pd.DataFrame
):
    my_evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=False)
    # Get the data frame from the experiment
    exp_df = paraboloid_lhs_sites
    # Create the space for the response values
    exp_df[my_evaluator.outputs] = np.nan
    # Evaluate the sites
    my_evaluator(exp_df)
    pd.testing.assert_frame_equal(exp_df, paraboloid_lhs_sites)


def test_paraboloid_evaluator_scan(
    om_paraboloid: om.Problem, paraboloid_lhs_sites_scan: pd.DataFrame
):
    my_evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=True)
    # Get the data frame from the experiment
    exp_df = paraboloid_lhs_sites_scan
    # Create the space for the response values
    exp_df[my_evaluator.outputs] = np.nan
    # Evaluate the sites
    my_evaluator(exp_df)
    pd.testing.assert_frame_equal(exp_df, paraboloid_lhs_sites_scan)


def test_paraboloid_scan_defined_problem_false(
    om_paraboloid: om.Problem,
):
    my_evaluator = OpenMDAOEvaluator(
        om_paraboloid, scan_model=True, use_defined_problem=False
    )
    # make sure the right problem is stored in the evaluator via opt_problem
    opt = my_evaluator.opt_problem
    assert sorted([v.name for v in opt.variables]) == ["x", "y"]
    assert sorted([r.name for r in opt.responses]) == ["const.g", "const2.h", "parab.f_xy"]
    assert opt.objectives == []
    assert opt.constraints == []


def test_beam_group_evaluator(om_beam_group: om.Problem):
    from standard_evaluator.problem import ArrayVariable

    # The beam group model has array-shaped variables; it should no longer raise TypeError
    my_evaluator = OpenMDAOEvaluator(om_beam_group, scan_model=True)
    opt = my_evaluator.opt_problem
    # Verify the evaluator was created successfully (no TypeError raised)
    assert opt is not None
    # The scanned problem should contain ArrayVariable instances for array-shaped vars
    scanned = my_evaluator._scanned_problem
    h_scanned = [v for v in scanned.variables if v.name == "h"]
    assert len(h_scanned) == 1
    assert isinstance(h_scanned[0], ArrayVariable)


def test_actuator_disk_evaluator(om_actuator_disc: om.Problem):
    my_evaluator = OpenMDAOEvaluator(om_actuator_disc, scan_model=True)
    assert my_evaluator.inputs == ["a", "Area", "rho", "Vu"]


# ---------------------------------------------------------------------------
# Requirement 7.1, 7.3, 7.4 — Backward compatibility for scalar-only models
# ---------------------------------------------------------------------------


class TestScalarOnlyBackwardCompatibility:
    """Verify scalar-only models produce identical FloatVariable instances and evaluation output.

    Requirements: 7.1, 7.2, 7.3, 7.4
    """

    def test_scalar_model_produces_only_float_variables(self, om_paraboloid: om.Problem):
        """Scalar-only model scanning produces only FloatVariable instances (not ArrayVariable)."""
        from standard_evaluator.problem import FloatVariable, ArrayVariable

        evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=True)
        opt = evaluator.opt_problem

        for var in opt.variables:
            assert isinstance(var, FloatVariable), (
                f"Variable {var.name} should be FloatVariable"
            )
            assert not isinstance(var, ArrayVariable), (
                f"Variable {var.name} should not be ArrayVariable"
            )

        for resp in opt.responses:
            assert isinstance(resp, FloatVariable), (
                f"Response {resp.name} should be FloatVariable"
            )
            assert not isinstance(resp, ArrayVariable), (
                f"Response {resp.name} should not be ArrayVariable"
            )

    def test_scalar_model_variable_bounds_defaults_scaling(self, om_paraboloid: om.Problem):
        """Scalar-only model variables have correct bounds, defaults, and scaling."""
        evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=False)
        opt = evaluator.opt_problem

        # Design vars from the paraboloid fixture: x in [-50, 50], y in [-50, 50]
        var_dict = {v.name: v for v in opt.variables}
        assert "x" in var_dict
        assert "y" in var_dict

        x_var = var_dict["x"]
        assert x_var.bounds[0] == -50.0
        assert x_var.bounds[1] == 50.0

        y_var = var_dict["y"]
        assert y_var.bounds[0] == -50.0
        assert y_var.bounds[1] == 50.0

    def test_scalar_model_evaluation_produces_correct_output(
        self, om_paraboloid: om.Problem, paraboloid_lhs_sites: pd.DataFrame
    ):
        """Evaluation of scalar-only model produces identical DataFrame values."""
        evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=False)

        # Copy the input sites
        exp_df = paraboloid_lhs_sites.copy()
        # Set outputs to NaN to be filled by evaluation
        exp_df[evaluator.outputs] = np.nan
        evaluator(exp_df)

        # Compare against expected values
        pd.testing.assert_frame_equal(exp_df, paraboloid_lhs_sites)

    def test_no_new_mandatory_constructor_params(self, om_paraboloid: om.Problem):
        """OpenMDAOEvaluator still works with just the om_problem argument.

        Requirements: 7.4
        """
        # This should work with just the single positional argument
        evaluator = OpenMDAOEvaluator(om_paraboloid)
        assert evaluator is not None
        assert evaluator.opt_problem is not None

    def test_scanned_problem_scalar_variables_are_float(self, om_paraboloid: om.Problem):
        """When scanning a scalar-only model, _scanned_problem contains only FloatVariable."""
        from standard_evaluator.problem import FloatVariable, ArrayVariable

        evaluator = OpenMDAOEvaluator(om_paraboloid, scan_model=True)
        scanned = evaluator._scanned_problem

        for var in scanned.variables:
            assert isinstance(var, FloatVariable)
            assert not isinstance(var, ArrayVariable)

        for resp in scanned.responses:
            assert isinstance(resp, FloatVariable)
            assert not isinstance(resp, ArrayVariable)
