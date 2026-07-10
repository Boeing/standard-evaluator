"""A Python module to provide access to CantileveredBeamContinuous optimization test problems"""

# pylint: disable=W0223

import pandas as pd
from standard_evaluator.problem import OptProblem
import standard_evaluator as se
from standard_evaluator.evaluators.test_evaluator import TestEvaluator


class CantileveredBeamContinuous(TestEvaluator):
    """Cantilevered Beam Example Evaluator with only continuous variables

    This example is making the first variable (x0) double instead of integer, so it
    is a relaxation of the CantileveredBeam example.

    The problem has four design variables with types as follows:

        - x0 - double
        - x1 - double
        - x2 - double
        - x3 - double

    There are three responses:

        - volume
        - stress
        - deflection

    .. note:: The source for this instantiation of this evaluator is
        a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_
        called *"SHERPA - An Efficient and Robust Optimization/Search
        Algorithm"*.
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for the Cantilevered Beam example with continuous variables.

        This method initializes an optimization problem with four design variables and three responses.
        The first variable is defined as a double, representing a relaxation of the original CantileveredBeam example.

        Design Variables:
            - hc: Height of the cantilever beam (double)
            - b1: Width of the first section of the beam (double)
            - b2: Width of the second section of the beam (double)
            - H: Height of the beam (double)

        Responses:
            - deflection: The deflection of the beam (bounded between 0.0 and 0.1)
            - stress: The stress experienced by the beam (bounded between 0.0 and 5000.0)
            - volume: The volume of the beam (bounded between 0.0 and 1200.0)

        Objectives:
            - Minimize the volume of the beam.

        Constraints:
            - The volume, stress, and deflection of the beam must meet specified limits.

        Description:
            The problem is described in detail, including the source of the evaluator's instantiation, which is a white paper from
            Red Cedar Technology titled "SHERPA - An Efficient and Robust Optimization/Search Algorithm".

        Returns:
            OptProblem: An instance of the optimization problem configured with the specified variables, responses, objectives,
            constraints, and description.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=4, num_dependent=3, name="cantilevered_beam_continuous"
        )

        var_names = ["hc", "b1", "b2", "H"]
        # Define default values
        defaults = self._def_initial_guess()
        var_bounds = ([0.1, 1.0], [2.0, 12.0], [0.1, 2.0], [3.0, 7.0])
        for var, local_default, local_bound, name in zip(
            new_prob.variables, defaults, var_bounds, var_names
        ):
            var.default = local_default
            var.name = name
            var.bounds = local_bound

        scales = [100, 0.001, 0.01]
        resp_bounds = ([0.0, 0.1], [0.0, 5000.0], [0.0, 1200.0])
        resp_names = ["deflection", "stress", "volume"]
        # Define the bounds, scales on the responses
        for resp, local_scale, name, local_bounds in zip(
            new_prob.responses, scales, resp_names, resp_bounds
        ):
            resp.scale = local_scale
            resp.name = name
            resp.bounds = local_bounds

        # Define objectives and constraints
        new_prob.objectives = ["volume"]
        new_prob.constraints = ["volume", "stress", "deflection"]

        # Define the description of the problem
        new_prob.description = """\
Cantilevered Beam Example Evaluator with only continuous variables

This example is making the first variable (x0) double instead of integer, so it
is a relaxation of the CantileveredBeam example.

The problem has four design variables with types as follows:

- hc: Height of the cantilever beam (double)
- b1: Width of the first section of the beam (double)
- b2: Width of the second section of the beam (double)
- H: Height of the beam (double)

There are three responses:

- deflection: The deflection of the beam (bounded between 0.0 and 0.1)
- stress: The stress experienced by the beam (bounded between 0.0 and 5000.0)
- volume: The volume of the beam (bounded between 0.0 and 1200.0)

Objectives:

- Minimize the volume of the beam.

Constraints:

- The volume, stress, and deflection of the beam must meet specified limits.

.. note:: The source for this instantiation of this evaluator is
    a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_
    called *"SHERPA - An Efficient and Robust Optimization/Search
    Algorithm"*."""

        # Define the citation
        new_prob.cite = 'The source for this instantiation of this evaluator is a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_ called *"SHERPA - An Efficient and Robust Optimization/Search Algorithm"*.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns:
            list: The initial guess values.
        """
        return [0.55, 7.0, 1.05, 5.0]

    def _evaluate(self, sites: pd.DataFrame):
        """Call to the CantileveredBeamContinuous function.

        Args:
            sites: The dataframe containing input values, updated with responses.
        """
        length = 60.0
        tip_load = 1000.00
        elasticity = 1.0e7

        moment_of_inertia = (1.0 / 12.0) * sites.b2 * (sites.H - 2.0 * sites.hc) * (
            sites.H - 2.0 * sites.hc
        ) * (sites.H - 2.0 * sites.hc) + 2.0 * (
            (1.0 / 12.0) * sites.b1 * sites.hc * sites.hc * sites.hc
            + sites.b1 * sites.hc * (sites.H - sites.hc) * (sites.H - sites.hc) / 4.0
        )  # moment of inertia for an I beam cross-section

        # Set a filter to ensure only values for correct sites are calculated
        moment_filter = moment_of_inertia > 1.0e-10
        sites.loc[moment_filter, "volume"] = (
            (2.0 * sites.hc[moment_filter] * sites.b1[moment_filter])
            + (
                (sites.H[moment_filter] - 2 * sites.hc[moment_filter])
                * sites.b2[moment_filter]
            )
        ) * length
        sites.loc[moment_filter, "stress"] = (
            tip_load
            * length
            * sites.H[moment_filter]
            / (2.0 * moment_of_inertia[moment_filter])
        )
        sites.loc[moment_filter, "deflection"] = (
            tip_load * length * length * length / (3.0 * elasticity * moment_of_inertia)
        )

    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.

        Returns:
            list: The variable values of the optimal solution.
        """
        return [0.255325, 12.0, 0.1, 7.0]
