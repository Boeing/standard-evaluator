import numpy as np
import pandas as pd

from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class ConstrainedBetts(TestEvaluator):
    """Implement the linearly constrained Betts function defined as

    min f(x) = 0.01 * x1^2 + x2^2 - 100

    s.t.

    10 * x1 - x2 >= 10
    2 <= x1 <= 50
    -50 <= x2 <= 50

    This function has an optimal solution of

    x* = (2, 0), f(x*) = -99.96

    verified using the unopy optimizer (Uno v2.7.3, IPOPT preset).
    See ``experimenting/solve_constrained_betts_with_unopy.ipynb`` for the
    full verification workflow.

    The initial point x0 = (-1, -1) is infeasible (violates variable bounds
    and the linear constraint).
    """

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates a linearly constrained optimization problem based on the Betts function.

        This method initializes an optimization problem with two independent variables and two dependent variables.
        It sets up the variable names, bounds, default values, response names, and their respective bounds.
        The objective function and constraints are defined according to the specifications of the Betts function.

        The optimization problem is defined as follows:

        - Objective: Minimize the function
        $$
        f(x) = 0.01 \cdot x_1^2 + x_2^2 - 100
        $$

        - Subject to the constraints:
        $$
        10 \cdot x_1 - x_2 \geq 10
        $$
        $$
        2 \leq x_1 \leq 50
        $$
        $$
        -50 \leq x_2 \leq 50
        $$

        The optimal solution for this problem is:
        $$
        x^* = (2, 0)
        $$

        An infeasible initial point is defined as:
        $$
        x_0 = (-1, -1)
        $$

        Returns:
            OptProblem: An instance of the OptProblem class representing the defined optimization problem.
        """
        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=2, num_dependent=2, name="constrained_betts"
        )

        var_names = [f"x{items + 1}" for items in range(2)]
        var_bounds = ([2.0, 50.0], [-50.0, 50.0])
        # Define default values
        defaults = self._def_initial_guess()

        for var, local_default, local_bound, name in zip(
            new_prob.variables, defaults, var_bounds, var_names
        ):
            var.default = local_default
            var.bounds = local_bound
            var.name = name

        resp_names = ["f", "c1"]
        resp_bounds = ([-np.inf, np.inf], [10.0, np.inf])
        # Define the bounds on the responses
        for resp, local_bounds, name in zip(
            new_prob.responses, resp_bounds, resp_names
        ):
            resp.bounds = local_bounds
            resp.name = name

        # Define objectives and constraints
        new_prob.objectives = ["f"]
        new_prob.constraints = ["c1"]

        # Define the description of the problem
        new_prob.description = r"""$$
Implement the linearly constrained Betts function defined as

    min f(x) = 0.01 * x1^2 + x2^2 - 100

    s.t.

    10 * x1 - x2 >= 10
    2 <= x1 <= 50
    -50 <= x2 <= 50

    This function has an optimal solution of

    x* = (2, 0)

    and an infeasible initial point

    x0 = (-1, -1)
$$"""
        # Define the citation
        new_prob.cite = ""
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns:
            list: The initial guess values.
        """
        return [-1.0, -1.0]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the constrained Betts function.

        Args:
            sites: The dataframe containing input values, updated with responses.
        """
        sites["f"] = 0.01 * sites.x1**2 + sites.x2**2 - 100
        sites["c1"] = 10.0 * sites.x1 - sites.x2

    def _def_known_solution(self) -> pd.Series:
        """Provide the known optimal solution obtained via unopy.

        Returns:
            pd.Series: The variable values of the optimal solution.
        """
        return pd.Series(data={"x1": 2.0, "x2": 0.0})

    # ---------------------------------------------------------------------
    # Analytic derivatives
    # ---------------------------------------------------------------------
    def evaluate_analytic_gradient(self, sites: pd.DataFrame, opt_problem: OptProblem) -> np.ndarray:
        """Compute analytic gradients of the objective(s) at the provided sites.

        Args:
            sites: DataFrame of input sites (one row per site).
            opt_problem: OptProblem instance providing mapping information.

        Returns:
            np.ndarray: Gradient array with shape
                (num_sites, num_objs_total, num_flat_vars).
        """

        num_sites = len(sites)

        # Number of flattened variable elements (should be 2 here: x1, x2)
        n_flat_vars = opt_problem.num_flat_vars

        # Number of objectives (1)
        n_objs = opt_problem.num_objs_total

        # Allocate gradient array
        gradient = np.zeros((num_sites, n_objs, n_flat_vars), dtype=float)

        # Extract x1 and x2 as numpy arrays
        x1 = sites["x1"].to_numpy(dtype=float)
        x2 = sites["x2"].to_numpy(dtype=float)

        # Analytic derivatives for f = 0.01*x1^2 + x2^2 - 100
        # df/dx1 = 0.02 * x1
        # df/dx2 = 2.0 * x2
        # Place into gradient: objective index is 0, variable flat indices assumed order [x1, x2]
        gradient[:, 0, 0] = 0.02 * x1
        gradient[:, 0, 1] = 2.0 * x2

        return gradient

    def evaluate_analytic_jacobian(self, sites: pd.DataFrame, opt_problem: OptProblem) -> np.ndarray:
        """Compute analytic Jacobian of constraints at the provided sites.

        Args:
            sites: DataFrame of input sites (one row per site).
            opt_problem: OptProblem instance providing mapping information.

        Returns:
            np.ndarray: Jacobian array with shape
                (num_sites, n_cons_total, num_flat_vars).
        """
        
        num_sites = len(sites)
        n_flat_vars = opt_problem.num_flat_vars

        # Number of flattened constraint elements
        n_cons = int(opt_problem.res_map["constraint"].sum())

        # Allocate jacobian array
        jacobian = np.zeros((num_sites, n_cons, n_flat_vars), dtype=float)

        # Analytic derivatives for c1 = 10*x1 - x2
        # dc1/dx1 = 10, dc1/dx2 = -1
        # Constraint index in this problem should be 0 (only one constraint element)
        if n_cons > 0:
            jacobian[:, 0, 0] = 10.0
            jacobian[:, 0, 1] = -1.0

        return jacobian
