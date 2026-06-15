"""A Python module to provide access to CantileveredBeam optimization test problems"""

# pylint: disable=W0223

import pandas as pd
import standard_evaluator as se
from standard_evaluator.problem import OptProblem
from standard_evaluator import (
    Variable,
    FloatVariable,
    IntVariable,
)
from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator.evaluators.test.cantilevered_beam_continuous import (
    CantileveredBeamContinuous,
)


class CantileveredBeam(CantileveredBeamContinuous):
    """Cantilevered Beam Example Evaluator

    The problem has four design variables with types as follows:

    - x0 - integer
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

    ::

        |<------ b1 ------>|
         __________________
        |                  |
        |                  | h1
        |_____        _____|
              |      |
              |      |
              |<-b2->|   H - 2 * h1
              |      |
         _____|      |_____
        |                  |
        |                  | h1
        |__________________|

    - x0 -> h1 in [0.1, 1.0] (accessed by indicies [1, 2, 3, 4, 5, 6, 7, 8] into
      a lookup table { .1, .25, .35, .5, .65, .75, .9, 1.0 } in this domain)
    - x1 -> b1 in [2.0, 12.0]
    - x2 -> b2 in [0.1, 2.0]
    - x3 -> H  in [3.0, 7.0]

    |

    - W = Load (hard coded to 1000 pounds)
    - E = Modulus of Elasticity (hard coded to 1.0e7 psi)
    - L = length (hard coded to 60 inches)
    - Z = I / z (section modulus of the cross-section of the beam)
    - z = H / 2 (Distance from neutral axis to extreme fiber (edge))
    - I = Moment of Inertia (of cross section about neutral axis)

    I = (1/12) * ((H - 2 * h1) * b2 ^ 3) + (1/12) * (b1 * h1 ^ 3) + (1/12) * (b1 * h1 ^ 3) + b1 * h1 * ((H - h1) / 2 ) ^ 2 + b1 * h1 * ((H - h1) / 2) ^ 2
    = (1/12) * ((H - 2 * h1) * b2 ^ 3) + 2 * (b1 * h1 ^ 3)) + (b1 * h1 *(H - h1) ^ 2) / 2

    V = (2 * b1 * h1 + b2 * (H - 2 * h1)) * L
    stress(x) = W *(L - x) / Z = W * (L - x) * z / I = W * (L - x) * H / (2 * I)
    max stress = stress(0) = W * L * H / (2 * I)
    deflection = W * L ^ 3 / (3 * E * I)

    """

    def _create_opt_problem(self) -> OptProblem:
        """Create and return an OptProblem describing the Cantilevered Beam optimization.

        This method extends the base implementation returned by super()._create_opt_problem() to customize the problem for the Cantilevered Beam example. It performs the following high-level operations:

        Obtains a copy of the base optimization problem via super()._create_opt_problem().
        Removes the original first design variable (index 0).
        Appends a new integer design variable "x0" with bounds [1, 8].
        Sets default values for all variables using self._def_initial_guess().
        Configures the problem objectives, constraints, description, and citation.
        Returns: OptProblem: The configured optimization problem object for the Cantilevered Beam example. The returned object will have: - variables: A list of variable objects (first three are double variables followed by the remaining integer x0). - objectives: ['volume'] - constraints: ['volume', 'stress', 'deflection'] - description: A multi-line description string (includes LaTeX). - cite: A citation string for the problem source.

        Notes: - The new integer variable x0 represents a discrete choice for the beam top/bottom thickness h1 and indexes a lookup table of values {0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0} mapped from indices 1..8. - The remaining design variables are: - x1 -> b1 in [2.0, 12.0] - x2 -> b2 in [0.1, 2.0] - x3 -> H in [3.0, 7.0] - Hard-coded physical parameters used in the problem description (not modified here) include: - W = 1000 (load, pounds) - E = 1.0e7 (modulus of elasticity, psi) - L = 60 (length, inches) - The problem description includes the following LaTeX block (kept here to document the evaluator):
            $$ Cantilevered Beam Example Evaluator

                The problem has four design variables with types as follows:

                - x0: integer variable representing h1 in [1,8]
                - x1: continuous variable representing b1 in [2.0, 12.0]
                - x2: continuous variable representing b2 in [0.1, 2.0]
                - x3: continuous variable representing H in [3.0, 7.0]

                There are three responses:

                - volume
                - stress
                - deflection
            $$
        - Caller should ensure that self._def_initial_guess() returns a list of
        default values that aligns with the final new_prob.variables order and
        length.
        """
        new_prob = super()._create_opt_problem()
        del new_prob.variables[0]
        # Add the new variable "x0"
        new_prob.variables.append(IntVariable(name="x0", bounds=[1, 8]))
        # Define default values
        defaults = self._def_initial_guess()

        # Update defaults for the variables
        for var, local_default in zip(new_prob.variables, defaults):
            var.default = local_default

        # Define objectives and constraints
        new_prob.objectives = ["volume"]
        new_prob.constraints = ["volume", "stress", "deflection"]

        # Define the description of the problem
        new_prob.description = r"""$$
Cantilevered Beam Example Evaluator

    The problem has four design variables with types as follows:

    - x0 - integer
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

    ::

        |<------ b1 ------>|
         __________________
        |                  |
        |                  | h1
        |_____        _____|
              |      |
              |      |
              |<-b2->|   H - 2 * h1
              |      |
         _____|      |_____
        |                  |
        |                  | h1
        |__________________|

    - x0 -> h1 in [0.1, 1.0] (accessed by indicies [1, 2, 3, 4, 5, 6, 7, 8] into
      a lookup table { .1, .25, .35, .5, .65, .75, .9, 1.0 } in this domain)
    - x1 -> b1 in [2.0, 12.0]
    - x2 -> b2 in [0.1, 2.0]
    - x3 -> H  in [3.0, 7.0]

    |

    - W = Load (hard coded to 1000 pounds)
    - E = Modulus of Elasticity (hard coded to 1.0e7 psi)
    - L = length (hard coded to 60 inches)
    - Z = I / z (section modulus of the cross-section of the beam)
    - z = H / 2 (Distance from neutral axis to extreme fiber (edge))
    - I = Moment of Inertia (of cross section about neutral axis)    

    I = (1/12) * ((H - 2 * h1) * b2 ^ 3) + (1/12) * (b1 * h1 ^ 3) + (1/12) * (b1 * h1 ^ 3) + b1 * h1 * ((H - h1) / 2 ) ^ 2 + b1 * h1 * ((H - h1) / 2) ^ 2
    = (1/12) * ((H - 2 * h1) * b2 ^ 3) + 2 * (b1 * h1 ^ 3)) + (b1 * h1 *(H - h1) ^ 2) / 2

    V = (2 * b1 * h1 + b2 * (H - 2 * h1)) * L
    stress(x) = W *(L - x) / Z = W * (L - x) * z / I = W * (L - x) * H / (2 * I)
    max stress = stress(0) = W * L * H / (2 * I)
    deflection = W * L ^ 3 / (3 * E * I)
    $$"""
        # Define the citation
        new_prob.cite = 'The source for this instantiation of this evaluator is a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_ called *"SHERPA - An Efficient and Robust Optimization/Search Algorithm"*.'
        return new_prob

    def _create_opt_problem(self) -> OptProblem:
        """Create and return an OptProblem describing the Cantilevered Beam optimization.

        This method extends the base implementation returned by super()._create_opt_problem() to customize the problem for the Cantilevered Beam example. It performs the following high-level operations:

        Obtains a copy of the base optimization problem via super()._create_opt_problem().
        Removes the original first design variable (index 0).
        Appends a new integer design variable "x0" with bounds [1, 8].
        Sets default values for all variables using self._def_initial_guess().
        Configures the problem objectives, constraints, description, and citation.
        Returns: OptProblem: The configured optimization problem object for the Cantilevered Beam example. The returned object will have: - variables: A list of variable objects (first three are double variables followed by the remaining integer x0). - objectives: ['volume'] - constraints: ['volume', 'stress', 'deflection'] - description: A multi-line description string (includes LaTeX). - cite: A citation string for the problem source.

        Notes: - The new integer variable x0 represents a discrete choice for the beam top/bottom thickness h1 and indexes a lookup table of values {0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0} mapped from indices 1..8. - The remaining design variables are: - x1 -> b1 in [2.0, 12.0] - x2 -> b2 in [0.1, 2.0] - x3 -> H in [3.0, 7.0] - Hard-coded physical parameters used in the problem description (not modified here) include: - W = 1000 (load, pounds) - E = 1.0e7 (modulus of elasticity, psi) - L = 60 (length, inches) - The problem description includes the following LaTeX block (kept here to document the evaluator):
            $$ Cantilevered Beam Example Evaluator

                The problem has four design variables with types as follows:

                - x0: integer variable representing h1 in [1,8]
                - x1: continuous variable representing b1 in [2.0, 12.0]
                - x2: continuous variable representing b2 in [0.1, 2.0]
                - x3: continuous variable representing H in [3.0, 7.0]

                There are three responses:

                - volume
                - stress
                - deflection
            $$
        - Caller should ensure that self._def_initial_guess() returns a list of
        default values that aligns with the final new_prob.variables order and
        length.
        """
        new_prob = super()._create_opt_problem()
        del new_prob.variables[0]
        # Add the new variable "x0"
        new_prob.variables.append(IntVariable(name="x0", bounds=[1, 8]))
        # Define default values
        defaults = self._def_initial_guess()

        # Update defaults for the variables
        for var, local_default in zip(new_prob.variables, defaults):
            var.default = local_default

        # Define objectives and constraints
        new_prob.objectives = ["volume"]
        new_prob.constraints = ["volume", "stress", "deflection"]

        # Define the description of the problem
        new_prob.description = r"""$$
Cantilevered Beam Example Evaluator

    The problem has four design variables with types as follows:

    - x0 - integer
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

    ::

        |<------ b1 ------>|
         __________________
        |                  |
        |                  | h1
        |_____        _____|
              |      |
              |      |
              |<-b2->|   H - 2 * h1
              |      |
         _____|      |_____
        |                  |
        |                  | h1
        |__________________|

    - x0 -> h1 in [0.1, 1.0] (accessed by indicies [1, 2, 3, 4, 5, 6, 7, 8] into
      a lookup table { .1, .25, .35, .5, .65, .75, .9, 1.0 } in this domain)
    - x1 -> b1 in [2.0, 12.0]
    - x2 -> b2 in [0.1, 2.0]
    - x3 -> H  in [3.0, 7.0]

    |

    - W = Load (hard coded to 1000 pounds)
    - E = Modulus of Elasticity (hard coded to 1.0e7 psi)
    - L = length (hard coded to 60 inches)
    - Z = I / z (section modulus of the cross-section of the beam)
    - z = H / 2 (Distance from neutral axis to extreme fiber (edge))
    - I = Moment of Inertia (of cross section about neutral axis)    

    I = (1/12) * ((H - 2 * h1) * b2 ^ 3) + (1/12) * (b1 * h1 ^ 3) + (1/12) * (b1 * h1 ^ 3) + b1 * h1 * ((H - h1) / 2 ) ^ 2 + b1 * h1 * ((H - h1) / 2) ^ 2
    = (1/12) * ((H - 2 * h1) * b2 ^ 3) + 2 * (b1 * h1 ^ 3)) + (b1 * h1 *(H - h1) ^ 2) / 2

    V = (2 * b1 * h1 + b2 * (H - 2 * h1)) * L
    stress(x) = W *(L - x) / Z = W * (L - x) * z / I = W * (L - x) * H / (2 * I)
    max stress = stress(0) = W * L * H / (2 * I)
    deflection = W * L ^ 3 / (3 * E * I)
    $$"""
        # Define the citation
        new_prob.cite = 'The source for this instantiation of this evaluator is a white paper from `Red Cedar Technology <http://www.redcedartech.com>`_ called *"SHERPA - An Efficient and Robust Optimization/Search Algorithm"*.'
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns:
            list: The initial guess values.
        """
        return [7, 1.05, 5.0, 4]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the CantileveredBeam function.

        Args:
            sites: The dataframe containing input values, updated with responses.
        """
        thickness = {1: 0.1, 2: 0.25, 3: 0.35, 4: 0.5, 5: 0.65, 6: 0.75, 7: 0.9, 8: 1.0}
        # Create the hc values by mapping the thickness index
        sites["hc"] = sites["x0"].map(thickness)
        super()._evaluate(sites)
        # Drop the extra hc column
        sites.drop(
            [
                "hc",
            ],
            axis="columns",
            inplace=True,
        )
