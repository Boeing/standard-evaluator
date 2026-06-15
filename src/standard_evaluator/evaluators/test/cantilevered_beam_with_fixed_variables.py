"""A Python module to provide access to CantileveredBeam fixed-variable test problems"""

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
from standard_evaluator.evaluators.test.cantilevered_beam import CantileveredBeam


class CantileveredBeamFixedVariable(CantileveredBeam):
    """Cantilevered Beam with Fixed Variables Example Evaluator

    This example initializes an optimization problem with six design variables,
    three responses, and specific objectives and constraints related to the
    structural analysis of a cantilevered beam. The design variables include
    both float and integer types, with defined bounds and default values.

    The problem has 6 design variables with types as follows:
    - b1: Width of the first section of the beam (double)
    - b2: Width of the second section of the beam (double)
    - H: Height of the beam (double)
    - x0: An integer variable
    - C0: A fixed float variable
    - C1: A fixed integer variable

    There are three responses:
    - deflection
    - stress
    - volume

    The objectives and constraints for the optimization problem are set to
    minimize the volume while ensuring that the stress and deflection
    remain within specified limits.

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
        """
        Creates an optimization problem for a cantilevered beam with fixed variables.

        This method extends the base optimization problem by adding two fixed design variables:
        - C0: a float variable fixed at 0.0
        - C1: an integer variable fixed at 1

        It sets the objectives and constraints related to the structural analysis of a cantilevered beam,
        focusing on minimizing volume while constraining stress and deflection.

        The problem includes six design variables:
            - b1: width of the first beam section (double)
            - b2: width of the second beam section (double)
            - H: height of the beam (double)
            - x0: integer variable representing thickness index h1
            - C0: fixed float variable (0.0)
            - C1: fixed integer variable (1)

        The responses measured are:
        - deflection
        - stress
        - volume

        The objectives and constraints for the optimization problem are set to
        minimize the volume while ensuring that the stress and deflection
        remain within specified limits.

        The problem is described in detail, including the relationships between
        the design variables and the physical parameters of the beam. The
        description also includes a citation for the source of the problem
        formulation.

        Returns:
            OptProblem: An instance of the optimization problem configured for
            the cantilevered beam example.

        Note:
            The source for this instantiation of this evaluator is a white paper
            from `Red Cedar Technology <http://www.redcedartech.com>`_ called
            *"SHERPA - An Efficient and Robust Optimization/Search Algorithm"*.
        """
        new_prob = super()._create_opt_problem()

        # Add the new variable "C0"  and "C1"
        new_prob.variables.append(
            FloatVariable(name="C0", default=0.0, bounds=[0.0, 0.0])
        )
        new_prob.variables.append(IntVariable(name="C1", default=1, bounds=[1, 1]))

        # Define objectives and constraints
        new_prob.objectives = ["volume"]
        new_prob.constraints = ["volume", "stress", "deflection"]

        # Define the description of the problem
        new_prob.description = r"""$$
Cantilevered Beam with Fixed Variables Example Evaluator

    This example initializes an optimization problem with six design variables, 
    three responses, and specific objectives and constraints related to the 
    structural analysis of a cantilevered beam. The design variables include 
    both float and integer types, with defined bounds and default values.

    The problem has 6 design variables with types as follows:
    - b1: Width of the first section of the beam (double)
    - b2: Width of the second section of the beam (double)
    - H: Height of the beam (double)
    - x0: An integer variable
    - C0: A fixed float variable
    - C1: A fixed integer variable

    There are three responses:
    - deflection
    - stress
    - volume

    The objectives and constraints for the optimization problem are set to 
    minimize the volume while ensuring that the stress and deflection 
    remain within specified limits.

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
        return [7.0, 1.05, 5.0, 4, 0.0, 1]

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the CantileveredBeam function.

        Args:
            sites: The dataframe containing input values, updated with responses.
        """
        super()._evaluate(sites)

    def _def_known_solution(self) -> list:
        """Provide the known optimal solution.

        Returns:
            list: The variable values of the optimal solution.
        """
        return [0.255325, 12.0, 0.1, 7.0, 0.0, 1]
