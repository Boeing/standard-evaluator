"""A Python module to define the base class for BoreholeMultiFi optimization test problem classes"""

# pylint: disable=W0223

from abc import abstractmethod
import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test_evaluator import TestEvaluator
from standard_evaluator.problem import OptProblem
import standard_evaluator as se


class BoreholeMultiFiBase(TestEvaluator):
    """The Borehole function models water flow through a borehole. Its simplicity and quick evaluation makes it a
    commonly used function for testing a wide variety of methods in computer experiments.
    The response is water flow rate in  𝑚3/𝑦𝑟 .

    Input Domain

    Radius of borehole ( 𝑚 ) -  𝑟𝑤∈[0.05,0.15]
    Radius of influence ( 𝑚 ) -  𝑟∈[100,50000]
    Transmissivity of upper aquifier ( 𝑚2/𝑦𝑟 ) -  𝑇𝑢∈[63070,115600]
    Potentiometric head of upper aquifier ( 𝑚 ) -  𝐻𝑢∈[990,1110]
    Transmissivity of lower aquifier ( 𝑚2/𝑦𝑟 ) -  𝑇𝑙∈[63.1,116]
    Potentiometric head of lower aquifier ( 𝑚 ) -  𝐻𝑙∈[700,820]
    Length of borehole ( 𝑚 ) -  𝐿∈[1120,1680]
    Hydraulic conductivity of borehole ( 𝑚/𝑦𝑟 ) -  𝐾𝑤∈[9855,12045]"""

    def _create_opt_problem(self) -> OptProblem:
        """
        Creates an optimization problem for modeling water flow through a borehole.

        This method initializes a new optimization problem with specified variables,
        their bounds, default values, objectives, and constraints. The Borehole function
        is commonly used for testing various methods in computer experiments due to its
        simplicity and quick evaluation.

        The optimization problem includes the following variables:
        - Radius of borehole (m)
        - Radius of influence (m)
        - Transmissivity of upper aquifer (m²/yr)
        - Potentiometric head of upper aquifer (m)
        - Transmissivity of lower aquifer (m²/yr)
        - Potentiometric head of lower aquifer (m)
        - Length of borehole (m)
        - Hydraulic conductivity of borehole (m/yr)

        The response variable is the water flow rate measured in m³/yr.

        Returns:
            OptProblem: An instance of the OptProblem class configured with the borehole
            optimization problem, including variables, responses, objectives, constraints,
            and a description of the problem.
        """

        # Create the basic problem
        new_prob = se.utilities.create_opt_problem(
            num_independent=8, num_dependent=1, name="borehole_multi_fi_base"
        )
        # Define default values
        defaults = self._def_initial_guess()
        var_bounds = (
            [0.05, 0.15],
            [100.0, 50000.0],
            [63070.0, 115600.0],
            [990.0, 1110.0],
            [63.1, 116.0],
            [700.0, 820.0],
            [1120.0, 1680.0],
            [9855.0, 12045.0],
        )
        var_names = [
            "rad_borehole",
            "rad_influence",
            "trans_upper",
            "pot_upper",
            "trans_lower",
            "pot_lower",
            "len_borehole",
            "hyd_con_borehole",
        ]

        var_unit_list = ["m", "m", "m**2/yr", "m", "m**2/yr", "m", "m", "m/yr"]

        for var, local_bound, local_default, name, local_unit in zip(
            new_prob.variables, var_bounds, defaults, var_names, var_unit_list
        ):
            var.bounds = local_bound
            var.default = local_default
            var.name = name
            var.units = local_unit

        resp_unit = ["m**3/yr"]
        for resp, local_unit in zip(new_prob.responses, resp_unit):
            resp.name = "water_flow_rate"
            resp.units = local_unit

        # Define objectives and constraints
        new_prob.objectives = ["water_flow_rate"]
        new_prob.constraints = []

        # Define th description of the problem
        new_prob.description = """The Borehole function models water flow through a borehole. Its simplicity and quick evaluation makes it a
commonly used function for testing a wide variety of methods in computer experiments.
The response is water flow rate in  𝑚3/𝑦𝑟 .

Input Domain

| Radius of borehole ( 𝑚 ) -  𝑟𝑤∈[0.05,0.15]
| Radius of influence ( 𝑚 ) -  𝑟∈[100,50000]
| Transmissivity of upper aquifier ( 𝑚2/𝑦𝑟 ) -  𝑇𝑢∈[63070,115600]
| Potentiometric head of upper aquifier ( 𝑚 ) -  𝐻𝑢∈[990,1110]
| Transmissivity of lower aquifier ( 𝑚2/𝑦𝑟 ) -  𝑇𝑙∈[63.1,116]
| Potentiometric head of lower aquifier ( 𝑚 ) -  𝐻𝑙∈[700,820]
| Length of borehole ( 𝑚 ) -  𝐿∈[1120,1680]
| Hydraulic conductivity of borehole ( 𝑚/𝑦𝑟 ) -  𝐾𝑤∈[9855,12045]"""
        # Define the citation
        new_prob.cite = ""
        return new_prob

    def _def_initial_guess(self) -> list:
        """Provide an initial guess for an optimizer.

        Returns:
            list: The initial guess values.
        """
        return [0.12899, 679.88, 72089.56, 999.45, 77.896, 777.18, 1340.55, 10015.24]
