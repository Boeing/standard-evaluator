"""A Python module to provide access to BoreholeMultiFiHi optimization test problems"""

# pylint: disable=W0223

import pandas as pd
import numpy as np
from standard_evaluator.evaluators.test.borehole_multi_fi_base import (
    BoreholeMultiFiBase,
)


class BoreholeMultiFiHi(BoreholeMultiFiBase):
    """Hi-fidelity class for a borehole multifidelity problem

    The Borehole function models water flow through a borehole. Its simplicity and quick evaluation makes it a
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

    # ToDo: Implement gradient information pylint: disable=fixme
    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Call to the Borehole multifidelity high function.

        Args:
            sites: The dataframe containing input values, updated with responses.
        """
        num_part = 2.0 * np.pi * sites.trans_upper * (sites.pot_upper - sites.pot_lower)
        den1 = np.log(sites.rad_influence / sites.rad_borehole)
        den2 = (
            1.0
            + 2.0
            * sites.len_borehole
            * sites.trans_upper
            / (den1 * sites.rad_borehole**2 * sites.hyd_con_borehole)
            + sites.trans_upper / sites.trans_lower
        )
        sites["water_flow_rate"] = num_part / (den1 * den2)
