import os
import matlab.engine
import numpy as np
import pandas as pd

from standard_evaluator.evaluators.abstract_evaluator import Evaluator


class MatlabEvaluator(Evaluator):
    """
    Runs a MATLAB function to evaluate the true function values. The
    ``sites`` dataframe passed to ``_evaluate`` is expected to be **modified in
    place**. Thus the MATLAB function must place the results back into this
    dataframe via the Python interface.

    .. note::
        Your MATLAB function should return a dictionary-like object (struct) where keys
        correspond to response names and values are the computed results.

    .. note::
        When running in parallel mode, the entire dataframe is passed at once,
        and the MATLAB function should be able to handle vectorized inputs.
        When running serially, each row is passed individually. See the examples
        in the documentation foe reccomended best practices in writing MATLAB
        functions to be used with this evaluator.

    .. warning::
        If the MATLAB function raises an error during evaluation, the outputs
        for the affected sites will be set to NaN.
    """

    def __init__(
        self,
        func_path: str | os.PathLike,
        name: str,
        nargout: int = 1, # For de we typically just want 1 dict-like back
        comp_cost: float = 100,
        parallel: bool= False,
        **kwargs,
    ) -> None:
        """Initialize the MatlabEvaluator.

        Args:
            func_path: Path to the directory containing the MATLAB function.
            name: Name of the MATLAB function to call.
            nargout: Number of output arguments expected. Defaults to 1.
            comp_cost: Computational cost estimate. Defaults to 100.
            parallel: Whether to pass all sites at once. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        
        super().__init__(name=name, comp_cost=comp_cost, **kwargs)
        
        # TODO: This may belong somewhere else to ensure that we are starting and stopping the engine in a way that makes sense.
        # start matlab engine
        # NOTE: matlab does not need to be running to start engine
        self.eng = matlab.engine.start_matlab()
        self.func_path = func_path
        self.eng.addpath(self.func_path, nargout=0)
        self.nargout = nargout
        self.matlab_func = getattr(self.eng, name)
        self.parallel = parallel

        self.eng.cd(self.func_path)


    def _evaluate(self, sites: pd.DataFrame):
        """Call the MATLAB function to evaluate response values.

        Args:
            sites: DataFrame containing sites to evaluate. Modified in place
                with response values from MATLAB.
        """

        # Convert dataframe to dict, which maps to struct in matlab
        # NOTE: Using a custom converter to ensure that array values remain arrays for matlab
        # Running in parallel using matlab parllelization
        if self.parallel:

            matlab_input = {var: sites[var].to_numpy() for var in self.inputs}
            
            try:
                matlab_output = self.matlab_func(matlab_input, nargout=self.nargout)
            except:
                matlab_output = {key:np.nan for key in self.outputs}

            #Update site df
            for key in self.outputs:
                # Handle case where only 1 site was run
                if type(matlab_output[key]) is float:
                    sites[key] = matlab_output[key]
                # Handle case where multiple sites were run
                else:
                    sites[key] = matlab_output[key][0]

        # Running serially
        else:
            for i, row in sites.iterrows():
                
                # Filter the matlab input to only those fields which are inputs
                matlab_input = {var: np.array(row[var]) for var in self.inputs}

                # Get the return dict from matlab
                try:
                    matlab_output = self.matlab_func(matlab_input, nargout=self.nargout)
                except:
                    matlab_output = {key:np.nan for key in self.outputs}

                # Update site df
                for key in self.outputs:
                    sites.at[i, key] = matlab_output[key]
