from pathlib import Path
import os
from typing import Union, List, Dict, Optional, Tuple
from logging import getLogger
import re
import warnings

import shutil
import tempfile
import secrets

from dask import delayed, compute

from pydantic import BaseModel, field_validator, ValidationError, Field
import numpy as np
import platform

if platform.system() == "Windows":
    from win32com.client import CDispatch, DispatchEx
    import pythoncom
else:

    class CDispatch:
        """Placeholder for win32com.client.CDispatch on non-Windows platforms."""

        pass


import pandas as pd

from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator import Variable
from standard_evaluator.evaluators.excel_utilities import (
    SpreadsheetModel,
    range_boundaries,
    get_explicit_definitions,
)
from standard_evaluator.evaluators.excel_utilities import MacroDefinition

def delete_files(paths):
    for p in paths:
        try:
            p.unlink(missing_ok=True)   # Python 3.8+: no error if file already removed
        except PermissionError:
            # Windows: file may be in use; handle as appropriate
            raise
        except Exception as exc:
            # other errors (log/raise/ignore)
            print(f"Could not remove {p}: {exc}")

class ExcelEvaluator(Evaluator):
    """
    Evaluator class that interfaces with an Excel spreadsheet to perform evaluations.

    This class manages the lifecycle of an Excel application instance, opens a specified workbook,
    and maps inputs and outputs to specific Excel sheets and cells. It supports executing macros
    at different stages of the evaluation process and optionally shows the Excel UI.

    Attributes:
        _spreadsheet (SpreadsheetModel): The spreadsheet model containing the file path.
        _input_mapping (Dict[str, Dict[str, str]]): Mapping of input variable names to Excel sheet and cell locations.
        _output_mapping (Dict[str, Dict[str, str]]): Mapping of output variable names to Excel sheet and cell locations.
        _excel (CDispatch): COM object representing the Excel application instance.
        _workbook (CDispatch): COM object representing the opened Excel workbook.
        _macros (MacroDefinition): Macros to be executed at various stages of evaluation.
        _logger (logging.Logger): Logger instance for logging events and errors.
    """

    def __init__(
        self,
        spreadsheet: Union[str, SpreadsheetModel],
        name: Optional[str] = None,
        comp_cost: float = 100,
        cache: Optional[str] = None,
        cache_options: Optional[dict] = None,
        logging: bool = False,
        interface: Optional[EvaluatorInfo] = None,
        opt_problem: Optional[OptProblem] = None,
        show_excel: bool = False,
        macros: Optional[MacroDefinition] = None,
        num_threads: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize the ExcelEvaluator.

        Args:
            spreadsheet (Union[str, SpreadsheetModel]):
                Path to the Excel spreadsheet file or a SpreadsheetModel instance.
            name (str, optional):
                Name to assign to the evaluator. Defaults to the evaluator type name.
                If provided, this value will be appended to the default name.
            comp_cost (float, optional):
                Estimated computational cost of the evaluator. Defaults to 100.
            cache (str, optional):
                Path to an SQLite database for caching evaluated sites. Enables caching functionality.
                Defaults to None.
            cache_options (dict, optional):
                Dictionary of options to customize caching behavior. Defaults to None.
                Supported options include:

                - `table_name` (str): Name of the database table storing cached sites (default: 'Design_Data').
                - `delta` (float): Unscaled distance threshold to identify duplicate sites (default: 1e-8).
                - `auto_update` (bool): If True, reloads the database before each evaluation (default: False).
            logging (bool, optional):
                Enable logging of all evaluated sites. Defaults to False.
            interface (EvaluatorInfo, optional):
                Interface information describing inputs and outputs. Defaults to None.
            opt_problem (OptProblem, optional):
                Optimization problem context. Takes priority over interface definition. Defaults to None.
            show_excel (bool, optional):
                Whether to make the Excel application visible during evaluation. Defaults to False.
            macros (MacroDefinition, optional):
                Macros to execute at different stages of the evaluation process.
                Defaults to an empty MacroDefinition instance.
            **kwargs:
                Additional keyword arguments passed to the base Evaluator class.
        """
        self._excel = None
        self._workbook = None

        super().__init__(
            name,
            comp_cost,
            cache,
            cache_options,
            logging,
            interface,
            opt_problem,
            **kwargs,
        )

        # Ensure spreadsheet is a SpreadsheetModel instance
        if isinstance(spreadsheet, str):
            spreadsheet = SpreadsheetModel(spreadsheet=spreadsheet)
        self._spreadsheet = spreadsheet

        # Map inputs and outputs to explicit Excel cell definitions
        self._input_mapping = get_explicit_definitions(self.interface.inputs)
        self._output_mapping = get_explicit_definitions(self.interface.outputs)

        # Use provided macros or default to empty MacroDefinition
        self._macros = macros or MacroDefinition()

        # Initialize logger
        self._logger = getLogger()

        # Check if on Windows:
        self._on_windows = platform.system() == "Windows"

        self.num_threads = num_threads
        self._show_excel = show_excel

        if self.num_threads == 1:
            if self._on_windows:
                # Start Excel application instance
                self._excel = self._start_excel()
            else:
                warnings.warn(
                    "ExcelEvaluator can only be used on Windows systems. "
                    "Calling this evaluator will raise an error.",
                )

            # Load workbook immediately only if no pre- or post-macros are defined
            if (len(self._macros.pre_macros) == 0) and (len(self._macros.post_macros) == 0):
                if self._on_windows:
                    # Only load the spreadsheet on Windows
                    self._workbook = self._load_spreadsheet(
                        self._excel, self._spreadsheet.spreadsheet
                    )
            else:
                self._workbook = None
                raise NotImplementedError(
                    "Running pre- and post-macros has not been implemented yet."
                )
        else:
            self._excel = None
            self._workbook = None

    def __del__(self):
        """
        Destructor to clean up Excel resources.

        Closes the workbook without saving changes and quits the Excel application.
        Suppresses exceptions that may occur during interpreter shutdown or if resources are already released.
        """
        try:
            if self._workbook is not None:
                self._workbook.Close(SaveChanges=False)
            if self._excel is not None:
                self._excel.Quit()
        except Exception:
            # Suppress exceptions during cleanup
            pass

        self._logger.info(f"{self.name} is being deleted")

    def _start_excel(self) -> CDispatch:
        """
        Start a new Excel application instance.

        Returns:
            CDispatch: COM object representing the Excel application.
        """
        if self._on_windows:
            self._logger.info("Starting Excel calculations.")
            excel = DispatchEx("Excel.Application")
            excel.Visible = self._show_excel  # Make the instance visible if desired
        else:
            excel = DispatchEx()
        return excel

    def _load_spreadsheet(self, excel: CDispatch, name: str) -> CDispatch:
        """
        Open the Excel workbook.

        Args:
            excel (CDispatch): Excel application COM object.
            name (str): Path to the Excel spreadsheet file.

        Returns:
            CDispatch: COM object representing the opened workbook.
        """
        if self._on_windows:
            self._logger.info(f"Loading spreadsheet {name}.")
            abs_path = os.path.abspath(name)
            return excel.Workbooks.Open(abs_path)
        else:
            # Return the fake class
            return CDispatch()

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """
        Perform the evaluation by writing inputs to the Excel spreadsheet and reading outputs.

        Modifies the input DataFrame in place by updating output columns.

        Args:
            sites (pd.DataFrame): DataFrame containing input values. Outputs will be added/updated in this DataFrame.
        """
        if not self._on_windows:
            raise RuntimeError(
                "_evaluate: win32com is unavailable on this platform (non-Windows)."
            )

        if self.num_threads > 1:
            # Run in parallel
            # sites: your original DataFrame
            # num_threads: integer number of splits

            # 1) split indices into roughly equal parts
            split_indices = np.array_split(sites.index, self.num_threads)

            # 2) build actual DataFrame splits (make copies so compute modifies the fragment only)
            split_sites = [sites.loc[idx].copy() for idx in split_indices]
            self._logger.info(f"Using {self.num_threads} splits to run as many threads" )

            # 3) run in parallel if the Dataframe is not empty.
            tasks = [self._worker(df_part) for df_part in split_sites if not df_part.empty]  # list of pd.DataFrame parts
            results = compute(*tasks)  # runs in parallel

            # 4) reassemble preserving original index order
            processed = pd.concat(results, ignore_index=True)
            processed = processed.loc[sites.index]     # ensure the same index order as original

            # 5) replace original DataFrame with processed results
            sites.drop(sites.index, inplace=True)
            for col in processed.columns:
                sites[col] = processed[col]
        else:
            # Serial execution
            self._logger.info(f"Running serially evaluating {len(sites)} sites" )
            # When running in serial we don't need the results since the sites are updated in place.
            # When running in parallel we need to send the results back.
            _ = self._serial_execution(sites, self._excel, self._workbook)

    @delayed
    def _worker(self, sites: pd.DataFrame) -> pd.DataFrame:
        pythoncom.CoInitialize()  # STA is typical for Excel
        try:
            if self._on_windows:
                # Start Excel application instance
                excel = self._start_excel()
            else:
                raise RuntimeError(
                    "ExcelEvaluator can only be used on Windows systems. ",
                    "Calling this evaluator will raise an error.",
                )
            temp_file_name = self._spreadsheet.make_temp_copies(1)
            wrkbk = self._load_spreadsheet(
                    excel, temp_file_name[0]
                )
            sites = self._serial_execution(sites, excel=excel, wrkbk=wrkbk)
            wrkbk.Close(SaveChanges=False)
            excel.Quit()

            # release python references (del) if needed
        finally:
            if self._on_windows:
                delete_files(temp_file_name)
                pythoncom.CoUninitialize()
        return sites

    def _serial_execution(self, sites: pd.DataFrame, excel: CDispatch=None, wrkbk: CDispatch=None) -> pd.DataFrame:
        if len(sites) == 0:
            return sites
        if excel is None:
            if self._on_windows:
                # Start Excel application instance
                excel = self._start_excel()
            else:
                raise RuntimeError(
                    "ExcelEvaluator can only be used on Windows systems. ",
                    "Calling this evaluator will raise an error.",
                )
            wrkbk = self._load_spreadsheet(
                    excel, self._spreadsheet.spreadsheet
                )
        orig_calc = excel.Calculation
        orig_enable_events = excel.EnableEvents
        orig_screen_updating = excel.ScreenUpdating
        orig_display_alerts = excel.DisplayAlerts
        # Some Excel versions have CalculateBeforeSave; check and save if present
        has_calc_before_save = hasattr(excel, "CalculateBeforeSave")
        orig_calc_before_save = (
            excel.CalculateBeforeSave if has_calc_before_save else None
        )

        try:
            # Turn off things that cause recalcs or UI interruptions
            excel.Calculation = -4135
            excel.EnableEvents = False
            excel.ScreenUpdating = False
            excel.DisplayAlerts = False
            if has_calc_before_save:
                excel.CalculateBeforeSave = False

            for output_name in self._output_mapping.keys():
                if output_name not in sites.columns:
                    sites[output_name] = None

            for index, row in sites.iterrows():
                # Set the new input values in the spreadsheet
                for input_name, info in self._input_mapping.items():
                    wrkbk.Worksheets(info["sheet"]).Range(
                        info["cell"]
                    ).Value = row[input_name]

                # Force calculation now that updates are done.
                # Choose one depending on how thorough you must be:
                # excel.Calculate()              -> normal recalculation (synchronous)
                # excel.CalculateFull()          -> forces a full recalculation of all formulas
                # excel.CalculateFullRebuild()   -> full recalc and rebuilds dependency tree
                excel.Calculate()  # usually sufficient; use CalculateFull if needed
                # Now run all the macros listed in the main_macros
                for local_macro in self._macros.main_macros:
                    excel.Application.Run(local_macro)

                # Extract the output values and store them in the DataFrame
                for output_name, info in self._output_mapping.items():
                    value = (
                        wrkbk.Worksheets(info["sheet"])
                        .Range(info["cell"])
                        .Value
                    )
                    # If the output is supposed to be an array we need to convert it
                    # into a NumPy array and apply the right shape.
                    if "shape" in info:
                        if isinstance(value, tuple):
                            # We need to convert ranges into numpy arrays
                            value = np.array(value).reshape(info["shape"])
                        else:
                            raise ValueError(
                                f"Expected a tuple to be returned, instead got {type(value)}"
                            )
                    sites.at[index, output_name] = value
        finally:
            # Always restore Excel settings
            excel.Calculation = orig_calc
            excel.EnableEvents = orig_enable_events
            excel.ScreenUpdating = orig_screen_updating
            excel.DisplayAlerts = orig_display_alerts
            if has_calc_before_save:
                excel.CalculateBeforeSave = orig_calc_before_save
        self._logger.debug(f"Results from the Excel evaluation:\n{sites}")
        return sites
