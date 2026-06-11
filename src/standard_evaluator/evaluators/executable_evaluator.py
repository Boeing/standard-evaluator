"""
Created Jun. 23, 2022

@author Mikel Woo
"""

from __future__ import annotations

import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import uuid
from collections.abc import Callable

import dask
import pandas as pd
from standard_evaluator import ArrayVariable

import standard_evaluator.utilities as utils
from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.problem import OptProblem

logger = logging.getLogger(__name__)


class ExecutableEvaluator(Evaluator):
    """Evaluator that can run an executable. Site values are passed to/from the
    executable using input/output files in whatever format is needed. Command
    line arguments may also be passed along with the input/output files. By
    default the input/output files will be formatted as a CSV file with headers.
    """

    def __init__(
        self,
        executable_name: pathlib.Path | str,
        *,
        interface: EvaluatorInfo | None = None,
        opt_problem: OptProblem | None = None,
        run_dir: pathlib.Path | str | None = None,
        pre_args: str | list[str] = "",
        post_args: str | list[str] = "",
        input_prepend: str = "",
        output_prepend: str = "",
        delimiter: str = ",",
        mult_evals: bool = False,
        run_parallel: bool = False,
        unroll_arrays: bool = True,
        input_writer: Callable[[pathlib.Path, list[str], pd.DataFrame], None] | None = None,
        output_reader: Callable[[pathlib.Path, list[str], str], pd.DataFrame] | None = None,
        name: str | None = None,
        comp_cost: float = 100,
        cache: str | None = None,
        cache_options: dict | None = None,
        logging: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the ExecutableEvaluator.

        Args:
            executable_name: Path to or name of the executable to run.
            interface: EvaluatorInfo defining inputs and outputs.
            opt_problem: OptProblem defining variables and responses.
            run_dir: Directory to run executable in.
            pre_args: Arguments placed before the input/output file paths.
            post_args: Arguments placed after the input/output file paths.
            input_prepend: String to prepend to the input file name.
            output_prepend: String to prepend to the output file name.
            delimiter: Single character used to delimit values in I/O files.
                Defaults to ','.
            mult_evals: If True, all sites are passed at once. Defaults to False.
            run_parallel: If True, sites are evaluated in parallel.
                Defaults to False.
            unroll_arrays: If True (default), array variables are unrolled
                to scalar columns before writing and rolled back after reading.
                If False, custom I/O functions receive raw rolled DataFrames.
            input_writer: Custom function to write the input file.
            output_reader: Custom function to read the output file.
            name: Name for identifying evaluator.
            comp_cost: Cost of running this evaluator. Defaults to 100.
            cache: Path to SQLite database for caching.
            cache_options: Options to modify caching behavior.
            logging: Enable logging of evaluated sites. Defaults to False.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        super().__init__(
            name=name,
            comp_cost=comp_cost,
            interface=interface,
            opt_problem=opt_problem,
            cache=cache,
            cache_options=cache_options,
            logging=logging,
            **kwargs,
        )

        # Make sure supplied directory and executable exist as well as valid
        self.run_dir, self.exe_path = self._format_paths(run_dir, executable_name)

        # Format pre and post arguments
        if isinstance(pre_args, str):
            self.pre_args = shlex.split(pre_args, posix=os.name == "posix")
        elif isinstance(pre_args, (list, tuple)):
            self.pre_args = list(pre_args)
        else:
            raise TypeError(
                "ExecutableEvaluator: pre_args must be a string, "
                "list, or tuple of strings! Received a "
                f"{type(pre_args).__name__} instead."
            )
        # Remove quotes since subprocess will add them
        for i, arg in enumerate(self.pre_args):
            self.pre_args[i] = arg.replace('"', "").replace("'", "")

        if isinstance(post_args, str):
            self.post_args = shlex.split(post_args, posix=os.name == "posix")
        elif isinstance(post_args, (list, tuple)):
            self.post_args = list(post_args)
        else:
            raise TypeError(
                "ExecutableEvaluator: post_args must be a string, "
                "list, or tuple of strings! Received a "
                f"{type(post_args).__name__} instead."
            )
        # Remove quotes since subprocess will add them
        for i, arg in enumerate(self.post_args):
            self.post_args[i] = arg.replace('"', "").replace("'", "")

        # Make sure input/output prepend is a string
        if isinstance(input_prepend, str):
            self.input_prepend = input_prepend
        else:
            raise TypeError(
                "ExecutableEvaluator: input_prepend must be a "
                f"string! Received a {type(input_prepend).__name__} instead."
            )
        if isinstance(output_prepend, str):
            self.output_prepend = output_prepend
        else:
            raise TypeError(
                "ExecutableEvaluator: output_prepend must be a "
                f"string! Received a {type(output_prepend).__name__} instead."
            )

        # Make sure delimeter is a one character string
        if isinstance(delimiter, str):
            if len(delimiter) == 1:
                self.delimiter = delimiter
            else:
                raise ValueError(
                    "ExecutableEvaluator: delimiter must be a "
                    f"one character long string! Was {len(delimiter)} instead."
                )
        else:
            raise TypeError(
                f"ExecutableEvaluator: delimiter must be a one "
                "character long string! Received a "
                f"{type(delimiter).__name__} instead."
            )

        # Make sure mult_eval is a bool
        if isinstance(mult_evals, bool):
            self.mult_eval = mult_evals
        else:
            raise TypeError(
                "ExecutableEvaluator: mult_evals must be a bool! "
                f"Received a {type(mult_evals).__name__} instead."
            )

        # Make sure run_parallel is a bool
        if isinstance(run_parallel, bool):
            self.run_parallel = run_parallel
        else:
            raise TypeError(
                "ExecutableEvaluator: run_parallel must be a "
                f"bool! Received a {type(run_parallel).__name__} instead."
            )

        self._unroll_arrays = unroll_arrays

        # Make sure custom IO functions are callables
        if callable(input_writer) or input_writer is None:
            self.input_writer = input_writer
        else:
            raise TypeError(
                "ExecutableEvaluator: input_writer must be a "
                f"callable! Received a {type(input_writer).__name__} instead."
            )

        if callable(output_reader) or output_reader is None:
            self.output_reader = output_reader
        else:
            raise TypeError(
                "ExecutableEvaluator: output_reader must be a "
                f"callable! Received a {type(output_reader).__name__} instead."
            )

        # Cache whether inputs/outputs contain array variables
        self._has_array_inputs = any(
            isinstance(var, ArrayVariable) for var in self._interface.inputs
        )
        self._has_array_outputs = any(
            isinstance(var, ArrayVariable) for var in self._interface.outputs
        )

        # Validate: unroll_arrays=False requires custom I/O for array sides
        if not unroll_arrays:
            if self._has_array_inputs and input_writer is None:
                raise ValueError(
                    "ExecutableEvaluator: unroll_arrays=False requires "
                    "input_writer when array variables are present "
                    "in the inputs."
                )
            if self._has_array_outputs and output_reader is None:
                raise ValueError(
                    "ExecutableEvaluator: unroll_arrays=False requires "
                    "output_reader when array variables are present "
                    "in the outputs."
                )

    def _evaluate(self, sites: pd.DataFrame) -> None:
        """Evaluate the given sites. Modifies the DataFrame in place.

        Args:
            sites: Sites to evaluate.
        """
        if self.mult_eval:
            # We simply assign the result back into the dataframe
            sites.loc[:, self.outputs] = self._run_exe(sites)[self.outputs].values
        else:
            # Need to do some work for single evals
            if self.run_parallel:
                runs = []
                for i in range(len(sites)):
                    runs.append(dask.delayed(self._run_exe)(sites.iloc[[i]]))
                results = dask.compute(*runs)
            else:
                results = []
                for i in range(len(sites)):
                    results.append(self._run_exe(sites.iloc[[i]]).loc[:, self.outputs])

            # Here we concatenate all the resulting dataframes before assigning
            sites.loc[:, self.outputs] = pd.concat(results)[self.outputs].values

    def _format_paths(
        self,
        run_dir: pathlib.Path | str | None,
        exe_name: pathlib.Path | str,
    ) -> tuple[pathlib.Path | None, pathlib.Path]:
        """Validate and resolve run directory and executable paths.

        Args:
            run_dir: Run directory path provided by user.
            exe_name: Path to executable provided by user.

        Returns:
            Tuple of (resolved run directory or None, resolved executable path).

        Raises:
            ValueError: If the directory or executable does not exist.
        """
        mod_run_dir: pathlib.Path | None = None

        # Make sure run directory exists
        if run_dir is not None:
            run_dir_path = pathlib.Path(run_dir)
            # Doesn't exist
            if not run_dir_path.exists():
                raise ValueError(
                    'ExecutableEvaluator: Running directory "'
                    f'{run_dir}" does not exist!'
                )

            # Is not a directory
            if not run_dir_path.is_dir():
                raise ValueError(
                    'ExecutableEvaluator: Running directory "'
                    f'{run_dir}" is not a directory!'
                )

            mod_run_dir = run_dir_path.resolve()

        # Make sure the provided exe_name is a runnable program
        # First check if an executable exists in two cases:
        #   1) exe_name is a relative path so check if it is in run_dir
        #   2) exe_name is an absolute path so check if it exists
        search_path = str(mod_run_dir) if mod_run_dir else None
        mod_exe_path = shutil.which(str(exe_name), os.F_OK | os.X_OK, search_path)
        if mod_exe_path is None:
            # Otherwise check if exe_name points to an executable somewhere on
            # the PATH variable.
            mod_exe_path = shutil.which(str(exe_name), os.F_OK | os.X_OK)
            if mod_exe_path is None:
                raise ValueError(
                    f'ExecutableEvaluator: "{exe_name}" is '
                    'either not an executable or was not found in the "'
                    f'{mod_run_dir}" directory or PATH!'
                )

        return (mod_run_dir, pathlib.Path(mod_exe_path).resolve())

    def _generate_file_names(self) -> tuple[pathlib.Path, pathlib.Path]:
        """Generate unique names for input/output files.

        Returns:
            Tuple of absolute paths to input and output files.
        """
        uid = uuid.uuid4().hex
        infile = f"de_exe_eval_{uid}.in"
        outfile = f"de_exe_eval_{uid}.out"

        if self.run_dir is None:
            return (pathlib.Path(infile).resolve(), pathlib.Path(outfile).resolve())
        return (self.run_dir / infile, self.run_dir / outfile)

    def _run_exe(self, sites: pd.DataFrame) -> pd.DataFrame:
        """Run the executable for the given sites.

        Args:
            sites: Sites to evaluate.

        Returns:
            pd.DataFrame: Response values for each site.
        """
        input_path, output_path = self._generate_file_names()

        if self.input_prepend.endswith(' '):
            inargs = [self.input_prepend.strip(), str(input_path)]
        else:
            inargs = [self.input_prepend + str(input_path)]

        if self.output_prepend.endswith(' '):
            outargs = [self.output_prepend.strip(), str(output_path)]
        else:
            outargs = [self.output_prepend + str(output_path)]

        cmd = (
            [self.exe_path]
            + self.pre_args
            + inargs
            + outargs
            + self.post_args
        )

        # Prepare input data — unroll array variables if needed
        if self._has_array_inputs and self._unroll_arrays:
            input_df = utils.unroll_data_frame_using_variables(
                sites[self.inputs],
                self._interface.inputs,
            )
            input_columns = list(input_df.columns)
        else:
            input_df = sites
            input_columns = self.inputs

        # Write input file
        if self.input_writer is None:
            input_df.to_csv(
                input_path,
                sep=self.delimiter,
                columns=input_columns,
                index=False,
            )
        else:
            self.input_writer(
                input_path,
                input_columns,
                input_df[input_columns],
            )

        # Run command and capture output
        result = subprocess.run(
            cmd,
            cwd=self.run_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        try:
            if self.output_reader is None:
                if result.returncode == 0 and output_path.exists():
                    ret = pd.read_csv(output_path, sep=self.delimiter)
                else:
                    if result.returncode != 0:
                        logger.warning(
                            "Executable returned non-zero exit code %d: %s",
                            result.returncode,
                            result.stdout.decode().strip(),
                        )
                    ret = pd.DataFrame(columns=self.outputs, index=sites.index)
            else:
                ret = self.output_reader(
                    output_path,
                    self.outputs,
                    result.stdout.decode(),
                )

            # Roll array output columns back if needed
            if self._has_array_outputs and self._unroll_arrays:
                try:
                    ret = utils.roll_data_frame_using_variables(ret, self._interface.outputs)
                except (TypeError, KeyError):
                    ret = pd.DataFrame(columns=self.outputs, index=sites.index)
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        return ret
