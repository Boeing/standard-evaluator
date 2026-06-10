from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from standard_evaluator import ArrayVariable, EvaluatorInfo, FloatVariable, OptProblem
from standard_evaluator.evaluators import ExecutableEvaluator


@pytest.fixture
def run_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "Executable Evaluator"


@pytest.fixture
def num_files(run_dir: pathlib.Path) -> int:
    return len(list(run_dir.iterdir()))


@pytest.fixture
def eval_info() -> EvaluatorInfo:
    inf = float("inf")
    return EvaluatorInfo(
        name="test_exe_eval",
        inputs=[
            FloatVariable(name="x1", bounds=(-2, 2)),
            FloatVariable(name="x2", bounds=(-2, 2)),
        ],
        outputs=[
            FloatVariable(name="obj"),
            FloatVariable(name="c1", bounds=(-inf, 0)),
            FloatVariable(name="c2", bounds=(-inf, 0)),
        ],
    )


@pytest.fixture
def pre_sites() -> pd.DataFrame:
    # Points to test
    return pd.DataFrame(
        data={
            "x1": [0, 1, 2e2, -3.499999999999, -(2**0.5)],
            "additional": ["a", "b", "c", "d", "e"],
            "x2": [0, -1, 6, -1.2, 0],
        }
    )


@pytest.fixture
def post_sites() -> pd.DataFrame:
    # Values to expect
    return pd.DataFrame(
        data={
            "x1": [0, 1, 2e2, -3.499999999999, -(2**0.5)],
            "additional": ["a", "b", "c", "d", "e"],
            "x2": [0, -1, 6, -1.2, 0],
            "obj": [0, 0, 206, -4.7, -(2**0.5)],
            "c1": [-2, 0, 40_034, 11.69, 0],
            "c2": [0, 1, -6, 1.2, 0],
        }
    )


# Modifications to responses for testing pre/post args
@pytest.fixture
def c1_plus_one() -> list[float]:
    return [-1, 3, 40_435, 5.69, 1 - 2**1.5]



# =============================================================================
# Construction Tests
# =============================================================================


def test_construct_with_evaluator_info(eval_info: EvaluatorInfo, run_dir: pathlib.Path) -> None:
    """Verify EvaluatorInfo construction produces correct inputs/outputs."""
    evaluator = ExecutableEvaluator("python", interface=eval_info, run_dir=run_dir)
    assert evaluator.inputs == ["x1", "x2"]
    assert evaluator.outputs == ["obj", "c1", "c2"]


def test_construct_with_opt_problem(run_dir: pathlib.Path) -> None:
    """Verify OptProblem construction produces correct inputs/outputs."""
    opt_prob = OptProblem(
        name="test",
        variables=[
            FloatVariable(name="x1", bounds=(-2, 2)),
            FloatVariable(name="x2", bounds=(-2, 2)),
        ],
        responses=[
            FloatVariable(name="obj"),
            FloatVariable(name="c1"),
            FloatVariable(name="c2"),
        ],
    )
    evaluator = ExecutableEvaluator("python", opt_problem=opt_prob, run_dir=run_dir)
    assert evaluator.inputs == ["x1", "x2"]
    assert evaluator.outputs == ["obj", "c1", "c2"]


def test_constructor_equivalence_interface_vs_opt_problem(run_dir: pathlib.Path) -> None:
    """EvaluatorInfo and equivalent OptProblem produce identical evaluators.
    """
    from standard_evaluator.converters import evaluator_info_to_opt_problem

    eval_info = EvaluatorInfo(
        name="equiv_test",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
            FloatVariable(name="s1", bounds=(-5, 5)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
            FloatVariable(name="total"),
        ],
    )
    opt_prob = evaluator_info_to_opt_problem(eval_info)

    eval_from_info = ExecutableEvaluator(
        "python", interface=eval_info, run_dir=run_dir
    )
    eval_from_prob = ExecutableEvaluator(
        "python", opt_problem=opt_prob, run_dir=run_dir
    )

    assert eval_from_info.inputs == eval_from_prob.inputs
    assert eval_from_info.outputs == eval_from_prob.outputs
    assert (
        [v.name for v in eval_from_info.interface.inputs]
        == [v.name for v in eval_from_prob.interface.inputs]
    )
    assert (
        [v.name for v in eval_from_info.interface.outputs]
        == [v.name for v in eval_from_prob.interface.outputs]
    )


# =============================================================================
# Initialization and Validation Tests
# =============================================================================


def test_format_paths(eval_info: EvaluatorInfo, run_dir: pathlib.Path) -> None:
    evaluator = ExecutableEvaluator("python", interface=eval_info)

    # ===============
    # |   run_dir   |
    # ===============
    # Doesn't exist
    with pytest.raises(ValueError, match="does not exist"):
        evaluator._format_paths("fake/folder", "python")

    # Not a directory
    with pytest.raises(ValueError, match="is not a directory"):
        evaluator._format_paths(__file__, "python")

    # Make sure returned directory is correct
    dir, _ = evaluator._format_paths(None, "python")
    assert dir is None

    dir, _ = evaluator._format_paths(".", "python")
    assert dir == pathlib.Path.cwd()

    # ================
    # |   exe_name   |
    # ================
    # Doesn't exist
    with pytest.raises(ValueError):
        evaluator._format_paths(None, "fake")

    # Relative path provided
    _, exe = evaluator._format_paths(".", os.path.relpath(sys.executable))
    assert exe == pathlib.Path(sys.executable).resolve()

    # Absolute path provided
    _, exe = evaluator._format_paths(".", sys.executable)
    assert exe == pathlib.Path(sys.executable).resolve()

    # Executable on PATH
    _, exe = evaluator._format_paths(run_dir, "python")
    # Need to check the PATH variable since sys.executable may be a
    # different version than the one on the PATH
    assert str(exe.parent) in os.environ["PATH"]


def test_init_checks(eval_info: EvaluatorInfo) -> None:
    # ==========================
    # |   Pre/post arguments   |
    # ==========================
    def check_args(args):
        expected_args = ["-arg1", "-arg2", "-arg3=5", "../path to/dir"]
        assert isinstance(args, list)
        for i, arg in enumerate(args):
            assert arg == expected_args[i]

    # Pre args
    # One string
    eval = ExecutableEvaluator(
        "python",
        interface=eval_info,
        pre_args='-arg1 -arg2 -arg3=5 "../path to/dir"',
    )
    check_args(eval.pre_args)
    # List of strings
    eval = ExecutableEvaluator(
        "python",
        interface=eval_info,
        pre_args=["-arg1", "-arg2", "-arg3=5", "'../path to/dir'"],
    )
    check_args(eval.pre_args)
    # Tuple of strings
    eval = ExecutableEvaluator(
        "python",
        interface=eval_info,
        pre_args=["-arg1", "-arg2", "-arg3=5", "'../path to/dir'"],
    )
    check_args(eval.pre_args)

    # Invalid type
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            pre_args={"arg1": None, "arg2": 2},
        )

    # Post args
    # One string
    eval = ExecutableEvaluator(
        "python",
        interface=eval_info,
        post_args='-arg1 -arg2 -arg3=5 "../path to/dir"',
    )
    check_args(eval.post_args)
    # List of strings
    eval = ExecutableEvaluator(
        "python",
        interface=eval_info,
        post_args=["-arg1", "-arg2", "-arg3=5", "'../path to/dir'"],
    )
    check_args(eval.post_args)
    # Tuple of strings
    eval = ExecutableEvaluator(
        "python",
        interface=eval_info,
        post_args=["-arg1", "-arg2", "-arg3=5", "'../path to/dir'"],
    )
    check_args(eval.post_args)

    # Invalid type
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            post_args={"arg1": None, "arg2": 2},
        )

    # ============================
    # |   Input/output prepend   |
    # ============================
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            input_prepend=26.5,
        )
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            output_prepend={1, 2, 3},
        )

    # =================
    # |   Delimiter   |
    # =================
    # Wrong type
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            delimiter=12,
        )

    # Wrong length
    with pytest.raises(ValueError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            delimiter="",
        )
    with pytest.raises(ValueError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            delimiter="sep",
        )

    # ==================
    # |   mult_evals   |
    # ==================
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            mult_evals="yes",
        )

    # ====================
    # |   run_parallel   |
    # ====================
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            run_parallel=[True],
        )

    # ====================================
    # |   Input Writer & Output Reader   |
    # ====================================
    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            input_writer=94.1,
        )

    with pytest.raises(TypeError):
        ExecutableEvaluator(
            "python",
            interface=eval_info,
            output_reader=False,
        )


def test_unroll_arrays_false_requires_custom_io() -> None:
    """Verify unroll_arrays=False raises without custom writer/reader."""
    array_inputs = EvaluatorInfo(
        name="array_in",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
        ],
        outputs=[
            FloatVariable(name="total"),
        ],
    )
    array_outputs = EvaluatorInfo(
        name="array_out",
        inputs=[
            FloatVariable(name="s1", bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
        ],
    )

    # Array inputs without writer
    with pytest.raises(ValueError, match="input_writer"):
        ExecutableEvaluator(
            "python",
            interface=array_inputs,
            unroll_arrays=False,
        )

    # Array outputs without reader
    with pytest.raises(ValueError, match="output_reader"):
        ExecutableEvaluator(
            "python",
            interface=array_outputs,
            unroll_arrays=False,
        )

    # Array inputs with writer but scalar outputs — should NOT raise
    ExecutableEvaluator(
        "python",
        interface=array_inputs,
        unroll_arrays=False,
        input_writer=lambda f, c, d: None,
    )


def test_gen_file_names(eval_info: EvaluatorInfo, run_dir: pathlib.Path) -> None:
    # No run dir, file relative to current working directory
    evaluator = ExecutableEvaluator("python", interface=eval_info)
    infile, outfile = evaluator._generate_file_names()

    assert infile.is_absolute()
    assert outfile.is_absolute()
    assert infile.parent == pathlib.Path.cwd()
    assert outfile.parent == pathlib.Path.cwd()

    # Run dir provided
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
    )
    infile, outfile = evaluator._generate_file_names()

    assert infile.is_absolute()
    assert outfile.is_absolute()
    assert infile.parent == run_dir.resolve()
    assert outfile.parent == run_dir.resolve()


# =============================================================================
# Scalar Evaluation Tests
# =============================================================================


def test_call(
    eval_info: EvaluatorInfo,
    run_dir: pathlib.Path,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="base_evaluator.py",
    )

    with pytest.raises(TypeError):
        evaluator(pre_sites.to_numpy())

    with pytest.raises(ValueError):
        evaluator(pre_sites.drop(columns="x1"))

    # Things function as expected
    sites = pre_sites.copy()
    # Response columns should not be in there already
    assert "obj" not in sites
    assert "c1" not in sites
    assert "c2" not in sites
    # Make sure nothing was returned
    assert evaluator(sites) is None
    # Make sure reponse values are in dataframe
    assert "obj" in sites
    assert "c1" in sites
    assert "c2" in sites
    # Make sure returned values are correct
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()

    # Evaluator doesn't break when input sites have random index values
    new_inds = [25, 16, 9, 4, 1]
    sites = pre_sites.copy()
    sites.rename(index={i: ind for i, ind in enumerate(new_inds)}, inplace=True)
    evaluator(sites)
    # Make sure returned values are correct
    for i, ind in enumerate(new_inds):
        assert sites.loc[ind, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[ind, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[ind, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[ind, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[ind, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
        assert sites.loc[ind, "additional"] == post_sites.loc[i, "additional"]


def test_run_exe(
    run_dir: pathlib.Path,
    num_files: int,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
    c1_plus_one: list[float],
) -> None:
    # Make sure points are evaluated
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="base_evaluator.py",
    )
    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Include pre args
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args=["base_evaluator.py", "-double"],
    )
    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(2 * post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Pre and post args
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args=["base_evaluator.py", "-plus_one"],
        post_args="-double",
    )
    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(2 * post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(c1_plus_one[i])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Input/output prepend with =
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="prepend_evaluator.py",
        input_prepend="-input=",
        output_prepend="-output=",
    )
    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Input/output prepend with space
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="prepend_evaluator.py",
        input_prepend="-input ",
        output_prepend="-output ",
    )
    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Output file not created so return NaN
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="fail_evaluator.py",
    )
    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    assert sites[["obj", "c1", "c2"]].isna().all(axis=None)
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))


def test_delimiter(
    run_dir: pathlib.Path,
    num_files: int,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="delim_evaluator.py",
        delimiter="~",
    )

    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))


def test_custom_read_write(
    run_dir: pathlib.Path,
    num_files: int,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    writer_used = False
    reader_used = False

    # Write variable names and values in rows separated by underscores
    def writer(fname: pathlib.Path, vars: list, vals: pd.DataFrame):
        nonlocal writer_used
        writer_used = True

        with open(fname, "w") as file:
            for i in range(len(vars)):
                val_str = [str(x) for x in vals[vars[i]].to_list()]
                file.write("_".join([vars[i]] + val_str) + "\n")

    # Read weird file output
    def reader(fname: pathlib.Path, resp_names: list, stdout: str):
        nonlocal reader_used
        reader_used = True

        # Read data from files
        offset_values = [[] for _ in range(len(resp_names))]
        with open(fname, "r") as file:
            # Values are reversed to flip them back
            lines = reversed(file.readlines())
            # Get response header
            resp_order = next(lines).split("sep")
            # Read in values
            for line in lines:
                for i, val in enumerate(line.strip().split("sep")):
                    offset_values[i].append(float(val))

        # Remove column offset
        values = [[] for _ in range(len(resp_names))]
        offset = len(offset_values[0])
        for i in range(len(resp_names)):
            values[i] = offset_values[(i + offset) % len(resp_names)]

        # Create DataFrame
        data = {name: vals for name, vals in zip(resp_order, values)}
        return pd.DataFrame(data=data)

    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="cust_io_evaluator.py",
        input_writer=writer,
        output_reader=reader,
    )

    sites = pre_sites.copy()
    evaluator(sites)
    # Make sure custom I/O functions were used
    assert writer_used
    assert reader_used
    # Make sure returned data is as expected
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))


def test_files_deleted(
    eval_info: EvaluatorInfo,
    run_dir: pathlib.Path,
    num_files: int,
    pre_sites: pd.DataFrame,
) -> None:
    def reader(fname, resp_names, stdout):
        # Raise error that shouldn't occur anywhere in code
        raise RecursionError

    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="base_evaluator.py",
        output_reader=reader,
    )

    # Make sure exception was raised and no files were left over
    with pytest.raises(RecursionError):
        evaluator(pre_sites)
    assert len(list(run_dir.iterdir())) == num_files


def test_mult_evals(
    run_dir: pathlib.Path,
    num_files: int,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    mult_run = False

    # Reads std output to make sure mult evals was used
    def reader(fname, resp_names, stdout):
        nonlocal mult_run
        mult_run = bool(stdout)

        return pd.read_csv(fname)

    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="mult_evaluator.py",
        mult_evals=True,
        output_reader=reader,
    )

    sites = pre_sites.copy()
    evaluator(sites)
    # Make sure multiple evaluations were performed
    assert mult_run
    # Make sure returned data is as expected
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Evaluator doesn't break when input sites have random index values
    new_inds = [25, 16, 9, 4, 1]
    sites = pre_sites.copy()
    sites.rename(index={i: ind for i, ind in enumerate(new_inds)}, inplace=True)
    evaluator(sites)
    # Make sure returned values are correct
    for i, ind in enumerate(new_inds):
        assert sites.loc[ind, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[ind, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[ind, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[ind, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[ind, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
        assert sites.loc[ind, "additional"] == post_sites.loc[i, "additional"]


def test_run_parallel(
    run_dir: pathlib.Path,
    num_files: int,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    start_end_times = []

    def reader(fname, resp_names, stdout):
        vals = stdout.strip().split("\n")

        start_vals = vals[0].strip().split(":")
        start_end_times.append((start_vals[0], float(start_vals[1])))
        end_vals = vals[1].split(":")
        start_end_times.append((end_vals[0], float(end_vals[1])))

        return pd.read_csv(fname)

    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="parallel_evaluator.py",
        run_parallel=True,
        output_reader=reader,
    )

    sites = pre_sites.copy()
    evaluator(sites)

    # Sort start/end times by time
    start_end_times.sort(key=lambda x: x[1])
    max_depth = 0
    active_jobs = []
    # Determine how many jobs were active at one time
    for id, t in start_end_times:
        if id in active_jobs:
            # Job must be ending so remove it from active job list
            active_jobs.remove(id)
        else:
            # Job is starting so add it to active job list
            active_jobs.append(id)
            # Update max job depth
            if len(active_jobs) > max_depth:
                max_depth = len(active_jobs)

    # Make sure evaluations happened in parallel
    assert max_depth > 1
    # Make sure returned data is as expected
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))

    # Evaluator doesn't break when input sites have random index values
    new_inds = [25, 16, 9, 4, 1]
    sites = pre_sites.copy()
    sites.rename(index={i: ind for i, ind in enumerate(new_inds)}, inplace=True)
    evaluator(sites)
    # Make sure returned values are correct
    for i, ind in enumerate(new_inds):
        assert sites.loc[ind, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[ind, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[ind, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[ind, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[ind, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
        assert sites.loc[ind, "additional"] == post_sites.loc[i, "additional"]


def test_logging(
    run_dir: pathlib.Path,
    num_files: int,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="delim_evaluator.py",
        delimiter="~",
        logging=True,
    )

    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "x1"] == pytest.approx(post_sites.loc[i, "x1"])
        assert sites.loc[i, "x2"] == pytest.approx(post_sites.loc[i, "x2"])
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
    assert (sites["additional"] == post_sites["additional"]).all()
    # Make sure no files were left over
    assert num_files == len(list(run_dir.iterdir()))
    # Call the evaluator a second time
    sites = pre_sites.copy()
    evaluator(sites)
    new_sites = evaluator.get_log()
    assert len(new_sites) == 2 * len(sites)


# =============================================================================
# Array Evaluation Tests
# =============================================================================


def test_array_variable_evaluation(run_dir: pathlib.Path) -> None:
    """Verify array variables unroll/roll correctly through a full evaluation."""
    eval_info = EvaluatorInfo(
        name="array_test",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
            FloatVariable(name="s1", bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
            FloatVariable(name="total"),
        ],
    )
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="array_evaluator.py",
    )

    sites = pd.DataFrame({
        "x": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        "s1": [10.0, 20.0],
    })
    evaluator(sites)

    # y[i] = x[i] * 2
    np.testing.assert_array_almost_equal(
        sites["y"].iloc[0], np.array([2.0, 4.0, 6.0])
    )
    np.testing.assert_array_almost_equal(
        sites["y"].iloc[1], np.array([8.0, 10.0, 12.0])
    )
    # total = sum(x) + s1
    assert sites["total"].iloc[0] == pytest.approx(16.0)
    assert sites["total"].iloc[1] == pytest.approx(35.0)


def test_array_variable_mult_evals(run_dir: pathlib.Path) -> None:
    """Verify array variables work with mult_evals=True (batch mode)."""
    eval_info = EvaluatorInfo(
        name="array_test",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
            FloatVariable(name="s1", bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
            FloatVariable(name="total"),
        ],
    )
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="array_evaluator.py",
        mult_evals=True,
    )

    sites = pd.DataFrame({
        "x": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        "s1": [10.0, 20.0],
    })
    evaluator(sites)

    np.testing.assert_array_almost_equal(
        sites["y"].iloc[0], np.array([2.0, 4.0, 6.0])
    )
    np.testing.assert_array_almost_equal(
        sites["y"].iloc[1], np.array([8.0, 10.0, 12.0])
    )
    assert sites["total"].iloc[0] == pytest.approx(16.0)
    assert sites["total"].iloc[1] == pytest.approx(35.0)


def test_array_variable_parallel(run_dir: pathlib.Path) -> None:
    """Verify array variables work with run_parallel=True."""
    eval_info = EvaluatorInfo(
        name="array_test",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
            FloatVariable(name="s1", bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
            FloatVariable(name="total"),
        ],
    )
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="array_evaluator.py",
        run_parallel=True,
    )

    sites = pd.DataFrame({
        "x": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        "s1": [10.0, 20.0],
    })
    evaluator(sites)

    np.testing.assert_array_almost_equal(
        sites["y"].iloc[0], np.array([2.0, 4.0, 6.0])
    )
    np.testing.assert_array_almost_equal(
        sites["y"].iloc[1], np.array([8.0, 10.0, 12.0])
    )
    assert sites["total"].iloc[0] == pytest.approx(16.0)
    assert sites["total"].iloc[1] == pytest.approx(35.0)


def test_array_variable_custom_io(run_dir: pathlib.Path) -> None:
    """Verify custom I/O functions receive unrolled data with array variables."""
    received_columns: list[str] = []

    def writer(fname: Path, columns: list[str], data: pd.DataFrame) -> None:
        received_columns.extend(columns)
        # Write as CSV (same as default behavior)
        data.to_csv(fname, columns=columns, index=False)

    def reader(fname: Path, outputs: list[str], stdout: str) -> pd.DataFrame:
        return pd.read_csv(fname)

    eval_info = EvaluatorInfo(
        name="array_test",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
            FloatVariable(name="s1", bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
            FloatVariable(name="total"),
        ],
    )
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="array_evaluator.py",
        input_writer=writer,
        output_reader=reader,
    )

    sites = pd.DataFrame({
        "x": [np.array([1.0, 2.0, 3.0])],
        "s1": [10.0],
    })
    evaluator(sites)

    # Writer should have received unrolled column names
    assert "x[0]" in received_columns
    assert "x[1]" in received_columns
    assert "x[2]" in received_columns
    assert "s1" in received_columns
    assert "x" not in received_columns  # Rolled name should NOT be passed

    # Results should still be correct (rolled back)
    np.testing.assert_array_almost_equal(
        sites["y"].iloc[0], np.array([2.0, 4.0, 6.0])
    )
    assert sites["total"].iloc[0] == pytest.approx(16.0)


def test_2d_array_variable_evaluation(run_dir: pathlib.Path) -> None:
    """Verify 2D array variables unroll/roll correctly (shape (2,3))."""
    eval_info = EvaluatorInfo(
        name="array_2d_test",
        inputs=[
            ArrayVariable(name="x", shape=(2, 3), bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(2, 3)),
        ],
    )
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="array_2d_evaluator.py",
    )

    input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sites = pd.DataFrame({"x": [input_array]})
    evaluator(sites)

    expected = input_array * 3.0
    np.testing.assert_array_almost_equal(sites["y"].iloc[0], expected)
    assert sites["y"].iloc[0].shape == (2, 3)


def test_array_variable_custom_io_unroll_false(run_dir: pathlib.Path) -> None:
    """Verify unroll_arrays=False passes rolled data to custom I/O."""
    received_columns: list[str] = []
    received_data: list[pd.DataFrame] = []

    def writer(
        fname: pathlib.Path,
        columns: list[str],
        data: pd.DataFrame,
    ) -> None:
        received_columns.extend(columns)
        received_data.append(data.copy())
        # Writer must handle serialization itself; for this test
        # we just write unrolled CSV manually from the rolled data
        rows = []
        for _, row in data.iterrows():
            x_arr = row["x"]
            vals = [str(x_arr[0]), str(x_arr[1]), str(x_arr[2]), str(row["s1"])]
            rows.append(",".join(vals))
        with open(fname, "w") as f:
            f.write("x[0],x[1],x[2],s1\n")
            for r in rows:
                f.write(r + "\n")

    def reader(
        fname: pathlib.Path,
        outputs: list[str],
        stdout: str,
    ) -> pd.DataFrame:
        # Reader returns rolled data directly
        raw = pd.read_csv(fname)
        result = pd.DataFrame({
            "y": [
                np.array([
                    raw["y[0]"].iloc[i],
                    raw["y[1]"].iloc[i],
                    raw["y[2]"].iloc[i],
                ])
                for i in range(len(raw))
            ],
            "total": raw["total"].values,
        })
        return result

    eval_info = EvaluatorInfo(
        name="array_test",
        inputs=[
            ArrayVariable(name="x", shape=(3,), bounds=(-10, 10)),
            FloatVariable(name="s1", bounds=(-10, 10)),
        ],
        outputs=[
            ArrayVariable(name="y", shape=(3,)),
            FloatVariable(name="total"),
        ],
    )
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="array_evaluator.py",
        input_writer=writer,
        output_reader=reader,
        unroll_arrays=False,
    )

    sites = pd.DataFrame({
        "x": [np.array([1.0, 2.0, 3.0])],
        "s1": [10.0],
    })
    evaluator(sites)

    # Writer should have received rolled column names (not unrolled)
    assert "x" in received_columns
    assert "s1" in received_columns
    assert "x[0]" not in received_columns

    # The data passed to writer should have numpy arrays in the x column
    assert hasattr(received_data[0]["x"].iloc[0], "shape")

    # Results should still be correct (reader returned rolled data)
    np.testing.assert_array_almost_equal(
        sites["y"].iloc[0], np.array([2.0, 4.0, 6.0])
    )
    assert sites["total"].iloc[0] == pytest.approx(16.0)


# =============================================================================
# Edge Case and Regression Tests
# =============================================================================


def test_nonzero_exit_code_returns_nan_and_logs(
    run_dir: pathlib.Path,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify non-zero exit code produces NaN outputs and emits a warning log."""
    evaluator = ExecutableEvaluator(
        "python",
        interface=eval_info,
        run_dir=run_dir,
        pre_args="fail_evaluator.py",
    )

    sites = pre_sites.copy()
    with caplog.at_level("WARNING", logger="standard_evaluator.evaluators.executable_evaluator"):
        evaluator(sites)

    assert sites[["obj", "c1", "c2"]].isna().all(axis=None)
    assert "non-zero exit code" in caplog.text


def test_construct_with_pathlib_path(
    run_dir: pathlib.Path,
    eval_info: EvaluatorInfo,
    pre_sites: pd.DataFrame,
    post_sites: pd.DataFrame,
) -> None:
    """Verify Path objects work directly for executable_name and run_dir."""
    exe_path = pathlib.Path(sys.executable)
    evaluator = ExecutableEvaluator(
        exe_path,
        interface=eval_info,
        run_dir=run_dir,
        pre_args="base_evaluator.py",
    )

    sites = pre_sites.copy()
    evaluator(sites)
    for i in range(len(sites)):
        assert sites.loc[i, "obj"] == pytest.approx(post_sites.loc[i, "obj"])
        assert sites.loc[i, "c1"] == pytest.approx(post_sites.loc[i, "c1"])
        assert sites.loc[i, "c2"] == pytest.approx(post_sites.loc[i, "c2"])
