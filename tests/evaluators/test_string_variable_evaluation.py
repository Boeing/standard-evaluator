import pandas as pd
from pandas.testing import assert_frame_equal

from standard_evaluator import EvaluatorInfo, FloatVariable, StringVariable
from standard_evaluator.evaluators import PyEvaluator
from standard_evaluator.utilities import apply_types_from_evaluator_info


def my_example(df: pd.DataFrame) -> None:
    """Modify ``df`` in place."""
    df["out"] = df["input"] + df["val"].astype(str)
    df["val_out"] = df["val"] * 4.0


def _make_interface() -> EvaluatorInfo:
    return EvaluatorInfo(
        name="string_example",
        inputs=[
            StringVariable(name="input"),
            FloatVariable(name="val"),
        ],
        outputs=[
            StringVariable(name="out"),
            FloatVariable(name="val_out"),
        ],
    )


def test_pyevaluator_matches_direct_call_with_string_columns():
    interface = _make_interface()

    # Call the function directly, then normalize the column dtypes the
    # same way the evaluator does (string columns -> pandas "string" extension
    # dtype, float columns -> float64), so the comparison is against a
    # like-for-like baseline.
    direct = pd.DataFrame({"input": ["one", "two"], "val": [3.4, 5.6]})
    my_example(direct)
    apply_types_from_evaluator_info(direct, interface)

    # Wrapped: run the same function through a PyEvaluator.
    wrapped = pd.DataFrame({"input": ["one", "two"], "val": [3.4, 5.6]})
    evaluator = PyEvaluator(my_example, interface=interface)
    evaluator(wrapped)

    assert_frame_equal(wrapped, direct)


def test_string_input_and_output_values_are_preserved():
    interface = _make_interface()
    wrapped = pd.DataFrame({"input": ["one", "two"], "val": [3.4, 5.6]})

    evaluator = PyEvaluator(my_example, interface=interface)
    evaluator(wrapped)

    # The string input column is untouched.
    assert list(wrapped["input"]) == ["one", "two"]
    # The string output column holds the concatenated values, not NaN.
    assert list(wrapped["out"]) == ["one3.4", "two5.6"]
    # The float output is computed as expected.
    assert list(wrapped["val_out"]) == [13.6, 22.4]

    # String columns use the pandas "string" extension dtype (matching the
    # convention already used for IntVariable -> "Int64").
    assert wrapped["input"].dtype == "string"
    assert wrapped["out"].dtype == "string"
