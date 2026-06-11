import platform
import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError
import numpy as np
import numpy.testing as npt
import pandas as pd
import numdifftools as nd
from pandas.testing import assert_frame_equal

from standard_evaluator import FloatVariable, ArrayVariable, EvaluatorInfo
from standard_evaluator.utilities import (
    unroll_data_frame_using_variables,
    roll_data_frame_using_variables,
)

# Run with the following pytest command on a Windows system:
# pytest -v -p no:faulthandler .\tests\evaluators\test_excel.py

if platform.system() == "Windows":
    from standard_evaluator.evaluators import (
        ExcelEvaluator,
        SpreadsheetModel,
        MacroDefinition,
    )


@pytest.mark.skipif(platform.system() != "Windows", reason="Runs only on Windows")
class TestSpreadsheetModel:
    """
    Unit tests for the SpreadsheetModel Pydantic model.

    Tests cover validation of the spreadsheet file path, including:
    - Valid .xlsx file
    - Valid .xlsm file
    - Invalid file extension
    - Non-existent file path
    """

    def test_valid_xlsx_file(self):
        """
        Test that a valid .xlsx file path passes validation.

        Creates a temporary .xlsx file and verifies that the model accepts it.
        """
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            model = SpreadsheetModel(spreadsheet=tmp_path)
            assert model.spreadsheet == tmp_path
        finally:
            # Clean up the temporary file
            tmp_path.unlink()

    def test_valid_xlsm_file(self):
        """
        Test that a valid .xlsm file path passes validation.

        Creates a temporary .xlsm file and verifies that the model accepts it.
        """
        with tempfile.NamedTemporaryFile(suffix=".xlsm", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            model = SpreadsheetModel(spreadsheet=tmp_path)
            assert model.spreadsheet == tmp_path
        finally:
            # Clean up the temporary file
            tmp_path.unlink()

    def test_invalid_extension(self):
        """
        Test that a file with an invalid extension raises a ValidationError.

        Creates a temporary file with a .txt extension and verifies that validation fails.
        """
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with pytest.raises(ValidationError) as exc_info:
                SpreadsheetModel(spreadsheet=tmp_path)
            # Check that the error message contains the expected text
            assert "File must have a .xlsx or .xlsm extension" in str(exc_info.value)
        finally:
            # Clean up the temporary file
            tmp_path.unlink()

    def test_nonexistent_file(self):
        """
        Test that a non-existent file path raises a ValidationError.

        Uses a fake file path and verifies that validation fails due to missing file.
        """
        fake_path = Path("nonexistent_file.xlsx")
        with pytest.raises(ValidationError) as exc_info:
            SpreadsheetModel(spreadsheet=fake_path)
        # Check that the error message contains the expected text
        assert f"File does not exist: {fake_path}" in str(exc_info.value)


@pytest.mark.skipif(platform.system() != "Windows", reason="Runs only on Windows")
class TestExcelEvaluatorRealExcel:

    @pytest.fixture(scope="class")
    def excel_file_path(self):
        # Adjust filename to your actual test Excel file
        return Path(__file__).parent / "test_spreadsheet.xlsm"

    @pytest.fixture(scope="class")
    def evaluator(self, excel_file_path):
        # Define variables matching your Excel file's cell locations
        var1 = FloatVariable(name="x1", options={"ExcelLocation": "Sheet1!A1"})
        var2 = FloatVariable(name="x2", options={"ExcelLocation": "Sheet1!B2"})
        output1 = FloatVariable(name="y1", options={"ExcelLocation": "Sheet1!C3"})

        interface = EvaluatorInfo(
            name="excelInfo", inputs=[var1, var2], outputs=[output1]
        )

        # Instantiate ExcelEvaluator with the real Excel file
        evaluator = ExcelEvaluator(
            spreadsheet=str(excel_file_path),
            interface=interface,
            show_excel=False,  # Set True if you want to see Excel open
        )
        yield evaluator
        # Cleanup explicitly (optional, __del__ should handle it)
        del evaluator

    def test_evaluate_single_row(self, evaluator):
        # Create a DataFrame with one row of inputs
        df = pd.DataFrame({"x1": [5.0], "x2": [10.0], "y1": [None]})

        # Call the evaluator (which calls _evaluate internally)
        evaluator(df)

        # After evaluation, output column 'y1' should be updated from Excel
        assert "y1" in df.columns
        # Check that output is a float (Excel returned a value)
        assert isinstance(df.loc[0, "y1"], (float, int))

    def test_evaluate_multiple_rows(self, evaluator):
        # Create a DataFrame with multiple rows of inputs
        df = pd.DataFrame(
            {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y1": [None, None, None]}
        )

        evaluator(df)

        # Check outputs for all rows
        assert all(isinstance(val, (float, int)) for val in df["y1"])

    @pytest.fixture(scope="class")
    def evaluator_macros(self, excel_file_path):
        # Define variables matching your Excel file's cell locations
        var1 = FloatVariable(name="x1", options={"ExcelLocation": "Sheet1!A1"})
        var2 = FloatVariable(name="x2", options={"ExcelLocation": "Sheet1!B2"})
        output1 = FloatVariable(name="y1", options={"ExcelLocation": "Sheet1!C3"})

        interface = EvaluatorInfo(
            name="excelInfo", inputs=[var1, var2], outputs=[output1]
        )
        macros = MacroDefinition(main_macros=["ThisWorkbook.updating"])

        # Instantiate ExcelEvaluator with the real Excel file
        evaluator = ExcelEvaluator(
            spreadsheet=str(excel_file_path),
            interface=interface,
            show_excel=False,  # Set True if you want to see Excel open
            macros=macros,
        )
        yield evaluator
        # Cleanup explicitly (optional, __del__ should handle it)
        del evaluator

    def test_evaluate_macro_single_row(self, evaluator_macros):
        # Create a DataFrame with one row of inputs
        df = pd.DataFrame({"x1": [5.0], "x2": [10.0], "y1": [None]})

        # Call the evaluator (which calls _evaluate internally)
        evaluator_macros(df)

        # After evaluation, output column 'y1' should be updated from Excel
        assert "y1" in df.columns
        # Check that output is a float (Excel returned a value)
        assert isinstance(df.loc[0, "y1"], (float, int))

    @pytest.fixture(scope="class")
    def excel_file_path_range(self):
        # Adjust filename to your actual test Excel file
        return Path(__file__).parent / "Range_example.xlsm"

    @pytest.fixture(scope="class")
    def evaluator_range(self, excel_file_path_range):
        # Define variables matching your Excel file's cell locations
        var1 = FloatVariable(name="x1", options={"ExcelLocation": "Sheet1!A1"})
        var2 = FloatVariable(name="x2", options={"ExcelLocation": "Sheet1!B2"})
        output1 = FloatVariable(name="y1", options={"ExcelLocation": "Sheet1!C3"})
        range_1d_horizontal = ArrayVariable(
            name="range_1d_horizontal",
            shape=(5, 1),
            options={"ExcelLocation": "Range!A4:A8"},
            default=np.array([1.2, 0.03243, 210.0, 20.0, 0.0]).reshape((5, 1)),
        )
        range_1d_vertical = ArrayVariable(
            name="range_1d_vertical",
            shape=(4,),
            options={"ExcelLocation": "Range!A12:D12"},
        )
        range_2d = ArrayVariable(
            name="range_2d", shape=(3, 4), options={"ExcelLocation": "Range!A17:D19"}
        )

        sum1 = FloatVariable(name="sum1", options={"ExcelLocation": "Range!A9"})
        average1 = FloatVariable(
            name="average1", options={"ExcelLocation": "Range!E12"}
        )
        sum_column = ArrayVariable(
            name="sum_column", shape=(3, 1), options={"ExcelLocation": "Range!E17:E19"}
        )
        average_row = ArrayVariable(
            name="average_row", shape=(4,), options={"ExcelLocation": "Range!A20:D20"}
        )
        outer_product = ArrayVariable(
            name="outer_product",
            shape=(4, 4),
            options={"ExcelLocation": "Range!H10:K13"},
        )

        interface = EvaluatorInfo(
            name="excelInfo",
            inputs=[var1, var2, range_1d_horizontal, range_1d_vertical, range_2d],
            outputs=[output1, sum1, average1, sum_column, average_row, outer_product],
        )

        # Instantiate ExcelEvaluator with the real Excel file
        evaluator = ExcelEvaluator(
            spreadsheet=str(excel_file_path_range),
            interface=interface,
            show_excel=False,  # Set True if you want to see Excel open
        )
        yield evaluator
        # Cleanup explicitly (optional, __del__ should handle it)
        del evaluator

    def test_evaluate_single_row_range(self, evaluator_range):
        # Create a DataFrame with one row of inputs by using the initial guess
        guess = evaluator_range.initial_guess()
        guess.loc[0, "x1"] = 2134.423
        guess.at[0, "range_1d_horizontal"] = np.array(
            [1.2, 0.03243, 210.0, 20.0, 0.0]
        ).reshape((5, 1))
        guess.at[0, "range_1d_vertical"] = np.array([31.2, 33.243, 211.0, 23.0])
        guess.at[0, "range_2d"][1, 2] = 4.0
        guess.at[0, "range_2d"][0, 3] = 1.0

        # Call the evaluator (which calls _evaluate internally)
        evaluator_range(guess)

        assert guess.at[0, "sum1"] == 231.23243
        assert guess.at[0, "sum1"] == np.sum(guess.at[0, "range_1d_horizontal"])
        assert guess.at[0, "average1"] == 74.61075
        assert guess.at[0, "average1"] == np.average(guess.at[0, "range_1d_vertical"])

        npt.assert_allclose(guess.at[0, "sum_column"], np.array([[1.0], [4.0], [0.0]]))
        npt.assert_allclose(
            guess.at[0, "sum_column"],
            np.sum(guess.at[0, "range_2d"], axis=1, keepdims=True),
        )

        npt.assert_allclose(
            guess.at[0, "average_row"], np.average(guess.at[0, "range_2d"], axis=0)
        )
        npt.assert_allclose(
            guess.at[0, "outer_product"],
            np.outer(
                guess.at[0, "range_1d_horizontal"][:4], guess.at[0, "range_1d_vertical"]
            ),
        )

    def test_evaluate_multiple_row_range(self, evaluator_range):
        # Create a DataFrame with one row of inputs by using the initial guess
        guess = evaluator_range.initial_guess()
        # Assuming guess is your original DataFrame with one row
        unrolled = unroll_data_frame_using_variables(
            guess, evaluator_range.interface.inputs
        )
        # Get the columns from guess
        columns = unrolled.columns

        # Generate random data: 10 rows, number of columns same as guess
        random_data = np.random.uniform(low=-10.0, high=2.5, size=(10, len(columns)))

        # Create the new DataFrame with the same columns
        multiple = pd.DataFrame(random_data, columns=columns)

        # Roll the DataFrame so we can use it with the evaluator
        multiple = roll_data_frame_using_variables(
            multiple, evaluator_range.interface.inputs
        )

        # Evaluate all 10 sites
        evaluator_range(multiple)

        for _, row in multiple.iterrows():
            assert row["sum1"] == np.sum(row["range_1d_horizontal"])
            assert row["average1"] == np.average(row["range_1d_vertical"])
            npt.assert_allclose(
                row["sum_column"],
                np.sum(row["range_2d"], axis=1, keepdims=True),
            )

            npt.assert_allclose(row["average_row"], np.average(row["range_2d"], axis=0))
            npt.assert_allclose(
                row["outer_product"],
                np.outer(row["range_1d_horizontal"][:4], row["range_1d_vertical"]),
            )
