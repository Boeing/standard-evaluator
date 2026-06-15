from pathlib import Path
import logging
from typing import Dict, List, Any, Union, Tuple, Literal, Type, Optional
import shutil
import tempfile
import secrets

import numpy as np
from openpyxl.workbook import Workbook
from openpyxl.utils import range_boundaries
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from pydantic import BaseModel, field_validator, Field

from standard_evaluator.evaluator import EvaluatorInfo
from standard_evaluator import (
    IntVariable,
    FloatVariable,
    ArrayVariable,
    Variable,
)

logger = logging.getLogger()

# A convenient alias for the allowed var_type values:
VarType = Union[Literal["int", "float"], Type[Union[int, float]]]

# Excel limits (xlsx)
_MAX_ROWS = 1_048_576
_MAX_COLS = 16_384


class MacroDefinition(BaseModel):
    """
    Pydantic model describing macro execution stages for a spreadsheet.

    This model organizes macros into three distinct stages of execution:

    Attributes:
        pre_macros (Optional[List[str]]):
            Macros that are executed after the spreadsheet is loaded and before any evaluations are done.
        main_macros (Optional[List[str]]):
            Macros that are executed every time after the inputs are updated and the spreadsheet has been recalculated.
            After these macros have been executed, the outputs are read.
        post_macros (Optional[List[str]]):
            Macros that are executed after the outputs are read and before the spreadsheet is saved and closed.
    """

    pre_macros: Optional[List[str]] = Field(
        default_factory=list,
        description="Macros that are executed after the spreadsheet is loaded and before any evaluations are done",
    )
    main_macros: Optional[List[str]] = Field(
        default_factory=list,
        description="Macros that are executed every time after the inputs are updated and the spreadsheet has been recalculated. After these macros have been executed the outputs are read.",
    )
    post_macros: Optional[List[str]] = Field(
        default_factory=list,
        description="Macros that are executed after the outputs are read and before the spreadsheet is saved and closed.",
    )


class SpreadsheetModel(BaseModel):
    """
    Pydantic model representing a spreadsheet file path.

    Attributes:
        spreadsheet (Path): Path to the spreadsheet file. Must have a .xlsx or .xlsm extension and exist on disk.
    """

    spreadsheet: Path

    @field_validator("spreadsheet")
    def check_xlsx_extension(cls, v: Path) -> Path:
        """
        Validator to ensure the spreadsheet file has a valid Excel extension and exists.

        Args:
            v (Path): The path to validate.

        Returns:
            Path: The validated path.

        Raises:
            ValueError: If the file extension is not .xlsx or .xlsm, or if the file does not exist.
        """
        if not (v.suffix in [".xlsx", ".xlsm"]):
            raise ValueError("File must have a .xlsx or .xlsm extension")
        if not v.is_file():
            raise ValueError(f"File does not exist: {v}")
        return v

    def make_temp_copies(self, count=1, dest_dir=None, name_prefix="tmp_"):
        """
        Create `count` temporary copies of `src_path` with random names preserving
        the same extension(s). Returns a list of pathlib.Path objects.

        Parameters:
        count: int - number of copies to create
        dest_dir: str or pathlib.Path - optional directory in which to create the copies
                (default: system temp directory)
        name_prefix: str - optional prefix for random filenames
        """
        src = Path(self.spreadsheet)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src!s}")

        # Preserve all suffixes (handles .tar.gz etc.)
        suffix = ''.join(src.suffixes)
        if not suffix:
            # If there is no extension, we just produce random names without extension
            suffix = ''

        if dest_dir is None:
            dest_dir = Path(tempfile.gettempdir())
        else:
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

        created_paths = []
        for _ in range(count):
            # Use a crypto-random token to minimize collision risk
            rand = secrets.token_hex(8)
            dest_name = f"{name_prefix}{rand}{suffix}"
            dest_path = dest_dir / dest_name

            # If an unlikely collision happens, regenerate the name
            while dest_path.exists():
                rand = secrets.token_hex(8)
                dest_name = f"{name_prefix}{rand}{suffix}"
                dest_path = dest_dir / dest_name

            shutil.copy2(src, dest_path)  # copy2 preserves metadata where possible

            created_paths.append(dest_path)

        return created_paths

def get_cell_shape(coord: str) -> Tuple[bool, Tuple[int, ...]]:
    """
    Determine whether a given worksheet coordinate refers to a single cell and
    return the shape (dimensions) of the referenced range in a compact form.

    Purpose
    - Parse an Excel-style coordinate (for example "A1", "B2:C4", "D5:F5") and
      compute:
      1) whether it refers to a single cell, and
      2) a tuple describing the shape of the referenced block.
    - If the referenced block is a single row with multiple columns, the shape
      is returned as a 1-D tuple (columns,). Otherwise it is returned as a
      2-D tuple (rows, columns).

    Parameters
    - coord: str
      An Excel range string or single-cell coordinate in openpyxl format
      (examples: "A1", "C3:D10", "G2:G2", "A1:Z1").

    Returns
    - Tuple[bool, Tuple[int, ...]]
      A tuple (single_cell, shape):
      - single_cell: bool
        True if coord specifies exactly one cell (e.g., "A1" or "B2:B2"),
        False otherwise.
      - shape: Tuple[int, ...]
        If the range is a single cell, shape will be (1, 1).
        If the range is NxM (multiple rows and columns), shape will be
        (rows, columns).
        If the range is a single row spanning multiple columns, shape will be
        returned as 1-D: (columns,).

    Validation
    - Ensures parsed min/max row/column indices are >= 1.
    - Ensures indices do not exceed Excel .xlsx limits:
      rows <= 1,048,576 and columns <= 16,384.
    - Raises ValueError with a descriptive message on invalid coordinates.

    Example usages
    - get_cell_shape("A1")        -> (True, (1, 1))
    - get_cell_shape("A1:B2")     -> (False, (2, 2))
    - get_cell_shape("A1:D1")     -> (False, (4,))
    - get_cell_shape("C3:C5")     -> (False, (3, 1))
    - get_cell_shape("B2:B2")     -> (True, (1, 1))
    """
    min_col, min_row, max_col, max_row = range_boundaries(coord)

    # Basic validation: indices must be >= 1
    if min_row < 1 or min_col < 1:
        raise ValueError(
            f"Invalid coordinate '{coord}': "
            "row and column indices must be >= 1."
        )
    # Ensure min <= max
    if min_row > max_row or min_col > max_col:
        raise ValueError(
            f"Invalid coordinate '{coord}': " "minimum row/column greater than maximum."
        )

    # Excel limits
    if max_row > _MAX_ROWS or max_col > _MAX_COLS:
        raise ValueError(
            f"Coordinate '{coord}' exceeds Excel limits: "
            f"max rows {_MAX_ROWS}, max columns {_MAX_COLS}."
        )
    local_shape: Tuple[int, ...] = (max_row - min_row + 1, max_col - min_col + 1)
    single_cell = local_shape == (1, 1)
    if not single_cell:
        # Convert 1xN rows to 1-D shape
        if local_shape[0] == 1:
            local_shape = (local_shape[1],)
    return single_cell, local_shape


def get_defined_names_info(
    local_workbook: Workbook,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract information about defined names in an openpyxl Workbook.

    For every defined name in the workbook, this function collects one or more
    destination descriptors. Each descriptor contains:
    - "address": the worksheet-qualified address (e.g. "Sheet1!A1" or "Sheet1!A1:B3")
    - "output": a boolean indicating whether any cell in the destination is a
      formula (data_type == "f"); if any cell in the range is a formula the
      whole destination is considered an output.
    - "shape": included only for multi-cell destinations; a tuple describing the
      shape of the returned data. A 2-D range yields (rows, cols). A 1xN row
      range is converted to a 1-D shape of length N: (N,). Single-cell
      destinations omit the "shape" key.

    Parameters
    ----------
    local_workbook : openpyxl.workbook.Workbook
        The workbook from which to read defined names.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A mapping from defined-name string to a list of destination descriptors.
        Each destination descriptor is a dict as described above.

    Notes
    -----
    - The function attempts to parse each defined name destination coordinate
      using openpyxl.utils.range_boundaries. If parsing fails (for example for
      an unsupported coordinate form), the function falls back to indexing the
      worksheet with the coordinate (e.g. ws[coord]) and emits a warning.
    - The detection of "output" is conservative: if any cell in a multi-cell
      range contains a formula, the entire range is marked as output.
    - The function relies on openpyxl APIs and the structure of defined names
      returned by openpyxl.workbook. Behavior may vary with openpyxl versions.

    Warnings
    --------
    - If a defined name destination raises an exception when parsed as a range,
      a warning is emitted with the defined-name key. This is to surface
      unusual/unsupported coordinate formats for later handling.

    Examples
    --------
    >>> # Assuming wb is an openpyxl.Workbook with a defined name "MYRANGE"
    >>> info = get_defined_names_info(wb)
    >>> info["MYRANGE"][0]["address"]
    'Sheet1!A1:B3'
    >>> info["MYRANGE"][0]["output"]
    False"""
    info: Dict[str, List[Dict[str, Any]]] = {}
    for local_key in local_workbook.defined_names:
        destinations: List[Dict[str, Any]] = []
        defined_name = local_workbook.defined_names[local_key]
        # defined_name.destinations is an iterator; iterate safely
        for sheet_name, coord in defined_name.destinations:
            try:
                ws = local_workbook[sheet_name]
            except KeyError:
                # Worksheet not found: record diagnostic and skip
                logger.warning(
                    "Worksheet %s referenced by defined name %s not found",
                    sheet_name,
                    local_key,
                )
                destinations.append(
                    {
                        "address": f"{sheet_name}!{coord}",
                        "output": False,
                        "error": "sheet_not_found",
                    }
                )
                continue

            output = False
            try:
                # Attempt to parse as a standard range coordinate
                min_col, min_row, max_col, max_row = range_boundaries(coord)
                local_shape: Tuple[int, ...] = (
                    max_row - min_row + 1,
                    max_col - min_col + 1,
                )
                single_cell = local_shape == (1, 1)
                if not single_cell:
                    # Convert 1xN rows to 1-D shape
                    if local_shape[0] == 1:
                        local_shape = (local_shape[1],)
                # Iterate cells (values_only=False) to check for formulas
                rows = []
                for row in ws.iter_rows(
                    min_row=min_row,
                    max_row=max_row,
                    min_col=min_col,
                    max_col=max_col,
                    values_only=False,
                ):
                    rows.append(list(row))
                    for cell in row:
                        # cell.data_type == "f" indicates a formula in openpyxl
                        if cell.data_type == "f":
                            output = True
                result: Dict[str, Any] = {
                    "address": f"{sheet_name}!{coord}",
                    "output": output,
                }
                if not single_cell:
                    result["shape"] = local_shape
                destinations.append(result)

            except (ValueError, TypeError) as e:
                # range_boundaries raises ValueError for invalid ranges; TypeError may occur too.
                # Narrowly catch these and attempt a safe fallback.
                logger.debug(
                    "range_boundaries could not parse '%s' for defined name %s: %s",
                    coord,
                    local_key,
                    e,
                )

                # Fallback: attempt to index worksheet using ws[coord]
                try:
                    cell_or_cells = ws[coord]
                except Exception as fallback_exc:
                    logger.warning(
                        "Failed fallback indexing for defined name %s at %s!%s: %s",
                        local_key,
                        sheet_name,
                        coord,
                        fallback_exc,
                    )
                    destinations.append(
                        {
                            "address": f"{sheet_name}!{coord}",
                            "output": False,
                            "error": "invalid_coordinate",
                        }
                    )
                    continue

                # ws[coord] can be a single Cell or a tuple of tuples for ranges; normalize
                # Check for a single cell
                if hasattr(cell_or_cells, "data_type"):
                    # Single cell
                    single_cell_output = cell_or_cells.data_type == "f"
                    destinations.append(
                        {
                            "address": f"{sheet_name}!{coord}",
                            "output": single_cell_output,
                        }
                    )
                else:
                    # Possibly a multi-cell structure (openpyxl can return tuples)
                    # Flatten and inspect for formula presence, but don't invent shapes here
                    seen_output = False
                    try:
                        # attempt to traverse nested sequences
                        for row in cell_or_cells:
                            for cell in row:
                                if getattr(cell, "data_type", None) == "f":
                                    seen_output = True
                    except Exception:
                        # If traversal fails, record diagnostic
                        logger.debug(
                            "Cannot iterate fallback cell structure for %s!%s",
                            sheet_name,
                            coord,
                        )
                    destinations.append(
                        {
                            "address": f"{sheet_name}!{coord}",
                            "output": seen_output,
                            # We don't attempt to produce a reliable shape here
                        }
                    )
        info[local_key] = destinations
    return info


def check_types(var_type: VarType = None) -> Literal["int", "float"]:
    """
    Validate and normalize a requested numeric type.

    Parameters
    - var_type: one of int, float, "int", or "float".
      If a type object is provided, it must be int or float. If a string is
      provided, it must be "int" or "float".

    Returns
    - A normalized string: either "int" or "float".

    Raises
    - TypeError if var_type is not one of the allowed values.
    """
    mapping = {int: "int", float: "float"}
    # Normalize string names to the corresponding type object
    if isinstance(var_type, str):
        if var_type in ["int", "float"]:
            target_type = var_type
        else:
            raise TypeError(f"Unsupported type name: {var_type!r}")
    # If the caller passed a type object, validate it's one of the allowed ones
    elif var_type in (int, float):
        target_type = mapping[var_type]
    else:
        raise TypeError("var_type must be one of: int, float, 'int', or 'float'")
    return target_type


def create_variable_from_excel_address(
    name: str, address: str, var_type: VarType = float
) -> Variable:
    """
    Create a Variable representing an Excel cell or range, validating the address.

    Parameters
    - name: str
      Variable name to create.
    - address: str
      Excel address in the form "SheetName!A1" or "SheetName!A1:B2".
      This function validates that the address contains a sheet name, a "!"
      separator, and a valid Excel-style cell or range on the right-hand side.
    - var_type: VarType (int, float, "int", or "float"), optional
      Element type for single-cell addresses. For single cells an IntVariable
      or FloatVariable is created; for ranges an ArrayVariable is created and
      var_type is ignored.

    Returns
    - Variable
      An IntVariable, FloatVariable, or ArrayVariable configured with the
      provided name and an "ExcelLocation" option set to the original address.

    Raises
    - ValueError for malformed addresses or invalid Excel coordinates.
    - TypeError for invalid var_type (propagated from check_types).
    """
    # Basic address syntax checking: must contain exactly one '!' and non-empty parts
    if not isinstance(address, str) or "!" not in address:
        raise ValueError(
            f"Invalid address '{address}': must be a string containing 'SheetName!Coord'."
        )

    parts = address.split("!")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid address '{address}': expected a single '!' separator."
        )
    sheet_name, coord = parts
    if not sheet_name:
        raise ValueError(f"Invalid address '{address}': sheet name is empty.")
    if not coord:
        raise ValueError(f"Invalid address '{address}': coordinate part is empty.")

    # Validate coord using existing range parsing logic (raises on invalid coords)
    try:
        # get_cell_shape internally calls openpyxl.utils.range_boundaries and performs Excel limits checks
        single_cell, local_shape = get_cell_shape(coord=coord)
    except Exception as exc:
        raise ValueError(
            f"Invalid Excel coordinate '{coord}' in address '{address}': {exc}"
        ) from exc

    mapping = {"int": IntVariable, "float": FloatVariable}
    if single_cell:
        var_type_norm = check_types(var_type)
        local_var = mapping[var_type_norm](
            name=name,
            options={"ExcelLocation": address},
            class_type=var_type_norm,
        )
    else:
        local_var = ArrayVariable(
            name=name,
            shape=local_shape,
            options={"ExcelLocation": address},
        )
    return local_var


def _read_default_value(
    wb_data: Optional[Workbook],
    address: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Read the current cell value(s) from a data-only workbook.

    Args:
        wb_data: Workbook opened with ``data_only=True``, or None.
        address: Worksheet-qualified address (e.g. ``Sheet1!A1:B3``).
        shape: Shape tuple for multi-cell ranges, or None for scalars.

    Returns:
        A float for scalar cells, a numpy array for ranges, or None if
        the value could not be read.
    """
    if wb_data is None:
        return None
    try:
        sheet_name, coord = address.split("!")
        ws = wb_data[sheet_name]
        min_col, min_row, max_col, max_row = range_boundaries(coord)
        if shape is None:
            # Scalar cell
            val = ws.cell(row=min_row, column=min_col).value
            return float(val) if val is not None else None
        # Multi-cell range
        rows = []
        for row in ws.iter_rows(
            min_row=min_row, max_row=max_row,
            min_col=min_col, max_col=max_col,
            values_only=True,
        ):
            rows.append([float(v) if v is not None else 0.0 for v in row])
        arr = np.array(rows)
        # Squeeze single-row ranges to 1-D to match the shape convention
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.squeeze(axis=0)
        return arr
    except Exception:
        logger.debug("Could not read default value for %s", address, exc_info=True)
        return None


def get_interface_from_excel_named_ranges(
    spreadsheet: Union[str, SpreadsheetModel],
    name: str,
    *,
    allow_noncontiguous: bool = False,
) -> Tuple[EvaluatorInfo, List[Dict[str, Any]]]:
    """
    Convert Excel named ranges into an EvaluatorInfo interface.

    Returns a tuple: (EvaluatorInfo, diagnostics), where diagnostics is a list of dicts describing issues/warnings.

    Parameters:
      - spreadsheet: either a path string or a validated SpreadsheetModel
      - name: name for EvaluatorInfo
      - allow_noncontiguous: if True, attempt to handle multiple destinations; otherwise skip them
    """
    # Normalize spreadsheet input
    if isinstance(spreadsheet, str):
        spreadsheet = SpreadsheetModel(spreadsheet=Path(spreadsheet))
    diagnostics: List[Dict[str, Any]] = []

    # Load workbook (allow macros)
    try:
        local_wb = load_workbook(
            spreadsheet.spreadsheet, keep_vba=True, data_only=False
        )
    except InvalidFileException as e:
        logger.error("Failed to open workbook %s: %s", spreadsheet.spreadsheet, e)
        raise

    info = get_defined_names_info(local_workbook=local_wb)

    # Load a data_only copy to read cached cell values for defaults
    try:
        local_wb_data = load_workbook(
            spreadsheet.spreadsheet, keep_vba=True, data_only=True
        )
    except Exception:
        logger.warning(
            "Could not load workbook in data_only mode; defaults will not be set."
        )
        local_wb_data = None

    inputs: List[Variable] = []
    outputs: List[Variable] = []

    for local_key, local_value in info.items():
        # local_value is a list of destination descriptors
        if len(local_value) > 1:
            msg = (
                f"Defined name {local_key} has multiple destinations (non-contiguous)."
            )
            logger.warning(msg)
            diagnostics.append(
                {
                    "name": local_key,
                    "severity": "warning",
                    "message": msg,
                    "detail": local_value,
                }
            )
            if not allow_noncontiguous:
                continue
            # If allow_noncontiguous is True, you might implement combining logic here.
        # Process each destination entry
        # Note: current code supports single-destination; if multiple destinations and allow_noncontiguous True,
        # you can extend logic to combine shapes or create multiple variables.
        for dest in local_value:
            if dest.get("error"):
                diagnostics.append(
                    {
                        "name": local_key,
                        "severity": "error",
                        "message": "Destination error",
                        "detail": dest,
                    }
                )
                continue
            address = dest["address"]
            default = _read_default_value(local_wb_data, address, dest.get("shape"))
            if "shape" in dest:
                local_var = ArrayVariable(
                    name=local_key,
                    shape=dest["shape"],
                    default=default,
                    options={"ExcelLocation": address},
                )
            else:
                local_var = FloatVariable(
                    name=local_key,
                    default=default,
                    options={"ExcelLocation": address},
                )

            if dest.get("output"):
                outputs.append(local_var)
            else:
                inputs.append(local_var)

    interface = EvaluatorInfo(name=name, inputs=inputs, outputs=outputs)
    return interface, diagnostics


def range_shape(excel_range: str) -> Tuple[int, int]:
    """
    Calculate the number of rows and columns defined by an Excel range string.

    The input range should be in the format 'A3:B28', optionally containing '$' characters
    which will be ignored. The function raises a ValueError if the format is invalid or
    if the range defines zero or negative rows or columns.

    Args:
        excel_range (str): Excel range string (e.g., 'A3:B28', '$A$3:$B$28').

    Returns:
        Tuple[int, int]: A tuple (num_rows, num_columns) representing the shape of the range.

    Raises:
        ValueError: If the range format is invalid or if rows/columns count is <= 0.

    Examples:
        >>> range_shape('A3:B28')
        (26, 2)
        >>> range_shape('$A$3:$B$28')
        (26, 2)
    """

    (min_col, min_row, max_col, max_row) = range_boundaries(excel_range)

    num_rows = max_row - min_row + 1
    num_cols = max_col - min_col + 1
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("Range must have positive number of rows and columns")

    return (num_rows, num_cols)


def get_explicit_definitions(list_vars: List[Variable]) -> Dict[str, Dict[str, str]]:
    """
    Extract explicit Excel location definitions from a list of Variable objects.

    Args:
        list_vars (List[Variable]): List of Variable instances to extract Excel locations from.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping variable names to dictionaries containing
                                   'sheet' and 'cell' keys indicating Excel locations.
    """
    output = {}
    for var in list_vars:
        if "ExcelLocation" in var.options:
            sheet, cell = var.options["ExcelLocation"].split("!")
            output[var.name] = {"sheet": sheet, "cell": cell}
            if var.class_type == "floatarray":
                output[var.name]["shape"] = var.shape
                # We need to make sure that the range defined in the options has the correct shape
                excel_shape = range_shape(cell)
                if excel_shape[0] == 1:
                    # This should just be a simple 1D vector:
                    excel_shape = (excel_shape[1],)
                if excel_shape != var.shape:
                    raise ValueError(
                        f"Shape of {var.name} is {var.shape}, but Excel range is {excel_shape}"
                    )
    return output
