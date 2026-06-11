import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Sequence, Tuple, Generator

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def flat_element_list(name: str, shape: Optional[Sequence[int]]) -> Generator[Tuple[str, Optional[Tuple[int, ...]]], None, None]:
    """Yields (name, multi_index) for each element; multi_index is None for scalars.

    Args:
        name: Base variable name (e.g. "E" or "A").
        shape: Shape of the array variable (None for scalar).

    Yields:
        Tuple of (name, multi_index) where multi_index is a tuple of ints or None.
    """
    if shape is None:
        yield name, None
    else:
        for multi_idx in np.ndindex(tuple(shape)):
            yield name, multi_idx


def flatten_items_to_arrays(items: Iterable[Tuple[str, Optional[Sequence[int]]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattens (name, shape) items into three parallel arrays.

    Args:
        items: Iterable of (name, shape) pairs. ``shape`` is None for scalars
            or a sequence of ints for arrays.

    Returns:
        A tuple of (names_arr, multi_idx_arr, flat_indices):
            - names_arr: numpy object array of repeated names.
            - multi_idx_arr: numpy object array of multi-index tuples or None.
            - flat_indices: numpy integer array 0..N-1.
    """
    names: List[object] = []
    multi_idx: List[Optional[Tuple[int, ...]]] = []
    for name, shape in items:
        for nm, mi in flat_element_list(name, shape):
            names.append(nm)
            multi_idx.append(mi)
    names_arr = np.array(names, dtype=object)
    multi_idx_arr = np.array(multi_idx, dtype=object)
    flat_indices = np.arange(len(names_arr), dtype=int)
    return names_arr, multi_idx_arr, flat_indices


def compress_whitespace(s: str) -> str:
    """Removes all whitespace from a string.

    Args:
        s: Input string to compact (may contain spaces, tabs, newlines).

    Returns:
        String with all whitespace removed.
    """
    return "".join(s.split())


def res_element_to_string(name: str, multi_idx: Optional[Sequence[int]]) -> str:
    """Returns canonical string for an element, e.g. "E[0,0]" or "E".

    Args:
        name: Base variable/response name.
        multi_idx: Multi-index tuple for array elements or None for scalars.

    Returns:
        Canonical identifier string for the flattened element.
    """
    if multi_idx is None:
        return f"{name}"
    idx_text = ",".join(str(int(i)) for i in multi_idx)
    return f"{name}[{idx_text}]"
