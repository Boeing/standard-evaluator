"""Test executable for 2D array variable round-trip evaluation.

Reads CSV with unrolled 2D array columns (e.g., x[0,0], x[0,1], x[1,0], x[1,1]).
Computes: y[i,j] = x[i,j] * 3
Writes output CSV with unrolled column headers.
"""
from __future__ import annotations

import sys


def parse_args(args: list[str]) -> tuple[str, str]:
    """Parse command line arguments for input/output file paths."""
    infile = ""
    outfile = ""
    for arg in args:
        if arg.endswith(".in"):
            infile = arg
        elif arg.endswith(".out"):
            outfile = arg
    return infile, outfile


def main() -> None:
    infile, outfile = parse_args(sys.argv)

    import csv

    with open(infile, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Identify x[i,j] columns
    x_cols = [col for col in header if col.startswith("x[")]
    # Build output columns: y[i,j] for each x[i,j]
    y_cols = [col.replace("x[", "y[") for col in x_cols]

    output_header = y_cols
    output_rows = []

    for row in rows:
        y_vals = [float(row[header.index(col)]) * 3.0 for col in x_cols]
        output_rows.append(y_vals)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for row in output_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
