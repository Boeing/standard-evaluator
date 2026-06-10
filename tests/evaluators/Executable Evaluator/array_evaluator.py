"""Test executable for array variable round-trip evaluation.

Reads a CSV input file with unrolled array columns (e.g., x[0], x[1], x[2], s1).
Computes array outputs: y[i] = x[i] * 2 for each element, plus scalar total = sum(x) + s1.
Writes output CSV with unrolled column headers (e.g., y[0], y[1], y[2], total).
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

    # Read input CSV
    with open(infile, "r") as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            rows.append(line.strip().split(","))

    # Identify x[i] columns and s1 column
    x_cols = sorted(
        [col for col in header if col.startswith("x[")],
        key=lambda c: int(c.split("[")[1].rstrip("]")),
    )
    s1_idx = header.index("s1")

    # Compute outputs
    output_header = [f"y[{i}]" for i in range(len(x_cols))] + ["total"]
    output_rows = []

    for row in rows:
        x_vals = [float(row[header.index(col)]) for col in x_cols]
        s1_val = float(row[s1_idx])

        y_vals = [x * 2.0 for x in x_vals]
        total = sum(x_vals) + s1_val

        output_rows.append(y_vals + [total])

    # Write output CSV
    with open(outfile, "w") as f:
        f.write(",".join(output_header) + "\n")
        for row in output_rows:
            f.write(",".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
