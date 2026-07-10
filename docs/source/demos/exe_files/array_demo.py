"""Demo executable for array variable tutorial.

Reads CSV with unrolled columns: x[0], x[1], x[2], s1
Computes: y[i] = x[i] * 2, total = sum(x) + s1
Writes CSV with columns: y[0], y[1], y[2], total
"""
import sys

import numpy as np


def main() -> None:
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    # Read input CSV
    data = np.loadtxt(in_file, delimiter=",", skiprows=1, ndmin=2)
    header = open(in_file).readline().strip().split(",")

    # Find x columns and s1
    x_cols = sorted(
        [i for i, col in enumerate(header) if col.startswith("x[")],
    )
    s1_idx = header.index("s1")

    # Compute outputs
    out_header = [f"y[{i}]" for i in range(len(x_cols))] + ["total"]
    results = []

    for row in data:
        x_vals = [row[i] for i in x_cols]
        s1_val = row[s1_idx]
        y_vals = [x * 2.0 for x in x_vals]
        total = sum(x_vals) + s1_val
        results.append(y_vals + [total])

    # Write output CSV
    with open(out_file, "w") as f:
        f.write(",".join(out_header) + "\n")
        for row in results:
            f.write(",".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
