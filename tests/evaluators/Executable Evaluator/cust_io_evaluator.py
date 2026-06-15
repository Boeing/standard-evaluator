# Handle wacky input/output file formats

import sys

sys.dont_write_bytecode = True

from base_evaluator import *

infile, outfile, double_mod, plus_one = parse_args(sys.argv)

# Read data
var_inds = {}
var_vals = []
with open(infile, "r") as file:
    for i, line in enumerate(file):
        var, *val_str = line.strip().split("_")

        var_inds[var] = i
        var_vals.append([float(x) for x in val_str])

# Reshape array
var_vals = [
    [var_vals[j][i] for j in range(len(var_inds))] for i in range(len(var_vals[0]))
]

# Calculate responses
resp_vals = calc_resp(var_inds, var_vals)

# Write output
# Rows are inverted and columns are right shifted by number of sites
# Header row is left unaltered
# ie. offset of 2
# [0, 1, 2, 3, 4] -> [3, 4, 0, 1, 2]
sep = "sep"
with open(outfile, "w") as file:
    offset = len(resp_vals)
    num_resp = len(resp_vals[0])
    for i in range(len(resp_vals) - 1, -1, -1):
        offset_vals = [
            str(resp_vals[i][(j - offset) % num_resp]) for j in range(num_resp)
        ]
        file.write(sep.join(offset_vals) + "\n")

    file.write(sep.join(["obj", "c1", "c2"]))
