# Evaluator we can use to test multiple evaluations

import sys

sys.dont_write_bytecode = True

from base_evaluator import *

# Parse arguments
infile, outfile, double_mod, plus_one = parse_args(sys.argv)

# Get variable values
var_inds, var_vals = read_values(infile, ",")

# Let reader know that multiple values were passed
print(len(var_vals) > 1)

# Calculate response values
resp_vals = calc_resp(var_inds, var_vals, double_mod, plus_one)

# Write output file
write_output(outfile, resp_vals, ",")
