# Same as the base_evaluator but reads files with custom delimiter

import sys

sys.dont_write_bytecode = True

from base_evaluator import *

# Parse arguments
infile, outfile, double_mod, plus_one = parse_args(sys.argv)

# Get variable values
var_inds, var_vals = read_values(infile, "~")

# Calculate response values
resp_vals = calc_resp(var_inds, var_vals, double_mod, plus_one)

# Write output file
write_output(outfile, resp_vals, "~")
