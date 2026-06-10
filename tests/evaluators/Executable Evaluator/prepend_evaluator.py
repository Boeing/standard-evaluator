# Evaluator to test input/output files with argument prepends
# ie. input=path/to/file

import argparse
import sys

sys.dont_write_bytecode = True

from base_evaluator import *

parser = argparse.ArgumentParser()
parser.add_argument('-plus_one', default=0, action='store_const', const=1)
parser.add_argument('-double', default=1, action='store_const', const=2)
parser.add_argument('-input', required=True)
parser.add_argument('-output', required=True)

# Read arguments
args = parser.parse_args()

# Get variable values
var_inds, var_vals = read_values(args.input, ",")

# Calculate response values
resp_vals = calc_resp(var_inds, var_vals, args.double, args.plus_one)

# Write output file
write_output(args.output, resp_vals, ",")
