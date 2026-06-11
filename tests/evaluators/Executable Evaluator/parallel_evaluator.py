# Evaluator we can use to test parallel evaluations

import sys

sys.dont_write_bytecode = True

from base_evaluator import *
import uuid
import time

# Generate unique ID
id = uuid.uuid4()
# Print when we start
print(f"{id}:{time.time()}")

# Parse arguments
infile, outfile, double_mod, plus_one = parse_args(sys.argv)

# Get variable values
var_inds, var_vals = read_values(infile, ",")

# Calculate response values
resp_vals = calc_resp(var_inds, var_vals, double_mod, plus_one)

# Write output file
write_output(outfile, resp_vals, ",")

# Wait a second to ensure there is some overlap between jobs
time.sleep(0.25)

# Print when we end
print(f"{id}:{time.time()}")
