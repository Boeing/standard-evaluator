import sys

import numpy as np

# Read input
in_file = sys.argv[1]
print('in_file: ', in_file)

x1, x2 = np.loadtxt(in_file, delimiter=',', skiprows=1, unpack=True, ndmin=2)
print(f'x1: {x1}\nx2: {x2}')

# Compute
result = x1 + x2

# Write output
out_file = sys.argv[2]
print('out_file: ', out_file)

np.savetxt(out_file, result, delimiter='\n', header='sum', comments='')
