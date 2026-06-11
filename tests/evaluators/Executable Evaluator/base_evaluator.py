# Code to test the basic functionality of the executable evaluator


def parse_args(args):
    plus_one = 0
    double_mod = 1
    for arg in args:
        if arg[0] == "-":
            if arg == "-plus_one":
                plus_one = 1
            elif arg == "-double":
                double_mod = 2
        elif arg.endswith(".in"):
            infile = arg
        elif arg.endswith(".out"):
            outfile = arg

    return infile, outfile, double_mod, plus_one


def read_values(fname, delim):
    var_vals = []
    with open(fname, "r") as file:
        var_inds = {
            name: i for i, name in enumerate(file.readline().strip().split(delim))
        }

        for line in file.readlines():
            var_vals.append([float(val) for val in line.split(delim)])

    return var_inds, var_vals


def calc_resp(var_inds, var_vals, double_mod=1, plus_one=0):
    resp_vals = []
    for val in var_vals:
        x1 = val[var_inds["x1"]]
        x2 = val[var_inds["x2"]]

        resp_vals.append(
            [double_mod * (x1 + x2), (x1 + plus_one) ** 2 + x2**2 - 2, -x2]
        )

    return resp_vals


def write_output(fname, resp_vals, delim):
    with open(fname, "w") as file:
        file.write(f"obj{delim}c1{delim}c2\n")

        for resp in resp_vals:
            file.write(delim.join([str(val) for val in resp]) + "\n")


if __name__ == "__main__":
    import sys

    # Parse arguments
    infile, outfile, double_mod, plus_one = parse_args(sys.argv)

    # Get variable values
    var_inds, var_vals = read_values(infile, ",")

    # Calculate response values
    resp_vals = calc_resp(var_inds, var_vals, double_mod, plus_one)

    # Write output file
    write_output(outfile, resp_vals, ",")
