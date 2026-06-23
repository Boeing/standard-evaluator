"""Utility classes and methods used across the Standard Evaluator."""


def unique_names(var):
    """Validates that all variable/response names are unique.

    Args:
        var: A list of variable or response objects with a `name` attribute.

    Raises:
        ValueError: If any name is duplicated.
    """
    names = []
    for my_var in var:
        names.append(my_var.name)
    if len(names) != len(set(names)):
        dup = {x for x in names if names.count(x) > 1}
        raise ValueError(
            f"There is at least one variable / response name that is used multiple times. Names are {dup}"
        )


from standard_evaluator.utilities.utility import (
    apply_types,
    apply_types_from_evaluator_info,
    create_df_from_problem,
    create_df_from_evaluator_info,
    check_prob,
    get_types,
    get_types_from_evaluator_info,
    concat_w_empty,
    remove_duplicates,
    get_constant_vars,
    restrict_problem,
    get_shift_scale_value,
    update_bounds_to_optimizer_space,
    update_bounds_to_design_space,
)

from standard_evaluator.utilities.problem_dict_utility import (
    legacy_to_opt_problem,
    collect_names,
    opt_problem_to_legacy,
    create_opt_problem,
    create_evaluator_info,
    problem_calculate_fields,
)

from standard_evaluator.utilities.se_arrays import (
    generate_names,
    unroll_data_frame,
    unroll_data_frame_using_variables,
    unroll_data_frame_numpy,
    unroll_names,
    unroll_names_using_variables,
    roll_data_frame,
    roll_data_frame_using_variables,
    get_variable_shape_in_data_frame,
    check_rolled_data_frame_against_variable_shapes,
)

from standard_evaluator.utilities.shift_scale import ShiftAndScale

from standard_evaluator.utilities.option_merge import combine_instances

from standard_evaluator.utilities.mapping import (
    flat_element_list,
    flatten_items_to_arrays,
    compress_whitespace,
    res_element_to_string,
)

from standard_evaluator.utilities.opt_problem_utility import (
    get_opt_problem_constant_vars,
)


__all__ = [
    "unique_names",
    "legacy_to_opt_problem",
    "collect_names",
    "opt_problem_to_legacy",
    "create_opt_problem",
    "create_evaluator_info",
    "create_df_from_problem",
    "create_df_from_evaluator_info",
    "apply_types",
    "apply_types_from_evaluator_info",
    "check_prob",
    "get_types",
    "get_types_from_evaluator_info",
    "concat_w_empty",
    "ShiftAndScale",
    "problem_calculate_fields",
    "remove_duplicates",
    "get_constant_vars",
    "restrict_problem",
    "combine_instances",
    "get_shift_scale_value",
    "update_bounds_to_optimizer_space",
    "update_bounds_to_design_space",
    "generate_names",
    "unroll_data_frame",
    "unroll_data_frame_using_variables",
    "unroll_data_frame_numpy",
    "unroll_names",
    "unroll_names_using_variables",
    "roll_data_frame",
    "roll_data_frame_using_variables",
    "get_variable_shape_in_data_frame",
    "check_rolled_data_frame_against_variable_shapes",
    "flat_element_list",
    "flatten_items_to_arrays",
    "compress_whitespace",
    "res_element_to_string",
    "get_opt_problem_constant_vars",
]
