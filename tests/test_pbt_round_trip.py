"""Property-based test for interface serialization round-trip.

Feature: surrogate-replacement-demo
Property 3: Interface serialization round-trip preserves assembly functionality

**Validates: Requirements 2.4**

For any valid OpenMDAO Problem built from ExecComp components (with arbitrary
equations, shapes, and internal connections), calling get_interface() followed
by create_problem() shall produce a new Problem that runs without error and
produces numerically equivalent outputs for the same inputs.
"""

import numpy as np
import openmdao.api as om
from hypothesis import given, settings, assume
from hypothesis import strategies as st

import standard_evaluator as se


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Simple linear equations that are numerically stable.
# Each equation has the form: output = coeff1 * input1 + coeff2 * input2 + offset
# We generate components with unique input/output names to avoid conflicts.

# Strategy for coefficient values (avoid zero to keep equations meaningful)
_coefficients = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for input values to set on the problem
_input_values = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def exec_comp_assembly(draw):
    """Generate a random ExecComp-based assembly with varying equation counts.

    The assembly is a single Group with promoted ExecComp subsystems.
    Each ExecComp has a simple linear equation of the form:
        y_i = a_i * x_i + b_i

    All variables are scalars (shape=1) to avoid shape-related issues.
    Promotions at the top level expose all inputs/outputs.
    No internal connect() calls (those need special handling).

    Returns:
        tuple: (num_components, equations_info) where equations_info is a list of
               dicts with keys: comp_name, equation, input_name, output_name,
               coeff, offset
    """
    # Generate between 1 and 4 components
    num_components = draw(st.integers(min_value=1, max_value=4))

    components_info = []
    for i in range(num_components):
        # Each component gets a unique input and output name
        input_name = f"x{i}"
        output_name = f"y{i}"
        comp_name = f"comp{i}"

        # Generate coefficient and offset
        coeff = draw(_coefficients)
        offset = draw(_coefficients)

        # Simple linear equation: y = coeff * x + offset
        equation = f"{output_name} = {coeff} * {input_name} + {offset}"

        components_info.append({
            "comp_name": comp_name,
            "equation": equation,
            "input_name": input_name,
            "output_name": output_name,
            "coeff": coeff,
            "offset": offset,
        })

    return components_info


def build_assembly(components_info):
    """Build an OpenMDAO Problem from component info.

    Args:
        components_info: List of dicts describing each component.

    Returns:
        om.Problem: A setup and final-setup'd OpenMDAO Problem.
    """
    prob = om.Problem()
    model = prob.model

    for comp in components_info:
        model.add_subsystem(
            comp["comp_name"],
            om.ExecComp(comp["equation"]),
            promotes=["*"],
        )

    prob.setup()
    prob.final_setup()
    return prob


def run_problem_with_inputs(prob, components_info, input_values):
    """Set inputs on a problem, run it, and return outputs.

    Args:
        prob: OpenMDAO Problem (must be setup).
        components_info: List of component info dicts.
        input_values: List of float values to set for each input.

    Returns:
        dict: Mapping of output_name -> output_value
    """
    # Set input values
    for comp, val in zip(components_info, input_values):
        prob.set_val(comp["input_name"], val)

    # Run the problem
    prob.run_model()

    # Collect outputs
    outputs = {}
    for comp in components_info:
        val = prob.get_val(comp["output_name"])
        outputs[comp["output_name"]] = np.asarray(val).flatten()[0]

    return outputs


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


class TestInterfaceSerializationRoundTrip:
    """Property 3: Interface serialization round-trip preserves assembly functionality.

    For any valid OpenMDAO Problem built from ExecComp components, calling
    get_interface() followed by create_problem() produces a new Problem that
    runs without error and produces numerically equivalent outputs.
    """

    @given(
        components_info=exec_comp_assembly(),
        input_vals=st.lists(
            _input_values,
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_round_trip_produces_equivalent_outputs(self, components_info, input_vals):
        """get_interface() followed by create_problem() produces numerically
        equivalent outputs for the same inputs.

        **Validates: Requirements 2.4**
        """
        # Ensure we have enough input values for the number of components
        assume(len(input_vals) >= len(components_info))
        input_vals = input_vals[:len(components_info)]

        # Build the original assembly
        original_prob = build_assembly(components_info)

        # Run original with inputs
        original_outputs = run_problem_with_inputs(
            original_prob, components_info, input_vals
        )

        # Capture interface
        info = se.get_interface(original_prob.model)

        # Recreate problem from interface
        recreated_prob = se.create_problem(info)

        # Run recreated with same inputs
        recreated_outputs = run_problem_with_inputs(
            recreated_prob, components_info, input_vals
        )

        # Assert outputs are numerically equivalent
        for output_name in original_outputs:
            orig_val = original_outputs[output_name]
            recreated_val = recreated_outputs[output_name]
            assert np.isclose(orig_val, recreated_val, rtol=1e-10, atol=1e-12), (
                f"Output '{output_name}' differs: "
                f"original={orig_val}, recreated={recreated_val}"
            )
