"""Module defining a class that exposes any Standard evaluator as an OpenMDAO component."""

import copy

import numpy as np
import pandas as pd
import openmdao.api as om

from standard_evaluator.evaluators.abstract_evaluator import Evaluator
from standard_evaluator.problem import ArrayVariable
from standard_evaluator.surrogate_models.abstract_model import SurrogateModel

EVALUATOR_OPTION_NAME = "evaluator_options"


class EvaluatorOpenMdaoComponent(om.ExplicitComponent):
    """Exposes a Standard Evaluator as an OpenMDAO ExplicitComponent.

    This component wraps a Standard Evaluator so it can be used within an
    OpenMDAO model. It maps evaluator variables to component inputs and
    evaluator responses to component outputs.
    """

    def __init__(self, evaluator: Evaluator = None, **kwargs: dict):
        """Initialize the component and store the evaluator.

        Args:
            evaluator: The Standard evaluator that will be wrapped. If None,
                the evaluator is created from the evaluator_options in kwargs.
            **kwargs: Keyword arguments passed to the OpenMDAO ExplicitComponent
                constructor.

        Raises:
            TypeError: If evaluator is provided but is not an instance of Evaluator.
            AttributeError: If no evaluator is provided and no evaluator_options
                key exists in kwargs.
        """
        if evaluator is not None:
            # Type check: must be an instance of Evaluator
            if not isinstance(evaluator, Evaluator):
                raise TypeError(
                    f"The evaluator argument must be an instance of Evaluator. "
                    f"Received a {type(evaluator).__name__} instead."
                )
        else:
            # Fallback: reconstruct from evaluator_options
            if EVALUATOR_OPTION_NAME in kwargs:
                evaluator = SurrogateModel.from_dict(kwargs.pop(EVALUATOR_OPTION_NAME))
            else:
                raise AttributeError(
                    f"No evaluator and no {EVALUATOR_OPTION_NAME} defined."
                )

        # Detect kwarg conflicts with internal attributes
        conflicting_keys = [key for key in kwargs if hasattr(self, key) or key == "eval"]
        if conflicting_keys:
            raise ValueError(
                f"Keyword argument(s) {conflicting_keys} conflict with internal "
                f"Component attribute(s). Please rename them."
            )

        # Store a deep copy of the evaluator
        self.eval = copy.deepcopy(evaluator)
        super().__init__(**kwargs)

    def initialize(self):
        """Declare component options.

        Stores the evaluator information in the evaluator_options option
        if the evaluator has a ``to_dict`` method.
        """
        if hasattr(self.eval, "to_dict"):
            self.options.declare(
                EVALUATOR_OPTION_NAME,
                types=dict,
                desc="Surrogate model information",
            )
            self.options[EVALUATOR_OPTION_NAME] = self.eval.to_dict()

    def setup(self):
        """Set up component inputs and outputs from the evaluator's OptProblem.

        Maps opt_problem.variables to OpenMDAO inputs and opt_problem.responses
        to OpenMDAO outputs with bounds, scaling, and defaults.

        Raises:
            ValueError: If a response has a scale value of 0.0.
        """
        opt_problem = self.eval.opt_problem

        # Set all inputs from variables
        for var in opt_problem.variables:
            if isinstance(var, ArrayVariable):
                # ArrayVariable: compute val from default or midpoint of bounds
                if var.default is not None:
                    val = var.default
                else:
                    lower, upper = var.bounds
                    val = (lower + upper) / 2.0

                kwargs = {"shape": var.shape, "val": val}
                if var.units is not None:
                    kwargs["units"] = var.units
                self.add_input(var.name, **kwargs)
            else:
                # FloatVariable: preserve existing logic
                if var.default is not None:
                    default_value = var.default
                else:
                    # Calculate midpoint from bounds
                    lower, upper = var.bounds
                    default_value = (lower + upper) / 2.0

                kwargs = {"val": default_value}
                if var.units is not None:
                    kwargs["units"] = var.units
                self.add_input(var.name, **kwargs)

        # Set all outputs from responses
        for resp in opt_problem.responses:
            if isinstance(resp, ArrayVariable):
                # ArrayVariable: shaped output with array-aware bounds and scaling
                val = np.zeros(resp.shape)

                kwargs = {"shape": resp.shape, "val": val}

                # Determine lower bound
                lower_bound = np.asarray(resp.bounds[0])
                if np.all(np.isneginf(lower_bound)):
                    pass  # omit lower
                elif np.all(lower_bound == lower_bound.flat[0]):
                    kwargs["lower"] = float(lower_bound.flat[0])
                else:
                    kwargs["lower"] = lower_bound

                # Determine upper bound
                upper_bound = np.asarray(resp.bounds[1])
                if np.all(np.isposinf(upper_bound)):
                    pass  # omit upper
                elif np.all(upper_bound == upper_bound.flat[0]):
                    kwargs["upper"] = float(upper_bound.flat[0])
                else:
                    kwargs["upper"] = upper_bound

                # Compute ref0/ref from shift/scale if non-default
                shift = resp.shift if resp.shift is not None else 0.0
                scale = resp.scale if resp.scale is not None else 1.0
                shift_arr = np.asarray(shift)
                scale_arr = np.asarray(scale)

                if np.any(scale_arr == 0.0):
                    raise ValueError(
                        f"Response '{resp.name}' has a scale element equal to "
                        f"zero, which causes division by zero in OpenMDAO "
                        f"normalization."
                    )

                is_default_scaling = (
                    np.all(shift_arr == 0.0) and np.all(scale_arr == 1.0)
                )
                if not is_default_scaling:
                    ref0 = -shift_arr
                    ref = (1.0 / scale_arr) + ref0
                    kwargs["ref0"] = ref0
                    kwargs["ref"] = ref

                # Pass units if not None
                if resp.units is not None:
                    kwargs["units"] = resp.units

                self.add_output(resp.name, **kwargs)
            else:
                # FloatVariable: preserve existing logic with units support
                lower, upper = resp.bounds[0], resp.bounds[1]

                # Convert infinite bounds to None for OpenMDAO
                if not np.isfinite(lower):
                    lower = None
                if not np.isfinite(upper):
                    upper = None

                # Calculate ref0 and ref from shift and scale
                shift = resp.shift if resp.shift is not None else 0.0
                scale = resp.scale if resp.scale is not None else 1.0

                if scale == 0.0:
                    raise ValueError(
                        f"Response {resp.name} has a scale value of 0.0"
                    )

                ref0 = -shift
                ref = (1.0 / scale) + ref0

                kwargs = {
                    "lower": lower,
                    "upper": upper,
                    "ref0": ref0,
                    "ref": ref,
                    "val": 0.0,
                }

                if resp.units is not None:
                    kwargs["units"] = resp.units

                self.add_output(resp.name, **kwargs)

    def compute(
        self,
        inputs: dict,
        outputs: dict,
        discrete_inputs: dict = None,
        discrete_outputs: dict = None,
    ):
        """Compute responses using the Standard evaluator.

        Builds a DataFrame from OpenMDAO inputs, calls the evaluator, and
        extracts responses back to OpenMDAO outputs.

        Args:
            inputs: Unscaled, dimensional input variables read via inputs[key].
            outputs: Unscaled, dimensional output variables read via outputs[key].
            discrete_inputs: If not None, dict containing discrete input values.
            discrete_outputs: If not None, dict containing discrete output values.

        Raises:
            ValueError: If discrete_inputs or discrete_outputs are provided.
        """
        if discrete_inputs is not None:
            raise ValueError("At this point we do not yet support discrete inputs")
        if discrete_outputs is not None:
            raise ValueError("At this point we do not yet support discrete outputs")

        # Build a lookup of variable types by name
        var_by_name = {
            v.name: v for v in self.eval.opt_problem.variables
        }
        resp_by_name = {
            r.name: r for r in self.eval.opt_problem.responses
        }

        # Build input dictionary from OpenMDAO inputs
        input_dict = {}
        for var_name in self.eval.inputs:
            var = var_by_name[var_name]
            if isinstance(var, ArrayVariable):
                # Wrap in a list so pandas places the array as a single cell
                input_dict[var_name] = [inputs[var_name]]
            else:
                # FloatVariable: scalar value
                input_dict[var_name] = inputs[var_name]

        # Create the DataFrame and evaluate
        data_frame = pd.DataFrame(data=input_dict)
        self.eval(data_frame)

        # Extract responses to outputs
        for resp_name in self.eval.outputs:
            resp = resp_by_name[resp_name]
            if isinstance(resp, ArrayVariable):
                # ArrayVariable: assign full NumPy array from DataFrame cell
                outputs[resp_name] = data_frame[resp_name].iloc[0]
            else:
                # FloatVariable: assign scalar value
                outputs[resp_name] = data_frame[resp_name].iloc[0]
