import importlib
import re
import typing
from typing import Tuple, List, Set
import copy
import json
from enum import Enum
import pathlib
from collections import defaultdict

import numpy as np
import h5py
import json_numpy
import openmdao.api as om
import openmdao.core.system as oms
from openmdao.core.indepvarcomp import _AutoIndepVarComp

from standard_evaluator import Variable, FloatVariable, ArrayVariable
from standard_evaluator import GroupInfo, EvaluatorInfo, EquationInfo, JoinedInfo
from standard_evaluator import OptProblem
from standard_evaluator import AviaryEncoder

try:
    from aviary.utils.aviary_values import AviaryValues
    from aviary.subsystems.propulsion.engine_deck import EngineDeck
    _AVIARY_AVAILABLE = True
except ImportError:
    _AVIARY_AVAILABLE = False
    AviaryValues = None
    EngineDeck = None


def _require_aviary():
    """Raise ImportError if aviary is not available."""
    if not _AVIARY_AVAILABLE:
        raise ImportError(
            "aviary is required for this functionality. "
            "Install it with: pip install standard-evaluator[aviary]"
        )

def get_openmdao_options(om_component) -> dict:
    options = copy.deepcopy(om_component.options)
    if 'meta' in [inner_name for (inner_name,_) in options.items()]:
        options.undeclare('meta')
    if 'subsystems' in [inner_name for (inner_name,_) in options.items()]:
        options.undeclare('subsystems')
    
    openmdao_info = {
        "pathname": om_component.pathname,
        #"under_complex_step": om_component.under_complex_step,
        #"under_finite_difference": om_component.under_finite_difference,
        "iter_count": om_component.iter_count,
        "iter_count_apply": om_component.iter_count_apply,
        "iter_count_without_approx": om_component.iter_count_without_approx,
    }
    openmdao_options = json.loads(json.dumps(options.__getstate__(), indent=2, skipkeys=True, cls=AviaryEncoder, ))
    options = {'openmdao_info': openmdao_info,
                'openmdao_options': openmdao_options}
    return(options)

def get_names_to_promoted_names(om_component):
    local_dict = om_component.get_io_metadata(metadata_keys=['tags'], return_rel_names=False)
    local_map = {}
    for key, value in local_dict.items():
        if not "_auto_ivc" in value['prom_name']:
            # we do not include internal variables that are within a component.
            if not 'internal' in value['tags']:
                local_map[key] = value['prom_name']
    return local_map

def get_linkages(om_group: om.Group):
    # For now we only allow linkage between full components.
    # TODO Allow linkage between elements
    linkage = []
    for key, value in om_group._manual_connections.items():
        if len(value) > 2:
            if (value[1] is not None) | (value[2] is not None):
                print(f"Indexing used: {key}, {value}, {type(value[0])}, {type(value[1])}, {type(value[2])}")
        linkage.append((value[0], key))
    return linkage

def convert_om_var(info: dict) -> Variable:
    optional_fields_array = ['compute_shape',  'copy_shape',
        'discrete', 'distributed', 'global_shape',
        'global_size',  'has_src_indices',
        'shape_by_conn', 'size', ]
    input_dict = {
            'name': info['prom_name'],
            'units': info['units'],
            'description': info['desc'],}
    if 'val' in info:
        input_dict['default'] = info['val']
    additional_dict = {}
    if len(info['tags']):
        additional_dict['tags'] = info['tags']

    if info['shape'] == (1,):
        # This is a float variable. Store it as such
        input_dict['options'] = additional_dict
        output = FloatVariable.parse_obj(input_dict)
    else:
        # This is an array variable, and we need to capture its shape and
        # additional information
        input_dict['shape'] = info['shape']
        for field in optional_fields_array:
            if field in info:
                additional_dict[field] = info[field]
        input_dict['options'] = additional_dict
        output = ArrayVariable.parse_obj(input_dict)
    return(output)

def get_external_names(om_component) -> dict:
    info = {}
    for element_type in ["input", "output"]:
        local_list = []
        local_dict = om_component.get_io_metadata(return_rel_names=False, iotypes=element_type)
        external_names = set()
        for key, value in local_dict.items():
            if "_auto_ivc" in key or "_auto_ivc" in value['prom_name']:
                # Skip auto-IVC variables — they are internal to OpenMDAO and should
                # not appear in the interface description of a component/group.
                continue
            name = value['prom_name']
            if not name in external_names:
                var_info = convert_om_var(value)
                external_names.add(value['prom_name'])
                local_list.append(var_info)
        info[element_type] = list(external_names)
        info[f"{element_type}_vars"] = local_list
    return info


def get_interface_w_promotions(om_component: oms.System, parent_info: dict={}) -> Tuple[JoinedInfo, List[Set[str]]]:
    if isinstance(om_component, _AutoIndepVarComp):
        return
    component_name = om_component.name
    # Get the external facing names of the component. They should be the same no matter
    # where the component is in the hierarchy. We also return the list of instances of
    # the Variable instances which capture the types / shapes.
    external_names = get_external_names(om_component)
    # Get the map how the external facing names map to the current instantiation of the model
    local_map = get_names_to_promoted_names(om_component)
    # Now create a list of mappings between the external names for this component
    # and the names that are being used to add this component to their parent.
    promotions = []
    for key_name, value in local_map.items():
        if key_name in parent_info:
            promotions.append((value, parent_info[key_name]))
    # Get all the OpenMDAO options for this component
    options = get_openmdao_options(om_component)

    if isinstance(om_component, om.Group):
        components = {}
        component_order = []
        component_promotions = {}
        for local_comp in om_component._subsystems_myproc:
            local_name = local_comp.name
            if local_name == '_auto_ivc':
                # We don't want to save the auto_ivc class. It's only used by OpenMDAO
                continue
            components[local_name], component_promotions[local_name] = get_interface_w_promotions(local_comp, local_map)
            component_order.append(local_name)
        linkage = get_linkages(om_component)

        my_evaluator_info = GroupInfo(name=component_name, 
              inputs=external_names["input_vars"], 
              outputs=external_names["output_vars"],
              cite=om_component.cite,
              tool=str(type(om_component)),
              component_type='Group',
              options = options,
              component_order=component_order,
              components=components,
              promotions=component_promotions,
              linkage=linkage)
    elif isinstance(om_component, om.ExecComp):
        component_type = 'EquationComponent'
        #del(options['openmdao_options']['_dict']['distributed'])
        del(options['openmdao_options']['_dict']['derivs_method'])
        del(options['openmdao_options']['_dict']['use_jit'])
        del(options['openmdao_options']['_dict']['run_root_only'])
        del(options['openmdao_options']['_dict']['always_opt'])
        my_evaluator_info = EquationInfo(name=component_name, 
              inputs=external_names["input_vars"], 
              outputs=external_names["output_vars"],
              cite=om_component.cite,
              tool=str(type(om_component)),
              component_type=component_type,
              equations=om_component._exprs,
              options = options)
    else:
        if isinstance(om_component, om.ExplicitComponent):
            component_type = 'ExplicitComponent'
        elif isinstance(om_component, om.ImplicitComponent):
            component_type = 'ImplicitComponent'
        else:
            component_type = 'Other'
        #if isinstance(om_component, om.ExecComp):
        #    # For an OpenMDAO ExecComp we store the extra information about the expressions to use
        #    options['openmdao_options']['_dict']['exprs'] = {'val': om_component._exprs}
        my_evaluator_info = EvaluatorInfo(name=component_name, 
              inputs=external_names["input_vars"], 
              outputs=external_names["output_vars"],
              cite=om_component.cite,
              tool=str(type(om_component)),
              component_type=component_type,
              options = options)
    return(my_evaluator_info, promotions)

def get_state(om_problem: om.Problem, info: JoinedInfo) -> typing.Dict[str, np.array]:
    local_dict = {}
    for ele in info.inputs+info.outputs:
        local_dict[ele.name] = om_problem.get_val(ele.name)
    return(local_dict)

def save_state(om_problem: om.Problem, info: JoinedInfo, file_name: str):
    state = get_state(om_problem, info)
    # Save the dictionary to an HDF5 file
    with h5py.File(file_name, 'w') as hdf:
        for key, value in state.items():
            if len(value) > 1000:
                print(f"Compressing {key}")
                hdf.create_dataset(key, data=value, compression='gzip')
            else:
                hdf.create_dataset(key, data=value)

def set_state(om_problem: om.Problem, info: JoinedInfo, state: typing.Dict[str, np.array]):
    for ele in info.inputs+info.outputs:
        if ele.name in state:
            # We will only set state values if they are in the state dictionary.
            om_problem.set_val(ele.name, state[ele.name])

def load_state(om_problem: om.Problem, info: JoinedInfo, file_name: str):
    # Load the dictionary from the HDF5 file
    loaded_dict = {}
    with h5py.File(file_name, 'r') as hdf:
        for key in hdf.keys():
            loaded_dict[key] = hdf[key][:]
    set_state(om_problem, info, loaded_dict)


def get_interface(om_component: oms.System) -> JoinedInfo:
    my_evaluator, _ = get_interface_w_promotions(om_component)
    return(my_evaluator)

###### Methods to re-create an OpenMDAO assembly from a JoinedInfo class
def convert_enum(local_info: dict) -> Enum:
    """Create an Enum from the information that was stored in on JoinedInfo instance.
    This is undoing the work done in the AviaryEncoder

    Arguments:
        local_info {dict} -- Enum information generated by AviaryEncoder

    Returns:
        Enum -- Enum instance with the stored information
    """
    _require_aviary()
    results_module = local_info['__enum__']['module'].split("'")[1]
    results_class = local_info['__enum__']['type'].split("'")[1]
    results_value = local_info['__enum__']['value'].split("'")[1]
    local_library = importlib.import_module(results_module)
    local_class = getattr(local_library, results_class)
    return [local_class(results_value), 'unitless']

def convert_path(local_info: dict) -> pathlib.WindowsPath:
    """Create an WindowsPath from the information that was stored in on JoinedInfo instance.
    This is undoing the work done in the AviaryEncoder

    Arguments:
        local_info {dict} -- WindowsPath information generated by AviaryEncoder

    Returns:
        pathlib.WindowsPath -- WindowsPath instance with the stored information
    """
    local_info = pathlib.WindowsPath(local_info['__pathlib.WindowsPath__'])
    return(local_info)

def convert_engine_deck(info: dict) -> EngineDeck:
    """Create an EngineDeck from the information that was stored in on JoinedInfo instance.
    This is undoing the work done in the AviaryEncoder

    Arguments:
        info {dict} -- EngineDeck information generated by AviaryEncoder

    Returns:
        EngineDeck -- EngineDeck instance with the stored information
    """
    _require_aviary()
    local_info = copy.deepcopy(info)
    results_module = local_info[0]['__EngineDeck__']['module'].split("'")[1]
    results_class = local_info[0]['__EngineDeck__']['type'].split("'")[1]
    results_class = re.split(r"\.", results_class)[-1]
    print(f"Importing {results_class} from {results_module}")
    local_library = importlib.import_module(results_module)
    local_class = getattr(local_library, results_class)
    options = convert_aviary(local_info[0]['__EngineDeck__']['options']['__aviary_values__'])
    return local_class(options=options)


def process_value(value):
    # Define your custom processing logic here
    if isinstance(value, list):
        return [process_value(item) for item in value]
    elif isinstance(value, dict):
        if '__enum__' in value:
            return(convert_enum(value))
        elif '__pathlib.WindowsPath__' in value:
            return(convert_path(value))
        elif '__set__' in value:
            print("Translating set")
            return(set(value['__set__']))
        elif '__EngineDeck__' in value:
            return(convert_engine_deck(value))
        return iterate_nested_dict(value)
    else:
        return value

def iterate_nested_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            iterate_nested_dict(value)
        elif isinstance(value, list):
            d[key] = [process_value(item) for item in value]
        else:
            d[key] = process_value(value)
    return d

def convert_dict(local_av_options: dict) -> dict:
    # Update the dictionary
    return(iterate_nested_dict(local_av_options))

def convert_aviary(local_av_options: dict) -> AviaryValues:
    """Create an AviaryValues instance from the information that was stored in on JoinedInfo instance.
    This is undoing the work done in the AviaryEncoder

    Arguments:
        local_av_options {dict} -- AviaryValue information generated by AviaryEncoder

    Returns:
        AviaryValues -- AviaryValues instance with the stored information
    """
    _require_aviary()
    # Update the dictionary
    local_av_options=convert_dict(local_av_options)
    print("final:", local_av_options)
    return AviaryValues(local_av_options)

def create_openmdao_options(info_dict: dict) -> dict:
    """Create the information to generate the options passed to an OpenMDAO component

    Arguments:
        info_dict {dict} -- Dictionary with the information stored in the JoinedInfo instance

    Returns:
        dict -- The dictionary that is unrolled and passed to OpenMDAO component when instantiated
    """
    minimal_dict = {}
    # We only need the values from the options in `_dict`
    local_dict = info_dict['_dict']
    del_avairy = False
    for name, info in local_dict.items():
        if name == 'aviary_options':
            if '__aviary_values__' in info['val']:
                local_dict[name] = convert_aviary(info['val']['__aviary_values__'])
            else:
                del_avairy = True
        else:
            # Convert list values to tuples (JSON serialization turns tuples into lists)
            if isinstance(info['val'], list):
                local_dict[name] = tuple(info['val'])
            else:
                local_dict[name] = info['val']
    if del_avairy:
        del(local_dict['aviary_options'])
    return local_dict

def create_explicit_component(info : JoinedInfo) -> om.ExplicitComponent:
    """Add an OpenMDAO group to the passed in group based on the information in the JoinedInfo instance

    Args:
        model (om.Group): OpenMDAO group to add the new component to
        info (JoinedInfo): Information how to build the component
    """
    name = info.name
    print(f"Building class {name}")
    # Get the type of the component from the dictionary
    my_type = info.tool
    # Split the type information into package and class name
    results = re.split(r"\.", my_type.split("'")[1])
    package = ".".join(results[:-1])
    class_name = results[-1]
    print(f"Importing class {class_name} from {package}")
    # Import the library that defines the component
    local_library = importlib.import_module(package)
    local_class = getattr(local_library, class_name)
    openmdao_options = create_openmdao_options(copy.deepcopy(info.options['openmdao_options']))
    return(local_class(**openmdao_options))

def create_equation_component(info : JoinedInfo) -> om.ExecComp:
    """Add an OpenMDAO group to the passed in group based on the information in the JoinedInfo instance

    Args:
        model (om.Group): OpenMDAO group to add the new component to
        info (JoinedInfo): Information how to build the component
    """
    name = info.name
    print(f"Building Equation class {name}")
    openmdao_options = create_openmdao_options(copy.deepcopy(info.options['openmdao_options']))
    equations = info.equations
    # Add the information about shapes and units.
    for elements in info.inputs + info.outputs:
        temp_dict = {}
        if len(elements.options) > 0:
            if 'tags' in elements.options:
                temp_dict['tags'] = process_value(elements.options['tags'])
        if isinstance(elements, ArrayVariable):
            if not elements.shape is None:
                temp_dict['shape'] = elements.shape
        if not elements.units is None:
            temp_dict['units'] = elements.units
        if len(temp_dict) > 0:
            openmdao_options[elements.name] = temp_dict
    return(om.ExecComp(equations, **openmdao_options))

def clean_promotions(proms: list, name: str) -> list:
    # Remove duplicate elements
    proms = list(set(proms))
    # We also need to make sure that we avoid trying to promote
    # an element both with and without the name. It should only
    # be the promotion without the name
    grouped_tuples = defaultdict(list)
    for tup in proms:
        grouped_tuples[tup[0]].append(tup)
    # If we have two different types of promotions this means that one
    # component does promote the name (i.e. ('a', 'a')) and the other does not
    clean_proms = [key for key, value in grouped_tuples.items() if len(value) > 1]
    clean_proms = clean_proms + [value[0] for key, value in grouped_tuples.items() if len(value) == 1]

    # Filter out no-op promotions: ('var', 'name.var') where `name` is the
    # subsystem name. These are equivalent to not promoting (the variable is
    # already accessible as name.var without explicit promotion). Keeping them
    # creates dotted promoted names that trigger an infinite loop in
    # OpenMDAO >= 3.43's conn_graph.add_auto_ivc_nodes.
    filtered = []
    prefix = name + '.'
    for prom in clean_proms:
        if isinstance(prom, tuple):
            local_name, target_name = prom
            if target_name == prefix + local_name:
                # This is a no-op promotion — drop it
                continue
        filtered.append(prom)
    return filtered

def add_group(comp_info: JoinedInfo) -> oms.System:
    if comp_info.class_type == 'EquationInfo':
        final_component = create_equation_component(comp_info)
    elif comp_info.class_type == 'EvaluatorInfo':
        final_component = create_explicit_component(comp_info)
    else:
        final_component = om.Group()
        for local_name in comp_info.component_order:
            local_info = comp_info.components[local_name]
            if local_info.class_type == 'GroupInfo': 
                print(f"** This is a group, special handling needed for {local_name} in group {comp_info.name}")
                component = add_group(comp_info=local_info)
            else:
                # This is a component, so we directly create it.
                print(f"Adding {local_name} component")
                component = add_group(local_info)
            final_component.add_subsystem(name=local_name,
                subsys=component,
                promotes=clean_promotions(comp_info.promotions[local_name], local_name))
        # Now create the linkages between components
        linkage = comp_info.linkage
        for local_link in linkage:
            final_component.connect(local_link[0], local_link[1])
    return(final_component)

def create_problem(info: JoinedInfo, state: typing.Dict[str, np.array] = {}, opt_problem: OptProblem = None) -> om.Problem:
    """ Create an OpenMDAO problem based on the information in the JoinedInfo instance"""
    component = add_group(info)

    # prob = om.Problem(my_group)
    prob = om.Problem(component)
    if opt_problem is not None:
        # If an optimization problem is passed in we set it.
        set_opt_problem(om_problem=prob, opt_problem=opt_problem, run_setup=False)
    prob.setup()

    # Set the state of all he elements in the model if state was defined.
    if len(state) > 0:
        set_state(prob, info, state)

    prob.final_setup()
    return prob


def show_structure(component_info: JoinedInfo, indent : int=0, counter=0):
    """ Show the structure of the assembly stored in a JoinedInfo instance.

    This method calls itself recursively to traverse the full tree

    Arguments:
        component_info {JoinedInfo} -- The JoinedInfo instance to explore

    Keyword Arguments:
        indent {int} -- Indent for printing. Allows tree like display (default: {0})
    """
    counter = counter + 1
    print(' '*indent,counter, component_info.name, len(component_info.inputs), 
            len(component_info.outputs))
    input_names = [x.name for x in component_info.inputs]
    output_names = [x.name for x in component_info.outputs]
    print(' '*indent, f"Inputs:  {input_names}")
    print(' '*indent, f"Outputs: {output_names}")
    if component_info.class_type == "GroupInfo":
        print(' '*indent, f"Promotions: {component_info.promotions}")
        print(' '*indent, f"Linkage   : {component_info.linkage}")
    if component_info.class_type == 'GroupInfo':
        for name in component_info.component_order:
            counter = show_structure(component_info=component_info.components[name], indent=indent+2, counter=counter)
    return counter

def save_assembly(problem: om.Problem, assembly_name: str, state_name: str):
    info = get_interface(problem.model)

    # Convert and write JSON object to file
    with open(assembly_name, "w") as outfile:
        json.dump(info.model_dump(exclude_unset=True), outfile, indent=2, skipkeys=True, cls=AviaryEncoder, )

    # Save the state in the state file
    save_state(problem, info, state_name)

def load_assembly(assembly_name: str, state_name: str) -> om.Problem:
    with open(assembly_name, "r") as infile:
        info_dict = json_numpy.load(infile)
        info_dict = convert_dict(info_dict)
        info = GroupInfo.model_validate(info_dict)
    problem = create_problem(info)
    # We want to make sure we use the actual info from the problem
    info = get_interface(problem.model)
    load_state(problem, info, state_name)
    return(problem)

def convert_om_opt_vars(info: dict) -> Variable:
    optional_fields_array = ['cache_liner_solutions',  'parallel_deriv_color',
        'distributed', 'global_size' ]
    if info['scaler'] is None:
        scale = 1.0
    else:
        scale = info['scaler']
    if info['adder'] is None:
        shift = 0.0
    else:
        shift = info['adder']
    if 'lower' in info:
        if info['lower'] is None:
            lower = None
        else:
            lower = info['lower'] / scale - shift
    else:
        lower = None
    if 'upper' in info:
        if info['upper'] is None:
            upper = None
        else:
            upper = info['upper'] / scale - shift
    else:
        upper = None
    bounds = (lower, upper)
    if bounds == (None, None):
        bounds = None
    input_dict = {
            'name': info['name'],
            'units': info['units'],
            'shift': shift,
            'scale': scale,
            'bounds': bounds,
            }
    additional_dict = {}
    for field in optional_fields_array:
        if field in info:
            additional_dict[field] = info[field]
    input_dict['options'] = additional_dict
    if info['size'] == 1:
        # This is a float variable. Store it as such
        output = FloatVariable.model_validate(input_dict)
    else:
        # This is an array variable, and we need to capture its shape and
        # additional information
        print("Warning: More work is needed to support array variables")
        input_dict['shape'] = (info['size'],)
        output = ArrayVariable.model_validate(input_dict)
    return(output)

def get_opt_problem(om_component: oms.System) -> OptProblem:
    # Right now we need to set driver_scaling to True since otherwise the bounds get modified. This is a bug in OpenMDAO
    opt_list = ['ref', 'ref0', 'indices', 'adder', 'scaler', 'parallel_deriv_color', 
            'cache_linear_solution', 'units', 'lower', 'upper', ]
    obj_list = opt_list[:-2]
    local_dict = om_component.list_driver_vars(show_promoted_name=True, driver_scaling=True, desvar_opts=opt_list, 
        cons_opts=opt_list, objs_opts=obj_list, out_stream=None)
    vars = []
    responses = []
    for info in local_dict['design_vars']:
        vars.append(convert_om_opt_vars(info[1]))
    constraints = []
    for info in local_dict['constraints']:
        responses.append(convert_om_opt_vars(info[1]))
        constraints.append(info[0])
    objectives = []
    for info in local_dict['objectives']:
        responses.append(convert_om_opt_vars(info[1]))
        objectives.append(info[0])
    my_problem = OptProblem(variables=vars, responses=responses, constraints=constraints, objectives=objectives)
    return(my_problem)


def set_opt_problem(om_problem: om.Problem, opt_problem: OptProblem, run_setup: bool=True):
    # Adding all the variables
    for my_variable in opt_problem.variables:
        om_problem.model.add_design_var(my_variable.name, lower=my_variable.bounds[0], upper=my_variable.bounds[1], 
            scaler=my_variable.scale, adder=my_variable.shift)

    # Extract the information for the responses that will be used for the objectives and responses
    resp_dict = {}
    for my_resp in opt_problem.responses:
        resp_dict[my_resp.name] = {
            'lower': my_resp.bounds[0],
            'upper': my_resp.bounds[1],
            'scaler': my_resp.scale,
            'adder': my_resp.shift,
        }

    # Set the constraints for the problem
    for constraint_name in opt_problem.constraints:
        om_problem.model.add_constraint(constraint_name, **(resp_dict[constraint_name]))

    # Set the objectives
    for obj_name in opt_problem.objectives:
        om_problem.model.add_objective(obj_name, scaler=resp_dict[obj_name]['scaler'], adder=resp_dict[obj_name]['adder'])

    # For the changes to take effect we need to run setup() and final_setup()
    # We allow this step to be switched off
    if run_setup:
        om_problem.setup()
        om_problem.final_setup()


def set_variable_bounds(info: JoinedInfo, bounds: typing.Dict[str, Tuple[float, float]]) -> None:
    """Set bounds on the input variables of a JoinedInfo interface description.

    This modifies the info in place. Only variables whose names match keys in the
    bounds dictionary are updated; others are left unchanged.

    Arguments:
        info {JoinedInfo} -- The interface description to update.
        bounds {dict} -- Dictionary mapping variable names to (lower, upper) tuples.

    Example:
        >>> set_variable_bounds(aero_info, {'rho': (0.5, 2.0), 'v': (10, 100)})
    """
    for var in info.inputs:
        if var.name in bounds:
            var.bounds = bounds[var.name]


def build_opt_problem(info: JoinedInfo, om_prob: om.Problem = None) -> OptProblem:
    """Build an OptProblem from a JoinedInfo interface description.

    Creates an OptProblem where the info's inputs become variables and the
    info's outputs become responses. Bounds, units, defaults, and scaling
    from the Variable objects are preserved.

    Variables and responses tagged as 'internal' (via the options['tags'] field)
    are excluded since they represent intermediate values not meant to be
    externally set or observed.

    When an OpenMDAO Problem is provided, inputs that are driven by another
    component within the model (i.e., their source is NOT _auto_ivc) are also
    excluded from the variables list, since they are internal connections
    rather than free inputs.

    Arguments:
        info {JoinedInfo} -- The interface description to convert.
        om_prob {om.Problem, optional} -- A setup'd OpenMDAO Problem. When
            provided, used to determine which inputs are true free variables
            vs internal connections. Defaults to None.

    Returns:
        OptProblem -- An optimization problem with variables and responses
            derived from the interface description.
    """
    def _is_internal(var) -> bool:
        """Check if a variable has the 'internal' tag."""
        tags = var.options.get('tags', set())
        if isinstance(tags, set):
            return 'internal' in tags
        elif isinstance(tags, (list, tuple)):
            return 'internal' in tags
        return False

    def _is_free_input(var) -> bool:
        """Check if an input is a free variable (sourced by auto_ivc)."""
        if om_prob is None:
            # Without a live problem, we can't check — assume it's free
            return True
        try:
            source = om_prob.model.get_source(var.name)
            return "_auto_ivc" in source
        except (KeyError, RuntimeError):
            # Variable not found in the model — skip it
            return False

    variables = [
        copy.deepcopy(v) for v in info.inputs
        if not _is_internal(v) and _is_free_input(v)
    ]
    responses = [copy.deepcopy(v) for v in info.outputs if not _is_internal(v)]
    return OptProblem(
        name=info.name or "from_interface",
        variables=variables,
        responses=responses,
    )
