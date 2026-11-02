"""Dependency tracing and automated assembly pruning utilities.

Enables identifying upstream components required to compute specific outputs,
downstream components impacted by input modifications, and pruning of
framework-independent model dictionaries for study-specific workflows.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import networkx as nx


def _extract_connections(model_dict: dict) -> List[dict]:
    """Extract connections_list from either model_dict or model_dict['model_data']."""
    if "model_data" in model_dict and isinstance(model_dict["model_data"], dict):
        return model_dict["model_data"].get("connections_list", [])
    return model_dict.get("connections_list", [])


def _extract_sys_pathnames(model_dict: dict) -> List[str]:
    """Extract sys_pathnames_list from either model_dict or model_dict['model_data']."""
    if "model_data" in model_dict and isinstance(model_dict["model_data"], dict):
        return model_dict["model_data"].get("sys_pathnames_list", [])
    return model_dict.get("sys_pathnames_list", [])


def _collect_tree_components(tree_node: dict) -> Set[str]:
    """Recursively collect component and subsystem names from the tree."""
    components = set()
    if not isinstance(tree_node, dict):
        return components

    name = tree_node.get("name")
    subsystem_type = tree_node.get("subsystem_type")
    node_type = tree_node.get("type")

    # If this node represents a component or subsystem
    if name and (subsystem_type == "component" or node_type == "subsystem"):
        components.add(name)

    for child in tree_node.get("children", []):
        components.update(_collect_tree_components(child))

    return components


def _get_known_components(model_dict: dict) -> Set[str]:
    """Collect all known component/subsystem names in the model."""
    known = set(_extract_sys_pathnames(model_dict))

    # Also inspect tree if present
    tree = None
    if "model_data" in model_dict and isinstance(model_dict["model_data"], dict):
        tree = model_dict["model_data"].get("tree")
    elif "tree" in model_dict:
        tree = model_dict.get("tree")

    if tree:
        known.update(_collect_tree_components(tree))

    # Infer from connections if known is still empty
    if not known:
        for conn in _extract_connections(model_dict):
            src = conn.get("src", "")
            tgt = conn.get("tgt", "")
            for var in (src, tgt):
                if "." in var:
                    comp, _ = var.rsplit(".", 1)
                    if comp:
                        known.add(comp)

    return known


def resolve_component(var_or_comp: str, known_components: Set[str]) -> Optional[str]:
    """Match a variable or component name to a known component pathname.

    Matches exact component name first, then longest matching component prefix.
    """
    if not var_or_comp:
        return None

    if var_or_comp in known_components:
        return var_or_comp

    # Longest matching prefix
    candidates = [
        comp for comp in known_components
        if var_or_comp.startswith(comp + ".") or var_or_comp.endswith("." + comp) or var_or_comp == comp
    ]
    if candidates:
        # Prefer longest prefix
        prefix_matches = [c for c in candidates if var_or_comp.startswith(c + ".")]
        if prefix_matches:
            return max(prefix_matches, key=len)
        return max(candidates, key=len)

    # Fallback to rsplit by last dot
    if "." in var_or_comp:
        return var_or_comp.rsplit(".", 1)[0]

    return var_or_comp


def build_dependency_graph(model_dict: dict) -> Tuple[nx.DiGraph, Set[str]]:
    """Build a directed dependency graph from model connections.

    In the returned graph:
    - Nodes represent component pathnames.
    - An edge (A -> B) means component B depends on component A (data flows A -> B).
    - Edge attributes store the associated connection details (src variable, tgt variable).

    Parameters
    ----------
    model_dict : dict
        OpenMDAO system description dictionary.

    Returns
    -------
    graph : nx.DiGraph
        Directed graph representing component dependencies.
    known_components : Set[str]
        Set of all known component names.
    """
    graph = nx.DiGraph()
    known_components = _get_known_components(model_dict)

    for comp in known_components:
        graph.add_node(comp)

    connections = _extract_connections(model_dict)
    for conn in connections:
        src = conn.get("src", "")
        tgt = conn.get("tgt", "")
        if not src or not tgt:
            continue

        src_comp = resolve_component(src, known_components)
        tgt_comp = resolve_component(tgt, known_components)

        if src_comp and tgt_comp and src_comp != tgt_comp:
            graph.add_node(src_comp)
            graph.add_node(tgt_comp)
            if graph.has_edge(src_comp, tgt_comp):
                graph[src_comp][tgt_comp]["connections"].append((src, tgt))
            else:
                graph.add_edge(src_comp, tgt_comp, connections=[(src, tgt)])

    return graph, known_components


def trace_required_components(
    model_dict: dict,
    outputs: Sequence[str],
) -> List[str]:
    """Identify all components required to compute the specified outputs.

    Computes the upstream closure: starts with the components that own the
    requested output variables, and traverses backwards along data-flow edges
    to find all predecessor components required for calculation.

    Parameters
    ----------
    model_dict : dict
        OpenMDAO system description dictionary.
    outputs : Sequence[str]
        List of output variable pathnames (e.g. 'aero.drag') or component names.

    Returns
    -------
    List[str]
        Sorted list of component pathnames required to calculate the outputs.
    """
    graph, known_components = build_dependency_graph(model_dict)

    required: Set[str] = set()
    to_visit: List[str] = []

    for out in outputs:
        comp = resolve_component(out, known_components)
        if comp:
            required.add(comp)
            to_visit.append(comp)

    # Traverse backwards (predecessors in data-flow graph)
    while to_visit:
        curr = to_visit.pop(0)
        if curr in graph:
            for pred in graph.predecessors(curr):
                if pred not in required:
                    required.add(pred)
                    to_visit.append(pred)

    return sorted(required)


def trace_impacted_components(
    model_dict: dict,
    inputs: Sequence[str],
) -> List[str]:
    """Identify all components impacted by changes to the specified inputs.

    Computes the downstream closure: starts with the components that own the
    specified input variables, and traverses forward along data-flow edges to
    find all successor components affected by the input modifications.

    Parameters
    ----------
    model_dict : dict
        OpenMDAO system description dictionary.
    inputs : Sequence[str]
        List of input variable pathnames (e.g. 'geometry.q') or component names.

    Returns
    -------
    List[str]
        Sorted list of component pathnames impacted by the inputs.
    """
    graph, known_components = build_dependency_graph(model_dict)

    impacted: Set[str] = set()
    to_visit: List[str] = []

    for inp in inputs:
        comp = resolve_component(inp, known_components)
        if comp:
            impacted.add(comp)
            to_visit.append(comp)

    # Traverse forward (successors in data-flow graph)
    while to_visit:
        curr = to_visit.pop(0)
        if curr in graph:
            for succ in graph.successors(curr):
                if succ not in impacted:
                    impacted.add(succ)
                    to_visit.append(succ)

    return sorted(impacted)


def explain_dependencies(
    model_dict: dict,
    outputs: Sequence[str],
) -> Dict[str, Any]:
    """Provide a structured explanation of why each retained component is needed.

    Parameters
    ----------
    model_dict : dict
        OpenMDAO system description dictionary.
    outputs : Sequence[str]
        Requested target outputs.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping each required component to its downstream consumers,
        direct target outputs, and connecting variable pairs.
    """
    graph, known_components = build_dependency_graph(model_dict)
    required_comps = set(trace_required_components(model_dict, outputs))

    direct_targets = set()
    for out in outputs:
        comp = resolve_component(out, known_components)
        if comp:
            direct_targets.add(comp)

    explanation: Dict[str, Any] = {}
    for comp in sorted(required_comps):
        consumers = []
        conns_out = []
        if comp in graph:
            for succ in graph.successors(comp):
                if succ in required_comps:
                    consumers.append(succ)
                    conns_out.extend(graph[comp][succ].get("connections", []))

        is_direct = comp in direct_targets
        reason = "Direct provider of requested output" if is_direct else f"Supplies data to {', '.join(consumers)}"

        explanation[comp] = {
            "is_direct_target": is_direct,
            "downstream_consumers": consumers,
            "outgoing_connections": [{"src": s, "tgt": t} for s, t in conns_out],
            "reason": reason,
        }

    return explanation


def _prune_tree_node(node: dict, required_components: Set[str], preserve_parents: bool = True) -> Optional[dict]:
    """Recursively prune tree node to retain only required components and ancestors."""
    if not isinstance(node, dict):
        return None

    node_copy = copy.deepcopy(node)
    name = node_copy.get("name", "")

    # If this is a leaf component
    subsystem_type = node_copy.get("subsystem_type")
    if subsystem_type == "component":
        if name in required_components:
            return node_copy
        return None

    # For groups / containers, recursively prune children
    children = node_copy.get("children", [])
    pruned_children = []
    for child in children:
        pruned_child = _prune_tree_node(child, required_components, preserve_parents)
        if pruned_child is not None:
            pruned_children.append(pruned_child)

    if pruned_children or (name in required_components and preserve_parents):
        node_copy["children"] = pruned_children
        return node_copy

    return None


def prune_model_dict(
    model_dict: dict,
    outputs: Sequence[str],
    preserve_parents: bool = True,
) -> dict:
    """Prune an OpenMDAO model dictionary to retain only components needed for outputs.

    Automatically removes unneeded components, dangling connections, irrelevant
    sys_pathnames, and unused tree subtrees while preserving required parent group
    structures and valid model metadata.

    Parameters
    ----------
    model_dict : dict
        Original OpenMDAO system description dictionary.
    outputs : Sequence[str]
        Target output variables or component names to retain.
    preserve_parents : bool, optional
        Whether to preserve parent group hierarchy nodes for retained components,
        by default True.

    Returns
    -------
    dict
        Pruned deep-copy of model_dict containing only necessary components and connections.
    """
    required_components = set(trace_required_components(model_dict, outputs))
    pruned = copy.deepcopy(model_dict)

    target_data = pruned.get("model_data") if "model_data" in pruned else pruned

    # Filter connections_list
    orig_conns = target_data.get("connections_list", [])
    pruned_conns = []
    for conn in orig_conns:
        src = conn.get("src", "")
        tgt = conn.get("tgt", "")
        src_comp = resolve_component(src, required_components)
        tgt_comp = resolve_component(tgt, required_components)
        if src_comp in required_components and tgt_comp in required_components:
            pruned_conns.append(conn)
    target_data["connections_list"] = pruned_conns

    # Filter sys_pathnames_list
    if "sys_pathnames_list" in target_data:
        target_data["sys_pathnames_list"] = [
            p for p in target_data["sys_pathnames_list"]
            if p in required_components or (preserve_parents and any(c.startswith(p + ".") for c in required_components))
        ]

    # Filter tree
    if "tree" in target_data:
        pruned_tree = _prune_tree_node(target_data["tree"], required_components, preserve_parents)
        if pruned_tree:
            target_data["tree"] = pruned_tree

    # Filter responses / objectives
    if "responses" in target_data and isinstance(target_data["responses"], dict):
        target_data["responses"] = {
            k: v for k, v in target_data["responses"].items()
            if resolve_component(k, required_components) in required_components
        }

    # Filter design_vars
    if "design_vars" in target_data and isinstance(target_data["design_vars"], dict):
        target_data["design_vars"] = {
            k: v for k, v in target_data["design_vars"].items()
            if resolve_component(k, required_components) in required_components
        }

    # Filter Optional additional component options
    if "Optional" in pruned and isinstance(pruned["Optional"], dict):
        pruned["Optional"] = {
            k: v for k, v in pruned["Optional"].items()
            if k in required_components
        }

    return pruned
