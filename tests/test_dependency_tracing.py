"""Unit tests for dependency tracing and model pruning utilities.

Tests cover:
- build_dependency_graph
- trace_required_components
- trace_impacted_components
- explain_dependencies
- prune_model_dict
"""

import pytest
from standard_evaluator.utilities.dependency_tracing import (
    build_dependency_graph,
    trace_required_components,
    trace_impacted_components,
    explain_dependencies,
    prune_model_dict,
)


@pytest.fixture
def multi_branch_model():
    """A multi-branch model description with two independent sub-pipelines and one shared input."""
    return {
        "model_data": {
            "connections_list": [
                # Branch 1: aero -> propulsion -> performance
                {"src": "aero_comp.lift", "tgt": "prop_comp.drag_in"},
                {"src": "aero_comp.drag", "tgt": "perf_comp.drag_total"},
                {"src": "prop_comp.thrust", "tgt": "perf_comp.thrust_in"},
                # Branch 2: structures -> weights (independent)
                {"src": "struct_comp.mass_wing", "tgt": "weights_comp.wing_mass_in"},
                {"src": "struct_comp.mass_fuse", "tgt": "weights_comp.fuse_mass_in"},
            ],
            "sys_pathnames_list": [
                "aero_comp",
                "prop_comp",
                "perf_comp",
                "struct_comp",
                "weights_comp",
            ],
            "tree": {
                "name": "root",
                "type": "root",
                "children": [
                    {
                        "name": "aero_group",
                        "type": "subsystem",
                        "subsystem_type": "group",
                        "children": [
                            {"name": "aero_comp", "type": "subsystem", "subsystem_type": "component", "children": []},
                            {"name": "prop_comp", "type": "subsystem", "subsystem_type": "component", "children": []},
                        ],
                    },
                    {
                        "name": "perf_group",
                        "type": "subsystem",
                        "subsystem_type": "group",
                        "children": [
                            {"name": "perf_comp", "type": "subsystem", "subsystem_type": "component", "children": []},
                        ],
                    },
                    {
                        "name": "struct_group",
                        "type": "subsystem",
                        "subsystem_type": "group",
                        "children": [
                            {"name": "struct_comp", "type": "subsystem", "subsystem_type": "component", "children": []},
                            {"name": "weights_comp", "type": "subsystem", "subsystem_type": "component", "children": []},
                        ],
                    },
                ],
            },
            "responses": {
                "perf_comp.range": {"type": "objective"},
                "weights_comp.total_weight": {"type": "constraint"},
            },
            "design_vars": {
                "aero_comp.aoa": {},
                "struct_comp.thickness": {},
            },
        },
        "Optional": {
            "aero_comp": {"solver": "vlm"},
            "struct_comp": {"material": "aluminum"},
        },
    }


@pytest.fixture
def coupled_cyclic_model():
    """A model with cyclic feedback between two components feeding an output component."""
    return {
        "model_data": {
            "connections_list": [
                {"src": "comp_a.out1", "tgt": "comp_b.in1"},
                {"src": "comp_b.out1", "tgt": "comp_a.in1"},
                {"src": "comp_b.out2", "tgt": "comp_c.in1"},
            ],
            "sys_pathnames_list": ["comp_a", "comp_b", "comp_c", "comp_d"],
        }
    }


def test_trace_required_linear_branch(multi_branch_model):
    """Tracing requirements for perf_comp should only pull in aero_comp and prop_comp."""
    required = trace_required_components(multi_branch_model, outputs=["perf_comp.range"])
    assert required == ["aero_comp", "perf_comp", "prop_comp"]
    assert "struct_comp" not in required
    assert "weights_comp" not in required


def test_trace_required_weights_branch(multi_branch_model):
    """Tracing requirements for weights_comp should only pull in struct_comp."""
    required = trace_required_components(multi_branch_model, outputs=["weights_comp.total_weight"])
    assert required == ["struct_comp", "weights_comp"]
    assert "aero_comp" not in required
    assert "prop_comp" not in required
    assert "perf_comp" not in required


def test_trace_impacted_components(multi_branch_model):
    """Changing aero_comp.aoa should impact prop_comp and perf_comp, but not structures."""
    impacted = trace_impacted_components(multi_branch_model, inputs=["aero_comp.aoa"])
    assert impacted == ["aero_comp", "perf_comp", "prop_comp"]
    assert "struct_comp" not in impacted
    assert "weights_comp" not in impacted


def test_trace_coupled_cycle(coupled_cyclic_model):
    """In a feedback loop, tracing comp_c requires comp_b and comp_a, but not isolated comp_d."""
    required = trace_required_components(coupled_cyclic_model, outputs=["comp_c.in1"])
    assert required == ["comp_a", "comp_b", "comp_c"]
    assert "comp_d" not in required


def test_explain_dependencies(multi_branch_model):
    """Explanations correctly distinguish direct targets from upstream suppliers."""
    explanation = explain_dependencies(multi_branch_model, outputs=["perf_comp.range"])
    assert "perf_comp" in explanation
    assert explanation["perf_comp"]["is_direct_target"] is True

    assert "aero_comp" in explanation
    assert explanation["aero_comp"]["is_direct_target"] is False
    assert "perf_comp" in explanation["aero_comp"]["downstream_consumers"]
    assert "prop_comp" in explanation["aero_comp"]["downstream_consumers"]


def test_prune_model_dict(multi_branch_model):
    """Pruning removes unneeded components, connections, tree branches, and metadata."""
    pruned = prune_model_dict(multi_branch_model, outputs=["perf_comp.range"])
    model_data = pruned["model_data"]

    # Only aero, prop, and perf components retained in pathnames
    assert "struct_comp" not in model_data["sys_pathnames_list"]
    assert "weights_comp" not in model_data["sys_pathnames_list"]
    assert "perf_comp" in model_data["sys_pathnames_list"]

    # Connections to/from struct and weights components removed
    for conn in model_data["connections_list"]:
        assert "struct_comp" not in conn["src"] and "struct_comp" not in conn["tgt"]
        assert "weights_comp" not in conn["src"] and "weights_comp" not in conn["tgt"]

    # Responses pruned
    assert "perf_comp.range" in model_data["responses"]
    assert "weights_comp.total_weight" not in model_data["responses"]

    # Design vars pruned
    assert "aero_comp.aoa" in model_data["design_vars"]
    assert "struct_comp.thickness" not in model_data["design_vars"]

    # Optional options pruned
    assert "aero_comp" in pruned["Optional"]
    assert "struct_comp" not in pruned["Optional"]


def test_prune_tree_structure(multi_branch_model):
    """Pruning tree retains ancestor groups of retained components and removes unneeded groups."""
    pruned = prune_model_dict(multi_branch_model, outputs=["perf_comp.range"])
    tree = pruned["model_data"]["tree"]

    group_names = [child["name"] for child in tree["children"]]
    assert "aero_group" in group_names
    assert "perf_group" in group_names
    assert "struct_group" not in group_names
