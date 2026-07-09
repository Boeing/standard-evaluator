"""Integration tests for the replace-group workflow.

Tests that replacing a sub-group in a model's interface description with a
simplified component/group works correctly through create_problem(), including
the fix for OpenMDAO >= 3.43 compatibility.

Requirements: 2.4
"""

import numpy as np
import openmdao.api as om

import standard_evaluator as se


# ---------------------------------------------------------------------------
# Fixtures: Model definitions
# ---------------------------------------------------------------------------

GRID_SIZE = 20


class _Sub1(om.Group):
    def setup(self):
        self.add_subsystem('z1', om.ExecComp("y = a + b + c + d"),
                           promotes_inputs=['a', ('b', 'my_alias'), 'd'])
        self.add_subsystem('z2', om.ExecComp("v = a + b + c + d"),
                           promotes_inputs=['c', 'd'],
                           promotes_outputs=['*'])
        self.connect('z1.y', 'c')


class _Aero(om.Group):
    def setup(self):
        self.add_subsystem('geometry', om.ExecComp(
            [f"grid = (q + z *l)*ones({GRID_SIZE})",
             f"bla = outer(ones({GRID_SIZE}), ones({GRID_SIZE}))"],
            grid={'tags': 'internal', 'shape': (GRID_SIZE,)},
            bla={'tags': 'internal', 'shape': (GRID_SIZE, GRID_SIZE)}
        ), promotes_inputs=['q', 'z', 'l'])
        self.add_subsystem('cfd', om.ExecComp(
            f"drag = (inner(grid, ones({GRID_SIZE})) + r * k)/1000",
            grid={'tags': 'internal', 'shape_by_conn': True},
            drag={'units': 'N'}
        ), promotes_inputs=['r', 'k'], promotes_outputs=['drag'])
        self.connect('geometry.grid', 'cfd.grid')


class _OriginalModel(om.Group):
    def setup(self):
        self.add_subsystem('sub1', _Sub1(), promotes_inputs=['a'])
        self.add_subsystem('aero', _Aero())
        self.connect('sub1.v', 'aero.k')


class _Surrogate(om.Group):
    def setup(self):
        self.add_subsystem('aero',
                           om.ExecComp("drag = ((q + z *l)*20 + r * k)/1000",
                                       drag={'units': 'N'}),
                           promotes_inputs=['q', 'z', 'l', 'r', 'k'],
                           promotes_outputs=['drag'])


def _build_original_problem():
    """Build and setup the original model."""
    prob = om.Problem(model=_OriginalModel())
    prob.setup()
    prob.final_setup()
    prob.run_model()
    return prob


def _build_surrogate_problem():
    """Build and setup the surrogate model standalone."""
    prob = om.Problem(model=_Surrogate())
    prob.setup()
    return prob


# ---------------------------------------------------------------------------
# Tests: get_external_names auto-IVC filtering
# ---------------------------------------------------------------------------


class TestGetExternalNamesAutoIvcFiltering:
    """Test that auto-IVC outputs are excluded from interface descriptions."""

    def test_standalone_group_outputs_exclude_auto_ivc(self):
        """When get_interface is called on a standalone group, auto-IVC
        variables should not appear in the outputs list."""
        prob = _build_surrogate_problem()
        info = se.get_interface(prob.model)

        output_names = [v.name for v in info.outputs]
        assert 'drag' in output_names
        # These are auto-IVC outputs that should NOT be captured
        for name in ['k', 'l', 'q', 'r', 'z']:
            assert name not in output_names, (
                f"Auto-IVC variable '{name}' should not appear in outputs"
            )

    def test_standalone_group_inputs_are_preserved(self):
        """Inputs of a standalone group should still be captured."""
        prob = _build_surrogate_problem()
        info = se.get_interface(prob.model)

        input_names = [v.name for v in info.inputs]
        for name in ['k', 'l', 'q', 'r', 'z']:
            assert name in input_names, (
                f"Input '{name}' should appear in interface inputs"
            )

    def test_nested_group_outputs_unchanged(self):
        """For a nested group (not top-level), outputs should be unaffected
        since auto-IVC only exists at the top level."""
        prob = _build_original_problem()
        info = se.get_interface(prob.model)
        aero_info = info.components['aero']

        output_names = [v.name for v in aero_info.outputs]
        assert 'drag' in output_names


# ---------------------------------------------------------------------------
# Tests: Replace group workflow
# ---------------------------------------------------------------------------


class TestReplaceGroupWorkflow:
    """Integration tests for replacing a sub-group with a surrogate."""

    def test_create_problem_does_not_hang(self):
        """create_problem completes without infinite loop after group replacement."""
        prob = _build_original_problem()
        info = se.get_interface(prob.model)

        prob_replace = _build_surrogate_problem()
        surrogate_info = se.get_interface(prob_replace.model)

        info.components['aero'] = surrogate_info

        # This should complete without hanging
        new_prob = se.create_problem(info)
        assert new_prob is not None

    def test_replaced_model_produces_same_drag(self):
        """The surrogate produces the same drag value as the original model."""
        prob = _build_original_problem()
        info = se.get_interface(prob.model)
        state = se.get_state(prob, info)

        prob_replace = _build_surrogate_problem()
        surrogate_info = se.get_interface(prob_replace.model)

        info.components['aero'] = surrogate_info
        new_prob = se.create_problem(info)

        # Set matching inputs on both problems
        inputs = {
            'a': 34., 'sub1.my_alias': 5., 'sub1.z1.c': 24.,
            'sub1.d': 3., 'sub1.c': 12., 'aero.l': 400.,
            'aero.q': 5., 'aero.r': 6., 'aero.z': 7.,
        }
        for name, val in inputs.items():
            prob.set_val(name, val)
        prob.run_model()

        new_state = se.get_state(prob, se.get_interface(prob.model))
        new_info = se.get_interface(new_prob.model)
        se.set_state(new_prob, new_info, new_state)
        new_prob.run_model()

        orig_drag = prob.get_val('aero.drag')[0]
        new_drag = new_prob.get_val('aero.drag')[0]
        np.testing.assert_allclose(new_drag, orig_drag, rtol=1e-10,
                                   err_msg="Surrogate drag should match original")

    def test_replaced_model_inputs_accessible(self):
        """All expected inputs are accessible in the replaced model."""
        prob = _build_original_problem()
        info = se.get_interface(prob.model)

        prob_replace = _build_surrogate_problem()
        surrogate_info = se.get_interface(prob_replace.model)

        info.components['aero'] = surrogate_info
        new_prob = se.create_problem(info)

        # These should all be settable without error
        new_prob.set_val('a', 1.0)
        new_prob.set_val('sub1.my_alias', 2.0)
        new_prob.set_val('sub1.d', 3.0)
        new_prob.set_val('aero.k', 4.0)
        new_prob.set_val('aero.r', 5.0)
        new_prob.set_val('aero.l', 6.0)
        new_prob.set_val('aero.q', 7.0)
        new_prob.set_val('aero.z', 8.0)

    def test_replaced_model_connections_work(self):
        """The connect('sub1.v', 'aero.k') linkage works in the replaced model."""
        prob = _build_original_problem()
        info = se.get_interface(prob.model)

        prob_replace = _build_surrogate_problem()
        surrogate_info = se.get_interface(prob_replace.model)

        info.components['aero'] = surrogate_info
        new_prob = se.create_problem(info)

        # Set inputs that feed into sub1.v = a + b + c + d
        new_prob.set_val('a', 10.0)
        new_prob.set_val('sub1.my_alias', 5.0)
        new_prob.set_val('sub1.z1.c', 3.0)
        new_prob.set_val('sub1.d', 2.0)
        new_prob.run_model()

        # sub1.z1.y = a + my_alias + z1.c + d = 10 + 5 + 3 + 2 = 20
        # sub1.v (z2) = z2.a + z2.b + c + d, where c = z1.y = 20
        # z2.a and z2.b default to 1.0 each, d = 2.0
        # sub1.v = 1 + 1 + 20 + 2 = 24
        # aero.k should equal sub1.v via the connection
        sub1_v = new_prob.get_val('sub1.v')[0]
        aero_k = new_prob.get_val('aero.k')[0]
        np.testing.assert_allclose(aero_k, sub1_v, rtol=1e-10,
                                   err_msg="aero.k should be connected to sub1.v")
