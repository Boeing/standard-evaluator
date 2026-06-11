"""
Unit tests for Evaluator base class imports and interface.

Validates:
- Evaluator is importable from standard_evaluator.evaluators.abstract_evaluator
- Public methods and properties exist on the class
- No boeing_standard_evaluator in import chain

Requirements: 2.5, 2.6
"""

import sys
import inspect
from abc import ABC

import pytest


class TestEvaluatorImport:
    """Verify the Evaluator class is importable from the expected module path."""

    def test_import_from_abstract_evaluator_module(self):
        """Evaluator is importable from standard_evaluator.evaluators.abstract_evaluator."""
        from standard_evaluator.evaluators.abstract_evaluator import Evaluator

        assert Evaluator is not None

    def test_evaluator_is_abc(self):
        """Evaluator inherits from ABC."""
        from standard_evaluator.evaluators.abstract_evaluator import Evaluator

        assert issubclass(Evaluator, ABC)


class TestEvaluatorPublicMethods:
    """Verify that all public methods exist on the Evaluator class."""

    @pytest.fixture
    def evaluator_cls(self):
        from standard_evaluator.evaluators.abstract_evaluator import Evaluator
        return Evaluator

    def test_call_method_exists(self, evaluator_cls):
        """__call__ method is defined on Evaluator."""
        assert hasattr(evaluator_cls, "__call__")
        assert callable(getattr(evaluator_cls, "__call__"))

    def test_eval_np_method_exists(self, evaluator_cls):
        """eval_np method is defined on Evaluator."""
        assert hasattr(evaluator_cls, "eval_np")
        assert callable(getattr(evaluator_cls, "eval_np"))

    def test_eval_list_method_exists(self, evaluator_cls):
        """eval_list method is defined on Evaluator."""
        assert hasattr(evaluator_cls, "eval_list")
        assert callable(getattr(evaluator_cls, "eval_list"))

    def test_default_site_method_exists(self, evaluator_cls):
        """default_site method is defined on Evaluator."""
        assert hasattr(evaluator_cls, "default_site")
        assert callable(getattr(evaluator_cls, "default_site"))

    def test_initial_guess_method_exists(self, evaluator_cls):
        """initial_guess method is defined on Evaluator."""
        assert hasattr(evaluator_cls, "initial_guess")
        assert callable(getattr(evaluator_cls, "initial_guess"))


class TestEvaluatorPublicProperties:
    """Verify that all public properties exist on the Evaluator class."""

    @pytest.fixture
    def evaluator_cls(self):
        from standard_evaluator.evaluators.abstract_evaluator import Evaluator
        return Evaluator

    @pytest.mark.parametrize(
        "prop_name",
        [
            "opt_problem",
            "interface",
            "inputs",
            "outputs",
            "nind",
            "ndep",
            "name",
            "comp_cost",
        ],
    )
    def test_property_exists(self, evaluator_cls, prop_name):
        """Each expected property is defined on the Evaluator class."""
        assert hasattr(evaluator_cls, prop_name), (
            f"Evaluator is missing expected property: {prop_name}"
        )
        # Verify it's accessible as a descriptor (property) on the class
        attr = inspect.getattr_static(evaluator_cls, prop_name)
        assert isinstance(attr, property), (
            f"'{prop_name}' should be a property, got {type(attr).__name__}"
        )


class TestEvaluatorAbstractMethod:
    """Verify that _evaluate is abstract."""

    def test_evaluate_is_abstract(self):
        """_evaluate is decorated with @abstractmethod."""
        from standard_evaluator.evaluators.abstract_evaluator import Evaluator

        # Check that _evaluate is in the abstract methods set
        assert "_evaluate" in Evaluator.__abstractmethods__, (
            "_evaluate should be an abstract method"
        )


class TestNoBoeingImports:
    """Verify no boeing_standard_evaluator appears in sys.modules after import."""

    def test_no_boeing_in_sys_modules(self):
        """After importing Evaluator, no module named boeing_standard_evaluator is in sys.modules."""
        # Clear any cached modules that might have boeing in the name
        boeing_modules_before = {
            k for k in sys.modules if "boeing_standard_evaluator" in k
        }

        from standard_evaluator.evaluators.abstract_evaluator import Evaluator  # noqa: F401

        boeing_modules_after = {
            k for k in sys.modules if "boeing_standard_evaluator" in k
        }

        # There should be no new boeing modules loaded as a result of this import
        new_boeing_modules = boeing_modules_after - boeing_modules_before
        assert len(new_boeing_modules) == 0, (
            f"Importing Evaluator loaded boeing_standard_evaluator modules: {new_boeing_modules}"
        )

    def test_no_boeing_in_evaluator_module_file(self):
        """The abstract_evaluator.py source file contains no boeing_standard_evaluator references."""
        from standard_evaluator.evaluators import abstract_evaluator

        source_file = inspect.getfile(abstract_evaluator)
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "boeing_standard_evaluator" not in content, (
            "abstract_evaluator.py still contains references to boeing_standard_evaluator"
        )
