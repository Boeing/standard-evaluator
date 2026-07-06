Evaluators
==========

This page documents the evaluator class hierarchy in the Standard Evaluator library.
The abstract :class:`~standard_evaluator.evaluators.abstract_evaluator.Evaluator` base class
defines the common API, with concrete implementations for different execution backends.

Base Class
----------

.. autoclass:: standard_evaluator.evaluators.abstract_evaluator.Evaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Concrete Evaluators
-------------------

.. autoclass:: standard_evaluator.evaluators.numpy_evaluator.NumpyEvaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: standard_evaluator.evaluators.executable_evaluator.ExecutableEvaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: standard_evaluator.evaluators.shift_scale_evaluator.ShiftScaleEvaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: standard_evaluator.evaluators.open_mdao_evaluator.OpenMDAOEvaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: standard_evaluator.evaluators.excel_evaluator.ExcelEvaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

.. autoclass:: standard_evaluator.evaluators.matlab_evaluator.MatlabEvaluator
   :members:
   :show-inheritance:
   :special-members: __init__, __call__
