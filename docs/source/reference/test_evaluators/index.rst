Test Evaluators
===============

This section documents the benchmark test evaluators available in the
:mod:`standard_evaluator.evaluators.test` module. These evaluators implement
well-known mathematical optimization problems with known solutions, useful for
testing optimization algorithms or as stand-in functions while developing
``standard_evaluator`` workflows.

Each test evaluator has a :attr:`known_solution` property returning the optimal
input values and corresponding response values as a DataFrame.

Base Class
----------

.. autoclass:: standard_evaluator.evaluators.test_evaluator.TestEvaluator
   :members:
   :show-inheritance:

Categories
----------

Test evaluators are organized by their optimization goal:

.. toctree::
   :maxdepth: 1

   optimization
   multi_objective
   feasibility
