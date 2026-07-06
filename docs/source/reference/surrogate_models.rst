Surrogate Models
================

This section documents the surrogate model hierarchy in the Standard Evaluator library.
Surrogate models provide trained mathematical approximations of evaluators, supporting
training from data, prediction, serialization, and parallel training.

Base Class
----------

.. autoclass:: standard_evaluator.surrogate_models.abstract_model.SurrogateModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Polynomial Model
----------------

.. autoclass:: standard_evaluator.surrogate_models.polynomial_model.PolynomialModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

SMT-Based Models
----------------

The following models require the ``smt`` optional dependency.
Install with: ``pip install standard-evaluator[smt]``

Radial Basis Function Model (RBFModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: standard_evaluator.surrogate_models.smt_models.rbf_model_using_smt.RadialBasisFunctionModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Inverse Distance Weighting Model (IDWModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: standard_evaluator.surrogate_models.smt_models.idw_model_using_smt.InverseDistanceWeightingModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Gradient-Enhanced Neural Networks Model (GENNModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: standard_evaluator.surrogate_models.smt_models.genn_model_using_smt.GradientEnhancedNeuralNetworksModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Least Squares Approximation Model (LSModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: standard_evaluator.surrogate_models.smt_models.ls_model_using_smt.LeastSquaresApproximationModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Regularized Minimal-Energy Tensor-Product B-Splines (RMTSModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: standard_evaluator.surrogate_models.smt_models.rmts_model_using_smt.RegularizedMinimalEnergyTensorProductBSplines
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

Second Order Polynomial Approximation Model (SOPAModel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: standard_evaluator.surrogate_models.smt_models.sopa_model_using_smt.SecondOrderPolynomialApproximationModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__
