.. documentation master file, created by
   sphinx-quickstart on Fri Mar 11 17:22:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Standard Evaluator
==================

.. image:: ../assets/logo.svg
   :alt: Standard Evaluator Logo
   :align: center
   :width: 270px

Standard Evaluator is an open-source Python library (published on PyPI as ``standard-evaluator``) that provides a common API for defining, wrapping, and composing analysis codes and surrogate models. It was initially developed under NASA Contract 80GRC023CA045, and has been expanded since then.

The library has three main purposes:

1. **Common Evaluator API** — Expose analysis capabilities and surrogate models through a unified interface. End-users interact via Pandas DataFrames; developers can use a simplified NumPy-focused interface (``eval_np``, ``eval_list``).

2. **Integration Framework Bridge** — Expose all evaluators to integration frameworks like OpenMDAO. The architecture is designed to support additional integration frameworks in the future.

3. **Assembly Serialization** — Capture the structure of assemblies of analyses in Pydantic classes or JSON files, and rebuild those assemblies from the stored information. Currently supports OpenMDAO; designed for future multi-framework support.

Note that users of the library are responsible for installing the third party open source components and for complying with the terms and conditions of the respective open source licenses governing the third party open source components.

Installation
------------

``pip install standard-evaluator``

Optional extras:

.. code-block:: bash

   pip install standard-evaluator[smt]      # Surrogate Modeling Toolbox models
   pip install standard-evaluator[aviary]   # NASA Aviary integration
   pip install standard-evaluator[test]     # Testing dependencies

Optionally clone the repo and install locally for development:

.. code-block:: bash

   git clone <repo-url>
   cd standard-evaluator
   pip install -e .[test,smt]

Project Structure
-----------------

- ``src/`` — Source code of the standard evaluator library
- ``docs/`` — Documentation source (Sphinx + Jupyter notebooks)
- ``tests/`` — Unit and property-based tests

Contents
--------

.. toctree::
   :maxdepth: 2

   theory/index
   demos/index
   reference/index
   notes/index

