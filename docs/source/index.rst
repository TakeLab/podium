.. Podium documentation master file, created by
   sphinx-quickstart on Mon Oct  7 17:02:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TakeLab Podium
==================================

Data loading & preprocessing library for natural language processing.
Compatible with all deep learning frameworks, based in NumPy.

The goal of Podium is to be **lightweight**, in terms of code and dependencies, **flexible**, to cover most common use-cases and easily adapt to more specific ones and **clearly defined**, so new users can quickly understand the sequence of operations and how to inject their custom functionality.

Contents
---------------------------------
The documentation is organized in five parts:

- **Quickstart**: an quick preview of the library,
- **Walkthrough**: a description of how the basics work,
- **In-depth overview**: advanced usage options,
- **Examples**: full stand-alone examples of NLP models using Podium,
- **Core package Reference**: the documentation of methods and classes in Podium.


.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    installation
    quickstart
    walkthrough

.. toctree::
    :maxdepth: 2
    :caption: In Depth Overview

    advanced
    coming_soon

.. toctree::
    :maxdepth: 2
    :caption: Full examples

    examples/tfidf_example.rst
    examples/pytorch_rnn_example.rst

.. toctree::
    :maxdepth: 2
    :caption: Preprocessing Tools

    preprocessing

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   package_reference/vocab_and_fields
   package_reference/preprocessing
   package_reference/datasets
   package_reference/iterators
   package_reference/vectorizers

.. toctree::
   :maxdepth: 2
   :caption: Modules Under Development

   under_development
   model_implementations
