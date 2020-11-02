.. Podium documentation master file, created by
   sphinx-quickstart on Mon Oct  7 17:02:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TakeLab Podium
==================================

Data loading, preprocessing & batching library for natural language processing.
Compatible with all deep learning frameworks, based in NumPy.

Podium enforces a light amount of coupling between its preprocessing components which allows flexibility and hackability, but also accepts that the user might not want to use the full pipeline and enables him to at any point retrieve his data and use it with another library.

Contents
---------------------------------
The documentation is organized in four parts:

- **Getting started**: a quick walkthrough of installation and basic usage examples.
- **Walkthrough**: an overview of the scope of the library.
- **In-depth overview**: examples of advanced usage of Podium.
- **Package reference**: full documentation of all methods and classes present in Podium.


.. toctree::
    :maxdepth: 2
    :caption: Getting started

    installation
    walkthrough
    faq

.. toctree::
    :maxdepth: 2
    :caption: In depth overview

    advanced

.. toctree::
   :maxdepth: 2
   :caption: Core package Reference:

   vocab_and_fields
   datasets
   iterators

.. toctree::
   :maxdepth: 2
   :caption: Unstable

   under_development
   podium.models


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
