.. Podium documentation master file, created by
   sphinx-quickstart on Mon Oct  7 17:02:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TakeLab Podium
==================================

Data loading & preprocessing library for natural language processing.
Compatible with all deep learning frameworks, based in NumPy.

Podium enforces a light amount of coupling between its preprocessing components which allows flexibility and hackability, but also accepts that the user might not want to use the full extent of Podium functionalities and enables them to at any point retrieve their data and use it with another library.

Contents
---------------------------------
The documentation is organized in four parts:

- **Getting started**: an overview of the scope of the library,
- **In-depth overview**: examples of advanced usage of Podium,
- **Core package Reference**: documentation of methods and classes in Podium ready for use,
- **Modules under development**: parts of the library which are either stale/outdated or pending major revision.


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
    coming_soon

.. toctree::
    :maxdepth: 2
    :caption: Preprocessing tools

    preprocessing

.. toctree::
   :maxdepth: 2
   :caption: Core package Reference:

   vocab_and_fields
   datasets
   iterators
   vectorizers
   resources
   preproc

.. toctree::
   :maxdepth: 2
   :caption: Modules under development

   under_development
   model_implementations

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
