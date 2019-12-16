"""
Home of the TakeLab Podium project. Podium is a Python machine learning library
that helps users to accelerate use of natural language processing models.
GitHub repository: https://github.com/FilipBolt/takepod
"""
import logging
import logging.config
from . import dataload
from . import datasets
from . import examples
from . import metrics
from . import models
from . import preproc
from . import storage
from . import validation
from . import model_selection
from . import pipeline

__name__ = "takepod"

__all__ = ["dataload",
           "datasets",
           "examples",
           "metrics",
           "models",
           "preproc",
           "storage",
           "validation",
           "model_selection",
           "pipeline"]


# Reference for initialization of logging scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/__init__.py
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.WARNING)

# More information about logging can be found on project github
#  https://github.com/FilipBolt/takepod/wiki/Logging
