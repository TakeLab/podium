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

__name__ = "takepod"
__version__ = "0.1.0"

__all__ = ["dataload",
           "datasets",
           "examples",
           "metrics",
           "models",
           "preproc",
           "storage"]


# From documentation:
#   `fileConfig can be called several times from an application,`
#   `allowing an end user the ability to select from various`
#   ` pre-canned configurations.`
logging.config.fileConfig(fname='logging.ini', disable_existing_loggers=False)
