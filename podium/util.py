"""Module contains utility functions used across the codebase."""
import sys


def error(error_type, logger, msg):
    """
    Parameters
    ----------
    error_type : Type[Exception]
        Type of the error to be raised.
    logger : logging.Logger
        Logger instance that emits the message.
    msg : str
        Error message.
    """

    caller_frame = sys._getframe(1)

    extra = {
        'funcName': caller_frame.f_code.co_name,
        'lineno': caller_frame.f_lineno,
    }

    logger.error(msg, extra=extra)
    raise error_type(msg)
