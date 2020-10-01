"""Module contains utility functions used across the codebase."""
import inspect


def error(error_type, logger, msg):
    caller_frame = inspect.getcurrentframe().f_back

    extra = {
        'funcName': caller_frame.f_code.co_name,
        'lineno': caller_frame.f_lineno,
    }

    logger.error(msg, extra=extra)
    raise error_type(msg)
