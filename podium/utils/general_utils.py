"""
Module contains general utilites used throughout the codebase.
"""

import warnings


def load_spacy_model_or_raise(model, *, disable=None):
    disable = disable if disable is not None else []
    try:
        import spacy

        nlp = spacy.load(model, disable=disable)
    except OSError:
        OLD_MODEL_SHORTCUTS = (
            spacy.errors.OLD_MODEL_SHORTCUTS
            if hasattr(spacy.errors, "OLD_MODEL_SHORTCUTS")
            else {}
        )
        if model not in OLD_MODEL_SHORTCUTS:
            raise
        warnings.warn(
            f"Spacy model '{model}' not found, trying '{OLD_MODEL_SHORTCUTS[model]}'",
            stacklevel=2,
        )
        nlp = spacy.load(OLD_MODEL_SHORTCUTS[model])
    return nlp
