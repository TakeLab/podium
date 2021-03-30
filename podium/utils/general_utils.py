"""
Module contains general utilites used throughout the codebase.
"""

import textwrap
from typing import Any, Dict


def load_spacy_model_or_raise(model, *, disable=None):
    disable = disable if disable is not None else []
    try:
        import spacy

        try:
            nlp = spacy.load(model, disable=disable)
        except OSError:
            OLD_MODEL_SHORTCUTS = (
                spacy.errors.OLD_MODEL_SHORTCUTS
                if hasattr(spacy.errors, "OLD_MODEL_SHORTCUTS")
                else {}
            )
            if model not in OLD_MODEL_SHORTCUTS:
                print(
                    f"Spacy model '{model}' not found. Please install the model. "
                    "See the docs at https://spacy.io for more information."
                )
                raise
            print(
                f"Spacy model '{model}' not found, trying '{OLD_MODEL_SHORTCUTS[model]}'",
            )
            nlp = spacy.load(OLD_MODEL_SHORTCUTS[model])
    except ImportError:
        print(
            "Please install SpaCy. "
            "See the docs at https://spacy.io for "
            "more information."
        )
        raise
    except AttributeError:
        print(
            f"Spacy model '{model}' not found. Please install the model. "
            "See the docs at https://spacy.io for more information."
        )
        raise
    return nlp


def repr_type_and_attrs(
    self: Any,
    attrs: Dict[str, Any],
    with_newlines: bool = False,
    repr_values: bool = True,
) -> str:
    delim = ",\n" if with_newlines else ", "
    attrs_str = delim.join(
        f"{k}: {repr(v) if repr_values else str(v)}" for k, v in attrs.items()
    )
    attrs_str = (
        f"\n{textwrap.indent(attrs_str, ' ' * 4)}\n" if with_newlines else attrs_str
    )
    return f"{type(self).__name__}({{{attrs_str}}})"
