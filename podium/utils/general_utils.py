"""
Module contains general utilites used throughout the codebase.
"""


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
                raise
            print(
                f"Spacy model '{model}' not found, trying '{OLD_MODEL_SHORTCUTS[model]}'",
            )
            nlp = spacy.load(OLD_MODEL_SHORTCUTS[model])
    except ImportError:
        print(
            "Please install SpaCy and the SpaCy. "
            "See the docs at https://spacy.io for "
            "more information."
        )
    except AttributeError:
        print(
            f"Spacy model '{model}' not found. Please install the model. "
            "See the docs at https://spacy.io for more information."
        )
    return nlp
