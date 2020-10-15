import ctypes
import os


def is_admin():
    # on some systems/environments, admin privileges are required to download
    # a SpaCy model using the shorthand syntax
    # see: https://stackoverflow.com/questions/1026431/
    # cross-platform-way-to-check-admin-rights-in-a-python-script-under-windows
    try:
        _flag = os.getuid() == 0
    except AttributeError:
        _flag = ctypes.windll.shell32.IsUserAnAdmin() != 0

    return _flag


def has_spacy_model(language):
    import spacy

    try:
        spacy.load(language)
    except:
        return False

    return True
