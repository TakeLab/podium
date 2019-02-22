"""Module contains utility functions to preprocess text data"""

def capitalize_target_like_source(func):
    """Capitalization decorator of a method that processes a word.
    Method invokes the parameter function 
    with a lowercased input, then capitalizes 
    the return value such that capitalization corresponds
    to the original input provided

    Parameters
    ----------
    func : function
        function which gets called, MUST be a class member with one argument
        like def func(self, word)

    Returns
    -------
    wrapper : function
        decorator function to decorate func with
    """

    def _wrapper(*args, **kwargs):
        source = args[1]

        is_lower = source.islower()
        source_lower = source if is_lower else source.lower()

        target = func(args[0], source_lower)

        if is_lower:
            return target
        else:
            return _uppercase_target_like_source(source, target)
    return _wrapper


def _uppercase_target_like_source(source, target):
    uppercased_target = ''.join([
        target[i].upper()
        if s.isupper() and s.lower() == target[i] else target[i]
        for i, s in zip(range(len(target)), source)
    ])
    uppercased_target += target[len(source):]
    return uppercased_target
