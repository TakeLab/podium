"""Module contains utility functions to preprocess text data"""

TRIE_END_SYMBOL = '*'


def capitalize_target_like_source(func):
    """Capitalization decorator of a method that processes a word.
    Method invokes the parameter function
    with a lowercased input, then capitalizes
    the return value such that capitalization corresponds
    to the original input provided

    Parameters
    ----------
    func : function
        function which gets called, MUST be a class member with one
        positional argument (like def func(self, word), but may contain
        additional keyword arguments (like func(self, word, my_arg='my_value'))

    Returns
    -------
    wrapper : function
        decorator function to decorate func with
    """

    def _wrapper(*args, **kwargs):
        source = args[1]

        is_lower = source.islower()
        source_lower = source if is_lower else source.lower()

        target = func(args[0], source_lower, **kwargs)

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


def make_trie(words):
    """Creates a prefix trie data structure given a
    list of strings. Strings are split into chars
    and a char nested trie dict is returned

    Parameters
    ----------
    words : list(str)
        List of strings to create a trie structure from

    Returns
    -------
    trie : dict
        Nested dict trie data structure
    """
    trie = dict()

    for word in words:
        # sub-dict to process a word
        subdict = trie
        for char in word:
            # get sub-dict if exists on letter
            # else create new empty dict
            subdict = subdict.setdefault(char, {})

        # add END_SYMBOL to indicate word is finished
        subdict[TRIE_END_SYMBOL] = TRIE_END_SYMBOL
    return trie


def find_word_by_prefix(trie, word):
    """Searches through a trie data structure and
    returns an element of the trie is the word
    is a prefix or exact match of one of the trie elements.
    Otherwise returns None

    Parameters
    ----------
    trie : dict
        Nested dict trie data structure
    word : str
        String being searched for in the trie data structure

    Returns
    -------
    found_word : str
        String found which is either the exact word,
        it's prefix or None if not found in trie
    """
    found_word = []
    for char in word:
        if char in trie:
            trie = trie[char]
            found_word.append(char)
        # found a match in trie for the prefix,
        # haven't reached the end of the given word
        elif TRIE_END_SYMBOL in trie:
            return ''.join(found_word)
        else:
            return None

    # reached end of the given word
    if TRIE_END_SYMBOL in trie:
        return word
    else:
        # partial match in trie found
        return None
