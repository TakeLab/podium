"""Module contains simple stemmer for Croatian"""
# -*-coding:utf-8-*-
#
#    Simple stemmer for Croatian v0.1
#    Copyright 2012 Nikola Ljubešić and Ivan Pandžić
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re
import os
import functools

from podium.preproc.util import (
    capitalize_target_like_source,
    make_trie,
    find_word_by_prefix
)


class CroatianStemmer:
    """Simple stemmer for Croatian language"""
    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.__rules = [
            re.compile(r'^(' + base + ')(' + suffix + r')$') for
            base, suffix in [
                e.strip().split(' ')
                for e in
                open(os.path.join(dir_path, "data/rules.txt"),
                     encoding='utf-8')]
        ]
        self.__transform_map = self._load_transformations(
            os.path.join(dir_path, "data/transformations.txt")
        )
        self.__transform_trie = make_trie(
            list(self.__transform_map.keys())
        )
        self.__stop = set([
            e.strip()
            for e in
            open(os.path.join(dir_path, 'data/nostem-hr.txt'),
                 encoding='utf-8')
        ])

    def _load_transformations(self, file_path):
        transform_map = {}
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                (key, value) = line.split('\t')
                # we reverse the keys, since we search for prefixes
                # e.g. when searching for 'izam' in 'turizam'
                # we go from end to start
                transform_map[key[::-1]] = value
        return transform_map

    def _determine_r_vowel(self, string):
        '''
        Determines if 'r' is a vowel or not
        If it is => uppercase it.

        Parameters
        ----------
        string : str
            word in Croatian

        Returns
        -------
        string : str
            Croatian word with 'r' vowel uppercased
        '''
        return re.sub(r'(^|[^aeiou])r($|[^aeiou])', r'\1R\2', string)

    def _has_vowel(self, string):
        if re.search(r'[aeiouR]', self._determine_r_vowel(string)):
            return True
        else:
            return False

    def transform(self, word):
        """Method transforms given word from a dict, given it
        ending with a specific suffix

        Parameters
        ----------
        word : str
            word

        Returns
        -------
        transformed_word : str
            transformed word according to transformation mappings
        """

        # first, we reverse the word, such that we can
        # use the prefix search in trie
        found_prefix = find_word_by_prefix(self.__transform_trie, word[::-1])

        if found_prefix:
            replace = self.__transform_map[found_prefix]
            return word[:-len(found_prefix)] + replace
        else:
            return word

    def root_word(self, word):
        """Method returns root of a word.

        Parameters
        ----------
        word : str
            word string

        Returns
        -------
        root : str
            root of a word
        """
        for rule in self.__rules:
            division = rule.match(word)
            if division:
                root = division.group(1)
                if self._has_vowel(root) and len(root) > 1:
                    return root
        return word

    @capitalize_target_like_source
    def stem_word(self, word, **kwargs):
        '''
        Returns the root or roots of a word,
        together with any derivational affixes

        Parameters
        ----------
        word : str
            word in Croatian

        Returns
        -------
        string : str
            Croatian word root plus derivational morphemes
        '''

        if word in self.__stop:
            return word
        else:
            return self.root_word(self.transform(word))


def _stemmer_posttokenized_hook(raw, tokenized, stemmer):
    """Stemmer postokenized hook that can be used in field processing.
    It is intented for user to use `get_croatian_stemmer_hook` instead
    of this function as it hides Stemmer initialization and ensures that
    constructor is called once.

    Parameters
    ----------
    raw : str
        raw field content
    tokenized : iter(str)
        iterable of tokens that needs to be stemmed
    stemmer : CroatianStemmer
        croatian stemmer instance

    Returns
    -------
    raw, tokenized : tuple(str, iter(str))
        Method returns unchanged raw and stemmed tokens.
    """
    return raw, [stemmer.stem_word(token) for token in tokenized]


def get_croatian_stemmer_hook():
    """Method obtains croatian stemmer hook."""
    return functools.partial(_stemmer_posttokenized_hook,
                             stemmer=CroatianStemmer())
