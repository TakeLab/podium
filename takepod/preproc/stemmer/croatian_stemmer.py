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


class CroatianStemmer:

    # list of words that are it's own stem
    __stop = None
    __transformations = None
    __rules = None

    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.__rules = [
            re.compile(r'^(' + base + ')(' + suffix + r')$') for
            base, suffix in [
                e.strip().split(' ')
                for e in
                open(os.path.join(dir_path, "data/rules.txt"), encoding='utf-8')]
        ]
        self.__transformations = [e.strip().split(
            '\t') for e in open(os.path.join(dir_path,
                                             'data/transformations.txt'),
                                encoding='utf-8')]
        self.__stop = set([
            e.strip()
            for e in
            open(os.path.join(dir_path, 'data/nostem-hr.txt'), encoding='utf-8')
        ])

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
        for seek, replace in self.__transformations:
            if word.endswith(seek):
                return word[:-len(seek)] + replace
        return word

    def root_word(self, word):
        for rule in self.__rules:
            division = rule.match(word)
            if division:
                root = division.group(1)
                if self._has_vowel(root) and len(root) > 1:
                    return root
        return word

    def stem_word(self, word):
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

        if word.lower() in self.__stop:
            return word
        stem = self.root_word(self.transform(word.lower()))
        return "".join(list(
            stem[i].upper() if c.isupper() else stem[i] for i, c in
            zip(range(len(stem)), word)))
