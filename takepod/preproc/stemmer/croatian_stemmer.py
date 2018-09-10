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
    __stop = set(
        ['biti', 'jesam', 'budem', 'sam', 'jesi', 'budeš', 'si', 'jesmo',
         'budemo', 'smo', 'jeste', 'budete', 'ste',
         'jesu',
         'budu', 'su', 'bih', 'bijah', 'bjeh', 'bijaše', 'bi', 'bje', 'bješe',
         'bijasmo', 'bismo', 'bjesmo', 'bijaste',
         'biste', 'bjeste', 'bijahu', 'biste', 'bjeste', 'bijahu', 'bi', 'biše',
         'bjehu', 'bješe', 'bio', 'bili',
         'budimo',
         'budite', 'bila', 'bilo', 'bile', 'ću', 'ćeš', 'će', 'ćemo', 'ćete',
         'želim', 'želiš', 'želi', 'želimo',
         'želite',
         'žele', 'moram', 'moraš', 'mora', 'moramo', 'morate', 'moraju',
         'trebam', 'trebaš', 'treba', 'trebamo',
         'trebate',
         'trebaju', 'mogu', 'možeš', 'može', 'možemo', 'možete'])

    __transformations = None
    __rules = None

    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.__rules = [
            re.compile(r'^(' + osnova + ')(' + nastavak + r')$') for
            osnova, nastavak in [e.strip().split(' ') for e in
                                 open(os.path.join(dir_path,
                                                   "rules.txt"), encoding='utf-8')]]
        self.__transformations = [e.strip().split(
            '\t') for e in open(os.path.join(dir_path,
                                             'transformations.txt'),
                                encoding='utf-8')]

    # TODO make this a private method
    def istakniSlogotvornoR(self, niz):
        return re.sub(r'(^|[^aeiou])r($|[^aeiou])', r'\1R\2', niz)

    # TODO make this a private method
    def imaSamoglasnik(self, niz):
        if re.search(r'[aeiouR]',
                     self.istakniSlogotvornoR(
                             niz)) is None:
            return False
        else:
            return True

    def transformiraj(self, pojavnica):
        for trazi, zamijeni in self.__transformations:
            if pojavnica.endswith(trazi):
                return pojavnica[:-len(trazi)] + zamijeni
        return pojavnica

    # TODO write pydoc for this + rename to english
    def korjenuj(self, pojavnica):
        for pravilo in self.__rules:
            dioba = pravilo.match(pojavnica)
            if dioba is not None:
                if self.imaSamoglasnik(
                        dioba.group(1)) and len(dioba.group(1)) > 1:
                    return dioba.group(1)
        return pojavnica

    # TODO IMHO this does too much (lower case, stopword check...)
    def stem_word(self, word):
        if word.lower() in self.__stop:
            return word
        stem = self.korjenuj(self.transformiraj(word.lower()))
        return "".join(list(
            stem[i].upper() if c.isupper() else stem[i] for i, c in
            zip(range(len(stem)), word)))

    # TODO could be deleted all together 
    def stem_doc(self, doc_string):
        stemmed_string = " ".join(
            [self.stem_word(word) for word in
             doc_string.split(" ")])
        return stemmed_string
