import sys
import numpy as np
from collections import Counter, defaultdict

class Vocab(object):
  def __init__(self, max_size=None, min_freq=1, specials=['<unk>', '<pad>'],
               vectors=None, unk_init=None, default_unk_index=0, keep_freqs=False):
    self.freqs = Counter()
    self.finalized = False # flag to know if we're ready to numericalize
    self.itos = list(specials) # specials are always first right now

    def _default_unk_index():
      return default_unk_index

    # need to validate unk index with order of specials
    self.stoi = defaultdict(_default_unk_index) 
    self.stoi.update({k:v for v, k in enumerate(self.itos)})
    self.keep_freqs = keep_freqs # destroy freqs after finalization?
    self.min_freq = min_freq
    self.vectors = vectors
    self.max_size = max_size


  @property
  def pad_symbol(self):
    # TODO: move magic string to some enum or sth
    p = '<pad>'
    if p not in self.stoi:
      return -1
    return self.stoi[p]


  def __add__(self, values):
    # vocabs can be added together
    #   or a set of values can be added to a vocab
    if not self.finalized:
      if type(values) == type(self):
        self.freqs += values.freqs # add freqs to this instance
      else:
        try:
          self.freqs.update(values)
        except TypeError as e:
          # not iterable, maybe handle? check type
          print(type(values))
          raise e
      return self
    else:
      # TODO: do the same, but just for stoi and itos
      # now the words won't be in sorted order anymore
      # it would be recommended here to keep freqs
      pass

  def __iadd__(self, values):
    return self.__add__(values)

  def __sub__(self, values):
    # same logic, except subtract a list or a vocab from freqs
    if not self.finalized:
      if type(values) != type(self):
        try:
          values = Counter(values)
        except TypeError as e:
          # not iterable
          print(type(values))
          raise e
      self.freqs -= values.freqs
      return self
    else:
      # TODO: handle behavior when finalized
      pass

  def __isub__(self, values):
    return self.__sub__(values)

  def finalize(self):
    if self.finalized:
      # TODO: raise an error if this happens
      # what do we want to happen when we extend a vocab post-finalization?
      # probably should set the finalized to `False` so that this has to be rerun
      return

    # construct stoi and itos, sort by frequency
    words_and_freqs = sorted(self.freqs.items(), key=lambda tup: tup[0])
    # sort alphabetically (do we _really_ care?)
    words_and_freqs.sort(key=lambda tup: tup[1], reverse=True)

    if self.max_size is None:
      self.max_size = len(words_and_freqs) + len(self.itos) # vocab + specials
    for word, freq in words_and_freqs:
      if freq < self.min_freq or len(self.itos) >= self.max_size:
        break
      self.itos.append(word)
      self.stoi[word] = len(self.stoi)

    # Here we should load word vectors _only_ for the words in the vocab
    if not self.keep_freqs:
      self.freqs = None #  release memory
    if self.vectors:
      # TODO: load vectors
      pass
    self.finalized = True

  def numericalize(self, data):
    if not self.finalized:
      raise ValueError('Cannot numericalize if the vocabulary has not been finalized'
                       ' call `.finalize() on the Field`')

    return np.array([self.stoi[token] for token in data])

  def __len__(self):
    if finalized:
      return len(self.itos)
    else:
      return len(self.freqs)

  def __eq__(self, other):
    if self.finalized != other.finalized:
      return False
    if self.freqs != other.freqs:
      return False
    if self.stoi != other.stoi:
      return False
    if self.itos != other.itos:
      return False
    if self.vectors != other.vectors:
      return
    return True
