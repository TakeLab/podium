import math
import random

import numpy as np
from takepod.storage import util

class Iterator(object):
  """An iterator that batches data from a dataset post numericalization
  """

  def __init__(self, dataset, batch_size, sort_key=len, shuffle=True, train=True, seed=42):
    self.batch_size = batch_size
    self.dataset = dataset
    # don't shuffle test/valid
    self.shuffle = shuffle if train else False
    self.sort_key = sort_key

    self.epoch = 0
    self.iterations = 0
    self.seed = seed

    self.shuffler = util.RandomShuffler(seed)


  def __len__(self):
    return math.ceil(len(self.dataset) / self.batch_size)


  def __iter__(self):
    # want: get numericalized samples
    # can we do this with an example? The example would need to have fields.
    # only datasets have fields.
    self.create_batches()
    for _ in range(len(self)):
      yield self.batches[self.iterations]
      self.iterations += 1

    # prepare for new epoch
    self.iterations = 0
    self.epoch += 1

  def data(self):
    # shuffle / sort the data of the whole dataset
    if self.shuffle:
      # not completely sure how optimal this is
      xs = [self.dataset[i] for i in random.sample(range(len(self.dataset)), len(self.dataset))] 
    else:
      xs = self.dataset.examples[:] # copy, which is expenive, but the above does this too
    return np.array(xs)

  def create_batches(self):
    # split into len(self) chunks of not necessarily equal length
    self.batches = np.array_split(self.data(), len(self)) 
