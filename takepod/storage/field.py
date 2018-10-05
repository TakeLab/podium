import numpy as np
from collections import OrderedDict

from takepod.storage.vocab import Vocab

class Field(object):
  """ Preprocessing class for processing text data contained in columns
  """

  def __init__(self, name, 
               tokenizer='split', 
               language='en', 
               use_vocab=True, 
               sequential=True,
               lower=True,
               store_raw=True,
               custom_numericalize=float,
               **kwargs):
    """Create a Field from arguments

      Arguments:

        tokenizer: str, defining the tokenizer to use on the raw data
        language: str, the language for the tokenizer (if necessary)
        use_vocab: bool, defines should a vocab be built for this field (no if the data is aleady numeric)
        sequential: bool, defines is the data field textual (should it be tokenized)
        lower: bool, whether to lowercase the data post-tokenization
        stop_words (maybe pass to vocab?)
        **kwargs: remainder of arguments is passed over to the Vocab
    """
    self.name = name
    self.language = language
    self.use_vocab = use_vocab
    self.store_raw = store_raw
    self.sequential = sequential
    self.lower = lower
    self.vocab = None
    if self.use_vocab:
      self.vocab = Vocab(**kwargs)
    self.tokenizer = get_tokenizer(tokenizer, language)
    self.custom_numericalize = custom_numericalize

    self.pre_tokenize_hooks = OrderedDict()
    self.post_tokenize_hooks = OrderedDict()

  def add_pretokenize_hook(self, hook):
    """Add a pre-tokenization hook to the Field

    Pretokenize hooks have the following signature:
      >>> function pre_tok_hook(raw_data):
      >>>   raw_data = do_stuff(raw_data)
      >>>   return raw_data

      the raw data is then replaced with the preprocessed version.
      This can be used to eliminate encoding errors in data, replace
      numbers and names etc.

    returns: a callable class which removes the hook from the Field
    """
    h = hash(hook) # maybe do this a bit smarter
    self.pre_tokenize_hooks[h] = hook
    return HookControl(h, self.pre_tokenize_hooks)


  def add_posttokenize_hook(self, hook):
    """Add a post-tokenization hook to the Field

    Posttokenize hooks have the following signature:

      >>> function post_tok_hook(tokenized_data, raw_data):
      >>>   tokenized_data_out = do_stuff(tokenized_data, raw_data)
      >>>   raw_data_out = do_stuff(tokenized_data raw_data)
      >>>   return tokenized_data_out, raw_data_out

      Both the tokenized and raw data are then replaced based on the hook output.
      Hooks are called only if the Field is sequential (otherwise, pre-tokenize
       is the same as post-tokenize).

      returns: a callable class which removes the hook from the Field
    """
    h = hash(hook)
    self.post_tokenize_hooks[h] = hook
    return HookControl(h, self.post_tokenize_hooks)


  def preprocess(self, raw):
    """Preprocess a raw string input for a column.
    """
    tokenized = None
    # create two views for data row: tokenized and untokenized
    for hook in self.pre_tokenize_hooks:
      raw = hook(raw)
    if self.sequential:
      tokenized = self.tokenizer(raw)
      for hook in self.post_tokenize_hooks:
        tokenized, raw = hook(tokenized, raw)
      if self.lower:
        # TODO: this can essentially be a post hook, remove for cleanliness
        tokenized = [s.lower() for s in tokenized]
    if self.use_vocab:
      # update the vocab on-the-fly
      if self.sequential:
        self.vocab += tokenized
      else:
        # construct vocab from untokenized data
        # use-case: class labels
        # map raw string data to iterable (array)
        self.vocab += [raw]
    if not self.store_raw:
      # TODO: this can't be true if sequential is false
      # maybe don't store if sequential is True, and default to False?
      raw = None
    return raw, tokenized


  def finalize(self):
    """Signal that the vocabulary can be built
    """
    if self.use_vocab:
      self.vocab.finalize()


  def numericalize(self, data):
    """Numericalize the input data row based on the built vocab
      TODO: decide whether to use two args (*data has to be passed)
            or keep demapping here
        Arguments:
          data: tuple of (raw, tokenized) processed input text

    """
    raw, tokenized = data
    # raw data is just a string, so we need to wrap it into an iterable
    data = tokenized if self.sequential else [raw]

    if self.use_vocab:
      return self.vocab.numericalize(data)
    else:
      # handle numericalization of non-textual data 
      # (such as floating point data Fields)
      return np.array([self.custom_numericalize(tok) for tok in data])


  def pad_to_length(self, row, length):
    """Append pad tokens to match len, or truncate trailing data
    """
    # TODO: maybe leave all the numpy-related stuff to the vocab?
    # TODO: where do we handle <bos> and <eos> tokens?
    if len(row) > length:
      row = row[:length]
    if len(row) < length:
      diff = length - len(row)
      row = np.append(row, [self.vocab.pad_symbol]*diff)
    return row

class HookControl():
  """Shallow class storing data necessary to detach a hook
  """
  def __init__(self, hook_id, hook_dictionary):
    self.hook_id = hook_id
    self.hook_dictionary = hook_dictionary # this is just a view!

  def __call__(self):
    # TODO: raise error if if fails?
    if self.hook_id in self.hook_dictionary:
      del self.hook_dictionary[self.hook_id]

def get_tokenizer(tokenizer, language='en'):
  # Add every new tokenizer to this "factory" method
  if callable(tokenizer):
    # if arg is already a function, just return it
    return tokenizer

  if tokenizer == 'spacy':
    try:
      import spacy
      spacy_tokenizer = spacy.load(language)

      # closures instead of lambdas because they are serializable
      def spacy_tokenize(string):
        # need to wrap in a function to access .text
        return [token.text for token in spacy_tokenizer.tokenizer(string)]

      return spacy_tokenize
    except:
      print("Please install SpaCy and the SpaCy {} tokenizer. ".format(language)
            + "See the docs at https://spacy.io for more information.")
      raise
  # add other cases here
  else:
    # default tokenizer: string splitting
    return str.split
