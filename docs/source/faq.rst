FAQ
====

**Q: Can I just load and retrieve a numericalized dataset from Podium?**


**A:** Yes, you can use the :meth:`Podium.datasets.Dataset.batch()` method. You still need to define the Fields if you are not using a built-in dataset.


.. code-block:: python

  >>> from podium.datasets import SST
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits()
  >>> x, y = sst_train.batch()
  >>> print(f"{x.text.shape}\n{y.label.shape}")
  (6920, 52)
  (6920, 1)

Be aware that you will get a dataset as a matrix by default -- meaning that all the instances will be padded to the maximum length. If you wish to get a list of instances instead, set ``disable_batch_matrix=True`` in the constructor for the corresponding Field.

.. code-block:: python

  >>> from podium.datasets import SST
  >>> from podium import Vocab, Field, LabelField
  >>> text = Field(name='text', numericalizer=Vocab(), disable_batch_matrix=True)
  >>> label = LabelField(name='label')
  >>> fields = {'text':text, 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> x, y = sst_train.batch()
  >>> print(type(x.text), len(x.text))
  <class 'list'> 6920


**Q: I want to get multiple outputs from the same input text, is this possible?**

**A:** Yes, you can process one input data column with multiple fields by passing a tuple of Fields as arguments, as follows:

.. code-block:: python

  >>> from podium.datasets import SST
  >>> from podium import Vocab, Field, LabelField
  >>> char = Field(name='char', numericalizer=Vocab(), tokenizer=list)
  >>> text = Field(name='word', numericalizer=Vocab())
  >>> label = LabelField(name='label')
  >>> fields = {'text':(char, text), 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> print(sst_train[222].word, sst_train[222].char, sep='\n')
  (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])
  (None, ['A', ' ', 's', 'l', 'i', 'c', 'k', ' ', ',', ' ', 'e', 'n', 'g', 'r', 'o', 's', 's', 'i', 'n', 'g', ' ', 'm', 'e', 'l', 'o', 'd', 'r', 'a', 'm', 'a', ' ', '.'])

In case your outputs can share the same tokenizer, you should use :class:`podium.storage.MultioutputField` for efficiency:

.. code-block:: python

  >>> from podium.datasets import SST
  >>> from podium import Vocab, Field, LabelField
  >>> char = Field(name='char', numericalizer=Vocab(), tokenizer=list)
  >>> text = Field(name='word', numericalizer=Vocab())
  >>> label = LabelField(name='label')
  >>> fields = {'text':(char, text), 'label':label}
  >>> sst_train, sst_test, sst_dev = SST.get_dataset_splits(fields=fields)
  >>> print(sst_train[222].word, sst_train[222].char, sep='\n')
  (None, ['A', 'slick', ',', 'engrossing', 'melodrama', '.'])
  (None, ['A', ' ', 's', 'l', 'i', 'c', 'k', ' ', ',', ' ', 'e', 'n', 'g', 'r', 'o', 's', 's', 'i', 'n', 'g', ' ', 'm', 'e', 'l', 'o', 'd', 'r', 'a', 'm', 'a', ' ', '.'])

