Pytorch RNN classifier
=======================

In this example, we will cover a simple RNN-based classifier model implemented in Pytorch. We will use the IMDB dataset loaded from 🤗/datasets, preprocess it with Fields and train the model briefly.
While having a GPU is not necessary it is recommended as otherwise training the model, even for a single epoch, will take a while.

Loading a dataset from 🤗/datasets
-----------------------------------

As we have covered in :ref:`hf-loading`, we have implemented wrappers around 🤗 dataset classes to enable working with the plethora of datasets implemented therein. We will now briefly go through (1) loading a dataset from 🤗/datasets and (2) wrapping it in Podium classes.

.. code-block:: python

  >>> import datasets
  >>> imdb = datasets.load_dataset('imdb')
  >>> print(imdb)
  DatasetDict({
      train: Dataset({
          features: ['text', 'label'],
          num_rows: 25000
      })
      test: Dataset({
          features: ['text', 'label'],
          num_rows: 25000
      })
      unsupervised: Dataset({
          features: ['text', 'label'],
          num_rows: 50000
      })
  })
  >>> from pprint import pprint
  >>> pprint(imdb['train'].features)
  {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None),
   'text': Value(dtype='string', id=None)}

By calling ``load_dataset`` the dataset was downloaded and cached on disk through the ``datasets`` library. The dataset has two splits we are interested in (``train`` and ``test``).
The main thing we need to pay attention to are the ``features`` of the dataset, in this case ``text`` and ``label``. These features, or data columns, need to be mapped to (and processed by) Podium Fields.

For convenience, we have implemented automatic ``Field`` type inference from 🤗 dataset features -- however it is far from perfect as we have to make many assumptions on the way. We will now wrap the IMDB dataset in Podium and show the automatically inferred Fields.

.. code-block:: python

  >>> from podium.datasets.hf import HFDatasetConverter as HF
  >>> splits = HF.from_dataset_dict(imdb)
  >>> imdb_train, imdb_test = splits['train'], splits['test']
  >>> imdb_train.finalize_fields() # Construct the vocab
  >>> print(*imdb_train.fields, sep="\n")
  Field({
      name: 'text',
      keep_raw: False,
      is_target: False,
      vocab: Vocab({specials: ('<UNK>', '<PAD>'), eager: False, is_finalized: True, size: 280619})
  })
  LabelField({
      name: 'label',
      keep_raw: False,
      is_target: True
  })

Both of the Fields were constructed well, but there are a couple of drawbacks for this concrete dataset. Firstly, the size of the vocabulary is very large (``280619``) -- we would like to trim this down to a reasonable number as we won't be using subword tokenization in this example. 

.. code-block:: python

  >>> print(imdb_train[0])
  Example({
      text: (None, ['Bromwell', 'High', 'is', 'a', 'cartoon', 'comedy.', 'It', 'ran', 'at', 'the', 'same', 'time', 'as', 'some', 'other', 'programs', 'about', 'school', 'life,', 'such', 'as', '"Teachers".', 'My', '35', 'years', 'in', 'the', 'teaching', 'profession', 'lead', 'me', 'to', 'believe', 'that', 'Bromwell', "High's", 'satire', 'is', 'much', 'closer', 'to', 'reality', 'than', 'is', '"Teachers".', 'The', 'scramble', 'to', 'survive', 'financially,', 'the', 'insightful', 'students', 'who', 'can', 'see', 'right', 'through', 'their', 'pathetic', "teachers'", 'pomp,', 'the', 'pettiness', 'of', 'the', 'whole', 'situation,', 'all', 'remind', 'me', 'of', 'the', 'schools', 'I', 'knew', 'and', 'their', 'students.', 'When', 'I', 'saw', 'the', 'episode', 'in', 'which', 'a', 'student', 'repeatedly', 'tried', 'to', 'burn', 'down', 'the', 'school,', 'I', 'immediately', 'recalled', '.........', 'at', '..........', 'High.', 'A', 'classic', 'line:', 'INSPECTOR:', "I'm", 'here', 'to', 'sack', 'one', 'of', 'your', 'teachers.', 'STUDENT:', 'Welcome', 'to', 'Bromwell', 'High.', 'I', 'expect', 'that', 'many', 'adults', 'of', 'my', 'age', 'think', 'that', 'Bromwell', 'High', 'is', 'far', 'fetched.', 'What', 'a', 'pity', 'that', 'it', "isn't!"]),
      label: (None, 1)
  })

When inspecting a concrete instance, there are a few more things to note. Firstly, IMDB instances can be quite long (on average around 200 tokens per instance), secondly, the text wasn't tokenized properly near sentence boundaries (due to using the default ``str.split`` tokenizer) and lastly, the text has varying casing.
We will instead define our own Fields for the corresponding features, add posttokenization hooks which will transform the data, and use those Fields to replace the automatically inferred ones:

.. code-block:: python

  >>> from podium import Field, LabelField, Vocab
  >>> 
  >>> # Lowercasing as a post-tokenization hook
  >>> def lowercase(raw, tokenized):
  ...   return raw, [token.lower() for token in tokenized]
  >>>
  >>> # Truncating as a post-tokenization hook
  >>> def truncate(raw, tokenized, max_length=200):
  ...     return raw, tokenized[:max_length]
  >>>
  >>> vocab = Vocab(max_size=10000)
  >>> text = Field(name="text", 
  ...              numericalizer=vocab,
  ...              include_lengths=True,
  ...              tokenizer="spacy-en_core_web_sm",
  ...              posttokenize_hooks=[truncate, lowercase])
  >>> 
  >>> # The labels are already mapped to indices in /datasets so we will
  >>> # pass them through
  >>> label = LabelField(name="label", numericalizer=lambda x: x)
  >>> fields = {
  ...     'text': text,
  ...     'label': label
  ... }
  >>> 
  >>> # Use the given Fields to load the dataset again
  >>> splits = HF.from_dataset_dict(imdb, fields=fields)
  >>> imdb_train, imdb_test = splits['train'], splits['test']
  >>> imdb_train.finalize_fields()
  >>> print(imdb_train)
  HFDatasetConverter({
      dataset_name: imdb,
      size: 25000,
      fields: [
              Field({
                  name: 'text',
                  keep_raw: False,
                  is_target: False,
                  vocab: Vocab({specials: ('<UNK>', '<PAD>'), eager: False, is_finalized: True, size: 10000})
              }),
              LabelField({
                  name: 'label',
                  keep_raw: False,
                  is_target: True
              })
      
      ]
  })
  >>> print(imdb_train[0])
  Example({
      text: (None, ['bromwell', 'high', 'is', 'a', 'cartoon', 'comedy', '.', 'it', 'ran', 'at', 'the', 'same', 'time', 'as', 'some', 'other', 'programs', 'about', 'school', 'life', ',', 'such', 'as', '"', 'teachers', '"', '.', 'my', '35', 'years', 'in', 'the', 'teaching', 'profession', 'lead', 'me', 'to', 'believe', 'that', 'bromwell', 'high', "'s", 'satire', 'is', 'much', 'closer', 'to', 'reality', 'than', 'is', '"', 'teachers', '"', '.', 'the', 'scramble', 'to', 'survive', 'financially', ',', 'the', 'insightful', 'students', 'who', 'can', 'see', 'right', 'through', 'their', 'pathetic', 'teachers', "'", 'pomp', ',', 'the', 'pettiness', 'of', 'the', 'whole', 'situation', ',', 'all', 'remind', 'me', 'of', 'the', 'schools', 'i', 'knew', 'and', 'their', 'students', '.', 'when', 'i', 'saw', 'the', 'episode', 'in', 'which', 'a', 'student', 'repeatedly', 'tried', 'to', 'burn', 'down', 'the', 'school', ',', 'i', 'immediately', 'recalled', '.........', 'at', '..........', 'high', '.', 'a', 'classic', 'line', ':', 'inspector', ':', 'i', "'m", 'here', 'to', 'sack', 'one', 'of', 'your', 'teachers', '.', 'student', ':', 'welcome', 'to', 'bromwell', 'high', '.', 'i', 'expect', 'that', 'many', 'adults', 'of', 'my', 'age', 'think', 'that', 'bromwell', 'high', 'is', 'far', 'fetched', '.', 'what', 'a', 'pity', 'that', 'it', 'is', "n't", '!']),
      label: (None, 1)
  })

Here, we can see the effect of our hooks and using the spacy tokenizer. Now our dataset will be a bit cleaner to work with. Some data cleaning would still be desired, such as removing tokens which only contain punctuation, but we leave this exercise to the reader :)

Loading pretrained embeddings
-----------------------------
In most use-cases, we want to use pre-trained word embeddings along with our neural model. With Podium, this process is very simple. If your field uses a vocabulary, it has already built an inventory of tokens for your dataset.

For example, we will use the `GloVe <https://nlp.stanford.edu/projects/glove/>`__ vectors. You can read more about loading pretrained vectors in :ref:`pretrained`, but the procedure to load these vectors has two steps: (1) initialize the vector class, which sets all the required paths and (2) obtain the vectors for a pre-defined list of words by calling ``load_vocab``.

.. code-block:: python

  >>> from podium.vectorizers import GloVe
  >>> vocab = fields['text'].vocab
  >>> glove = GloVe()
  >>> embeddings = glove.load_vocab(vocab)
  >>> print(f"For vocabulary of size: {len(vocab)} loaded embedding matrix of shape: {embeddings.shape}")
  For vocabulary of size: 10000 loaded embedding matrix of shape: (10000, 300)
  >>> # We can obtain vectors for a single word (given the word is loaded) like this:
  >>> word = "sport"
  >>> print(f"Vector for {word}: {glove.token_to_vector(word)}")
  Vector for sport: [ 0.34566    0.15934    0.48444   -0.13693    0.18737    0.2678
 -0.39159    0.4931    -0.76111   -1.4586     0.41475    0.55837
  ...
  0.13802    0.36619    0.19734    0.35701   -0.42228   -0.25242
 -0.050651  -0.041129   0.15092    0.22084    0.52252   -0.27224  ]

Defining a simple neural model in Pytorch
------------------------------------------

In this section, we will implement a very simple neural classification model -- a 2-layer BiGRU with a single hidden layer classifier on top of its last hidden state. Many improvements to the model can be made, but this is not our current focus.

.. code-block:: python

  >>> import torch
  >>> import torch.nn as nn
  >>> import torch.nn.functional as F
  >>> 
  >>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
  >>> 
  >>> class RNNClassifier(nn.Module):
  ...     def __init__(self, embedding, embed_dim=300, hidden_dim=300, num_labels=2):
  ...       super(NLIModel, self).__init__()
  ...       self.embedding = embedding
  ...       self.encoder = nn.GRU(
  ...             input_size=embed_dim,
  ...             hidden_size=hidden_dim,
  ...             num_layers=2,
  ...             bidirectional=True,
  ...             dropout=0.3
  ...       )
  ...       self.decoder = nn.Sequential(
  ...             nn.Linear(2*hidden_dim, hidden_dim),
  ...             nn.Tanh(),
  ...             nn.Linear(hidden_dim, num_labels)
  ...       )
  ...
  ...     def forward(self, x, lengths):
  ...         e = self.embedding(x)
  ...         h_pack = pack_padded_sequence(e, 
  ...                                       lengths,
  ...                                       enforce_sorted=False,
  ...                                       batch_first=True)
  ...
  ...         _, h = self.encoder(h_pack) # [2L x B x H]
  ...
  ...         # Concat last state of left and right directions
  ...         h = torch.cat([h[-1], h[-2]], dim=-1) # [B x 2H]
  ...         return self.decoder(h)

There. We will now define the prerequisites for pytorch model training, where we will use a GPU for speed, however running the model for one epoch will is possible albeit time-consuing even without a GPU.

.. code-block:: python

  >>> embed_dim = 300
  >>> padding_index = text.vocab.get_padding_index()
  >>> embedding_matrix = nn.Embedding(len(text.vocab), embed_dim,
  ...                                 padding_idx=padding_index)
  >>> # Copy the pretrained GloVe word embeddings
  >>> embedding_matrix.weight.data.copy_(torch.from_numpy(embeddings))
  >>>
  >>> device = torch.device("cuda:0")
  >>> model = RNNClassifier(embedding_matrix)
  >>> model = model.to(device)
  >>> criterion = nn.CrossEntropyLoss()
  >>> optimizer = torch.optim.Adam(model.parameters())

Now that we have the model setup code ready, we will first define helper method to measure accuracy of our model after each epoch:

.. code-block:: python

  >>> import numpy as np
  >>> def update_stats(accuracy, confusion_matrix, logits, y):
  ...     _, max_ind = torch.max(logits, 1)
  ...     equal = torch.eq(max_ind, y)
  ...     correct = int(torch.sum(equal))
  ... 
  ...     for j, i in zip(max_ind, y):
  ...         confusion_matrix[int(i),int(j)]+=1
  ...     return accuracy + correct, confusion_matrix

and now the training loop for the model:

.. code-block:: python

  >>> import tqdm
  >>> def train(model, data, optimizer, criterion, num_labels):
  ...     model.train()
  ...     accuracy, confusion_matrix = 0, np.zeros((num_labels, num_labels), dtype=int)
  ...     for batch_num, batch in tqdm.tqdm(enumerate(data), total=len(data)):
  ...         x, lens = batch.text
  ...         y = batch.label
  ...         logits = model(x, lens)
  ...         accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
  ...         loss = criterion(logits, y.squeeze())
  ...         loss.backward()
  ...         optimizer.step()
  ...     print("[Accuracy]: {}/{} : {:.3f}%".format(
  ...           accuracy, len(data)*data.batch_size, accuracy / len(data) / data.batch_size * 100))
  ...     return accuracy, confusion_matrix

and now, we are done with our model code. Let's turn back to Podium and see how we can set up batching for our training loop to start ticking.

Minibatching data in Podium
--------------------------------

We have covered batching data in :ref:`minibatching` and advanced batching through bucketing in :ref:`bucketing`. We will use the plain Iterator and leave bucketing for you to change to see how much the model speeds up when minimizing padding. One change we would like to do when iterating over data is to obtain the data matrices as torch tensors on the ``device`` we defined previously. We will now demonstrate how to do this by setting the ``matrix_class`` argument of the :class:`podium.datasets.Iterator`\:

.. code-block:: python

  >>> from podium import Iterator
  >>> # Closure for converting data to given device
  >>> def gpu_tensor(data):
  ...     return torch.tensor(data).to(device)
  >>> # Initialize our iterator
  >>> train_iter = Iterator(imdb_train, batch_size=32, matrix_class=gpu_tensor)
  >>>
  >>> epochs = 5
  >>> for epoch in range(epochs):
  >>>     train(model, train_iter, optimizer, criterion, num_labels=2)
  [Accuracy]: 20050/25024 : 80.123%
  [Accuracy]: 22683/25024 : 90.645%
  [Accuracy]: 23709/25024 : 94.745%
  [Accuracy]: 24323/25024 : 97.199%
  [Accuracy]: 24595/25024 : 98.286%

And we are done! In our case, the model takes about one minute per epoch on a GPU, but this can be sped up by using bucketing, which we recommend you try out yourself.
