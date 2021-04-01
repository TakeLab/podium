
TFIDF + scikit-learn SVM
=========================

In this example, we will cover a once popular family of models -- support vector machines (SVMs) with TF-IDF representations. As a simple example, we will analyse binary classification on the Stanford sentiment treebank (SST) dataset.

First, we will implement a minimalistic example without much additional preprocessing. Since we're using TFIDF representation of our dataset, it is smart to limit the size of the vocabulary as each word needs to be present in the instance TFIDF representation. Let's load the SST dataset and convert it into a single batch:

.. code-block:: python

  >>> from podium import Vocab, Field, LabelField
  >>> from podium.datasets import SST
  >>> vocab = Vocab(max_size=5000, specials=())
  >>> text = Field(name='text', numericalizer=vocab, disable_batch_matrix=True)
  >>> label = LabelField(name='label')
  >>> fields = {'text': text, 'label': label}
  >>> train, dev, test = SST.get_dataset_splits(fields=fields)
  >>> train.finalize_fields()
  >>> x, y = train.batch(add_padding=True)

We have now loaded our dataset, finalized its Fields and obtained it as a batch of input and target data. What we need to do next is define the TF-IDF vectorization for each instance in the dataset. This is done by using our :class:`podium.vectorizers.TfIdfVectorizer`, which adapts the ``scikit-learn`` vectorizer to the Podium input data.

.. code-block:: python

  >>> from podium.vectorizers import TfIdfVectorizer
  >>> tfidf_vectorizer = TFIdfVectorizer()
  >>> tfidf_vectorizer.fit(train, field=train.field('text'))
  >>> tfidf_x = tfidf_vectorizer.transform(x.text)
  >>> print(type(tfidf_batch), tfidf_batch.shape)
  <class 'scipy.sparse.csr.csr_matrix'> (6920, 5000)

We have transformed the train dataset to a sparse matrix containing TF-IDF values for each word in the vocabulary in each instance. What is left to do now is to train our classification model:

.. code-block:: python

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.metrics import accuracy_score
  >>> # Train the SVM on the training set
  >>> svm = LinearSVC()
  >>> svm.fit(tfidf_batch, y.label.ravel())
  >>> # Obtain accuracy on the train set
  >>> y_hat = svm.predict(tfidf_batch)
  >>> acc = accuracy_score(y_hat, y.label.ravel())
  >>> print(f"Accuracy on the train set: {acc:.4f}")
  Accuracy on train set: 0.9597

And for a more accurate performance evaluation of our model we turn to the test set:

.. code-block:: python

  >>> test_x, test_y = test.batch()
  >>> tfidf_test_batch = tfidf_vectorizer.transform(test_x.text)
  >>> y_test_hat = svm.predict(tfidf_test_batch)
  >>> acc = accuracy_score(y_test_hat, test_y.label.ravel())
  >>> print(f"Accuracy on the test set: {acc:.4f}")
  Accuracy on the test set: 0.7946

Our basic unigram TF-IDF linear SVM performs pretty well on the SST dataset, reaching accuracy of almost ``0.8``. While this example encapsulates the basics of using Podium with ``scikit-learn``\s ``SVM``, we will delve a bit deeper and consider some additional preprocessing.

Using ngram features
---------------------

We have only considered basic unigram features for our model, and this is somewhat prohibitive. Apart from that, we have not implemented any preprocessing for our dataset, and our dataset is cased -- which might be detrimental for the performance of our model since we agressively trim the vocabulary size.

We will do two things: (1) implement a pre-tokenization hook to lowercase our data, which in our case is fine as we are using the case-insensitive ``str.split`` as a tokenizer, and (2) implement ngram extraction as a post-tokenization hook. For a more detailed overview of hooks and how to use them, check :ref:`fields`. We will first implement our lowercase hook:

.. code-block:: python

  >>> def lowercase(raw):
  ...   """Lowercases the input string"""
  ...   return raw.lower()

And then implement flexible ngram extraction where the ``n`` is an interval using ``nltk``\s ``ngrams`` function:

.. code-block:: python
  
  >>> from ntlk import ngrams
  >>> class NGramHook:
  ...   # Transforms a sequence of unigrams into a sequence of
  ...   # [min_n, max_n]-grams
  ...   def __init__(self, min_n, max_n):
  ...     self.min_n = min_n
  ...     self.max_n = max_n
  ...   def __call__(self, raw, tokenized):
  ...     tokenized_ngrams = []
  ...     for n in range(self.min_n, self.max_n+1):
  ...        tokenized_ngrams.extend(ngrams(tokenized, n))
  ...     return raw, tokenized_ngrams

We will now incorporate these two hooks into our text input Field:

.. code-block:: python

  >>> # Use [1-3]grams, inclusive
  >>> ngram_hook = NGramHook(1,3)
  >>> vocab = Vocab(max_size=5000, specials=())
  >>> text = Field(name='text', numericalizer=vocab, 
                   disable_batch_matrix=True,
                   pretokenization_hooks=[lowercase],
                   posttokenization_hooks=[ngram_hook]
                   )
  >>> label = LabelField(name='label')
  >>> fields = {'text': text, 'label': label}
  >>> train, dev, test = SST.get_dataset_splits(fields=fields)
  >>> train.finalize_fields()
  >>> print(text.vocab.itos[40:50])
  [('at',), ('from',), ('one',), ('have',), ('I',), ('like',), ('his',), ('in', 'the'), ('all',), ("'",)]

We can see that our new Vocab now contains tuples as its tokens -- as long as an item in a sequence is hashable, we can represent it as part of a Vocab! We can see that one 2-gram ``('in', 'the')`` has made its way into the 50 most frequent tokens.

As before, we need to train the TFIDF vectorizer and apply it to our data (which now includes 1-, 2- and 3-grams):

.. code-block:: python

  >>> dataset_batch = train.batch(add_padding=True)
  >>> tfidf_vectorizer = TfIdfVectorizer()
  >>> tfidf_vectorizer.fit(train, field=train.field('text'))
  >>> tfidf_batch = tfidf_vectorizer.transform(dataset_batch.text)
  >>> print(type(tfidf_batch), tfidf_batch.shape)
  <class 'scipy.sparse.csr.csr_matrix'> (6920, 5000)

We can now train our SVM classification model and evaluate it on the train and test set:

.. code-block:: python

  >>> svm = LinearSVC()
  >>> text, label = dataset_batch
  >>> svm.fit(tfidf_batch, label.ravel())
  >>> # Compute accuracy on the train set
  >>> y_hat = svm.predict(tfidf_batch)
  >>> acc = accuracy_score(y_hat, label.ravel())
  >>> print(f"Accuracy on the train set: {acc:.4f}")
  Accuracy on the train set: 0.9575
  >>>
  >>> # Compute accuracy on the test set
  >>> test_text, test_label = test.batch(add_padding=True)
  >>> tfidf_test_batch = tfidf_vectorizer.transform(test_text)
  >>> y_test_hat = svm.predict(tfidf_test_batch)
  >>> acc = accuracy_score(y_test_hat, test_label.ravel())
  >>> print(f"Accuracy on the test set: {acc:.4f}")
  Accuracy on the test set: 0.7743

Sadly, our new model didn't perform better than our initial one on the train set, but there are many avenues we can try further, such as tuning the hyperparameters of the LinearSVC model on the development set or filtering out stop words and punctuation. We encourage you to open this example in Colab and try some things yourself!
