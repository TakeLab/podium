Basic usage
============

Machine learning datasets in Podium are stored in `Dataset` classes. A `podium.Dataset` is a shallow container with the functionality of retrieving instances of the dataset, whether they are stored in-memory, disk-backed or streamed. Along with instance storage, the `Dataset` stores the preprocessing pipeline for each data field in :py:class:`podium.Field` classes, which we will cover later.

For a very basic walkthrough, we will use a pre-defined dataset: the Stanford Sentiment Treebank. Podium contains a number of pre-defined concrete datasets which are automatically downloaded and unpacked upon your request. Apart from pre-defined datasets, we wrap the popular `huggingface/datasets` repository and enable users to seamlessly use all therein present datasets as if they were Podium datasets. Lastly, we allow loading of custom datasets from a number of predefined formats through the :py:class:`podium.TabularDataset` interface.


Loading predefined datasets
----------------------------

.. code-block:: python

  from podium.datasets import SST
  sst_train, sst_test, sst_dev = SST.get_dataset_splits()

