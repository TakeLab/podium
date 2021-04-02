# Roadmap

If you are interested in making a contribution to Podium, this page outlines some changes we are planning to focus on in the near future. Feel free to propose improvements and moficiations either via [discussions](https://github.com/TakeLab/podium/discussions) or by raising an [issue](https://github.com/TakeLab/podium/issues).

Order does not reflect importance.

## Major changes

- Dynamic application of Fields
  - Right now, for every change in Fields the dataset needs to be reloaded. The goal of this change would be to allow users to replace or update a Field in a Dataset. The Dataset should be aware of this change (e.g. by keeping a hash of the Field object) and if it happens, recompute all the necessary data for that Field.

  The current pattern is:
  ```python
    # Load a dataset
    fields = {'text':text, 'label':label}
    dataset = load_dataset(fields=fields)

    # Decide to change something with one of the Fields
    text = Field(..., tokenizer=some_different_tokenizer)
    # Potentially expensive dataset loading is required again
    dataset = load_dataset(fields=fields)
  ```
  Dataset instances should instead detect changes in a Field and recompute values (Vocabs) for the ones that changed.

- Parallelization
  - For data preprocessing (apply Fields in parallel)
  - For data loading

- Conditional processing in Fields
  - Handle cases where the values computed in one Field are dependent on values computed in another Field 

- Experimental pipeline
  - `podium.experimental`, wrappers for model framework agnostic training & serving
  - Low priority 

## Minor changes

- Populate hooks & preprocessing utilities
  - Lowercase, truncate, extract POS, ...
- Populate pretrained vectors
  - Word2vec
- Improve Dataset coverage
  - Data wrappers / abstract loaders for other source libraries and input formats
- BucketIterator modifications
  - Simplify setting the sort key (e.g., in the basic case  where the batch should be sorted according to the length of a single Field, accept a Field name as the argument)
- Improve HF/datasets integration
  - Better automatic Field inference from features
  - Cover additional feature datatypes (e.g., image data)
  - Cleaner API?
- Centralized and intuitive download script
  - Low priority as most data loading is delegated to hf/datasets
- Add a Mask token for MLM (can be handled with posttokenization hooks right now, but not ideal)

## Documentation

- Examples
  - Language modeling
  - Tensorflow model
  - Various task types
- Chapters
  - Handling datasets with missing tokens
  - Loading data from pandas / porting data to pandas
  - Loading CoNLL datasets
  - Implementing your own dataset subclass
