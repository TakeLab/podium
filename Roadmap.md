# Roadmap

If you are interested in contributing to Podium, some of our 
Order does not reflect importance.

## Major changes

- Dynamic application of Fields
  - Right now, for every change in Fields the dataset needs to be reloaded. The goal of this change would be to allow users to replace or update a Field in a Dataset. The Dataset should be aware of this change (e.g. by keeping a hash of the Field object) and if it happens, recompute all the necessary data for that Field.
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
- Improve Dataset coverage
  - Data wrappers / abstract loaders for other source libraries and input formats
- BucketIterator modifications
  - Simplify setting the sort key (e.g., in the basic case  where the batch should be sorted according to the length of a single Field, accept a Field name as the argument)
- Improve HF/datasets integration
  - Better automatic Field inference from features
  - Cover additional feature datatypes (e.g., image data)
- Centralized and intuitive download script
  - Low priority as most data loading is delegated to hf/datasets
- Add a Mask token for MLM (can be handled with posttokenization hooks right now, but not ideal)
- Populate pretrained vectors
  - Word2vec

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
