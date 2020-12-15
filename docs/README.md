# Podium Documentation

## Generating documentation from source
First, make sure that your environment is set up as described in the [Getting started](../README.md#getting-started) section. To install the dependencies required to successfully build the documentation, run the following command:
```bash
pip install .[docs]
```
Finally, make sure you are positioned in the documentation root directory, i.e. `podium/docs`.

### Building the HTML docs
Run
```bash
make html
```
from the documentation root directory. The generated HTML source will be placed in `./build/html`. It contains the `index.html` file which you can then open in your favourite browser.

### Building the PDF docs
If you would like to build the PDF version of the documentation instead, make sure you have a working LaTeX install first. After that, build the latex sources using:
```bash
make latex
```
After that, change the directory to `./build/latex` and generate the resulting PDF from the LaTeX sources:
```bash
make all-pdf
```
This creates the resulting `podium.pdf` file.

## Notes
We adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) style for documentation.
