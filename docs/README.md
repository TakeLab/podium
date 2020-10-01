# Podium Documentation

## Generating documentation from source
First, make sure that your environment is set up as described in the [Getting started](https://github.com/mttk/podium/blob/master/README.md#getting-started) section. Additionally, install Sphinx using e.g.
```
pip install -U sphinx
```
Finally, make sure you are positioned in the documentation root directory, i.e. `podium/docs`.

### Building the HTML docs
Run
```
make html
```
from the documentation root directory. The generated HTML source will be placed in `./build/html`. It contains the `index.html` file which you can then open in your favourite browser.

### Building the PDF docs
If you would like to build the PDF version of the documentation instead, make sure you have a working LaTeX install first. After that, build the latex sources using:
```
make latex
```
After that, change the directory to `./build/latex` and generate the resulting PDF from the LaTeX sources:
```
make all-pdf
```
This creates the resulting `podium.pdf` file.

## Notes
We adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) style for documentation.
