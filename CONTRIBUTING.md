# Contributing

Thank you for thinking about contributing to Podium. Everyone is more than welcome to file issues, to contribute code via pull requests, to help triage and fix bugs, to improve our documentation or to help out in any other way.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Setup

To setup the project locally, follow the steps below:

1. Fork the repository by clicking on the *Fork* button in the top-right corner on the repository's page. This creates a copy of the repository under you GitHub account.

2. Clone the forked repository and connect it with the base repository.

   ```bash
   git clone git@github.com:<your GitHub handle>/podium.git
   cd podium
   git remote add upstream git@github.com:TakeLab/podium.git
   ```

3. Build the project in development mode.

   To install the minimal set of dependencies, run:

   ```bash
   pip install -e .
   ```

   To install the full set of dependencies for developing Podium, run:

   ```bash
   pip install -e ".[dev]"
   ```

## Submiting pull requests

If the change you wish to make is substantial, e.g. adding a new feature or fixing a bug, please file an issue first to discuss your proposal with the project maintainers. For smaller changes, e.g. fixing typos in documentation, you can submit a PR directly.

Follow the steps below to submit a PR to Podium:

1. Follow the steps in [Setup](#setup) to build the project locally.

2. Create a branch to hold you development changes.

   ```bash
   git checkout -b descriptive-branch-name
   ```

3. Implement your changes.

4. Apply the code style changes (for more details see our [Code and docstring style standards](#code-and-docstring-style-standards)):

   ```bash
   make style 
   ```

5. Run tests:

   ```bash
   python -m pytest -sv tests
   ```

6. If everything goes well, commit your changes and push them to the forked repository:

   ```bash
   git add .
   git commit -m "descriptive commit message"
   git push -u origin descriptive-branch-name
   ```

   Optionally, to be up with the latest changes, sync up with the project repository beforehand: 

   ```bash
   git pull --rebase upstream master
   ```

7. Click on *Pull request* on the webpage of the forked repository to request merging your changes with the project's master branch.

## Code and docstring style standards

In this repository, we use [black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/) and [flake8](http://flake8.pycqa.org/en/latest/) for code sytle. 

Commands to check black, isort and flake8 compliance for written code and tests.

```bash
black --check --line-length 90 --target-version py36 podium tests examples
isort --check-only podium tests examples
flake8 podium tests examples
```

We use [docfomatter](https://github.com/myint/docformatter) for docstring style.

Commands to check docformatter compliance for written docstrings.

```bash
docformatter podium tests examples --check --recursive \
   --wrap-descriptions 80 --wrap-summaries 80 \
   --pre-summary-newline --make-summary-multi-line
```

To check the code and doc style all at once, run:

```bash
make quality
```

Similarly, to apply the required code and doc style changes in the source, run:

```bash
make style
```

## Building and running unit tests

Commands to run tests.

```bash
pip install ".[tests]"
pytest -sv tests
```

## Writing documentation

We adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) style for documentation.

### Building documentation

1. Build and install Podium.

2. Install the dependencies.

   ```bash
   pip install ".[docs]"
   ```

3. Generate the documentation HTML files in `docs/build/html`.

   ```bash
   cd docs
   make html
   ```
