# TakeLab Podium

Home of the **TakeLab Podium** project. Podium is a Python machine learning library that helps users to accelerate use of NLP models. 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
Special notes for Windows systems are at the end of this chapter.

### Prerequisites
For building this project system needs to have installed the following:
- [```git```](https://git-scm.com/)
- [```virtualenv```](https://virtualenv.pypa.io/en/latest/installation/)
- [```python3.6```](https://www.python.org/downloads/release/python-360/)
- [```pip```](https://pypi.org/project/pip/)

### Building and running unit tests

You will work in a virtual environment and keep a list of required
dependencies in a ```requirements.txt``` file. The master branch of the 
project **must** be buildable with passing tests **all the time**. 
Code coverage should be kept as high as possible. 

Commands to setup virtual environment and run tests.
```
virtualenv -p python3.6 env
source env/bin/activate
pip install -r requirements.txt
py.test --cov=takepod test
```

If you intend to develop part of podium you should use following command to install podium.
```
python setup.py develop
```
In other cases it should be enough to run ```python setup.py``` for podium to be added to python environment.


The project is packaged according to official Python packaging [guidelines](https://packaging.python.org/tutorials/packaging-projects/).

We recommend use of [pytest](https://docs.pytest.org/en/latest/) and [pytest-mock](https://pypi.org/project/pytest-mock/) library for testing when developing new parts of the library.

### Adding new dependencies

Adding a new library to a project should be done via ```pip install <new_framework>```. **Don't forget to add it to requirements.txt** 

The best thing to do is to manually add dependencies to the
```requirements.txt``` file instead of using 
```pip freeze > requirements.txt```. 
See [here](https://medium.com/@tomagee/pip-freeze-requirements-txt-considered-harmful-f0bce66cf895)
why.

### Windows specifics
1. install python 3.6 64 bit with pip
(if needed update pip ``` python3 -m pip install --upgrade pip ```)
2. install and create virtual environment  
```
pip3 install virtualenv
virtualenv -p python3 env
```
3. Activate environment  
```
\path\to\env\Scripts\activate.bat -- using CMD
\path\to\env\Scripts\activate.ps1 -- using PowerShell
```

Note: To create a virtualenv under a path with spaces in it on Windows, you’ll need the win32api library installed.

3. Install requirements and run tests
Install pytorch.  
```
pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
```
Install other requirements.  
```
pip install -r requirements.txt
python setup.py install
```

4. Deactivate environment when needed  
```
.\env\Scripts\deactivate.bat
```

## Usage examples
For usage examples see examples in [takepod/examples](https://github.com/FilipBolt/takepod/tree/master/takepod/examples)

## Code style standards
In this repository we use [numpydoc](https://numpydoc.readthedocs.io/en/latest/) as a standard for documentation and Flake8 for code sytle. Code style references are [Flake8](http://flake8.pycqa.org/en/latest/) and [PEP8](https://www.python.org/dev/peps/pep-0008/).

Commands to check flake8 compliance for written code and tests.
```
flake8 takepod
flake8 test
```

For automatic flake8 compliance we recommend use of [Black](https://github.com/ambv/black).

## Deployment
Additional notes about how to deploy this library in production environment.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/FilipBolt/takepod/tags). 

## Authors

* Podium is currently maintained by [Ivan Smoković](https://github.com/ivansmokovic), [Domagoj Pluščec](https://github.com/domi385), [Filip Boltužić](https://github.com/FilipBolt) and [Martin Tutek](https://github.com/mttk). A non-exhaustive but growing list needs to mention: [Marin Kačan](https://github.com/mkacan), [Mate Mijolić](https://github.com/matemijolovic).
* Project made as part of TakeLab laboratory at Faculty of Electrical Engineering and Computing, University of Zagreb
* Laboratory url: http://takelab.fer.hr

See also the list of [contributors](https://github.com/FilipBolt/takepod/graphs/contributors) who participated in this project.

## License

This project is licensed under the (TODO) - see the [LICENSE.md](LICENSE.md) file for details

## List of References


## Project package TODOs

- The project still needs a to have a license picked. 
- OSX instructions on installation and building
- Add used references
- Add deployment notes


