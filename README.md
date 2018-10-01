Home of the **TakeLab Podium** project. 

## System prerequisites

- ```virtualenv```
- ```python3.6```
- ```pip```

## Building and running unit tests

You will work in a virtual environment and keep a list of required
dependencies in a ```requirements.txt``` file. The master branch of the 
project **must** be buildable with passing tests **all the time**. 
Code coverage should be kept as high as possible. 

```

virtualenv -p python3.6 env
source env/bin/activate
pip install -r requirements.txt
py.test --cov=takepod test

```

## Adding new dependencies

Adding a new library to a project should be done via ```pip install
<new_framework>```. **Don't forget to add it to requirements.txt** 

The best thing to do is to manually add dependencies to the
```requirements.txt``` file instead of using 
```pip freeze > requirements.txt```. 
See [here](https://medium.com/@tomagee/pip-freeze-requirements-txt-considered-harmful-f0bce66cf895)
why.


## Details

The project is packaged according to official Python packaging
[guidelines](https://packaging.python.org/tutorials/packaging-projects/).

## Project package TODOs

- The project still needs a to have a license picked. 
- Windows and OSX instructions on installation and building


## Windows specifics
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
Install pytorch, when changing pytorch version update url.

```

pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl

```

```
pip install -r requirements.txt
python setup.py install

```

4. Deactivate environment when needed

```

.\env\Scripts\deactivate.bat

```

