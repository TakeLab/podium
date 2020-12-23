@echo off

if /i "%1"=="quality" goto quality
if /i "%1"=="style" goto style
if /i "%1"=="test" goto test
if /i "%1"=="clean" goto clean
if "%1"!="" goto error

:quality
    echo Checking code quality.
    echo black --check --line-length 90 --target-version py36 podium tests examples
    black --check --line-length 90 --target-version py36 podium tests examples
    echo isort --check-only podium tests examples
    isort --check-only podium tests examples
    echo flake8 podium tests examples
    flake8 podium tests examples
    goto :EOF

:style
    echo Applying code style changes.
    echo black --line-length 90 --target-version py36 podium tests examples
    black --line-length 90 --target-version py36 podium tests examples
    echo isort podium tests examples
    isort podium tests examples
    goto :EOF

:test
    echo Running tests.
    echo python -m pytest -sv tests
    python -m pytest -sv tests
    goto :EOF

:clean
    echo Cleaning up the project.
    rmdir /s /q .pytest_cache 2>NUL
    rmdir /s /q podium.egg-info 2>NUL
    rmdir /s /q dist 2>NUL
    rmdir /s /q build 2>NUL
    for /f "delims=" %%a in ('dir /b /s . ^| findstr /e /r "__pycache__$" ') do rmdir /s /q "%%a" 2>NUL
    del /q *.pyc 2>NUL
    del /q *.pyo 2>NUL
    call docs\make.bat clean
    goto :EOF

:error
    echo make: *** No rule to make target '%1%'.  Stop.
    goto :EOF
