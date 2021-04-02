@echo off

if /i "%1"=="quality" goto quality
if /i "%1"=="style" goto style
if /i "%1"=="test" goto test
if /i "%1"=="clean" goto clean
if "%1"!="" goto error

:quality
    echo Checking code quality.
    echo black --check --line-length 90 --target-version py36 podium tests
    black --check --line-length 90 --target-version py36 podium tests
    echo isort --check-only podium tests
    isort --check-only podium tests
    echo docformatter podium tests --check --recursive --wrap-descriptions 80 --wrap-summaries 80 --pre-summary-newline --make-summary-multi-line
    docformatter podium tests --check --recursive --wrap-descriptions 80 --wrap-summaries 80 --pre-summary-newline --make-summary-multi-line
    echo flake8 podium tests
    flake8 podium tests
    goto :EOF

:style
    echo Applying code style changes.
    echo black --line-length 90 --target-version py36 podium tests
    black --line-length 90 --target-version py36 podium tests
    echo isort podium tests
    isort podium tests
    echo docformatter podium tests -i --recursive --wrap-descriptions 80 --wrap-summaries 80 --pre-summary-newline --make-summary-multi-line
    docformatter podium tests -i --recursive --wrap-descriptions 80 --wrap-summaries 80 --pre-summary-newline --make-summary-multi-line
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
