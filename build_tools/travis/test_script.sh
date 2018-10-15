#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version

run_tests() {
    if [[ "$RUN_SLOW" == "true" ]]; then
        TEST_CMD="py.test --runslow -s -v --cov=takepod --durations=20"
    else
        TEST_CMD="py.test -v --cov=takepod --durations=20"
    fi
    $TEST_CMD
}

if [[ "$SKIP_TESTS" != "true" ]]; then
    # need to install takepod as a library
    python setup.py install
    run_tests
fi

if [[ "$RUN_FLAKE8" == "true" ]]; then
    flake8
fi

