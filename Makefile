.PHONY: quality style test clean

# Check code quality
quality:
	@echo Checking code and doc quality.
	black --check --line-length 90 --target-version py36 podium tests
	isort --check-only podium tests
	docformatter podium tests --check --recursive \
    	--wrap-descriptions 80 --wrap-summaries 80 \
        --pre-summary-newline --make-summary-multi-line
	flake8 podium tests

# Enforce code quality in source 
style:
	@echo Applying code and doc style changes.
	black --line-length 90 --target-version py36 podium tests
	isort podium tests
	docformatter podium tests -i --recursive \
    	--wrap-descriptions 80 --wrap-summaries 80 \
        --pre-summary-newline --make-summary-multi-line

# Run tests
test:
	@echo Running tests.
	python -m pytest -sv tests

# Clean up the project
clean:
	@echo Cleaning up the project.
	rm -rf .pytest_cache/
	rm -rf podium.egg-info/
	rm -rf dist/
	rm -rf build/
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
	$(MAKE) -C docs clean
