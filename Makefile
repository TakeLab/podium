.PHONY: quality style test

# Check code quality
quality:
	@echo Checking code quality.
	black --check --line-length 90 --target-version py36 podium tests examples
	isort --check-only podium tests examples
	flake8 podium tests examples

# Enforce code quality in source 
style:
	@echo Applying code style changes.
	black --line-length 90 --target-version py36 podium tests examples
	isort podium tests examples

# Run tests
test:
	@echo Running tests.
	python -m pytest -sv tests
