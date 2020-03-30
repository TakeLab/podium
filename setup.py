from setuptools import setup, find_packages


# assumes no comments in requirements.txt could use internal pip `from
# pip._internal.req import parse_requirements`

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='takepod',
    version='0.0.1',
    description='TakeLab podium (takepod) project',
    author='TakeLab',
    author_email='takelab@fer.hr',
    license='MIT',
    packages=find_packages(),
    url="https://git.takelab.fer.hr/TakeLab/takepod",
    setup_requires=["pytest-runner"],
    # pytest version because of https://github.com/pytest-dev/pytest/issues/3950
    # tests need to be rewritten
    tests_require=["pytest==3.7.4", "pandas", "mock", "pytest_mock", "nltk"],
    install_requires=install_requires,
    package_data={'takepod': ['preproc/stemmer/data/*.txt']},
    zip_safe=False
)
