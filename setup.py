from setuptools import setup, find_packages

setup(name='takepod',
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
      tests_require=["pytest==3.7.4", "pandas", "mock", "pytest_mock"],
      install_requires=[
        "paramiko", "tqdm", "numpy", 
        "requests", "scipy", "sklearn", "dill"
      ],
      package_data={'takepod': ['preproc/stemmer/data/*.txt']},
      zip_safe=False)
