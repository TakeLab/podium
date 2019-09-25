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
      tests_require=["pytest", "pandas"],
      install_requires=[
        "paramiko", "tqdm", "numpy", 
        "requests", "scipy", "sklearn", "dill"
      ],
      package_data={'takepod': ['preproc/stemmer/data/*.txt']},
      zip_safe=False)
