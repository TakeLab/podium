from setuptools import setup, find_packages


setup(name='takepod',
      version='0.1',
      description='TakeLab podium (TakePod) project',
      author='TakeLab',
      author_email='takelab@fer.hr',
      license='MIT',
      packages=find_packages(),
      url="https://git.takelab.fer.hr/TakeLab/Podium",
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      zip_safe=False)
