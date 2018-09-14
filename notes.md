### Bayesian optimization 

- part of Sequential model-based optimization algorithm family
- multiple python libraries already available
 - [spearmint](https://github.com/HIPS/Spearmint)
 - [smac](https://www.cs.ubc.ca/labs/beta/Projects/SMAC/)
 - [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
 - [hyperopt](https://github.com/hyperopt/hyperopt)
 - [pygpgo](http://pygpgo.readthedocs.io/en/latest/)
 - [GPyOpt](https://sheffieldml.github.io/GPyOpt/)

Four preconditions for doing BO

1. Objective Function: takes in an input and returns a loss to minimize
2. Domain space: the range of input values to evaluate Optimization
3. Algorithm: the method used to construct the surrogate function and
   choose the next values to evaluate 
4. Results: score, value pairs that the algorithm uses to build the model

Benefits of Bayesian optimization over random search

- works better most of the time (less iterations)
- returns distributions of searched spaces (which learning rates are
  better than other)
- returns trials

Resources

- Excellent notebook on [Bayesian
  optimization](https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Introduction%20to%20Bayesian%20Optimization%20with%20Hyperopt.ipynb)
- [Blog post using scikit](https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)


### Active learning

The idea is to label part of the dataset and try to intelligently select
which data to label to maximize model performance. 

Procedure

1. Split data to seed(will be labelled) and unlabelled. 

2. Train the model using seed

3. Choose unlabelled instances to label based on one of many selection
   criteria (Pool-based, Stream-baseda) based on least confidence

4. Iterate  2. and 3. until some stopping criteria is met


Python libraries

- [google active learning](https://github.com/google/active-learning),
  6 sampling strategies
- [modAL](https://cosmic-cortex.github.io/modAL/#introduction)
- [libact](https://github.com/ntucllab/libact), Pool-based
  Active Learning in Python
- [acton](https://github.com/chengsoonong/acton)

Resources

- [Tutorial](https://towardsdatascience.com/active-learning-tutorial-57c3398e34d)
- [Begginer datacamp tutorial](https://www.datacamp.com/community/tutorials/active-learning)

### Framework part

- interchanging saved models -- use ONNX format (ici generalnije)
- continuous integration using Travis CI

### Development setup

- Python (you will likely use these)
  - [pytest](https://docs.pytest.org/en/latest/getting-started.html)
  - [Decorators in python](https://realpython.com/primer-on-python-decorators/)
  - [Debugger pdb](https://docs.python.org/2/library/pdb.html)
  - pandas
  - pytorch
  - [pytest](https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest)

https://docs.pytest.org/en/latest/example/simple.html
- Git 
  - [git basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
  - [how I operate in git](https://medium.com/@fredrikmorken/why-you-should-stop-using-git-rebase-5552bee4fed1)
- Linux (all TakeLab servers have Ubuntu installed)
  - [tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
    / [nohup](https://linux.101hacks.com/unix/nohup-command/)
  - [htop](http://www.deonsworld.co.za/2012/12/20/understanding-and-using-htop-monitor-system-resources/)
  - nvidia-smi
  

Advanced preconditions (also depending on personal choice)

- vim (vimtutor)
- [Tmux + vim](https://blog.bugsnag.com/tmux-and-vim/)


### Some general test principles

- unit test should run within a few seconds
- unit tests should mock external resources
- integration tests should be run separately from unit tests 
- unit tests should have extremely high coverage, integration tests should
  focus on only testing integration

Some general source writing practices

- discern between public and private methods ('\_' for private)
- proposed: write numpydoc-style documentation for public methods
- use flake8 to check code formatting

### Onboarding -- week 1 checklist

By the end of week 1 you should have:

- git access (use pull request)
- ssh access to all TakeLab servers (ask sysadmins on Trello to give you
  access)
- trello access (ask TakeLab members to give you access)
- create and launch a deep learning model on TakeLab GPU resources 
- checkout, build and run tests of TakeLab podium
- make a simple code change on TakeLab podium passing
- pick up a software implementation task
