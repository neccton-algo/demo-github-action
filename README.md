[![Build Status](https://github.com/neccton-algo/demo-github-action/workflows/CI/badge.svg)](https://github.com/neccton-algo/demo-github-action/actions)

# demo-github-action
Predict satellite data based on L3 data (including missing data) with a UNet

![image](https://github.com/neccton-algo/demo-github-action/assets/9881475/cda82b0f-337c-43e5-bbf8-b34dea9924e0)




# Github action

* Why continuous integration testing? Detect failures early.
* Test your code on every commit, pull request
* Possibly on multiple platforms (Ubuntu, Mac OS, Windows...) and different versions of julia/python/...
* All these combinations represent the test matrix
* However, even if GitHub action is free for open source projects, consider the environmental impact and avoid unnecessary combinations of OS/version/...

## Recommentations

* Automate the installation of all dependencies by declaring them in the `requirements.txt` or `pyproject.toml` files
* First make sure that you can instanticate the project from a clean environement and run all tests locally
* There is no GPU on github actions
* Tests should be short
* Badge for your `README.md`:
```
[![Build Status](https://github.com/neccton-algo/demo-github-action/workflows/CI/badge.svg)](https://github.com/neccton-algo/demo-github-action/actions)
```
* Notebooks can be run and tested with `nbconvert`
* There is an environement variable `CI` which is set to `true` on github action to adapt the code path to be tested.
* For python: consider testing frameworks like `py.test`, `Hypothesis`, `tox`, ... 
* For julia: write test in the `test/runtests.jl` which gets executed with `Pkg.test()`.
* Monitor code coverage with e.g. [codecov.io](http://codecov.io).


<!--  LocalWords:  github UNet julia
 -->
