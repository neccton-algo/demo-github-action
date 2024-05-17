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


<!--  LocalWords:  github UNet julia
 -->
