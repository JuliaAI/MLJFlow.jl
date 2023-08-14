# MLJFlow

| Branch | Build |
| :---: | :---: |
| dev | [![CI](https://github.com/pebeto/MLJFlow.jl/actions/workflows/CI.yml/badge.svg?branch=dev)](https://github.com/pebeto/MLJFlow.jl/actions/workflows/CI.yml) |

[MLJ](https://github.com/alan-turing-institute/MLJ.jl) is a Julia framework for
combining and tuning machine learning models. MLJFlow is a package that extends
the MLJ capabilities to use [MLFlow](https://mlflow.org/) as a backend for
model tracking and experiment management. To be specific, MLJFlow provides a
close to zero-preparation to use MLFlow with MLJ; by the usage of function
extensions that automate the MLFlow cycle (create experiment, create run, log
metrics, log parameters, log artifacts, etc.).

## Background
This project is part of the GSoC 2023 program. The proposal description can be
found [here](https://summerofcode.withgoogle.com/programs/2023/projects/iRxuzeGJ).
The entire workload is divided into three different repositories:
[MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl), 
[MLFlowClient.jl](https://github.com/JuliaAI/MLFlowClient.jl) and this one.

## Features
- [x] MLFlow cycle automation (create experiment, create run, log metrics, log
  parameters, log artifacts, etc.)
- [x] Wrapper type used by MLJ to store MLFlow metadata and client instance
  from MLFlowClient.jl
- [x] MLJ extended functions to allow MLFlow logging
- [x] Polished compatibility with composed models
- [ ] Polished compatibility with tuned models
- [ ] Polished compatibility with iterative models
