module MLJFlow

using Logging: with_logger, NullLogger
using MLJBase: Model, Machine, machine, name
using MLJModelInterface: flat_params
using MLFlowClient: MLFlow, logparam, logmetric, createrun, Run,
    updaterun, uploadartifact, downloadartifact, getexperiment,
    getexperimentbyname, createexperiment, restoreexperiment,
    RunStatus, Tag

import Base: show
import MLJBase: save, log_evaluation

include("types.jl")
include("base.jl")
include("service.jl")

end
