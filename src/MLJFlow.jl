module MLJFlow

using MLJBase: info, name, Model,
    Machine
using MLJModelInterface: flat_params
using MLFlowClient: MLFlow, logparam, logmetric,
    createrun, MLFlowRun, updaterun,
    healthcheck, logartifact, getorcreateexperiment

import MLJBase: save, log_evaluation

include("types.jl")
include("base.jl")
include("service.jl")

# types.jl
export MLFlowLogger

end
