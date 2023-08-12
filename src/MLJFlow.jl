module MLJFlow

using MLJBase: info, name, Model,
    Machine, deep_params, flat_params
using OrderedCollections: LittleDict
using MLFlowClient: MLFlow, logparam, logmetric,
    createrun, MLFlowRun, updaterun,
    healthcheck, logartifact, getorcreateexperiment

import MLJBase: save, log_evaluation

include("types.jl")
include("base.jl")
include("client.jl")

# types.jl
export MLFlowLogger

end
