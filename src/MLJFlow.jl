module MLJFlow

using MLJBase: Model, Machine, name
using MLJModelInterface: flat_params
using MLFlowClient: MLFlow, logparam, logmetric, createrun, MLFlowRun,
    updaterun, logartifact, getorcreateexperiment

import Base: show
import MLJBase: save, log_evaluation

include("types.jl")
include("base.jl")
include("service.jl")

end
