module MLJFlow

using MLJBase: info, name, Model,
      Machine, MLJLogger

using MLFlowClient: MLFlow, logparam, logmetric,
      createrun, MLFlowRun, updaterun,
      logartifact, getorcreateexperiment

using OrderedCollections: LittleDict

import MLJBase: save, log_evaluation

include("types.jl")
include("base.jl")
include("client.jl")
include("utilities.jl")

# base.jl
export log_evaluation, save

# client.jl
export runs

# types.jl
export MLFlowLogger

# utilities.jl
export flat_params

end
