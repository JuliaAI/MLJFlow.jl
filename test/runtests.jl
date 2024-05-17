using Test

using MLJFlow

using MLJBase
using MLJModels
using MLFlowClient
using MLJModelInterface
using StatisticalMeasures

# To run this tests, you need to set the URI of your MLFlow server. By default,
# you can set:
#
# ENV["MLFLOW_TRACKING_URI"] = "http://localhost:5000/api"
#
# For more information, see https://mlflow.org/docs/latest/quickstart.html#view-mlflow-runs-and-experiments
if ~haskey(ENV, "MLFLOW_TRACKING_URI")
    error("WARNING: MLFLOW_TRACKING_URI is not set. To run this tests, you need to set the URI of your MLFlow server API")
end

include("base.jl")
include("types.jl")
include("service.jl")

