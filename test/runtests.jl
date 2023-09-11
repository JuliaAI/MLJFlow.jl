using Test

using MLJFlow

using MLJBase
using MLJModels
using MLJTuning
using MLFlowClient
using MLJModelInterface

const X, y = make_moons(100)
const LOGGER = MLFlowLogger("http://localhost:5000";
    experiment_name="MLJFlow tests",
    artifact_location="/tmp/mlj-test")
const DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree

include("base.jl")
include("types.jl")
include("tunedmodels.jl")
