__precompile__()
module SpikingEquilibriumProp
include("equilibriumprop.jl")
include("backprop.jl")
#include("plotting.jl")

export EquipropConfig, getequipropnet, createandrunsim, getequipropnetandconf,
	learn!, StaticDenseConnection, TransposeDenseConnection, One2OneConnection,
	updatenet!, updatenetasync!, Network, BackpropConfig,
	getaverageinputtraceprediction
end

