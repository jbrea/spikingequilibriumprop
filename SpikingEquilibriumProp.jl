__precompile__()
module SpikingEquilibriumProp
include("equilibriumprop.jl")
include("backprop.jl")
#include("plotting.jl")

export EquipropConfig, getequipropnet, createandrunsim, getequipropnetandconf,
	learn! 
end

