__precompile__()
module SpikingEquilibriumProp
include("equilibriumprop.jl")
include("backprop.jl")

export EquipropConfig, getequipropnet, createandrunsim, getequipropnetandconf,
	learn! 
end

