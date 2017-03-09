include("../equilibriumprop.jl")

noiselevels = []
taus = []
losses = []

T = 10^6

for _ in 1:10
	noiselevel = 10^-(3*rand())
	tau = 198 * rand() + 2
	net = getequipropnet([2; 30; 2],
						 neuronparams = 
							MarkovNeuronParameters(tau_trace = tau,
												   noiselevel = noiselevel),
						 neuronparamsoutput = 
							ScellierOutputNeuronParameters(tau_trace = tau,
														   noiselevel = noiselevel))
	conf = EquipropConfig(net, stepsforward = 50 + 15*ceil(Int64, tau),
							   stepsbackward  = 2 + 4*ceil(Int64, tau),
							   n_ofsamples = T)
	push!(taus, tau)
	push!(noiselevels, noiselevel)
	push!(losses, learn!(net, conf))
end

using JLD
run(`mkdir -p $datapath/noisedependence`)
@save "$datapath/noisedependence/$(ARGS[1]).jld" noiselevels taus losses
