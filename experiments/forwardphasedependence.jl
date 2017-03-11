include("../equilibriumprop.jl")
include("../backprop.jl")

T0 = 5*10^6
if ARGS[2] == "1"
	ns = [2; 30; 2]
	stepsbackward = 2
elseif ARGS[2] == "2"
	ns = [2; 30; 30; 2]
	stepsbackward = 4
end

stepsf = []
losses = []
for stepsforward in [20; 50; 100; 500]
	for j in 1:2
		net = getequipropnet(ns, 
			neuronparamsoutput = ScellierOutputNeuronParameters(beta = .05))
		conf = EquipropConfig(net, stepsforward = stepsforward,
							  n_ofsamples = T0,
							  learningratefactor = 5.,
							  #stepsbackward = stepsbackward,
							  stepsbackward = stepsforward,
							  outputprocessor = getoutputprediction);
		push!(losses, learn!(net, conf))
		push!(stepsf, stepsforward)
	end
end
for j in 1:2
	net = BackpropNetwork(ns)
	conf = BackpropConfig(net, learningratefactor = .5, n_ofsamples = T0)
	push!(losses, learn!(net, conf))
	push!(stepsf, 0)
end

using JLD
run(`mkdir -p $datapath/forwardphasedependence`)
@save
"$datapath/forwardphasedependence/equiprop-stepsforward$(ARGS[1])-$(ARGS[2])P2.jld" stepsf losses
