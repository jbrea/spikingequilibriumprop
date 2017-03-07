include("../equilibriumprop.jl")

T0 = 5*10^6
ns = [2; 30; 2]

stepsf = []
losses = []
for stepsforward in [20; 50; 100; 500]
	for j in 1:5
		net = getequipropnet(ns)
		conf = EquipropConfig(net, stepsforward = stepsforward,
							  n_ofsamples = T0,
							  learningratefactor = 5.,
							  outputprocessor = getoutputprediction);
		push!(losses, learn!(net, conf))
		push!(stepsf, stepsforward)
	end
end
for j in 1:5
	net = BackpropNetwork(ns)
	conf = BackpropConfig(net, learningratefactor = .5, n_ofsamples = T0)
	push!(losses, learn!(net, conf))
	push!(stepsf, 0)
end

using JLD
run(`mkdir -p $datapath/forwardphasedependence`)
@save "$datapath/forwardphasedependence/equiprop-stepsforward$(ARGS[1]).jld" stepsf losses
