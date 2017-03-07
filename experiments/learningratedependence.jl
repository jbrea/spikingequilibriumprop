include("../equilibriumprop.jl")
include("../backprop.jl")
using Distributions

function shuffle(x)
	sample(x[:], size(x), replace = false)
end
function linearScaleT0(T0, i, range)
	T0*i*range
end

if ARGS[2] == "L"
	println("linear scaling")
	scaleT0 = linearScaleT0
else
	scaleT0(T0, i, range) = T0
end

T0 = 5*10^6
lr0 = 10.
ns = [2; 30; 2]
stepsforward = 50

lrs = []
lossesbp = []
lossesstatich = []
losseseq = []
for range in [1]
	for i in 2.^collect(0:5)
		for j in 1:5
			net = BackpropNetwork(ns)
			conf = BackpropConfig(net, 
								  learningratefactor = lr0/i/range/4, 
								  n_ofsamples = scaleT0(T0, i, range))
			push!(lossesbp, learn!(net, conf))
			push!(lrs, lr0/i/range)
			net2 = BackpropNetwork(ns)
			net2.b[1] = shuffle(net.b[1])
			net2.w[1] = shuffle(net.w[1])
			conf.learningrate[1] = 0;
			push!(lossesstatich, learn!(net2, conf))
			net = getequipropnet(ns)
			conf = EquipropConfig(net,
								  stepsforward = stepsforward,
								  n_ofsamples = scaleT0(T0, i, range),
								  learningratefactor = lr0/i/range,
								  outputprocessor = getoutputprediction)
			push!(losseseq, learn!(net, conf))
		end
	end
end

using JLD
run(`mkdir -p $datapath/learningratedependence`)
@save "$datapath/learningratedependence/backprop$(ARGS[2])-$(ARGS[1]).jld" lrs lossesbp
@save "$datapath/learningratedependence/statich$(ARGS[2])-$(ARGS[1]).jld" lrs lossesstatich
@save "$datapath/learningratedependence/equiprop$(ARGS[2])-$(ARGS[1]).jld" lrs losseseq
