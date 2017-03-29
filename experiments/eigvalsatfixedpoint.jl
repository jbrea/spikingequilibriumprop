#include("../equilibriumprop.jl")
#include("../backprop.jl")
#include("../plotting.jl")
#include("../SpikingEquilibriumProp.jl"); 
using SpikingEquilibriumProp

function islayermatch(layerRegex, layername)
    |(map(r -> ismatch(r, string(layername)), layerRegex)...)
end

function getlinearizedatfixedpoint(net)
	lastlayer = 0
	layerRegex = (r"hiddenlayer", r"outputlayer")
	w = Float64[]
	for (name, l) in net.layers
		if islayermatch(layerRegex, name)
			if lastlayer != 0
				offdiag = [zeros(size(w, 1)-net.layers[lastlayer].n_of, l.n_of);
						   (l.inputconnections[:default][lastlayer].w .* 
		((0 .< l.neurons.s .< 1) * 
		 (0 .< l.inputconnections[:default][lastlayer].pre.s .< 1)'))']
				w = [w offdiag; 
					offdiag' -eye(l.n_of)]
			else
				w = -eye(l.n_of)
			end
			lastlayer = name
		end
	end
	Symmetric(w)
end

getlargesteigvalatfixedpoint(net) =	maximum(eigvals(getlinearizedatfixedpoint(net)))
function getmeanlargesteigvalatfixedpoint(net, conf)
	evs = Float64[]
	for _ in 1:100
		forwardphase!(net, conf)
		push!(evs, getlargesteigvalatfixedpoint(net))
	end
	evs
end

function eigvalevolution(ns)
	net = getequipropnet(ns)
	conf = EquipropConfig(net, n_ofsamples = 10^4, stepsforward = 2000,
						  stepsbackward = 2 * (length(ns) - 1))
	eigvals = []
	push!(eigvals, getmeanlargesteigvalatfixedpoint(net, conf))
	for t in 1:50
		learn!(net, conf, silent = true)
		push!(eigvals, getmeanlargesteigvalatfixedpoint(net, conf))
	end
	eigvals
end

