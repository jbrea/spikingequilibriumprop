using SpikingEquilibriumProp, JLD, DataFrames

data = DataFrame(filename = AbstractString[], 
				 stepsforward = Int64[], 
				 stepsbackward = Int64[],
				 n_ofsamples = Int64[],
				 losses = Array{Float64, 1}[], 
				 endloss = Float64[], 
				 beta = Float64[], 
				 learningratefactor = Float64[],
				 noiselevel = Float64[],
				 nofhidden = Float64[])

datadir = SpikingEquilibriumProp.datapath * "/dump"
summarypath = SpikingEquilibriumProp.datapath * "/summary.jld"
data = isfile(summarypath) ? load(summarypath, "data") : data
loadedfiles = data[:filename]
for file in readdir(datadir)
	if file in loadedfiles
		nothing
	else
		net, conf, losses = load("$datadir/$file", "net", "conf", "losses")
		push!(data, [file, conf.stepsforward, conf.stepsbackward, conf.n_ofsamples,
			  losses, losses[end], conf.beta, conf.learningratefactor,
			  net.layers[:hiddenlayer1].neurons.p.noiselevel, 
			  length(net.layers) - 5])
	end
end
@save summarypath data
