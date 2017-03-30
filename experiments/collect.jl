using SpikingEquilibriumProp, JLD, DataFrames

data = DataFrame(filename = AbstractString[], 
				 stepsforward = Int64[], 
				 stepsbackward = Int64[],
				 n_ofsamples = Int64[],
				 losses = Array{Float64, 1}[], 
				 endloss = Float64[], 
				 beta = Float64[], 
				 learningratefactor = Float64[])

datadir = SpikingEquilibriumProp.datapath * "/dump"
for file in readdir(datadir)
	try
	net, conf, losses = load("$datadir/$file", "net", "conf", "losses")
	push!(data, [file, conf.stepsforward, conf.stepsbackward, conf.n_ofsamples,
			  losses, losses[end], conf.beta, conf.learningratefactor])
	end
end
@save SpikingEquilibriumProp.datapath * "/summary.jld" data
