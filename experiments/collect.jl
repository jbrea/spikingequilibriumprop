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
				 nofhidden = Float64[],
				 backwardconnection = AbstractString[],
				 nonlin = AbstractString[])

databp = DataFrame(filename = AbstractString[],
				   n_ofsamples = Int64[],
				   losses = Array{Float64, 1}[],
				   endloss = Float64[],
				   learningratefactor = Float64[],
				   nofhidden = Float64[])

datadir = SpikingEquilibriumProp.datapath * "/dump"
summarypath = SpikingEquilibriumProp.datapath * "/summary.jld"
data, databp = isfile(summarypath) ? load(summarypath, "data", "databp") : 
									 data, databp
loadedfiles = data[:filename]
file = ""

for file in readdir(datadir)
	if file in loadedfiles
		nothing
	else
		net, conf, losses = load("$datadir/$file", "net", "conf", "losses")
		if typeof(conf) == BackpropConfig
			push!(databp, [file, conf.n_ofsamples, losses, losses[end],
			conf.learningrate[1] * sqrt(length(net.x[1]) + 1), net.nl - 1])
		else
			nofhidden = length(net.layers) - 4
			if typeof(net.layers[Symbol("hiddenlayer", nofhidden)].inputconnections[:default][:outputlayer]) == SpikingEquilibriumProp.TransposeDenseConnection
				backwardconnection = "symm"
			else
				backwardconnection = "asym"
			end
			push!(data, [file, conf.stepsforward, conf.stepsbackward, conf.n_ofsamples,
				  losses, losses[end], conf.beta, conf.learningratefactor,
				  net.layers[:hiddenlayer1].neurons.p.noiselevel, 
				  nofhidden, backwardconnection, 
				  string(typeof(net.layers[:hiddenlayer1].neurons))])
		end
	end
end

@save summarypath data databp
