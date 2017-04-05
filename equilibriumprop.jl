const FloatXX = Float64
const datapath = "$(Base.source_dir())/data"
using NetSim
include("neurons.jl")
include("helper.jl")
include("targetfunctions.jl")
include("connectfunctions.jl")
include("getnets.jl")


# helper

function averageoutput(output::Array{FloatXX, 1},
					   groupsize::Int64,
					   n_ofgroups::Int64)
	[mean(output[groupsize * (i - 1) + 1 : groupsize * i]) 
		for i in 1:n_ofgroups]
end
function getinputtraceprediction(net)
	BLAS.scale!(0., net.layers[:outputlayer].neurons.prediction)
	for con in values(net.layers[:outputlayer].inputconnections[:default])
		BLAS.gemv!('N', 1., con.w, con.pre.trace, 1., 
					net.layers[:outputlayer].neurons.prediction)
	end
	clamp!(net.layers[:outputlayer].neurons.prediction, 0., Inf64)
	net.layers[:outputlayer].neurons.prediction
end
function getaverageinputtraceprediction(net)
	groupsize = div(net.layers[:outputlayer].n_of, net.layers[:targetlayer].n_of)
	n_ofgroups = net.layers[:targetlayer].n_of
	averageoutput(getinputtraceprediction(net), groupsize, n_ofgroups)
end
function getoutputprediction(net)
	net.layers[:outputlayer].neurons.outp
end
randinput() = rand(2)
scaledrandinput() = .8rand(2) + .1
scaledtargetfunction(x) = Z((x - .1)/.8)
# alias
outputlayerprediction = getoutputprediction
getlpfprediction = getinputtraceprediction
averageoutput(net) = getaverageinputtraceprediction(net)


# config

abstract Config
@savable type EquipropConfig <: Config
	stepsforward::Int64
	stepsbackward::Int64
	beta::Float64
	inputfunction::Function
	targetfunction::Function
	n_ofsamples::Int64
	learningratefactor::Float64
	learningrate::Array{FloatXX, 1}
	records::Int64
	outputprocessor::Function
	seed::UInt64
	backwardupdate::Function
end
function EquipropConfig(net, seed;
						stepsforward = 50,
						stepsbackward = 2,
						inputfunction = randinput,
						targetfunction = Z,
						n_ofsamples = 10^6,
						learningratefactor = .5, beta = .5,
						learningrate = learningratefactor/beta*getlrates(net),
						records = 10,
						outputprocessor = getinputtraceprediction,
						backwardupdate = updatenet!,
						vargs...)
	EquipropConfig(stepsforward, stepsbackward, beta, inputfunction, 
				targetfunction, n_ofsamples, learningratefactor,
				learningrate, records, 
				outputprocessor, seed, backwardupdate)
end
type EquipropConfigStorableOld
    stepsforward::Int64
	stepsbackward::Int64
	beta::Float64
	inputfunction::AbstractString
	targetfunction::AbstractString
	n_ofsamples::Int64
	learningratefactor::Float64
	learningrate::Array{FloatXX, 1}
	records::Int64
	outputprocessor::AbstractString
	seed::UInt64
end

JLD.readas(x::EquipropConfigStorableOld) = 
EquipropConfigStorable(x.stepsforward,
			  x.stepsbackward, x.beta, x.inputfunction, 
			  x.targetfunction, x.n_ofsamples, x.learningratefactor,
			  x.learningrate, x.records, 
			  x.outputprocessor, x.seed,
			  "updatenet!")

translate("EquipropConfigStorable", "EquipropConfigStorableOld")

function getequipropnetandconf(ns; 
							   seed = rand(0:typemax(UInt64) - 1),
							   vargs...)
	srand(seed)
	net = getequipropnet(ns; vargs...)
	conf = EquipropConfig(net, seed; vargs...)
	net, conf
end
function createandrunsim(ns; vargs...)
	net, conf = getequipropnetandconf(ns; vargs...)
	learn!(net, conf, save = true)
end

function getlrates(net)
	lrates = FloatXX[]
	for c in net.plasticconnections
		nin = 0
		for (name, pre) in net.layers[c.postname].inputconnections[:default]
			nin += net.layers[name].n_of
		end
		push!(lrates, 1/sqrt(nin))
	end
	lrates
end

# learning

function forwardphase!(net::SimpleNetwork, conf::Config)
	forwardphase!(net, conf, conf.inputfunction()) 
end

function forwardphase!(net::SimpleNetwork, conf::EquipropConfig, input::Array{FloatXX, 1})
	if typeof(net.layers[:outputlayer].neurons.p) == ScellierOutputNeuronParameters
		net.layers[:outputlayer].neurons.p.beta = 0.
	elseif typeof(net.layers[:outputlayer].neurons) == SRM0TwoCompNeuron 
		net.layers[:outputlayer].neurons.p.gI = 0.
	end
	net.layers[:inputlayer].neurons.outp[:] = input
	net.layers[:targetlayer].neurons.outp[:] = zeros(net.layers[:targetlayer].n_of)
	for t in 1:conf.stepsforward
		updatenet!(net)
	end
	conf.outputprocessor(net)
end

function backwardphase!(net::SimpleNetwork, conf::EquipropConfig, 
						target::Array{FloatXX, 1})
	if typeof(net.layers[:outputlayer].neurons.p) == ScellierOutputNeuronParameters
		net.layers[:outputlayer].neurons.p.beta = conf.beta
	elseif typeof(net.layers[:outputlayer].neurons) == SRM0TwoCompNeuron 
		net.layers[:outputlayer].neurons.p.gI = 1
	end
	net.layers[:targetlayer].neurons.outp[:] = target
	for t in 1:conf.stepsbackward
		conf.backwardupdate(net)
	end
end

function delayedantihebbianupdate!(net, pretraces, posttraces, learningrate)
	for (i, con) in enumerate(net.plasticconnections)
		delayedantihebbianupdate!(con, pretraces[i], posttraces[i],
								  learningrate[i])
	end
end

function delayedantihebbianupdate!(con::PlasticDenseConnection, 
								   pretraces, posttraces, learningrate)
	BLAS.ger!(learningrate, con.post.trace, con.pre.trace, con.w)
	BLAS.ger!(-learningrate, posttraces, pretraces, con.w)
end

function delayedantihebbianupdate!(con::PlasticSparseConnection, 
								   pretraces, posttraces, learningrate)
	for i in 1:con.w.nnz
		con.w.V[i] += learningrate * 
				       (con.post.trace[con.w.I[i]] * con.pre.trace[con.w.J[i]] -
						posttraces[con.w.I[i]] * pretraces[con.w.J[i]])
	end
end

function copytraces!(pretraces, posttraces, net)
	@inbounds for i in 1:length(pretraces)
		copy!(pretraces[i], net.plasticconnections[i].pre.trace)
		copy!(posttraces[i], net.plasticconnections[i].post.trace)
	end
end

function savesim(net, conf, losses)
	run(`mkdir -p $datapath/dump`)
	filename = "$datapath/dump/" * string(hash(losses)) * ".jld"
	@save filename net conf losses
end

using ProgressMeter
function learn!(net::SimpleNetwork, conf::EquipropConfig; 
				silent = false, save = false)
	losses = FloatXX[]
	loss = FloatXX(0.)
	divisor = div(conf.n_ofsamples, conf.records)
	pretraces = []; posttraces = []
	for con in net.plasticconnections
		push!(pretraces, deepcopy(con.pre.trace))
		push!(posttraces, deepcopy(con.post.trace))
	end
	@showprogress for t in 1:conf.n_ofsamples
		input = conf.inputfunction()
		target = conf.targetfunction(input)
		prediction = forwardphase!(net, conf, input)
		loss += norm(prediction - target)
		if t % divisor == 0
			push!(losses, loss/divisor)
			silent ? nothing : println("\t$(loss/divisor)")
			if loss/divisor > 1
				break
			end
			loss = 0.
		end
		copytraces!(pretraces, posttraces, net)
		backwardphase!(net, conf, target)
		delayedantihebbianupdate!(net, pretraces, posttraces, conf.learningrate)
	end
	save ? savesim(net, conf, losses) : nothing
	losses
end
