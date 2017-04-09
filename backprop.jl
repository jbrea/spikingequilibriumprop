type BackpropNetwork <: Network
	nl::Int64
	x::Array{Array{FloatXX, 1}, 1}
	ax::Array{Array{FloatXX, 1}, 1}
	e::Array{Array{FloatXX, 1}, 1}
	w::Array{Array{FloatXX, 2}, 1}
	b::Array{Array{FloatXX, 1}, 1}
end

function BackpropNetwork(ns::Array{Int64, 1})
	nl = length(ns)
	BackpropNetwork(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(FloatXX, ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[rand(FloatXX, ns[i])/10 for i in 2:nl])
end

type BackpropNetworkFixedFeedback <: Network
	nl::Int64
	x::Array{Array{FloatXX, 1}, 1}
	ax::Array{Array{FloatXX, 1}, 1}
	e::Array{Array{FloatXX, 1}, 1}
	w::Array{Array{FloatXX, 2}, 1}
	wb::Array{Array{FloatXX, 2}, 1}
	b::Array{Array{FloatXX, 1}, 1}
end

function BackpropNetworkFixedFeedback(ns::Array{Int64, 1})
	nl = length(ns)
	BackpropNetworkFixedFeedback(nl - 1,
		    [zeros(ns[i]) for i in 1:nl],
			[zeros(ns[i]) for i in 2:nl],
			[zeros(ns[i]) for i in 2:nl],
			[randn(FloatXX, ns[i+1], ns[i])/(10*sqrt(ns[i])) for i in 1:nl - 1],
			[randn(FloatXX, ns[i], ns[i+1])/(10*sqrt(ns[i])) for i in 1:nl - 1], 
			[rand(FloatXX, ns[i])/10 for i in 2:nl])
end

function relu!(inp, outp)
	for i in 1:length(inp)
		outp[i] = max(0, inp[i])
	end
end

function forwardprop!(net::Union{BackpropNetwork,BackpropNetworkFixedFeedback};
					  nonlinearity = relu!)
	for i in 1:net.nl
		BLAS.gemv!('N', FloatXX(1.), net.w[i], net.x[i], FloatXX(0.), net.ax[i])
		BLAS.axpy!(FloatXX(1.), net.b[i], net.ax[i])
		nonlinearity(net.ax[i], net.x[i+1])
	end
end

function predict!(net::Union{BackpropNetwork,BackpropNetworkFixedFeedback},
				  x::Array{FloatXX, 1})
	net.x[1] = x
	forwardprop!(net)
	net.x[end]
end

function heaviside!(x::Array{FloatXX, 1})
	for i in 1:length(x)
		x[i] = x[i] > 0
	end
end

function relu_diff!(ax, error)
	for i in 1:length(ax)
		error[i] *= ax[i] > 0
	end
end

function getbackproplrates(net::Union{BackpropNetwork,BackpropNetworkFixedFeedback})
	getbackproplrates([length(net.x[i]) for i in 1:net.nl + 1])
end

function getbackproplrates(layerdims)
	convert(Array{FloatXX, 1}, 1./sqrt(layerdims[1:end-1] + 1))
end

function backprop!(net::BackpropNetwork, learningrate;
				   nonlinearity_diff = relu_diff!)
	for i in net.nl:-1:2
		nonlinearity_diff(net.ax[i], net.e[i])
		BLAS.gemv!('T', FloatXX(1.), net.w[i], net.e[i], FloatXX(0.), net.e[i-1])
	end
	updateweights!(net, learningrate, nonlinearity_diff)
end

function updateweights!(net::Network, learningrate, nonlinearity_diff)
	nonlinearity_diff(net.ax[1], net.e[1])
	for i in 1:net.nl
		BLAS.ger!(learningrate[i], net.e[i], net.x[i], net.w[i])
		BLAS.axpy!(learningrate[i], net.e[i], net.b[i])
	end
end


function backprop!(net::BackpropNetworkFixedFeedback, learningrate;
				   nonlinearity_diff = relu_diff!)
	for i in net.nl:-1:2
		nonlinearity_diff(net.ax[i], net.e[i])
		BLAS.gemv!('N', FloatXX(1.), net.wb[i], net.e[i], FloatXX(0.), net.e[i-1])
	end
	updateweights!(net, learningrate, nonlinearity_diff)
end

type BackpropConfig <: Config
	inputfunction::Function
	targetfunction::Function
	n_ofsamples::Int64
	noiselevel::FloatXX
	learningrate::Array{FloatXX}
	records::Int64
	seed::UInt64
end

function BackpropConfig(net, seed; 
						targetfunction = Z,
						inputfunction  = () -> rand(FloatXX, 2),
						n_ofsamples = 10^6,
						learningratefactor = 1.,
						records = 10,
						noiselevel = 0.,
						vargs...)
	BackpropConfig(inputfunction, targetfunction, n_ofsamples, noiselevel,
				   getbackproplrates(net) * learningratefactor, records, seed)
end

function getbackpropnetandconf(ns; seed = rand(0:typemax(UInt64) - 1),
							       vargs...)
	srand(seed)
	net = BackpropNetwork(ns)
	conf = BackpropConfig(net, seed; vargs...)
	net, conf
end

function setrandominput!(input)
	for i in 1:length(input)
		input[i] = rand()
	end
end

function getslope(losses, i)
    ([collect(1:10) ones(10)]\log10(losses[i-9:i]))[1]
end

function breaking(losses, crit)
    for i in 10:length(losses)
		if getslope(losses, i) > -crit
			return i
		end
    end
    return length(losses)
end

using ProgressMeter
function learn!(net::Network, conf::BackpropConfig; save = false)
	losses = FloatXX[]
	loss = FloatXX(0.)
	divisor = div(conf.n_ofsamples, conf.records)
	@showprogress for i in 1:conf.n_ofsamples
		net.x[1] = conf.inputfunction()
		forwardprop!(net)
		target = conf.targetfunction(net.x[1])
		net.e[end] = target - net.x[end] #+ conf.noiselevel * randn(net.nl[end])
		loss += norm(net.e[end])
		if i % divisor == 0
			push!(losses, loss/divisor)
			if length(losses) > 10 && getslope(losses, length(losses)) > -1e-3
				break
			end
			loss = FloatXX(0.)
		end
		backprop!(net, conf.learningrate)
	end
	save ? savesim(net, conf, losses) : nothing
	losses
end
