# helper
import NetSim.updateneuron!
import NetSim.collectmessages!

function twocompurbsenn!(n, u, con, gamma)
	BLAS.scale!(FloatXX(0.), n.dendritic)
	weightedprerates!(n.dendritic, con[:default], FloatXX(1.))
	BLAS.scale!(FloatXX(0.), n.somatic)
	weightedprerates!(n.somatic, con[:somatic], FloatXX(1.))
	@inbounds for i in 1:length(u)
	if n.in_refractoriness[i] <= 0
		lambda = n.p.gI/2
		u[i] = lambda * n.somatic[i] + (1 - lambda) * n.dendritic[i]
	end
end
end


function twocompscellier!(n, s, con, gamma)
	BLAS.scale!(FloatXX(0.), n.dendritic)
	weightedprerates!(n.dendritic, con[:default], 1.)
	if n.p.beta > 0
		BLAS.scale!(FloatXX(0.), n.somatic)
		weightedprerates!(n.somatic, con[:somatic], 1.)
	end
	@inbounds for i in 1:length(s)
		s[i] = (gamma * (1+n.p.beta) - n.p.beta) * s[i] + 
				    (1 - gamma) * (n.p.beta * n.somatic[i] +  n.dendritic[i])
	end
end


function addnoise!(x::Array{FloatXX, 1}, noiselevel::FloatXX)
	for i in 1:length(x)
		x[i] += noiselevel * randn(FloatXX)
	end
end

function getextrafieldstwocomp(twocomp)
	if twocomp
		extrafields = :(somatic::Array{FloatXX, 1};
						dendritic::Array{FloatXX, 1};
						prediction::Array{FloatXX, 1})
		noffields = 3
	else
		extrafields = :(;)
		noffields = 0
	end
	noffields, extrafields
end

# neuron parameter types

type NoNeuronParameters <: NeuronParameters
end
export NoNeuronParameters

type MarkovNeuronParameters <: NeuronParameters
	gamma::FloatXX
	noiselevel::FloatXX
	tau_trace::FloatXX
end
function MarkovNeuronParameters(;gamma = .5, 
								 noiselevel = FloatXX(0),
								 tau_trace = FloatXX(100))
	MarkovNeuronParameters(gamma, noiselevel, tau_trace)
end
export MarkovNeuronParameters

type ScellierOutputNeuronParameters <: NeuronParameters
	gamma::FloatXX
	beta::FloatXX
	beta0::FloatXX
	noiselevel::FloatXX
	tau_trace::FloatXX
end
function ScellierOutputNeuronParameters(;gamma = .5,
										 beta = .5,
										 noiselevel = 0.,
										 tau_trace = 100)
	ScellierOutputNeuronParameters(gamma, beta, beta, noiselevel, tau_trace)
end
export ScellierOutputNeuronParameters

# rate neurons
function createrateneuron(name, paramtype; twocomp = false)
	noffields, extrafields = getextrafieldstwocomp(twocomp)
	noffields += 3
	@eval begin
		type $name <: Neuron
			p::$paramtype
			s::Array{FloatXX, 1}
			outp::Array{FloatXX, 1}
			trace::Array{FloatXX, 1}
			$extrafields
		end
		function $name(n_of::Int64, params::$paramtype)
			n = $name(params, [rand(n_of) for i in 1:$noffields]...)
			if !(:noiselevel in fieldnames(params)) || params.noiselevel == 0 
				n.trace = n.outp
			end
			n
		end
		export $name
	end
end

function createupdateneuron(name, func; updater = :singlecomp)
	if updater == :singlecomp
		updateline = :(BLAS.scale!(n.p.gamma, n.s);
				       weightedprerates!(n.s, c[:default], FloatXX(1 - n.p.gamma)))
	else
		updateline = :($updater(n, n.s, c, n.p.gamma))
	end
	@eval begin
		function collectmessages!(n::$name, c)
			$updateline
		end
		function updateneuron!(n::$name)
			if n.p.noiselevel > 0
				addnoise!(n.s, n.p.noiselevel)	
				$func(n.s, n.outp)
				BLAS.scale!(FloatXX(1 - 1/n.p.tau_trace), n.trace)
				BLAS.axpy!(FloatXX(1/n.p.tau_trace), n.outp, n.trace)
			else
				$func(n.s, n.outp)
			end
		end
	end
end


# spiking neurons
# spike response
type SRM0NeuronParameters <: NeuronParameters
	eta0::FloatXX
	tau_eta::FloatXX
	tau_syn::FloatXX
	tau_memb::FloatXX
	tau_trace::FloatXX
	epsilon0::FloatXX
	threshold::FloatXX
	sigma::FloatXX
	refractory_period::FloatXX
	gI::FloatXX
end
export SRM0NeuronParameters
function SRM0NeuronParameters(; tau_syn = 3.,
							    tau_memb = 15.,
							    tau_trace = 100.,
							    tau_eta = tau_memb,
								eta0 = -1.,
								epsilon0 = .5,
							    threshold = 0.,
							    sigma = 1e-2,
								refractory_period = 5,
								gI = 1)
	SRM0NeuronParameters(eta0, tau_eta, tau_syn, tau_memb, tau_trace, 
					     epsilon0, threshold, sigma, refractory_period, gI)
end

function createSRM0neuron(name; twocomp = false)
	noffields, extrafields = getextrafieldstwocomp(twocomp)
	noffields += 8
	@eval begin
		type $name <: Neuron
			p::SRM0NeuronParameters
			trace_syn::Array{FloatXX, 1}
			trace_memb::Array{FloatXX, 1}
			potential::Array{FloatXX, 1}
			in_refractoriness::Array{FloatXX, 1}
			eta::Array{FloatXX, 1}
			trace::Array{FloatXX, 1}
			spike::Array{FloatXX, 1}
			outp::Array{FloatXX, 1}
			$extrafields
		end
		export $name
		function $name(n_of::Int64, params::SRM0NeuronParameters)
			$name(params, [zeros(n_of) for _ in 1:3]...,
					rand(0:1, n_of) .* rand(1:params.refractory_period, n_of), 
					[zeros(n_of) for _ in 1:$noffields - 4]...)
		end
	end
end

function approxsig(u, θ, σ)
	if u <= θ - σ
		return 0.
	elseif u >= θ + σ
		return 1.
	else
		return 1/2 * ((u - θ)/σ + 1)
	end
end

function urbsenntransfer(u, θ, σ)
	1/(1+exp((θ - u)/σ))
end

function createSRM0update(name; updater = :singlecomp, urbsenn = false)
	if updater == :singlecomp
		updateline = :(for i in 1:length(n.potential);
					   n.potential[i] = 0;
					   end;
					   weightedprerates!(n.potential,  c[:default], 1.))
	else
		updateline = :($updater(n, n.potential, c, 0.))
	end
	if urbsenn
		transferfunc = urbsenntransfer
		addrefandbaseline = :(;)
	else
		transferfunc = approxsig
		addrefandbaseline = :(n.potential[i] += n.eta[i])
	end
	@eval begin
		function collectmessages!(n::$name, c)
			$updateline
		end
		function updateneuron!(n::$name)
			@inbounds for i in 1:length(n.potential)
				if n.in_refractoriness[i] > 0
					n.in_refractoriness[i] -= 1
					n.potential[i] = n.p.eta0
					n.spike[i] = 0.
				else
					n.eta[i] *= (1 - 1/n.p.tau_eta)
					$addrefandbaseline
					if n.p.sigma == 0
						n.spike[i] = n.potential[i] > n.p.threshold
					else
						n.spike[i] = rand() < $transferfunc(n.potential[i],
														n.p.threshold,
														n.p.sigma)
					end
				end
				if n.spike[i] == 1
					n.in_refractoriness[i] = n.p.refractory_period
					n.eta[i] = n.p.eta0
				end
				n.trace_memb[i] *= 1 - 1/n.p.tau_memb
				n.trace_memb[i] += n.spike[i]
				n.trace_syn[i] *= 1 - 1/n.p.tau_syn
				n.trace_syn[i] += n.spike[i]
				n.outp[i] = n.p.epsilon0 * (n.trace_memb[i] - n.trace_syn[i])
				n.trace[i] *= 1 - 1/n.p.tau_trace
				n.trace[i] += 1/n.p.tau_trace * n.spike[i] * 
					n.p.epsilon0 * (n.p.tau_memb - n.p.tau_syn)
			end
		end
	end
end

# create all neurons
include("createneurontypes.jl")
