# RateNeurons
macro defineinlinefunction(funname, fun, clamptodomain)
	if clamptodomain == 1
		clampline = :()
	else
		clampline = :($clamptodomain(s))
	end
	if fun == 1
		funline = :(x[:] = s)
	else
		funline = :(@inbounds for i in 1:length(x); x[i] = $fun(s[i]); end)
	end
	quote
		function $(esc(funname))(s::Array{FloatXX, 1}, x::Array{FloatXX, 1})
			$clampline
			$funline
		end
	end
end

function liffi(x::FloatXX)
	if x <= 0
		return 0.
	else
		return @fastmath  1./(log((x + 1.)/x) + 1/3.)
	end
end

@defineinlinefunction(sigmoid!, x -> 1./(1. + exp(-x)), 1)
@defineinlinefunction(tanh!, tanh, 1)
@defineinlinefunction(exp!, exp, 1)
@defineinlinefunction(liffi!, liffi, x -> clamp!(x, 0, Inf64))
@defineinlinefunction(reli!, 1, x -> clamp!(x, 0, Inf64))
@defineinlinefunction(relire!, 1, x -> clamp!(x, 0, 1))
@defineinlinefunction(linear!, 1, 1)


functions = Dict("Linear" => :(linear!),
				 "ReLi"   => :(reli!),
				 "ReLiRe" => :(relire!),
				 "Sigmoid" => :(sigmoid!),
				 "Tanh"  =>  :(tanh!),
				 "Exp"  => :(exp!),
				 "Liffi" => :(liffi!))
compartements = Dict("Rate" => Dict("updater" => :singlecomp,
									"params" => MarkovNeuronParameters),
					 "TwoCompRate" => Dict("updater" => twocompurbsenn!,
											"params" => MarkovNeuronParameters),
					 "ScellierRate" => Dict("updater" => twocompscellier!,
											"params" => ScellierOutputNeuronParameters)
					 )

# rate neurons
for f in functions
	for c in compartements
		name = Symbol(f[1] * c[1] * "Neuron")
		createrateneuron(name, c[2]["params"], twocomp = c[1] != "Rate")
		createupdateneuron(name, f[2], updater = c[2]["updater"])
	end
end
			

# InputRateNeuron
createrateneuron(:InputRateNeuron, NoNeuronParameters)

# spiking neurons
createSRM0neuron(:SRM0Neuron)
createSRM0update(:SRM0Neuron)
createSRM0neuron(:SRM0TwoCompNeuron, twocomp = true)
createSRM0update(:SRM0TwoCompNeuron, updater = twocompurbsenn!)
