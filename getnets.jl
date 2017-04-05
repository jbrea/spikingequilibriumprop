function getlayersdeepnet!(net::SimpleNetwork,
						   layerdimensions::Array{Int64, 1};
						   neurontypeinput = InputRateNeuron,
						   neuronparamsinput = NoNeuronParameters(),
						   neurontype = ReLiRateNeuron,
					       neuronparams = NoNeuronParameters(),
					       neurontypeoutput = ReLiRateNeuron,
						   neuronparamsoutput = NoNeuronParameters(),
						   firstlayername = :firstlayer)
	addlayer!(net, Layer(:biaslayer, InputRateNeuron, NoNeuronParameters(), 1))
	addlayer!(net, Layer(firstlayername, neurontypeinput, 
					  neuronparamsinput, layerdimensions[1]))
	net.layers[:biaslayer].neurons.outp[1] = 1.
	nh = length(layerdimensions) - 2
	for i in 1:nh
		addlayer!(net, Layer(Symbol(:hiddenlayer, i), neurontype, 
					   neuronparams, layerdimensions[i+1]))
	end
	addlayer!(net, Layer(:outputlayer, neurontypeoutput, 
					  neuronparamsoutput, layerdimensions[end]))
end

function initilizebiases(net)
	for layer in values(net.layers)
		if haskey(layer.inputconnections, :default) &&
		   haskey(layer.inputconnections[:default], :biaslayer)
		    layer.inputconnections[:default][:biaslayer].w[:] = 
				rand(layer.n_of, 1)/10
		end
	end
end

function getequipropnet(layerdimensions::Array{Int64, 1};
					    neurontype = ReLiReRateNeuron,
						noiselevel = 0.,
						tau_trace = 1.,
					    neuronparams = 
							MarkovNeuronParameters(noiselevel = noiselevel,
												   tau_trace = tau_trace),
						neurontypeinput = InputRateNeuron,
						neuronparamsinput = NoNeuronParameters(),
					    neurontypeoutput = ReLiReScellierRateNeuron,
					    neuronparamsoutput = 
							ScellierOutputNeuronParameters(noiselevel = noiselevel,
														   tau_trace = tau_trace),
					    forwardconnectiontype = PlasticDenseConnection,
						backwardconnectiontype = TransposeDenseConnection,
						inputdimension = 2, targetdimension = 2,
						inputconnectiontype = One2OneConnection,
						targetconnectiontype = One2OneConnection, vargs...)
	#srand(1)
	net = SimpleNetwork()
	if neurontypeinput == InputRateNeuron 
		firstlayername =:inputlayer
	else
		firstlayername = :firstlayer
		addlayer!(net, Layer(:inputlayer, InputRateNeuron, 
							 NoNeuronParameters(), inputdimension))
	end
	getlayersdeepnet!(net, layerdimensions, 
					  neurontypeinput = neurontypeinput,
				      neuronparamsinput = neuronparamsinput,
					  neurontype = neurontype,
					  neuronparams = neuronparams,
					  neurontypeoutput = neurontypeoutput,
					  neuronparamsoutput = neuronparamsoutput,
					  firstlayername = firstlayername)
	addlayer!(net, Layer(:targetlayer, InputRateNeuron, 
					  NoNeuronParameters(), targetdimension))
	
	nh = length(layerdimensions) - 2
	connectforward!(net, nh, connectiontype = forwardconnectiontype, 
					firstlayername = firstlayername)
	connectbackward!(net, nh, connectiontype = backwardconnectiontype)
	if firstlayername == :firstlayer
		connect!(net, :inputlayer, :firstlayer, inputconnectiontype)
	end
	connect!(net, :targetlayer, :outputlayer, targetconnectiontype, label = :somatic)
	
	initilizebiases(net)	
	reverselayerorder!(net)
	net
end
