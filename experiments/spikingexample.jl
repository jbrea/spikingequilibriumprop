include("../equilibriumprop.jl")

tau_trace = eval(parse(ARGS[1]))

if ARGS[3] == "mnist"
	include("../mnistfunctions.jl")
	ni = 28^2
	nh = 500
	no = 10
	ninput = 28^2
	ntarget = 10
	inputfunction = getimg
	targetfunction = getlabel
	outputprocessor = getinputtraceprediction
	n_ofsamples = 10^5
else
	ni = 2 	
	no = 2
	nh = 30 
	ninput = 2
	ntarget = 2
	inputfunction = scaledrandinput
	targetfunction = scaledtargetfunction
	n_ofsamples = 10^6
	if no != ntarget 
		outputprocessor = getaverageinputtraceprediction
	else
		outputprocessor = getinputtraceprediction
	end
end

BLAS.set_num_threads(2)
nparams  = nparams1 = SRM0NeuronParameters(sigma = 0., tau_trace = tau_trace);
ntype = SRM0Neuron;
ntype2c = SRM0TwoCompNeuron;
#nparams = ScellierOutputNeuronParameters(gamma = 1 - 1/15); 
#nparams1 = MarkovNeuronParameters(gamma = 1 - 1/15);
#ntype = LiffiRateNeuron;
#ntype2c = LiffiScellierRateNeuron;;
net = getequipropnet([ni, nh, no], 
			targetdimension = ntarget,
			inputdimension = ninput,
            neurontypeinput = ntype, neuronparamsinput = nparams1, 
            neurontype = ntype, neuronparams = nparams1, 
			neurontypeoutput = ntype2c, neuronparamsoutput = nparams,
			backwardconnectiontype = TransposeDenseConnection,
			targetconnectiontype = ntarget == no ? One2OneConnection : StaticDenseConnection,
			inputconnectiontype = ninput == ni ? One2OneConnection : StaticDenseConnection); 
conf = EquipropConfig(net, stepsforward = 15*tau_trace, stepsbackward = 8*tau_trace, 
					  n_ofsamples = n_ofsamples, learningratefactor = .1,
					  outputprocessor = outputprocessor,
					  inputfunction = inputfunction,
					  targetfunction = targetfunction,
					  records = 100);
if ntarget != no
	net.layers[:outputlayer].inputconnections[:somatic][:targetlayer].w[:] = 
        FloatXX[div(i-1, div(no, ntarget)) == j - 1 for i in 1:no, j in 1:ntarget];
end
if ninput != ni
	net.layers[:firstlayer].inputconnections[:default][:inputlayer].w[:] = 
		rand(ni, ninput)/2.
        #FloatXX[div(i-1, div(ni, ninput)) == j - 1 for i in 1:ni, j in 1:ninput];
end
l = learn!(net, conf)
using JLD
run(`mkdir -p $datapath/spiking`)
@save "$datapath/spiking/spikingnet$(ARGS[1])-$(ARGS[2])-$(ARGS[3]).jld" net conf l
