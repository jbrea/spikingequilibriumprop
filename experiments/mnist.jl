using SpikingEquilibriumProp, JLD
include("../mnistfunctions.jl")
BLAS.set_num_threads(2)

net, conf = getequipropnetandconf([28^2; 500; 500; 10],
								  targetdimension = 10,
								  stepsforward = 100, 
								  stepsbackward = 1,
								  learningratefactor = .005, 
								  inputfunction = getimg, 
								  targetfunction = getlabel, 
								  n_ofsamples = 10^4, records = 10,
								  outputprocessor = 
								  SpikingEquilibriumProp.getoutputprediction,
								  beta = 2,
								  backwardupdate =
								  SpikingEquilibriumProp.NetSim.updatenetasync!);

l = Float64[]
traininge = Float64[]
vales = Float64[]
for _ in 1:10
	l = [l; learn!(net, conf)]
	#push!(traininge, trainingerror(net, conf))
	vale = valerror(net, conf)
	println(vale)
	push!(vales, vale)
end
run(`mkdir -p $(SpikingEquilibriumProp.datapath)/mnist`)
@save "$(SpikingEquilibriumProp.datapath)/mnist/mnistratenet2.jld" net conf l traininge vales
