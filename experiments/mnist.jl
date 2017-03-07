include("../equilibriumprop.jl")
include("../mnistfunctions.jl")
BLAS.set_num_threads(4)

net = getequipropnet([28^2; 500; 10], targetdimension = 10);
conf = EquipropConfig(net, stepsforward = 50, 
						   stepsbackward = 4,
						   learningratefactor = 1., 
						   inputfunction = getimg, 
						   targetfunction = getlabel, 
						   n_ofsamples = 10^5, records = 10,
						   outputprocessor = getoutputprediction);

l = Float64[]
traininge = Float64[]
teste = Float64[]
for _ in 1:10
	l = [l; learn!(net, conf)]
	push!(traininge, trainingerror(net, conf))
	push!(teste, testerror(net, conf))
end
run(`mkdir -p $datapath/mnist`)
@save "$datapath/mnist/mnistratenet.jld" net conf l traininge teste
