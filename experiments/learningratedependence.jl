include("../equilibriumprop.jl")
include("../backprop.jl")
using Distributions
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--scaling", "-s"
        help = "constant or linear"
		arg_type = AbstractString
		default = "constant"
    "--nhidden", "-n"
		help = "number of hidden layers (1 or 2)"
        arg_type = Int
        default = 1
	"--stepsforward", "-f"
		help = "number of steps in forward phase"
		arg_type = Int
		default = 50
    "--withbackprop", "-b"
        help = "run also backprop"
        action = :store_true
	"--nofsamples", "-t"
		help = "number of samples"
		arg_type = Int
		default = 5*10^6
    "id"
        help = "simulation id"
        required = true
end
pa = parse_args(s)

function shuffle(x)
	sample(x[:], size(x), replace = false)
end
function linearScaleT0(T0, i, range)
	T0*i*range
end

if pa["scaling"] == "linear"
	println("linear scaling")
	scaleT0 = linearScaleT0
else
	scaleT0(T0, i, range) = T0
end

T0 = pa["nofsamples"]
lr0 = 10.
ns = pa["nhidden"] == 1 ? [2; 30; 2] : [2; 30; 30; 2]
stepsforward = pa["stepsforward"]

lrs = []
lossesbp = []
lossesstatich = []
losseseq = []
for range in [1]
	for i in 2.^collect(0:5)
		for j in 1:2
			push!(lrs, lr0/i/range)
			if pa["withbackprop"]
				net = BackpropNetwork(ns)
				conf = BackpropConfig(net, 
									  learningratefactor = lr0/i/range/4, 
									  n_ofsamples = scaleT0(T0, i, range))
				push!(lossesbp, learn!(net, conf))
				net2 = BackpropNetwork(ns)
				net2.b[1] = shuffle(net.b[1])
				net2.w[1] = shuffle(net.w[1])
				conf.learningrate[1] = 0;
				push!(lossesstatich, learn!(net2, conf))
			end
			net = getequipropnet(ns)
			conf = EquipropConfig(net,
								  stepsforward = stepsforward,
								  n_ofsamples = scaleT0(T0, i, range),
								  learningratefactor = lr0/i/range,
								  outputprocessor = getoutputprediction)
			push!(losseseq, learn!(net, conf))
		end
	end
end

using JLD
run(`mkdir -p $datapath/learningratedependence`)
fileid = "-$(pa["id"])-$(pa["nhidden"])-$(pa["nofsamples"])-$(pa["scaling"])"
if pa["withbackprop"]
	@save "$datapath/learningratedependence/backprop$fileid.jld" lrs lossesbp
	@save "$datapath/learningratedependence/statich$fileid.jld" lrs lossesstatich
end
fileid *= "-$(pa["stepsforward"])"
@save "$datapath/learningratedependence/equiprop$fileid.jld" lrs losseseq
