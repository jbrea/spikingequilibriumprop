const figspath = "figs"
using PyPlot
using PyCall
@pyimport matplotlib.lines as mlines
@pyimport matplotlib.patches as mpatches
plt[:style][:use]("mystyle")
plt[:rc]("axes", labelsize = 18)
plt[:rc]("xtick", labelsize = 17)
plt[:rc]("ytick", labelsize = 17)
plt[:rc]("legend", fontsize = 16)
plt[:rc]("text", usetex=true)
plt[:rc]("font", family="sans-serif")

macro Ls_str(s) ismatch(r"[^\\]\$|^\$", s) ? LaTeXString(s) :  LaTeXString(string("\$\\sf{", s, "}\$")) end

import PyPlot.scatter
function scatter(ax, points::Array{Array{FloatXX,1}, 1}; 
				 color = "black", 
				 label = "")
	x = hcat(points...)'[:,1]; y = hcat(points...)'[:,2]
	ax[:scatter](x, y, c=color, label = label)
end

function plotsomepoints(net::Network, 
						conf::BackpropConfig; 
						n_ofpoints = 10^3,
						ax = nothing)
	inputs = [rand(length(net.x[1])) for i in 1:n_ofpoints]
	predictions = map(x -> deepcopy(predict!(net, x)), inputs)
	targets = map(x -> conf.targetfunction(x), inputs)
	plotsomepoints(inputs, predictions, targets, ax = ax)
end

function plotsomepoints(net::Network, conf::EquipropConfig; 
						n_ofpoints = 10^3, ax = nothing)
	inputs = [conf.inputfunction() for i in 1:n_ofpoints]
	predictions = map(x -> deepcopy(forwardphase!(net, conf, x)), inputs)
	targets = map(x -> conf.targetfunction(x), inputs)
	plotsomepoints(inputs, predictions, targets, ax = ax)
end

function plotsomepoints(inputs, predictions, targets; ax = nothing)
	if ax == nothing
		fig = plt[:figure](figsize = [6; 5])
		ax =  fig[:add_axes]([.2; .18; .7; .78])
	end
#	for i in 1:length(targets)
#		d = targets[i] - predictions[i]
#		ax[:arrow](predictions[i][1], predictions[i][2], d[1], d[2], fc = "green",
#			 ec = "green")
#	end
	scatter(ax, targets, label = "target", color = "gray")
	scatter(ax, predictions, color="red", label = "prediction")
	ax[:legend](loc = 4)
	ax[:set_xticks]([0.25; .5; .75])
	ax[:set_yticks]([.25; .5; .75])
	ax[:set_ylim]([0; 1])
	ax[:set_xlim]([0; 1])
	ax[:set_ylabel](L"y")
	ax[:set_xlabel](L"x")
end

function gatherdata(name, range; key1 = "lrs", key2 = "losses", suff = "")
	lrs = []; losses = [];
	for i in collect(range)
    res = load("$datapath/$name$i$suff.jld");
	push!(lrs, res[key1])
    push!(losses, res[key2])
    end
	lrs = vcat(lrs...)
	losses = vcat([[i[end] for i in l] for l in losses]...)
	lrs, losses
end

function sortdataforboxplot(lrs, losses)
	per = sortperm(lrs)
	dat = ()
	setsize = div(length(lrs), length(union(lrs)))
	for i in 0:div(length(lrs), setsize) - 1
       dat = (dat..., [l[end] for l in losses[per][i*setsize+1:(i+1)*setsize]])
    end
	reverse(dat)
end

function set_box_color(bp, color)
	plt[:setp](bp["boxes"], color=color)
    plt[:setp](bp["whiskers"], color=color)
    plt[:setp](bp["fliers"], markeredgecolor=color)
    plt[:setp](bp["caps"], color=color)
    plt[:setp](bp["medians"], color=color)
end


function plotlearningratedependence(; flag = "C", 
									  save = false,
									  suff = "")
	fig = figure(figsize = (5, 4))
	lrs, losses = gatherdata("learningratedependence/equiprop-", 
							 1:8, 
							 suff = suff,
							 key2 = "losseseq")
	dat = sortdataforboxplot(lrs, losses)
	bp1 = boxplot(dat)
	set_box_color(bp1, "green")
	#lrs, losses = gatherdata("statich$flag-", 1:8, key2 = "lossesstatich")
	#dat = sortdataforboxplot(lrs, losses)
	#bp3 = boxplot(dat)
	#set_box_color(bp3, "black")
	lrs, losses = gatherdata("learningratedependence/backprop$flag-", 
							 1:8, key2 = "lossesbp")
	dat = sortdataforboxplot(lrs, losses)
	bp2 = boxplot(dat, labels = [Ls"1", Ls"\frac12", Ls"\frac14",
							  Ls"\frac18", Ls"\frac1{16}", Ls"\frac1{32}"])
	set_box_color(bp2, "blue")
	plt[:yscale]("log")
	plt[:ylim]([1e-3; 9e-1])
	plt[:xlabel]("learning rate (normalized)")
	plt[:ylabel]("final error")
	bl = mlines.Line2D([], [], color = "blue", label = "backprop")
	bg = mlines.Line2D([], [], color = "green", label = "equiprop 50")
	plt[:legend](handles=[bg; bl], loc = 2)
	if flag == "C"
		plt[:title](Ls"\# samples = 5\cdot10^6")
	else
		plt[:title](Ls"\# samples\propto 1/learningrate")
	end
	plt[:tight_layout]()
	save ? savefig("$figspath/learningratedependence$flag.png") : nothing
end


function plotforwardphasedependence(; save = false, suff = "")
	fig = figure(figsize = (5, 4))
	ax = axes()
	stepsf, losses = gatherdata("forwardphasedependence/equiprop-stepsforward",
								1:8, key1 = "stepsf", suff = "")
	dat = reverse(sortdataforboxplot(stepsf, losses))
	#lrs, losses = gatherdata("backprop-", 5:12)
	#bp = sortdataforboxplot(lrs, losses)[1]
	bp1 = boxplot((dat[2:end]..., dat[1]))
	set_box_color(bp1, "green")
	stepsf, losses = gatherdata("forwardphasedependence/equiprop-stepsforward",
								1:24, key1 = "stepsf", suff = "-2")
	dat = reverse(sortdataforboxplot(stepsf, losses))
	bp2 = boxplot((dat[2:end]..., dat[1]), 
				labels = [Ls"20", Ls"50", Ls"100", Ls"500", Ls"backprop"])
	set_box_color(bp2, "blue")
	ax[:set_yscale]("log")
	ax[:set_ylim]([1.e-3; 1])
	bl = mlines.Line2D([], [], color = "blue", label = "2 hidden")
	bg = mlines.Line2D([], [], color = "green", label = "1 hidden")
	plt[:legend](handles=[bg; bl], loc = 1)
	#ax[:set_yticks]([2.5e-3, 5e-3, 1e-2, 2.5e-2])
	#ax[:set_yticklabels]([Ls"2.5\cdot10^{-3}", Ls"5\cdot10^{-3}", 
	#				   Ls"10^{-2}", Ls"2.5\cdot10^{-2}"])
	ax[:set_ylabel]("final error")
	ax[:set_xlabel]("duration forward phase")
	plt[:tight_layout]()
	save ? savefig("$figspath/forwardphasedependence.png") : nothing
end

function plotnoisedependence(; save = false)
	x = []; y = []; z = Float64[];
	for i in 1:48
		try
		dat = load("$datapath/noisedependence/$i.jld")
		x = [x; dat["taus"]]
		y = [y; dat["noiselevels"]]
		z = [z; [i[end] for i in dat["losses"]]]
		catch
		end
	end
	fig = figure(figsize = (5,4))
	#ax = fig[:add_subplot](111, projection = "3d")
	#sc = ax[:scatter](x, y, min(0, log10(z)), c = min(0, log10(z)), marker = "o", cmap = "RdYlBu_r")
	ax = fig[:add_subplot](111)
	sc = ax[:scatter](x, y, c = min(0, log10(z)), marker = "o", cmap = "RdYlBu_r")
	ax[:set_xlabel](L"\tau")
	ax[:set_ylabel]("noise level")
	ax[:set_yscale]("log")
	ax[:set_ylim]([9e-4, 1])
	cb = plt[:colorbar](sc)
	cb[:set_ticks]([-2,-1,0])
	cb[:set_ticklabels]([Ls"10^{-2}", Ls"10^{-1}", Ls"10^0"])
	cb[:set_label]("final error")
	tau = linspace(1,200)
	sig = sqrt(2tau)
	for i in [1e-4; 3e-4; 1e-3; 3e-3; 1e-2; 3e-2;  1e-1]
		plot(tau, i*sig, c = "black", linestyle = "dashed")
	end
	plt[:tight_layout]()
	save ? savefig("$figspath/noisedependence.png") : nothing
end

function plotspikes(net, conf; 
					input = [.1; .9], 
					save1 = false, save2 = false,
					aspect = 1,
					T1 = conf.stepsforward,
					T2 = conf.stepsbackward,
					T = T1 + T2,
					range = T1 - 150 : min(T1 + 150, T))
	if haskey(net.layers, :firstlayer)
		ni = net.layers[:firstlayer].n_of
		firstvar = ("si", net.layers[:firstlayer].neurons.spike, 1:ni)
	else
		ni = net.layers[:inputlayer].n_of
		firstvar = ("si", net.layers[:inputlayer].neurons.outp, 1:ni)
	end
	nh = net.layers[:hiddenlayer1].n_of
	no = net.layers[:outputlayer].n_of
	n = ni + no + nh
	variables = [firstvar;
                    ("sh", net.layers[:hiddenlayer1].neurons.spike, 1:nh);
                    ("so", net.layers[:outputlayer].neurons.spike, 1:no);
                    ("u", net.layers[:outputlayer].neurons.potential, 1);
                    ("s", net.layers[:outputlayer].neurons.spike, 1);
                    ("dend", net.layers[:outputlayer].neurons.dendritic, 1);
                    ("som", net.layers[:outputlayer].neurons.somatic, 1);
                    ("t", net.layers[:outputlayer].neurons.trace, 1)];
	net.layers[:inputlayer].neurons.outp[:] = input;
    net.layers[:targetlayer].neurons.outp[:] = conf.targetfunction(input);
    net.layers[:outputlayer].neurons.p.gI = 0;
    resfw = monitor(net, variables, conf.stepsforward);
    net.layers[:outputlayer].neurons.p.gI = 1;
    resbw = monitor(net, variables, conf.stepsbackward);
	figure(figsize = (6,4))
	ax = subplot(111)
	ax[:matshow]([resfw["si"] resfw["sh"] resfw["so"]; 
				  resbw["si"] resbw["sh"] resbw["so"]]'[:, range], 
				  cmap = "Greys", origin = (0,1), aspect = aspect)
	ax[:add_patch](mpatches.Rectangle((0, - .5), T, ni/2, alpha = .1, fc = "b"))
	ax[:add_patch](mpatches.Rectangle((0, ni/2 - .5), T, ni/2, alpha = .1, 
								   fc = "darkblue"))
	ax[:add_patch](mpatches.Rectangle((0, ni - .5), T, nh, alpha = .1, fc = "g"))
	ax[:add_patch](mpatches.Rectangle((0, ni+nh - .5), T, no/2, alpha = .1, fc = "r"))
	ax[:add_patch](mpatches.Rectangle((0, ni+nh+no/2 - .5), T, no/2, alpha = .1, fc
								   = "darkred"))
	ax[:set_yticks]([])
	ax[:get_xaxis]()[:set_ticks_position]("bottom")	
	ax[:set_xticks]([0; 100; 200; 300])
	ax[:set_xticklabels](T1 - 150 + collect(0:3)*100)
	ax[:set_xlabel]("time [ms]")
	#ax[:set_ylabel]("neurons")
	plt[:tight_layout]()
	save1 ? savefig("$figspath/spiking.png", dpi = 100, bbox_inches="tight") : nothing
	fig, (ax1, ax2, ax3) = subplots(3,1, figsize = (6,4))
	ax1[:plot]([resfw["u"]; resbw["u"]][range] + 
				[resfw["s"]; resbw["s"]][range])
	ax1[:set_xticks]([])
	ax1[:set_yticks]([])
	ax1[:set_ylabel]("potential")
	ax2[:plot]([resfw["t"]; resbw["t"]][range]/6.*10^3)
	ax2[:set_xticks]([])
	ax2[:set_yticks]([0; 150])
	ax2[:set_ylim]([0; 200])
	ax2[:set_ylabel]("rate est. [Hz]")
	ax3[:plot]([resfw["dend"]; resbw["dend"]][range], label = "dendritic")
	ax3[:plot]([resfw["som"]*0; resbw["som"]][range], label = "somatic")
	ax3[:set_yticks]([])
	ax3[:set_ylabel]("input")
	ax3[:set_xlabel]("time [ms]")
	ax3[:legend](loc = 3)
	ax3[:set_xticks]([0; 100; 200; 300])
	ax3[:set_xticklabels](T1 - 150 + collect(0:3)*100)
	plt[:tight_layout]()
	save2 ? savefig("$figspath/exampletrace.png") : nothing
	resfw
end

function plottask(; n_ofpoints = 10^3, save = false)
	fig = figure(figsize = (9, 4))
	ax1 = fig[:add_axes]([.1, .2, 1/3, .75])
	ax2 = fig[:add_axes]([.6, .2, 1/3, .75])
	input = rand(n_ofpoints, 2)
	output = hcat([Z(input[i, :]) for i in 1:n_ofpoints]...)
	ax1[:scatter](input[:, 1], input[:, 2], c = "gray")
	ax1[:set_ylabel](L"\theta")
	ax1[:set_xlabel](L"\phi")
	ax1[:set_ylim]([-.2; 1.2])
	ax1[:set_xlim]([-.2; 1.2])
	ax2[:scatter](output[1, :], output[2, :], c = "gray")
	ax2[:set_ylim]([.1; .9])
	ax2[:set_xlim]([.1; .9])
	ax2[:set_ylabel](L"y")
	ax2[:set_xlabel](L"x")
	ax1tr = ax1[:transData]
	ax2tr = ax2[:transData]
	figtr = fig[:transFigure][:inverted]()
	ax = fig[:add_axes]([0,0,1,1])
	ax[:set_frame_on](false)
	ax[:set_zorder](1000)
	specialpoints = ([0.;0.], [0.;1.], [1.;0.], [1.;1.], [0; .5], [.5; 0])
	for p in specialpoints
		ptA = figtr[:transform](ax1tr[:transform](p))
		ptB = figtr[:transform](ax2tr[:transform](Z(p)))
		ax[:add_patch](mpatches.FancyArrowPatch(
					ptA, ptB, transform=fig[:transFigure],  
					arrowstyle="simple",
					mutation_scale = 20., shrinkB = 0, shrinkA = 0))
	end
	save ? savefig("$figspath/task.png") : nothing
end

function plottaudependencespiking(; basename ="$datapath/spiking",
									save = false)
	dat = ()
	ls, lsend = load("$basename/benchmark1e6.jld", "ls", "lsend")
	dat = (dat..., lsend[2])
	taus = (50, 150, 250)
	for tau in taus
		l = []
		for i in 1:5
			try
				push!(l, load("$basename/spikingnet-$tau-$i-Y.jld", "l")[end])
			end
		end
		dat = (dat..., l)
	end
	dat = (dat..., lsend[1])
	dat = (dat[1:3]..., [load("$basename/spikingnetoutpoutensemble-150-$i-Y.jld", "l")[end] 
		for i in 1:5], dat[4:end]...)
	fig = figure(figsize = (5, 4))
	ax1 = fig[:add_axes]([.2, .2, .76, .8])
	#ax2 = fig[:add_axes]([.6, .2, .35, .8])
	ax1[:boxplot](dat, labels = ["no hidden", "50", "150", "150e", "250", "backprop"])
	ax1[:set_yscale]("log")
	ax1[:set_ylim]([2e-3; 2e-1])
	ax1[:set_ylabel]("final error")
	ax1[:set_xlabel](Ls"\tau")
	save ? savefig("$figspath/taudependencespiking.png") : nothing
#	dat = dat[[1; 3]]
#	dat = (dat..., [load("$datapath/spikingnetnewoutgroup150-$i-Y-1.jld", "l")[end] 
#						for i in 1:5])
#	dat = (dat..., [load("$datapath/spikingnetnewoutgroup2150-$i-Y-1.jld", "l")[end] 
#						for i in 1:5])
#	dat =  (dat..., lsend[1])
#	ax2[:boxplot](dat, labels = ["no hidden", "1", "10", "20", "backprop"])
#	ax2[:set_yscale]("log")
#	ax2[:set_ylim]([2e-3; 2e-1])
#	ax2[:set_ylabel]("final error")
#	ax2[:set_xlabel]("output neurons per dimension")
#	
#	figure()
#	colors = ("black", "blue", "green", "red", "orange")
#	for (k, tau) in enumerate(taus)
#        for i in 1:5
#			try
#				lcs = load("$basename$tau-$i-Y-1.jld", "l")
#				plot(lcs, color = colors[k], label = tau)
#			end
#		end
#    end
#	plt[:yscale]("log")
#	plt[:legend](loc = 2)
end

function plotexamples(; save = false)
	fig = figure(figsize = (9, 4))
	ax1 = fig[:add_axes]([0, .1, 1/3, .75])
	ax2 = fig[:add_axes]([1/3, .1, 1/3, .75])
	ax3 = fig[:add_axes]([2/3, .1, 1/3, .75])
	netnoh,	confnoh, neteq, confeq = 
		load("$datapath/examples.jld", "netnoh", "confnoh", "neteq", "confeq")
	plotsomepoints(netnoh, confnoh, ax = ax1)
	ax1[:axis]("off")
	ax1[:get_legend]()[:set_visible](false)
	ax1[:set_title]("no hidden layer")
	plotsomepoints(neteq, confeq, ax = ax2)
	ax2[:axis]("off")
	ax2[:set_title]("rate neurons")
	ax2[:get_legend]()[:set_visible](false)
	net, conf, l = 
		load("$datapath/spiking/spikingnet-250-4-Y.jld", "net", "conf", "l")
	plotsomepoints(net, conf, ax = ax3)
	ax3[:axis]("off")
	ax3[:set_title]("spiking neurons")
	save ? savefig("$figspath/examples.png") : nothing
end

function plotmnistrate(; save = false)
	res = load("$datapath/mnist/mnistratenet.jld")
	x = 10^5*collect(1:10)
	plot(x, res["traininge"]*100, label = "trainingerror")
	plot(x, res["teste"]*100, label = "testerror")
	plt[:ylabel]("error in percent")
	plt[:xlabel]("training samples")
	plt[:legend]()
	plt[:tight_layout]()
	save ? savefig("$figspath/mnistrate.png") : nothing
end
