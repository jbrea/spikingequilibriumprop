using JLD, DataFrames, PyPlot # Gadfly

dat = load("data/summary.jld", "data")
sort!(dat, cols=:learningratefactor)
groups = groupby(dat, [:beta, :stepsforward])
#colors = [colorant"red",colorant"purple",colorant"green"]
colors = ["red","purple","green"]

function learningratedependence()
	ax = subplot(111)
	# plots = []
	for g in groups
		#for g in ginner
			#push!(plots, layer(g, x = :learningratefactor, y = :endloss, 
			#			 color = :beta, Geom.point, Theme(default_color=colors[i])))
			ax[:scatter](g[:learningratefactor], g[:endloss], 
						 label = "$(g[:beta][1])-$(g[:stepsforward][1])")
		#end
	end
	#feh(plot(plots..., Scale.y_log10, Scale.x_log10, 
	#	     Scale.color_discrete_manual(colorant"red",colorant"purple",colorant"green")))
	ax[:set_yscale]("log", basey = 10)
	ax[:set_xscale]("log", basex = 10)
	ax[:legend]()
	ax
end

