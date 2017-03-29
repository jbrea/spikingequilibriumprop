using SpikingEquilibriumProp
vargs = []
ns = [2; 30; 2]
for arg in ARGS
	eq = search(arg, '=')
	if arg[1:eq-1] == "ns"
		ns = eval(parse(arg[eq+1:end]))
	else
		push!(vargs, (Symbol(arg[1:eq-1]), eval(parse(arg[eq+1:end]))))
	end
end
createandrunsim(ns; vargs...)
