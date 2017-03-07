function Z(x::Array{FloatXX, 1})
	[(cos(pi*x[1]) + cos(pi*(x[1]+x[2])) + 3)/6; 
     (sin(pi*x[1]) + sin(pi*(x[1]+x[2])) + 3)/6]
end

