macro savable(t)
	ts = deepcopy(t)
	name = typeof(t.args[2]) == Expr ? t.args[2].args[1] : t.args[2]
	namesavable = Symbol(name, :Storable)
	ts.args[2] = namesavable
	fieldswrite = :()
	fieldsread = :()
	for arg in ts.args[3].args
		if typeof(arg) != Expr
			error("Do all fields have a type?")
		end
		if arg.head == :(::) 
			if arg.args[2] == :Function
				arg.args[2] = :AbstractString
				fieldswrite = :($fieldswrite..., 
				ismatch(r"#", string(o.$(arg.args[1]))) ? 
					error("Encountered # in function name. 
						   Do you want to store an nonymous function?
						   (unsupported)"):
					string(o.$(arg.args[1])))
				fieldsread = :($fieldsread..., eval(parse(o.$(arg.args[1]))))
			else
				fieldswrite = :($fieldswrite..., o.$(arg.args[1]))
				fieldsread = :($fieldsread..., o.$(arg.args[1]))
			end
		end
	end
	funcwrite = :(JLD.writeas(o::$name) = $namesavable($fieldswrite...))
	funcread = :(JLD.readas(o::$namesavable) = $name($fieldsread...))
	return :(using JLD; $t; $ts; $funcwrite; $funcread)
end

