function connectforward!(net::SimpleNetwork, nh::Int64;
						 connectiontype = PlasticDenseConnection,
						 withbias = true,
						 firstlayername = :firstlayer)
	if withbias
		connect!(net, :biaslayer, :hiddenlayer1, PlasticDenseConnection)
		connect!(net, :biaslayer, :outputlayer, PlasticDenseConnection)
	end
	connect!(net, firstlayername, :hiddenlayer1, connectiontype)
	connect!(net, Symbol(:hiddenlayer, nh), :outputlayer, connectiontype)
	for i in 2:nh
		if withbias
			connect!(net, :biaslayer, Symbol(:hiddenlayer, i), connectiontype)
		end
		connect!(net, Symbol(:hiddenlayer, i-1), Symbol(:hiddenlayer, i), 
				 connectiontype)
	end
end

function connectbackward!(net::SimpleNetwork, nh::Int64;
						 connectiontype = PlasticDenseConnection)
	if connectiontype == TransposeDenseConnection
		return connectbackwardsymmetric!(net, nh)
	end
	connect!(net, :outputlayer, Symbol(:hiddenlayer, nh), connectiontype)
	for i in 2:nh
		connect!(net, Symbol(:hiddenlayer, i), 
					  Symbol(:hiddenlayer, i-1), connectiontype)
	end
end

function connectbackwardsymmetric!(net::SimpleNetwork, 
								   nh::Int64)
	connectbackwardsymmetric!(net, net, nh)
end

function connectbackwardsymmetric!(net::SimpleNetwork, 
								   netforward::SimpleNetwork,
								   nh::Int64)
	con = TransposeDenseConnection(netforward.layers[:outputlayer].inputconnections[:default][Symbol(:hiddenlayer, nh)], net)
	connect!(net, :outputlayer, Symbol(:hiddenlayer, nh), con)
	for i in 1:nh - 1
		con = TransposeDenseConnection(netforward.layers[Symbol(:hiddenlayer, i+1)].inputconnections[:default][Symbol(:hiddenlayer, i)], net)
		connect!(net, Symbol(:hiddenlayer, i), Symbol(:hiddenlayer, i), con)
	end
end
