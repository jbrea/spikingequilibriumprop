using MNIST
imgs, labels = traindata()
imgstest, labelstest = testdata()

function getimg()
	global patternindex = rand(1:60000)
	imgs[:, patternindex]
end

function getlabel(x)
    Float64[labels[patternindex] == i for i in 0:9]
end

function geterror(net, conf, imgs, labels)
	error = 0
	for i in 1:length(labels)
		error += indmax(forwardphase!(net, conf, imgs[:, i])) != labels[i] + 1
	end
	error/length(labels)
end
function trainingerror(net, conf)
	geterror(net, conf, imgs, labels)
end
function testerror(net, conf)
	geterror(net, conf, imgstest, labelstest)
end


