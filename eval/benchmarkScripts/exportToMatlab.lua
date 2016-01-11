require 'mattorch'
--Export from Lua to Matlab file
data = torch.load('/data/EXPDATA/4decPAMAPslidingWindowsDifferentPartitions.dat') 
list = {trainingData = data.training.inputs, trainingLabels = data.training.targets, valData=data.validation.inputs, valLabels=data.validation.targets, testingData=data.test.inputs, testingLabels=data.test.targets}
mattorch.save('pamapfrom4dec.mat',list)
