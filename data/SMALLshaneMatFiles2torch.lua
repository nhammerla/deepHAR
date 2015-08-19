require 'mattorch'

data = {}

data.classes={}
data.training={}
data.test={}
data.validation={}

data.training['inputs'] = mattorch.load('SMALLtrainingData.mat')['trainingData']
data.training['targets'] = mattorch.load('SMALLtrainingLabels.mat')['trainingLabels']

data.test['inputs'] = mattorch.load('SMALLtestingData.mat')['testingData']
data.test['targets']=mattorch.load('SMALLtestingLabels.mat')['testingLabels']

data.validation['inputs'] = mattorch.load('SMALLvalData.mat')['valData']
data.validation['targets']=mattorch.load('SMALLvalLabels.mat')['valLabels']

torch.save('SMALLopportunityShane.dat', data)
