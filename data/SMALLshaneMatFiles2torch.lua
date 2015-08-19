require 'mattorch'

data = {}

data.classes={}
data.training={}
data.test={}
data.validation={}

data.training['inputs'] = mattorch.load('trainingData.mat')['trainingData']
data.training['targets'] = mattorch.load('trainingLabels.mat')['trainingLabels']

data.test['inputs'] = mattorch.load('testingData.mat')['testingData']
data.test['targets']=mattorch.load('testingLabels.mat')['testingLabels']

data.validation['inputs'] = mattorch.load('valData.mat')['valData']
data.validation['targets']=mattorch.load('valLabels.mat')['valLabels']

torch.save('SMALLopportunityShane.dat', data)
