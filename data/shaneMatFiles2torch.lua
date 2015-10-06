require 'mattorch'
loading = mattorch.load('~/opp2.mat')

data = {}

data.classes={}
data.training={}
data.test={}
data.validation={}

data.classes = torch.totable(loading.classes)[1]

data.training.inputs = loading.trainingData
data.training.targets = torch.squeeze(loading.trainingLabels)

-- Make test and validation sets the same for now...
data.test.inputs = loading.testingData
data.test.targets = torch.squeeze(loading.testingLabels)

data.validation.inputs = data.test.inputs
data.validation.targets = data.test.targets

torch.save('~/opp2.dat',data)
