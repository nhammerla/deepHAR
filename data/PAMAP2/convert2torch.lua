cmd = torch.CmdLine()
cmd:option('-datafile','set1.mat', 'Path where MAT file is saved')
cmd:option('-saveAs','set1.dat', 'Path to save DAT file')
params = cmd:parse(arg)

require 'mattorch'

loading = mattorch.load(params.datafile)

data={}

data.classes={}
data.training={}
data.validation={}
--data.test={}

--data.classes = torch.totable(loading.classes)[1]

data.training.inputs = loading.trainingData
data.training.targets=torch.squeeze(loading.trainingLabels)

data.validation.inputs=loading.valData
data.validation.targets=torch.squeeze(loading.valLabels)

torch.save(params.saveAs, data)
