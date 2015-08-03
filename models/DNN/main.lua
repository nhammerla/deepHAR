-- --[[
-- main file to train a deep neural network.
-- run as:
-- >> th main.lua -data mydata.dat
-- to see additional parameters:
-- >> th main.lua --help
-- --]]

-- require 'cutorch'
-- require 'cunn'

-- cmd = torch.CmdLine()
-- cmd:text()
-- cmd:text()
-- cmd:text('Deep neural network for HAR')
-- cmd:text()
-- cmd:text('Options')
-- cmd:option('-seed',                   123,    'initial random seed')
-- cmd:option('-logdir',                 'exp',  'path to store model progress, results, and log file')
-- cmd:option('-data',                   '',     'data-set to run on (DP datasource)')
-- cmd:option('-gpu',                    0,      'GPU to run on (default: 0)')
-- cmd:option('-cpu',                    false,  'Run on CPU')
-- cmd:option('-numLayers',              1,      'Number of hidden layers')
-- cmd:option('-layerSize',              512,    'Number of units in hidden layers')
-- cmd:option('-learningRate',           0.5,    'Learning rate')
-- cmd:option('-batchSize',              64,     'Batch-size')
-- cmd:option('-dropout',                0.5,    'Dropout')
-- cmd:option('-momentum',               0.9,    'Momentum')
-- cmd:option('-learningRateDecay',      5e-4,   'Learning rate decay')
-- cmd:option('-maxInNorm',              3,      'Max-in-norm for regularisation')
-- cmd:option('-patience',               10,     'Patience in early stopping')
-- cmd:option('-minEpoch',               30,     'Minimum number of epochs before check for convergence')
-- cmd:option('-maxEpoch',               150,    'Stop after this number of epochs even if not converged')
-- cmd:option('-ignore',                 false,  'Is there a class we should ignore?')
-- cmd:option('-ignoreClass',            0,      'Class to ignore for analysis')

-- cmd:text()

-- -- parse input params
-- params = cmd:parse(arg)

-- params.rundir = cmd:string(params.logdir, params, {dir=true})
-- paths.mkdir(params.rundir)

-- -- create log file
-- cmd:log(params.rundir .. '/log', params)

-- -- Read in data-set and (maybe) store on GPU

-- -- Define model

-- -- helper functions

-- -- define training function

-- -- define test function

-- -- main training loop

-- -- cleaning up

-- -- done!

params={}
params.numLayers = 2
params.layerSize = 3
params.dropout=0.5
params.cpu = true
params.seed=1
params.gpu= 1
params.layerSize = 3
params.batchSize = 64
params.maxEpoch=10
require 'cunn'
require 'nn'
require 'cutorch'
require 'optim'

data = torch.load('opp1.dat' )
inputLayerSize = data.training.inputs:size(2)*data.training.inputs:size(3)
outputSize = #data.classes

torch.manualSeed(params.seed)
cutorch.setDevice(params.gpu)
cutorch.manualSeed(params.seed, params.gpu)
torch.setnumthreads(16)
epochPerformance = {} -- table to store progress
testPerformance = {} -- table to store progress

setmetatable(data.training, 
    {__index = function(t, i) 
                    return {t.inputs[i], t.targets[i]} 
                end}
);

function data.training:size() 
    return self.inputs:size(1) 
end

net = nn.Sequential()

data.training.inputs = nn.Reshape(data.training.inputs:size(2)*data.training.inputs:size(3)):forward(data.training.inputs)

--double-check inputLayerSize
layerToAdd = nn.Linear(inputLayerSize, params.layerSize)
net:add(layerToAdd)
net:add(nn.Dropout(dropout))
net:add(nn.ReLU())

-- Output Layer
outputSize = #data.classes
net:add(nn.Linear(params.layerSize, outputSize))
net:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

require 'stratBatchIter'
labels = data.training.targets
-- AIM: use stratified batches on the GPU for one epoch, and then calcualte the epoch performance
batchCounter = 1
net:training()
nbatch = torch.floor(data.training.targets:size(1) / params.batchSize)
for batchIndex in stratBatchIter(labels, params.batchSize) do
	inputBatch = data.training.inputs:index(1, batchIndex)
	outputBatch = labels:index(1,batchIndex)
	criterion:forward(net:forward(inputBatch), outputBatch)--3 seconds per epoch
	net:zeroGradParameters()
	net:backward(inputBatch, criterion:backward(net.output, outputBatch))
	net:updateParameters(0.01)
	xlua.progress(batchCounter, nbatch)
	batchCounter = batchCounter +1
end

