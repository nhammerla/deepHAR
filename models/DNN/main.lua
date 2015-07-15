--[[
main file to train a deep neural network.

run as:
>> th main.lua -data mydata.dat

to see additional parameters:
>> th main.lua --help

--]]

require 'cutorch'
require 'cunn'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Deep neural network for HAR')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'initial random seed')
cmd:option('-logdir',                 'exp',  'path to store model progress, results, and log file')
cmd:option('-data',                   '',     'data-set to run on (DP datasource)')
cmd:option('-gpu',                    0,      'GPU to run on (default: 0)')
cmd:option('-cpu',                    false,  'Run on CPU')
cmd:option('-numLayers',              1,      'Number of hidden layers')
cmd:option('-layerSize',              512,    'Number of units in hidden layers')
cmd:option('-learningRate',           0.5,    'Learning rate')
cmd:option('-batchSize',              64,     'Batch-size')
cmd:option('-dropout',                0.5,    'Dropout')
cmd:option('-momentum',               0.9,    'Momentum')
cmd:option('-learningRateDecay',      5e-4,   'Learning rate decay')
cmd:option('-maxInNorm',              3,      'Max-in-norm for regularisation')
cmd:option('-patience',               10,     'Patience in early stopping')
cmd:option('-minEpoch',               30,     'Minimum number of epochs before check for convergence')
cmd:option('-maxEpoch',               150,    'Stop after this number of epochs even if not converged')

cmd:text()

-- parse input params
params = cmd:parse(arg)

params.rundir = cmd:string(params.logdir, params, {dir=true})
paths.mkdir(params.rundir)

-- create log file
cmd:log(params.rundir .. '/log', params)

-- Read in data-set and (maybe) store on GPU

-- Define model

-- helper functions

-- define training function

-- define test function

-- main training loop

-- cleaning up

-- done!
