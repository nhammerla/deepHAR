--[[
main file to train a convolutional neural network.

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
cmd:text('Convolutional neural network for HAR')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'initial random seed')
cmd:option('-logdir',                 'exp',  'path to store model progress, results, and log file')
cmd:option('-data',                   '',     'data-set to run on (DP datasource)')
cmd:option('-gpu',                    0,      'GPU to run on (default: 0)')
cmd:option('-cpu',                    false,  'Run on CPU')
cmd:option('-numConv',                2,      'Number of convolution + maxpool layers (min 1, max 3, default 2)')
cmd:option('-numFull',                0,      'Number of additional fully connected layers (default 0)')
cmd:option('-layerSize',              512,    'Number of units in fully connected layer')
cmd:option('-kW1',                    5,      'Width of first kernel')
cmd:option('-kW2',                    5,      'Width of second kernel')
cmd:option('-kW3',                    5,      'Width of third kernel')
cmd:option('-nF1',                    16,     'Number of featuremaps in first conv layer')
cmd:option('-nF2',                    16,     'Number of featuremaps in second conv layer')
cmd:option('-nF3',                    16,     'Number of featuremaps in third conv layer')
cmd:option('-mW1',                    3,      'Width of first pooling')
cmd:option('-mW2',                    3,      'Width of second pooling')
cmd:option('-mW3',                    3,      'Width of third pooling')
cmd:option('-dropout1',               0.1,    'Dropout for first conv layer')
cmd:option('-dropout2',               0.25,   'Dropout for second conv layer')
cmd:option('-dropout3',               0.25,   'Dropout for second conv layer')
cmd:option('-dropoutFull',            0.5,    'Dropout for fully connected layers')
cmd:option('-learningRate',           0.5,    'Learning rate')
cmd:option('-batchSize',              64,     'Batch-size')
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
