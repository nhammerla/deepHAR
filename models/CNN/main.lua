require 'torch'
require 'nn'
require 'optim'
require 'meanF1score'
require 'stratBatchIter'
require 'hdf5'
json = require 'json'

-- parse command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Convolutional neural network for scratch detection')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'initial random seed')
cmd:option('-logdir',                 'exp',  'path to store model progress, results, and log file')
cmd:option('-data',                   '',     'data-set to run on')
cmd:option('-gpu',                    0,      'GPU to run on (default: 1)')
cmd:option('-cpu',                    false,  'Run on CPU')
cmd:option('-numConv',                3,      'Number of convolution + maxpool layers (min 1, max 3, default 3)')
cmd:option('-numFull',                0,      'Number of additional fully connected layers (default 0)')
cmd:option('-layerSize',              512,    'Number of units in fully connected layer')
cmd:option('-kW1',                    9,      'Width of first kernel')
cmd:option('-kW2',                    5,      'Width of second kernel')
cmd:option('-kW3',                    5,      'Width of third kernel')
cmd:option('-kW4',                    5,      'Width of fourth kernel')
cmd:option('-nF1',                    64,     'Number of featuremaps in first conv layer')
cmd:option('-nF2',                    64,     'Number of featuremaps in second conv layer')
cmd:option('-nF3',                    64,     'Number of featuremaps in third conv layer')
cmd:option('-nF4',                    64,     'Number of featuremaps in fourth conv layer')
cmd:option('-mW1',                    3,      'Width of first pooling')
cmd:option('-mW2',                    3,      'Width of second pooling')
cmd:option('-mW3',                    3,      'Width of third pooling')
cmd:option('-mW4',                    3,      'Width of fourth pooling')
cmd:option('-dropout1',               0.1,    'Dropout for first conv layer')
cmd:option('-dropout2',               0.25,   'Dropout for second conv layer')
cmd:option('-dropout3',               0.25,   'Dropout for third conv layer')
cmd:option('-dropout4',               0.25,   'Dropout for fourth conv layer')
cmd:option('-dropoutFull',            0.5,    'Dropout for fully connected layers')
cmd:option('-learningRate',           0.5,    'Learning rate')
cmd:option('-batchSize',              64,     'Batch-size')
cmd:option('-momentum',               0.9,    'Momentum')
cmd:option('-learningRateDecay',      5e-4,   'Learning rate decay')
cmd:option('-maxInNorm',              3,      'Max-in-norm for regularisation')
cmd:option('-patience',               10,     'Patience in early stopping')
cmd:option('-minEpoch',               30,     'Minimum number of epochs before check for convergence')
cmd:option('-maxEpoch',               150,    'Stop after this number of epochs even if not converged')
cmd:option('-expand',                 false,  'Expand input data with normalised dim')
cmd:option('-fbconv',                 false,   'Use facebook convolution layers (much faster!)')
cmd:option('-scaleInput',             1,      'Divide each data dimension by scaleInput [1]')
cmd:option('-classWeights',           '',     'Weights for each class')
cmd:text()

params = cmd:parse(arg)

-- override fbconv for convenience
params.fbconv = false

paths.mkdir(params.logdir)

-- create log file
cmd:log(params.logdir .. '/log', params)

-- preliminaries
torch.manualSeed(params.seed)
if not params.cpu then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpu)
    cutorch.manualSeed(params.seed, params.gpu)
end
torch.setnumthreads(4)
epochPerformance = {} -- table to store progress
testPerformance = {} -- table to store progress

if params.classWeights:len() > 0 then
    params.weights = {}
    for i,v in ipairs(params.classWeights:split(',')) do table.insert(params.weights, tonumber(v)) end
    params.weights = torch.CudaTensor(params.weights):view(1,#params.weights)
end

-- Read in data-set and (maybe) store on GPU
f = hdf5.open(params.data, 'r')
data = f:read('/'):all()
f:close()
data.classes = json.load(params.data .. '.classes.json')

if not params.scaleInput == 1 then
    data.training.inputs:mul(1/params.scaleInput);
    data.test.inputs:mul(1/params.scaleInput);
    data.validation.inputs:mul(1/params.scaleInput);
end

-- check if data has the right dimensions
if data.training.inputs:dim() == 2 then
    -- just two -> put in third 'empty' dimension
    data.training.inputs = data.training.inputs:reshape(data.training.inputs:size(1), data.training.inputs:size(2), 1)
    data.validation.inputs = data.validation.inputs:reshape(data.validation.inputs:size(1), data.validation.inputs:size(2), 1)
    data.test.inputs = data.test.inputs:reshape(data.test.inputs:size(1), data.test.inputs:size(2), 1)
end

-- Expand one-dimensional data by adding a scaled version
if params.expand then
    local scale = function(data)
        data:add(-data:mean(2):expandAs(data)):cdiv(data:std(2):expandAs(data))
        return data
    end
    local expand = function(data)
        local D = torch.zeros(data:size(1), data:size(2), data:size(3)+1)
        D[{{},{},1}] = data[{{},{},1}]
        D[{{},{},2}] = scale(data[{{},{},1}])
        return D
    end
    data.training.inputs = expand(data.training.inputs)
    data.validation.inputs = expand(data.validation.inputs)
    data.test.inputs = expand(data.test.inputs)
end

if not params.cpu then
    -- put data on gpu
    data.training.inputs = data.training.inputs:cuda()
    data.validation.inputs = data.validation.inputs:cuda()
    data.test.inputs = data.test.inputs:cuda()
end

-- define model
model = nn.Sequential()

local convLayer = nn.TemporalConvolution
if params.fbconv then
    print('using fbconv')
    require 'fbcunn'
    convLayer = nn.TemporalConvolutionFB
end

local dim -- store last dim of layer before
-- we use facebook's temporal convolution
model:add(convLayer(data.training.inputs:size(3), params.nF1, params.kW1, 1))
dim = (data.training.inputs:size(2) - params.kW1) + 1
nf = params.nF1

--model:add(nn.Dropout(0.1))
model:add(nn.TemporalMaxPooling(params.mW1,params.mW1))
model:add(nn.ReLU())
model:add(nn.Dropout(params.dropout1))

dim = torch.floor((dim-params.mW1)/params.mW1+1)
-- check if we need to add another layer
if params.numConv > 1 then
    model:add(convLayer(params.nF1, params.nF2, params.kW2, 1))
    dim = (dim - params.kW2) + 1
    nf = params.nF2
    model:add(nn.TemporalMaxPooling(params.mW2,params.mW2))
    model:add(nn.ReLU())
    model:add(nn.Dropout(params.dropout2))
    dim = torch.floor((dim - params.mW2)/params.mW2+1)
end
-- check if we need to add another layer
if params.numConv > 2 then
    model:add(convLayer(params.nF2, params.nF3, params.kW3, 1))
    dim = (dim - params.kW3) + 1
    nf = params.nF3
    model:add(nn.TemporalMaxPooling(params.mW3,params.mW3))
    model:add(nn.ReLU())
    model:add(nn.Dropout(params.dropout3))
    dim = torch.floor((dim - params.mW3)/params.mW3+1)
end
-- check if we need to add another layer
if params.numConv > 3 then
    model:add(convLayer(params.nF3, params.nF4, params.kW4, 1))
    dim = (dim - params.kW4) + 1
    nf = params.nF4
    model:add(nn.TemporalMaxPooling(params.mW4,params.mW4))
    model:add(nn.ReLU())
    model:add(nn.Dropout(params.dropout4))
    dim = torch.floor((dim - params.mW4)/params.mW4+1)
end
-- fully connected part
model:add(nn.Reshape(dim*nf, true))
model:add(nn.Linear(dim*nf, params.layerSize))
model:add(nn.ReLU())
model:add(nn.Dropout(params.dropoutFull))

-- add additional fully connected layers
for i=1,params.numFull do
    model:add(nn.Linear(params.layerSize, params.layerSize))
    model:add(nn.ReLU())
    model:add(nn.Dropout(params.dropoutFull))
end

-- final softmax
model:add(nn.Linear(params.layerSize, #data.classes))
model:add(nn.LogSoftMax())

-- training criterion
criterion = nn.ClassNLLCriterion()

if not params.cpu then
    model:cuda()
    criterion:cuda()
end

-- get parameter references and initialise
parameters, gradParameters = model:getParameters()
parameters:uniform(-0.08,0.08)

-- helper functions
batchIter = stratBatchIter
if params.imbalanced then
    batchIter = stratBatchIterRepeated
end

-- max-in norm regularization
renormW = function(mod)
    -- rescale to incoming 2-norm of maxInNorm for each hidden unit
    local par = mod:parameters()

    -- skip layers without parameters (e.g. ReLU())
    if not par then
        return
    end
    for i,p in pairs(par) do
        if p:dim() > 1 then
            -- layer weights
            p:renorm(2,p:dim(),params.maxInNorm)
        end
    end
end

local checkConvergence = function(epoch, win)
    -- check for convergence over last win epochs (on validation set)
    for p=(epoch-win+1),epoch do
        if (epochPerformance[p].meanF1score) >
           (epochPerformance[epoch-win].meanF1score) then
           -- if performance has increased relatively to last epoch at least once,
           -- then we have not converged!
           return false
        end
    end
    -- all epochs decreased in performance, we have converged
    return true
end

-- training function
function train(data, labels)
    -- epoch tracker
    epoch = epoch or 1

    -- set model to training mode
    model:training()

    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. params.batchSize .. ']')

    local nbatch = torch.floor(data:size(1) / params.batchSize)
    local cnt = 1

   -- cycle through stratified batches
    for batchIndex in stratBatchIter(labels, params.batchSize) do
        -- create mini batch
        local inputs = data:index(1,batchIndex):view(batchIndex:size(1),data:size(2),data:size(3))

        local targets = labels:index(1,batchIndex)

        if not params.cpu then
            targets = targets:cuda()
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
             -- just in case:
             collectgarbage()

             -- get new parameters
             if x ~= parameters then
                parameters:copy(x)
             end

             -- reset gradients
             gradParameters:zero()

             -- evaluate function for complete mini batch
             local outputs = model:forward(inputs)
             local f = criterion:forward(outputs, targets:view(targets:size(1)))

             -- estimate df/dW
             local df_do = criterion:backward(outputs, targets:view(targets:size(1)))

             if params.weights then
                 df_do:cmul(params.weights:expandAs(df_do))
             end

             -- backpropagate
             model:backward(inputs, df_do)

             -- return f and df/dX
             return f,gradParameters
        end

        sgdState = sgdState or {
            learningRate = params.learningRate,
            momentum = params.momentum,
            learningRateDecay = params.learningRateDecay
        }

        optim.sgd(feval, parameters, sgdState)

        -- renormalise weights
        for i,mod in ipairs(model.modules) do
            renormW(mod)
        end

        xlua.progress(cnt, nbatch)
        cnt = cnt + 1
    end

    -- time taken
    time = sys.clock() - time
    time = time / data:size(1)

    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    epoch = epoch + 1
end

function test(data, labels, classes, isValidationSet)
    -- local vars
    local time = sys.clock()
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()

    -- set to test mode
    model:evaluate()

    -- test over given data
    print('<trainer> on testing Set:')
    local nbatch = torch.floor(labels:size(1) / params.sequenceLength)
    local cnt = 1

    for x,y in linearDataIter(data, labels) do
        -- x,y are tables with an entry per input

        -- disp progress
        xlua.progress(cnt, nbatch)

        -- test samples
        local preds = model:forward(x)

        -- preds is table

        -- confusion:
        for i = 1,batchIndex:size(1) do
            confusion:add(preds[i], y[i][1])
        end
        cnt = cnt + 1
   end

    -- timing
    time = sys.clock() - time
    time = time / data:size(1)
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    local perf = {}
    perf.confusion = confusion
    perf.meanF1score = meanF1score(confusion)
    perf.TN = confusion.mat[1][1] / confusion.mat[1]:sum()
    perf.TP = confusion.mat[2][2] / confusion.mat[2]:sum()

    if isValidationSet then
        table.insert(epochPerformance, perf)
    else
        table.insert(testPerformance, perf)
    end

    print('meanF1score: ' .. meanF1score(confusion))
    return meanF1score(confusion)
end

local best = 0
local progress = {}
progress.epochPerformance = epochPerformance
progress.testPerformance = testPerformance
for e=1,params.maxEpoch do
    train(data.training.inputs, data.training.targets)
    local score = test(data.validation.inputs, data.validation.targets, data.classes, true)
    local scoreT = test(data.test.inputs, data.test.targets, data.classes, false)

    if score > best then
        best = score
        epochPerformance.best = e
        torch.save(params.logdir .. '/model.dat', model)
    end

    -- save progress
    torch.save(params.logdir .. '/progress.dat', progress)

    if e > params.minEpoch then
        -- check for convergence
        if checkConvergence(e,params.patience) == true then
            break
        end
    end
end
