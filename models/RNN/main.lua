--[[
main file to train a LSTM network.

run as:
>> th main.lua -data mydata.dat

to see additional parameters:
>> th main.lua --help

--]]

require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
LSTM = require 'LSTM'
model_utils = require 'model_utils'
require 'meanF1score'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('LSTM network for HAR')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'Initial random seed')
cmd:option('-logdir',                 'exp',  'Path to store model progress, results, and log file')
cmd:option('-data',                   '',     'Data-set to run on (DP datasource)')
cmd:option('-gpu',                    0,      'GPU to run on (default: 0)')
cmd:option('-cpu',                    false,  'Run on CPU')
cmd:option('-numLayers',              1,      'Number of LSTM layers')
cmd:option('-layerSize',              64,     'Number of cells in LSTM')
cmd:option('-learningRate',           0.1,    'Learning rate')
--cmd:option('-dropout',                0.5,    'Dropout (dropout == 0 -> no dropout)')
cmd:option('-momentum',               0.9,    'Momentum')
cmd:option('-learningRateDecay',      5e-4,   'Learning rate decay')
cmd:option('-maxInNorm',              0,      'Max-in-norm for regularisation')
cmd:option('-maxOutNorm',             0,      'Max-out-norm for regularisation')
cmd:option('-patience',               10,     'Patience in early stopping')
cmd:option('-minEpoch',               30,     'Minimum number of epochs before check for convergence')
cmd:option('-maxEpoch',               150,    'Stop after this number of epochs even if not converged')
cmd:option('-batchSize',              64,     'Batch-size (number of sequences in each batch)')
cmd:option('-stepSize',               8,      'Step-size when iterating through sequence')
cmd:option('-sequenceLength',         64,     'Sequence-length that is looked at in each batch')
cmd:option('-carryOverProb',          0.5,    'Probability to carry over hidden states between batches')
cmd:option('-ignore',                 false,  'Is there a class we should ignore?')
cmd:option('-ignoreClass',            0,      'Class to ignore for analysis')
cmd:option('-classWeights',           '',     'Weightings for classes. Must be string of weights separated with ","')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- parse possible weights for classes
if params.classWeights:len() > 0 then
    params.weights = {}
    for i,v in ipairs(params.classWeights:split(',')) do table.insert(params.weights, tonumber(v)) end
    params.weights = torch.CudaTensor(params.weights):view(1,#params.weights)
end

paths.mkdir(params.logdir)

-- create log file
cmd:log(params.logdir .. '/log', params)

-- preliminaries
torch.manualSeed(params.seed)
torch.setnumthreads(16)
epochPerformance = {} -- table to store progress

-- Read in data-set and (maybe) store on GPU
data = torch.load(params.data)

-- transpose data (for later convenience), labels are just one-dimensional
data.training.inputs = data.training.inputs:t()
data.test.inputs = data.test.inputs:t()
data.validation.inputs = data.validation.inputs:t()

-- assumes that
--   data.training.{inputs,targets,subjectIds}
--   data.validation.{inputs,targets,subjectIds}
--   data.test.{inputs,targets,subjectIds}
--   data.classes
-- are all set now

-- Define model
-- we define prototypes that will be replicated through time
protos = {}
-- first: linear layer for data -> LSTM
protos.embed = nn.Sequential():add(nn.Linear(data.training.inputs:size(1), params.layerSize)):cuda()
-- second: LSTM layers
for i=1,params.numLayers do
    protos[i] = LSTM.lstm(params):cuda()
end
-- third: output (softmax for classification)
protos.softmax = nn.Sequential():add(nn.Linear(params.layerSize, #data.classes)):add(nn.LogSoftMax()):cuda()
-- lastly add the criterion
protos.criterion = nn.ClassNLLCriterion():cuda()

local components = {}
table.insert(components, protos.embed)
--for i,v in pairs(protos) do
--    if i ~= 'criterion' then
for i=1,params.numLayers do
    table.insert(components, protos[i])
end
table.insert(components, protos.softmax)
--    end
--end
-- get reference to params and gradparams
local parameters, grad_params = model_utils.combine_all_parameters(components)
parameters:uniform(-0.08, 0.08)
print('#parameters: ' .. parameters:size(1))

-- make a bunch of clones, AFTER flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, params.sequenceLength, not proto.parameters)
end

-- LSTM states for carryover between batches
local initstate_c = {}
local initstate_h = {}
local initstateTest_c = {}
local initstateTest_h = {}
local dfinalstate_c = {}
local dfinalstate_h = {}
for i=1,params.numLayers do
    initstate_c[i] = torch.zeros(params.batchSize, params.layerSize):cuda()
    initstate_h[i] = torch.zeros(params.batchSize, params.layerSize):cuda()
    initstateTest_c[i] = torch.zeros(1, params.layerSize):cuda()
    initstateTest_h[i] = torch.zeros(1, params.layerSize):cuda()
    dfinalstate_c[i] = torch.zeros(params.batchSize, params.layerSize):cuda()
    dfinalstate_h[i] = torch.zeros(params.batchSize, params.layerSize):cuda()
end

-- helper functions

-- Helper function to get most common element of a list (for smoothing predictions later on)
mode = function(list)
    set = {}
    -- get unique elements
    for _, l in ipairs(list) do
        set[l] = 0
    end
    -- count each occurrence
    for _, l in ipairs(list) do
        set[l] = set[l]+1
    end
    -- find element with most occurrences (or first one of them if equally distributed)
    local maxInd
    for k,v in pairs(set) do
        if maxInd then
            if v>set[maxInd] then maxInd = k end
        else
            maxInd = k
        end
    end
    return maxInd
end

-- max in-norm regularisation
function renormWeights(mod)
    if mod.modules then
        for i,module in ipairs(mod.modules) do
            renormWeights(module)
        end
    else
        local parameters = mod:parameters()
        if not parameters or gradParams then
            return
        end
        for k,param in pairs(parameters) do -- pairs for sparse params
            -- By default, only affects non-1D params.
            if param:dim() > 1 then
                if params.maxOutNorm > 0 then
                    -- rows feed into output neurons
                    param:renorm(2, 1, params.maxOutNorm)
                end
                if params.maxInNorm > 0 then
                    -- cols feed out from input neurons
                    param:renorm(2, param:dim(), params.maxInNorm)
                end
            end
        end
    end
end


-- iterator through the data-set (one particle per batch)
local dataIter = function(data,labels)
    --local pos = 1
    -- random start positions for a particle for each batch
    local posList = torch.rand(params.batchSize)*(data:size(2)-params.sequenceLength)
--    local posList = torch.linspace(1,data:size(2)-params.sequenceLength,params.batchSize)

    posList:ceil()

    local ndim = data:size(1)
    local nelem = params.batchSize * params.sequenceLength

    local d = torch.CudaTensor(params.batchSize, ndim, params.sequenceLength):fill(0)
    local l = torch.CudaTensor(params.batchSize, params.sequenceLength):fill(0)

    return function()
        -- generate random batch of sequences
        for i=1,params.batchSize do
            local pos

            -- continue going until we have usable labels (TODO)
            if params.ignore == true then
                repeat
                    pos = posList[i]
                    d[i] = data[{{}, {pos, pos+params.sequenceLength-1}}]
                    l[i] = labels[{{pos, pos+params.sequenceLength-1}}]

                    posList[i] = posList[i] + params.stepSize
                    if posList[i] > (data:size(2)-params.sequenceLength) then
                        posList[i] = 1
                    end
                until mode(l[i]:double():totable()) ~= params.ignoreClass
            else
                pos = posList[i]
                d[i] = data[{{}, {pos, pos+params.sequenceLength-1}}]
                l[i] = labels[{{pos, pos+params.sequenceLength-1}}]

                posList[i] = posList[i] + params.stepSize
                if posList[i] > (data:size(2)-params.sequenceLength) then
                    posList[i] = 1
                end
            end
        end
        return d,l
    end
end


-- define training function
function train(data, labels)
    epoch = epoch or 1
    print('Training epoch: ' .. epoch)

    -- make iterator for sampling sequences
    local batchIter = dataIter(data, labels)

    -- reset initstate
    for i=1,params.numLayers do
        initstate_c[i]:zero()
        initstate_h[i]:zero()
    end

    local maxIter = math.floor((data:size(2)-params.sequenceLength)/(params.stepSize*params.batchSize))

    -- one pass through the data-set
    local cnt = 1
    for iter=1, maxIter do
        -- get next batch of data
        local x,y = batchIter()

        ------------------- closure for gradient calculation -------------------
        function feval(p)
            -- just in case
            collectgarbage()

            if p ~= parameters then
                parameters:copy(p)
            end
            grad_params:zero()

            ------------------- forward pass -------------------
            local embeddings = {}            -- input embeddings
            local lstm_c = {}
            local lstm_h = {}
            for i=1,params.numLayers do
                lstm_c[i] = {[0]=initstate_c[i]}
                lstm_h[i] = {[0]=initstate_h[i]}
            end

            local predictions = {}           -- softmax outputs
            local loss = 0

            -- forward pass through forward layer
            for t=1, params.sequenceLength do
                embeddings[t] = clones.embed[t]:forward(x[{{}, {}, {t}}]:squeeze())
                local input = embeddings[t]
                for i=1,params.numLayers do
                    lstm_c[i][t], lstm_h[i][t] = unpack(clones[i][t]:forward{input, lstm_c[i][t-1], lstm_h[i][t-1]})
                    input = lstm_h[i][t]
                end

                predictions[t] = clones.softmax[t]:forward(lstm_h[params.numLayers][t])
                loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
            end

            loss = loss / params.sequenceLength

            ------------------ backward pass -------------------
            -- complete reverse order of the above
            local dembeddings = {}                              -- d loss / d input embeddings
            local dlstm_c = {}
            local dlstm_h = {}
            for i=1,params.numLayers do
                dlstm_c[i] = {[params.sequenceLength]=dfinalstate_c[i]}    -- internal cell states of LSTM
                dlstm_h[i] = {}                                  -- output values of LSTM
            end

            -- backward pass through forward layer (i.e. right to left)
            for t=params.sequenceLength,1,-1 do
                -- backprop through loss, and softmax/linear
                local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
                if params.weights then
                    doutput_t:cmul(params.weights:expandAs(doutput_t))
                end

                dlstm_h[params.numLayers][t] = clones.softmax[t]:backward(lstm_h[params.numLayers][t], doutput_t)

                local deriv
                for i=params.numLayers,2,-1 do
                    dlstm_h[i-1][t], dlstm_c[i][t-1], dlstm_h[i][t-1] = unpack(clones[i][t]:backward(
                        {lstm_h[i-1][t], lstm_c[i][t-1], lstm_h[i][t-1]},
                        {dlstm_c[i][t], dlstm_h[i][t]}
                        ))
                end
                dembeddings[t], dlstm_c[1][t-1], dlstm_h[1][t-1] = unpack(clones[1][t]:backward(
                    {embeddings[t], lstm_c[1][t-1], lstm_h[1][t-1]},
                    {dlstm_c[1][t], dlstm_h[1][t]}
                    ))

                -- backprop through embeddings
                clones.embed[t]:backward(x[{{}, {}, {t}}]:squeeze(), dembeddings[t])
            end

            ------------------------ misc ----------------------
            -- transfer final state to initial state (BPTT) with probability params.carryOverProb
            for i=1,params.numLayers do
                if torch.rand(1)[1] <= params.carryOverProb then
                    initstate_c[i]:copy(lstm_c[i][params.stepSize])
                    initstate_h[i]:copy(lstm_h[i][params.stepSize])
                end
            end



            -- use a drop-out layer to drop some of the previous context
--            initstate_c:copy(dropper:forward(lstm_c[params.stepSize]))
--            initstate_h:copy(dropper:forward(lstm_h[params.stepSize]))

            -- clip gradient element-wise
            --grad_params:clamp(-5, 5)

            return loss, grad_params
        end

        -- optimizer state
        sgdState = sgdState or {
            learningRate = params.learningRate,
            momentum = params.momentum,
            learningRateDecay = params.learningRateDecay
         }

         -- run optimizer
         optim.adagrad(feval, parameters, sgdState)

         -- renorm weights if required
         if params.maxInNorm > 0 or params.maxOutNorm > 0 then
             renormWeights(protos.embed)
             --renormWeights(protos.lstm)
             for i=1,params.numLayers do
                 renormWeights(protos[i])
             end
             renormWeights(protos.softmax)
         end

         xlua.progress(cnt, maxIter)

         cnt = cnt + 1
    end

    epoch = epoch + 1
    -- done
end

-- define test function
function test(data,labels,classes)
    -- run through test-set
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()

    for i=1,params.numLayers do
        initstateTest_c[i]:zero()
        initstateTest_h[i]:zero()
    end

    local nelem = params.sequenceLength
    local maxPos = data:size(2)-nelem

    -- run through whole set, with a batch-size of 1
    for pos=1,maxPos,nelem do
        xlua.progress(pos, maxPos)
        -- get data for this window
        --local dat = data[{{},{pos+(b-1)*params.sequenceLength, pos+b*params.sequenceLength-1}}]
        --local target = labels[{{},{pos+(b-1)*params.sequenceLength, pos+b*params.sequenceLength-1}}]
        local dat = data[{{},{pos, pos+params.sequenceLength-1}}]:cuda()
        local target = labels[{{pos, pos+params.sequenceLength-1}}]

        -- do forward pass through network
        local embeddings = {}            -- input embeddings
        local lstm_c = {}
        local lstm_h = {}
        for i=1,params.numLayers do
            lstm_c[i] = {[0]=initstateTest_c[i]}
            lstm_h[i] = {[0]=initstateTest_h[i]}
        end

        local predictions = {}           -- softmax outputs
        local loss = 0

        -- forward through forward layer
        for t=1, params.sequenceLength do
            embeddings[t] = clones.embed[t]:forward(dat[{{}, t}])
            local input = embeddings[t]
            for i=1,params.numLayers do
                lstm_c[i][t], lstm_h[i][t] = unpack(clones[i][t]:forward{input, lstm_c[i][t-1], lstm_h[i][t-1]})
                input = lstm_h[i][t]
            end

            predictions[t] = clones.softmax[t]:forward(lstm_h[params.numLayers][t])

            if params.ignore then
                if target ~= params.ignoreClass then
                    confusion:add(predictions[t],target[t]);
                end
            else
                confusion:add(predictions[t],target[t]);
            end
        end

        -- save state for next window
        for i=1,params.numLayers do
            initstateTest_c[i]:copy(lstm_c[i][#lstm_c[i]])
            initstateTest_h[i]:copy(lstm_h[i][#lstm_h[i]])
        end

        -- done for this window
    end

    local perf = {}
    perf.epoch = epoch

    -- frame by frame performance
    print('Frame-by-frame performance:')
    print(confusion)
    print('Mean f1-score: ' .. meanF1score(confusion))
    perf.raw = meanF1score(confusion)
    perf.conf = confusion

    -- save progress
    table.insert(epochPerformance, perf)

    return meanF1score(confusion)
end

-- main training loop
local best = 0
for e=1,params.maxEpoch do
    -- train for one epoch
    train(data.training.inputs, data.training.targets)
    -- test performance (TODO: on validation set)
    local perf = test(data.test.inputs, data.test.targets, data.classes)

    -- save model if best so far
    if perf > best then
        best = perf
        epochPerformance.best = e
        torch.save(params.logdir .. '/model.dat', protos)
    end

    -- save progress
    torch.save(params.logdir .. '/progress.dat', epochPerformance)
    -- TODO: check for convergence
end

-- cleaning up

-- done!
