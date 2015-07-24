require 'cutorch'
require 'mattorch'
require 'fbcunn'
require 'optim'
require 'meanF1score'
require 'stratBatchIter'


-- parse command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Convolutional neural network for HAR')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'initial random seed')
cmd:option('-logdir',                 'exp',  'path to store model progress, results, and log file')
cmd:option('-data',                   '',     'data-set to run on (DP datasource)')
cmd:option('-gpu',                    1,      'GPU to run on (default: 1)')
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
cmd:option('-ignore',                 false,  'Is there a class we should ignore?')
cmd:option('-ignoreClass',            0,      'Class to ignore for analysis')

cmd:text()

params = cmd:parse(arg)

paths.mkdir(params.logdir)

-- create log file
cmd:log(params.logdir .. '/log', params)

-- preliminaries
torch.manualSeed(params.seed)
cutorch.setDevice(params.gpu)
torch.setnumthreads(16)
epochPerformance = {} -- table to store progress
testPerformance = {} -- table to store progress

-- parse possible weights for classes
if params.classWeights:len() > 0 then
    params.weights = {}
    for i,v in ipairs(params.classWeights:split(',')) do table.insert(params.weights, tonumber(v)) end
    params.weights = torch.CudaTensor(params.weights):view(1,#params.weights)
end

-- Read in data-set and (maybe) store on GPU
data = torch.load(params.data)

-- check if data has the right dimensions
if data.dim == 2 then
    -- just two -> put in third 'empty' dimension
    data.training.inputs = data.training.inputs:view(data.training.inputs:size(1), data.training.inputs:size(2), 1)
    data.validation.inputs = data.validation.inputs:view(data.validation.inputs:size(1), data.validation.inputs:size(2), 1)
    data.test.inputs = data.test.inputs:view(data.test.inputs:size(1), data.test.inputs:size(2), 1)
end

-- put data on gpu
data.training.inputs = data.training.inputs:cuda()
data.test.inputs = data.test.inputs:cuda()
data.validation.inputs = data.validation.inputs:cuda()

-- define model
model = nn.Sequential()
local dim -- store last dim of layer before
local nf
-- we use facebook's temporal convolution
model:add(nn.TemporalConvolutionFB(data.training.inputs:size(3), params.nF1, params.kW1, 1))
dim = (data.training.inputs:size(2) - params.kW1) + 1
nf = params.nF1
--model:add(nn.Dropout(0.1))
model:add(nn.TemporalMaxPooling(params.mW1,params.mW1))
model:add(nn.ReLU())
model:add(nn.Dropout(params.dropout1))

dim = torch.floor((dim-params.mW1)/params.mW1+1)

-- check if we need to add another layer
if params.numConv > 1 then
    model:add(nn.TemporalConvolutionFB(params.nF1, params.nF2, params.kW2, 1))
    dim = (dim - params.kW2) + 1
    nf = params.nf2
    model:add(nn.TemporalMaxPooling(params.mW2,params.mW2))
    model:add(nn.ReLU())
    model:add(nn.Dropout(params.dropout2))
    dim = torch.floor((dim - params.mW2)/params.mW2+1)
end

-- check if we need to add another layer
if params.numConv > 2 then
    model:add(nn.TemporalConvolutionFB(params.nF2, params.nF3, params.kW3, 1))
    dim = (dim - params.kW3) + 1
    nf = params.nf3
    model:add(nn.TemporalMaxPooling(params.mW3,params.mW3))
    model:add(nn.ReLU())
    model:add(nn.Dropout(params.dropout3))
    dim = torch.floor((dim - params.mW3)/params.mW3+1)
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
model:cuda()

-- training criterion
criterion = nn.ClassNLLCriterion():cuda()

-- get parameter references and initialise
parameters, gradParameters = model:getParameters()
parameters:uniform(-0.08,0.08)

-- helper functions

-- max in norm
renormW = function(mod)
    -- rescale to incoming 2-norm of maxInNorm for each hidden unit
    local params = mod:parameters()

    -- skip layers without parameters (e.g. ReLU())
    if not params then
        return
    end
    for i,param in pairs(params) do
        if param:dim() > 1 then
            -- layer weights
            param:renorm(2,param:dim(),maxInNorm)
        end
    end
end

local checkConvergence = function(epoch, win)
    -- check for convergence over last win epochs (on validation set)
    for p=(epoch-win+1),epoch do
        if (epochPerformance[p].TP+epochPerformance[p].TN) >
           (epochPerformance[epoch-win].TP+epochPerformance[epoch-win].TN) then
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
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

   local nbatch = torch.floor(data:size(1) / batchSize)
   local cnt = 1

   -- cycle through stratified batches
   for batchIndex in stratBatchIter(labels, batchSize) do

      -- create mini batch
      local inputs = data:index(1,batchIndex):view(batchIndex:size(1),data:size(2),data:size(3))
      local targets = labels:index(1,batchIndex):cuda()

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
             df_do:cmul(weights:expandAs(df_do))
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

function test(data, labels, testing)
   -- local vars
   local time = sys.clock()
   local confusion = optim.ConfusionMatrix(classes)
   confusion:zero()

   -- set to test mode
   model:evaluate()

   -- test over given data
   print('<trainer> on testing Set:')
    local nbatch = torch.floor(labels:size(1) / batchSize)
    local cnt = 1

   for batchIndex in stratBatchIter(labels:view(labels:size(1)), batchSize) do
      -- disp progress
      xlua.progress(cnt, nbatch)

      -- create mini batch
--      local inputs = data:index(1,batchIndex):cuda()
      local inputs = data:index(1,batchIndex):view(batchIndex:size(1),data:size(2),data:size(3))
      local targets = labels:index(1,batchIndex):view(batchIndex:size(1)):cuda()

      -- test samples
      local preds = model:forward(inputs)

      -- turn into softmax
      preds:exp()
      preds:cdiv(preds:sum(2):expandAs(preds))

      -- confusion:
      for i = 1,batchIndex:size(1) do
        confusion:add(preds[i], targets[i])
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
   perf.TN = confusion.mat[1][1] / confusion.mat[1]:sum()
   perf.TP = confusion.mat[2][2] / confusion.mat[2]:sum()

   if testing == true then
      table.insert(testPerformance, perf)
    else
      table.insert(epochPerformance, perf)
   end

   return meanF1score(confusion)
end

local best = 0
local progress = {}
progress.epochPerformance = epochPerformance
progress.testPerformance = testPerformance

for e=1,params.maxEpoch do
    train(D.train_data, D.train_labels)
    local score = test(D.val_data, D.val_labels, false)
    test(D.test_data, D.test_labels, true)

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
