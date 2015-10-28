--[[
main file to train a deep neural network.
run as:
>> th main.lua -data mydata.dat
to see additional parameters:
>> th main.lua --help
]]

require 'cunn'
require 'nn'
require 'cutorch'
require 'stratBatchIter'
require 'meanF1score'
require 'optim'


cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Deep neural network for HAR')
cmd:text()
cmd:text('Options')
cmd:option('-seed',                   123,    'initial random seed')
cmd:option('-logdir',                 'exp',  'path to store model progress, results, and log file')
cmd:option('-datafile',                   'opp1.dat',     'data-set to run on (DP datasource)')
cmd:option('-gpu',                    1,      'GPU to run on (default: 1)') -- NOTE:1
cmd:option('-cpu',                    false,  'Run on CPU')
cmd:option('-numLayers',              1,      'Number of hidden layers')
cmd:option('-layerSize',              512,    'Number of units in hidden layers')
cmd:option('-learningRate',           0.0001,    'Learning rate')
cmd:option('-batchSize',              64,     'Batch-size')
cmd:option('-dropout',                0.5,    'Dropout')
cmd:option('-momentum',               0.9,    'Momentum')
cmd:option('-learningRateDecay',      5e-4,   'Learning rate decay')
cmd:option('-maxInNorm',              3,      'Max-in-norm for regularisation')
cmd:option('-patience',               10,     'Patience in early stopping')
cmd:option('-minEpoch',               30,     'Minimum number of epochs before check for convergence')
cmd:option('-maxEpoch',               150,    'Stop after this number of epochs even if not converged')
-- cmd:option('-ignore',                 false,  'Is there a class we should ignore?')
-- cmd:option('-ignoreClass',            0,      'Class to ignore for analysis')

cmd:text()

-- parse input params
params = cmd:parse(arg)
print(params.datafile)



--This can be replaced by command line parameters:
--params={}
-- 1.	Parameters about filesystem:
--params.datafile = 'opp2.dat'
--params.logdir = 'exp'
-- 2.	Parameters for defining model:
--params.numLayers = 3 --changed
--params.layerSize = 512 --changed
--params.dropout=0.5
--params.maxInNorm = 2
--params.maxOutNorm = 0.5 -- NOT IMPLEMENTED
-- 3.	Parameters related to computation
--params.cpu = false
--params.seed=1
--params.gpu= 1
---- 4.	Parameters related to model training/backpropagation:
--params.batchSize = 64
--params.learningRate = 0.1
--params.momentum = 0.9
--params.learningRateDecay = 5e-5
--params.patience = 50 -- changed
--params.minEpoch = 100 --changed
--params.maxEpoch = 1000 --changed
--params.batchSize = 64
----params.stepSize = 8
---- 5.	Parameters related to class selection
--params.ignore = false
--params.ignoreClass = 0
--params.classWeights = '' -- Won't run while this is here?
--end

-- 1.	FILESYSTEM: IMPORT DATA
data = torch.load(params.datafile)

setmetatable(data.training, 
    {__index = function(t, i) 
                    return {t.inputs[i], t.targets[i]} 
                end}
);

function data.training:size() 
    return self.inputs:size(1) 
end

-- 2.	DEFINE MODEL
model = nn.Sequential()

--Assume the data is 3D, where:
-- Dimension 1: the index of the sliding window (e.g. 11437)
-- Dimension 2: the number of timepoints per sliding window (e.g. 30)
-- Dimension 3: the number of sensor features (e.g. 113)
if data.training.inputs:size():size()==3 then
	--add reshape layer to model
	inputLayerSize = data.training.inputs:size(2)*data.training.inputs:size(3)
	model:add(nn.Reshape(inputLayerSize))
	firstLayer = nn.Linear(inputLayerSize, params.layerSize)
end

model:add(firstLayer)
model:add(nn.Dropout(params.dropout))
model:add(nn.ReLU())

--Adding hidden layers:
for layer=1,params.numLayers do
	model:add(nn.Linear(params.layerSize, params.layerSize))
	model:add(nn.Dropout(params.dropout))
	model:add(nn.ReLU())
end

outputSize = 18
model:add(nn.Linear(params.layerSize, outputSize))
model:add(nn.LogSoftMax())

classWeights = {0.695899274285215, 0.0126781498644749, 0.0134650695112355, 0.0104922619568069, 0.0181865873917985, 0.0190609425548658, 0.0105796974731136, 0.0146017312232229, 0.0116289236687943, 0.0126781498644749, 0.014251989157996, 0.0112791816035674, 0.0195855556527061, 0.0200227332342397, 0.0106671329894203, 0.015825828451517, 0.072484043018274, 0.0166127480982775}
classWeights = torch.Tensor(classWeights)
classWeights = torch.ones(18):cdiv(classWeights)
criterion = nn.ClassNLLCriterion(classWeights)
print(model)

-- 3.	COMPUTATION SETTINGS
cutorch.setDevice(params.gpu)
cutorch.manualSeed(params.seed, params.gpu)
torch.manualSeed(params.seed)
if params.cpu==false then
	criterion:cuda()
	data.training.inputs = data.training.inputs:cuda()
	data.test.inputs = data.test.inputs:cuda()
	data.validation.inputs = data.validation.inputs:cuda()
	data.training.targets=data.training.targets:cuda()
	data.test.targets=data.test.targets:cuda()
	data.validation.targets=data.validation.targets:cuda()
	model:cuda()
end

torch.setnumthreads(16)

--4.	TRAINING / BACKPROPAGATION

epochPerformance={}
testPerformance={}
-- get parameter references and initialise
parameters, gradParameters = model:getParameters()
parameters:uniform(-0.08,0.08)

-- helper functions
batchIter = stratBatchIter
if params.imbalanced then
    batchIter = stratBatchIterRepeated
end

-- max in norm
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
   for batchIndex in batchIter(labels, params.batchSize) do

      -- create mini batch
      if data:nDimension()==3 then  
            inputs = data:index(1,batchIndex):view(batchIndex:size(1),data:size(2),data:size(3))  
      end
      local targets = = labels:index(1,batchIndex)
      
	if params.cpu==false then
		targets:cuda()
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
	 --print(f)
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

function test(data, labels, classes, testing)
   -- local vars
   local time = sys.clock()
   local confusion = optim.ConfusionMatrix(classes)
   confusion:zero()

   -- set to test mode
   model:evaluate()

   -- test over given data
   print('<trainer> on testing Set:')
    local nbatch = torch.floor(labels:size(1) / params.batchSize)
    local cnt = 1

   for batchIndex in batchIter(labels:view(labels:size(1)), params.batchSize) do
      -- disp progress
      xlua.progress(cnt, nbatch)

      -- create mini batch
--      local inputs = data:index(1,batchIndex):cuda()
      local inputs = data:index(1,batchIndex):view(batchIndex:size(1),data:size(2),data:size(3))
      local targets = labels:index(1,batchIndex):view(batchIndex:size(1))
      if params.cpu==false then
	      targets:cuda()
      end

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
   perf.meanF1score = meanF1score(confusion)
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
    train(data.training.inputs, data.training.targets)
    local score = test(data.validation.inputs, data.validation.targets, data.classes, false)
    test(data.test.inputs, data.test.targets, data.classes, true)

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
