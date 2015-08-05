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
cmd:option('-learningRate',           0.5,    'Learning rate')
cmd:option('-batchSize',              64,     'Batch-size')
cmd:option('-dropout',                0.5,    'Dropout')
--cmd:option('-momentum',               0.9,    'Momentum')
-- cmd:option('-learningRateDecay',      5e-4,   'Learning rate decay')
cmd:option('-maxInNorm',              3,      'Max-in-norm for regularisation')
cmd:option('-patience',               10,     'Patience in early stopping')
cmd:option('-minEpoch',               30,     'Minimum number of epochs before check for convergence')
cmd:option('-maxEpoch',               150,    'Stop after this number of epochs even if not converged')
-- cmd:option('-ignore',                 false,  'Is there a class we should ignore?')
-- cmd:option('-ignoreClass',            0,      'Class to ignore for analysis')

cmd:text()

-- parse input params
params = cmd:parse(arg)

-- FOR CREATING LOG FILE
params.rundir = cmd:string(params.logdir, params, {dir=true})
paths.mkdir(params.rundir)

-- -- create log file
cmd:log(params.rundir .. '/log', params)

-- -- Read in data-set and (maybe) store on GPU
data = torch.load(params.datafile)
-- -- Define model

-- -- helper functions

-- -- define training function

-- -- define test function

-- -- main training loop

-- -- cleaning up

-- -- done!

inputLayerSize = data.training.inputs:size(2)*data.training.inputs:size(3)
outputSize = #data.classes

torch.manualSeed(params.seed)
cutorch.setDevice(params.gpu)
cutorch.manualSeed(params.seed, params.gpu)
torch.setnumthreads(16)

setmetatable(data.training, 
    {__index = function(t, i) 
                    return {t.inputs[i], t.targets[i]} 
                end}
);

function data.training:size() 
    return self.inputs:size(1) 
end

net = nn.Sequential()

--data.training.inputs = nn.Reshape(data.training.inputs:size(2)*data.training.inputs:size(3)):forward(data.training.inputs)

--double-check inputLayerSize
net:add(nn.Reshape(data.training.inputs:size(2)*data.training.inputs:size(3)))
layerToAdd = nn.Linear(inputLayerSize, params.layerSize)
net:add(layerToAdd)
net:add(nn.Dropout(dropout))
net:add(nn.ReLU())

-- Output Layer
outputSize = #data.classes
net:add(nn.Linear(params.layerSize, outputSize))
net:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion(torch.ones(18))

if params.cpu==false then
	--criterion = 
	criterion:cuda()
	data.training.inputs = data.training.inputs:cuda()
	data.test.inputs = data.test.inputs:cuda()
	data.validation.inputs = data.validation.inputs:cuda()
	net:cuda()
end

parameters, gradParameters = net:getParameters()
parameters:uniform(-0.08,0.08)

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

lastValueOfList = function(list1)
	return list1[#list1]
end

lowestOfLast_n_Values = function(list1,n) 
	if not list1 then return nil end
	lowest = list1[#list1-n+1]
	for i=(#list1-n+1),#list1 do
		if list1[i]<lowest then
			lowest = list1[i]
		end
	end
	return lowest
end

hasConverged = function(epoch_loss_values_list, n)
	if #epoch_loss_values_list<11 then
		return false
	elseif lastValueOfList(epoch_loss_values_list)<lowestOfLast_n_Values(epoch_loss_values_list, n) then
		return true
	else
		return false
	end
end

epochCounter = 1
epoch_loss_values_list = {}
bestPerformance = 0

for epochNumber=1,params.maxEpoch do
	converged = hasConverged(epoch_loss_values_list,params.patience)
	if converged and not epochNumber<params.minEpoch then
		break
	else
		--EPOCH BEGINS HERE
		--print('Training epoch '..epochNumber)
		--TRAINING PHASE OF EPOCH
		batchCounter = 1
		net:training()
		nbatch = torch.floor(data.training.targets:size(1) / params.batchSize)
		for batchIndex in stratBatchIter(data.training.targets, params.batchSize) do
			local batchInputs = data.training.inputs:index(1, batchIndex)
			local batchTargets = data.training.targets:index(1,batchIndex)
			if params.cpu == false then
				batchTargets = batchTargets:cuda()
			end
			local batchPredictions = net:forward(batchInputs)
			--print(batchPredictions)

			-- WHAT DO THESE 4 LINES OF CODE DO?
			--criterion:forward(batchPredictions, batchTargets)
			--net:zeroGradParameters()
			--net:backward(batchInputs, criterion:backward(net.output, batchTargets))
			--net:updateParameters(0.01)

			--print(batchTargets)

			err = criterion:forward(batchPredictions, batchTargets)
			gradCriterion = criterion:backward(batchPredictions, batchTargets)
			net:zeroGradParameters()
			net:backward(batchInputs, gradCriterion)
			net:updateParameters(params.learningRate)

			-- renormalise weights
     		for i,mod in ipairs(net.modules) do
       			renormW(mod)
     		end
			--xlua.progress(batchCounter, nbatch)
			--batchCounter = batchCounter + 1
		end
		--TESTING PHASE OF EPOCH
		--in this epoch, construct the confusion matrix

		--print('Testing epoch...')
		
		cmatrix = optim.ConfusionMatrix(data.classes)
		cmatrix:zero()

		nbatch = torch.floor(data.validation.targets:size(1) / params.batchSize)
		for batchIndex in stratBatchIter(data.validation.targets, params.batchSize) do
			local testBatchInputs = data.validation.inputs:index(1, batchIndex)
			local testBatchTargets = data.validation.targets:index(1,batchIndex):view(batchIndex:size(1))
			local testBatchPredictions = net:forward(testBatchInputs)
			--turn into softmax...
			testBatchPredictions:exp() -- exponentiate
			testBatchPredictions:cdiv(testBatchPredictions:sum(2):expandAs(testBatchPredictions)) -- divide by sum

			--modify confusion matrix...
			--cmatrix:batchAdd(testBatchPredictions, testBatchTargets)--batchAdd doesn't work?
			--TRY GOING THROUGH IT INDIVIDUALLY INSTEAD...
			for i=1,testBatchPredictions:size(1) do
				cmatrix:add(testBatchPredictions[i], testBatchTargets[i])
			end
		end

		epochPerformance = meanF1score(cmatrix)
		--print('epoch performance ' .. epochPerformance)
		table.insert(epoch_loss_values_list, epochPerformance)

		--check if this is the best performance so far,
		-- and if so, then save the net to a file...
		if epochPerformance>bestPerformance then
			bestPerformance = epochPerformance
			torch.save(params.logdir .. '/model.dat',net)
		end

		--EPOCH ENDS HERE
	end
	xlua.progress(epochCounter, params.maxEpoch)
	epochCounter = epochCounter+1
end
print()
print(cmatrix)
print(epoch_loss_values_list)




