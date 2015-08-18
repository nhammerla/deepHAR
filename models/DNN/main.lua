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

-- FOR CREATING LOG FILE
params.rundir = cmd:string(params.logdir, params, {dir=true})
paths.mkdir(params.rundir)

-- -- create log file
cmd:log(params.rundir .. '/log', params)

-- -- Read in data-set and (maybe) store on GPU
data = torch.load(params.datafile)
-- -- Define model

-- -- helper functions
-- from http://lua-users.org/wiki/TableUtils
function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

function table.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end

header = table.tostring(params)
log_file_name = header:gsub('%W','')
logger = optim.Logger(params.logdir .. '/shanesLogs/' .. log_file_name)

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

--rescale: subtract mean and divide by standard deviation:
mean = {}
stdv = {}

--For training set:
for i=1,data.training.inputs:size(3) do -- over all our variables
	mean[i] = data.training.inputs[{ {},{},{i} }]:mean() -- calculate mean of variable i
	print('Feature ' .. i .. ', Mean: ' .. mean[i]) -- print this mean
	data.training.inputs[{ {},{},{1} }]:add(-mean[i]) -- subtract this mean from variable values
	stdv[i] = data.training.inputs[{ {},{},{i} }]:std() -- std estimation
	print('Feature ' .. i .. ', Standard Deviation: ' .. stdv[i]) -- print this standard dev
	data.training.inputs[{ {},{},{1} }]:div(stdv[i])
end

-- data.training.inputs:add(mean:expandAs(data.training.inputs)):cdiv(stdv:expandAs(data.training.inputs))

--For val set:
for i=1,data.training.inputs:size(3) do
	data.validation.inputs[{ {},{},{i} }]:add(-mean[i]) -- subtract VALIDATION means from test set
	data.validation.inputs[{ {},{},{i} }]:div(stdv[i]) -- divide VALIDATION by training stdvs
end
--For test set:
for i=1,data.training.inputs:size(3) do
	data.test.inputs[{ {},{},{i} }]:add(-mean[i]) -- subtract training means from test set
	data.test.inputs[{ {},{},{i} }]:div(stdv[i]) -- divide testset by training stdvs
end

model = nn.Sequential()

--data.training.inputs = nn.Reshape(data.training.inputs:size(2)*data.training.inputs:size(3)):forward(data.training.inputs)




--need todouble-check inputLayerSize

--If the training data is 3D, then add reshape layer
--We will assume that the validation and test data are of the same dimensions
if data.training.inputs:nDimension()==3 then
	model:add(nn.Reshape(data.training.inputs:size(2)*data.training.inputs:size(3)))
end

layerToAdd = nn.Linear(inputLayerSize, params.layerSize)
model:add(layerToAdd)
model:add(nn.Dropout(dropout))
model:add(nn.ReLU())

-- Output Layer
outputSize = #data.classes
model:add(nn.Linear(params.layerSize, outputSize))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion(torch.ones(18))
--criterion = nn.ClassNLLCriterion()

if params.cpu==false then
	--criterion = 
	criterion:cuda()
	data.training.inputs = data.training.inputs:cuda()
	data.test.inputs = data.test.inputs:cuda()
	data.validation.inputs = data.validation.inputs:cuda()
	model:cuda()
end

parameters, gradParameters = model:getParameters()
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
		model:training()
		nbatch = torch.floor(data.training.targets:size(1) / params.batchSize)

		local trconf = optim.ConfusionMatrix(data.classes)

		for batchIndex in stratBatchIter(data.training.targets, params.batchSize) do
			local batchInputs = data.training.inputs:index(1, batchIndex)
			local batchTargets = data.training.targets:index(1,batchIndex)
			if params.cpu == false then
				batchTargets = batchTargets:cuda()
			end

			--BEGIN SHANE'S TRAINING PART
			-- model:zeroGradParameters()

			-- local batchPredictions = model:forward(batchInputs)
			-- --print(batchPredictions)

			-- -- WHAT DO THESE 4 LINES OF CODE DO?
			-- --criterion:forward(batchPredictions, batchTargets)
			-- --model:zeroGradParameters()
			-- --model:backward(batchInputs, criterion:backward(model.output, batchTargets))
			-- --model:updateParameters(0.01)

			-- --print(batchTargets)
			

			-- local loss = criterion:forward(batchPredictions, batchTargets)
			-- gradCriterion = criterion:backward(batchPredictions, batchTargets)
			
			-- model:backward(batchInputs, gradCriterion)

			-- model:updateParameters(params.learningRate)

			--END SHANE'S TRAINING PART
			--BEGIN NILS' ALTERNATIVE

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
		         local batchPredictions = model:forward(batchInputs)
		         local f = criterion:forward(batchPredictions, batchTargets:view(batchTargets:size(1)))

		         -- estimate df/dW
		         local df_do = criterion:backward(batchPredictions, batchTargets:view(batchTargets:size(1)))

		         if params.weights then
		             df_do:cmul(params.weights:expandAs(df_do))
		         end

		         -- backpropagate
		         model:backward(batchInputs, df_do)

		         -- return f and df/dX
		         return f,gradParameters
		     end

		     sgdState = sgdState or {
		        learningRate = params.learningRate,
		        momentum = params.momentum,
		        learningRateDecay = params.learningRateDecay
		     }

		     optim.sgd(feval, parameters, sgdState)
		     -- END NILS' ALTERNATIVE

			-- renormalise weights
     		for i,mod in ipairs(model.modules) do
       			renormW(mod)
     		end

     		--print(loss)	

     		-- for i=1,batchPredictions:size(1) do
     		-- 	trconf:add(batchPredictions[i], batchTargets[i])
     		-- end
			--xlua.progress(batchCounter, nbatch)
			--batchCounter = batchCounter + 1
		end

		-- print(trconf)
		--TESTING PHASE OF EPOCH
		--in this epoch, construct the confusion matrix

		--print('Testing epoch...')
		
		model:evaluate()
		cmatrix = optim.ConfusionMatrix(data.classes)
		cmatrix:zero()

		nbatch = torch.floor(data.validation.targets:size(1) / params.batchSize)
		for batchIndex in stratBatchIter(data.validation.targets, params.batchSize) do
			local testBatchInputs = data.validation.inputs:index(1, batchIndex)
			local testBatchTargets = data.validation.targets:index(1,batchIndex):view(batchIndex:size(1))
			local testBatchPredictions = model:forward(testBatchInputs)
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

		print(cmatrix)

		epochPerformance = meanF1score(cmatrix)
		print('epoch'.. epochCounter .. ' performance ' .. epochPerformance)
		table.insert(epoch_loss_values_list, epochPerformance)

		--check if this is the best performance so far,
		-- and if so, then save the net to a file...
		if epochPerformance>bestPerformance then
			bestPerformance = epochPerformance
			torch.save(params.logdir .. '/model.dat',model)
		end


    	-- save progress
    	logger:add{['MeanF1:'..header] = epochPerformance, ['epoch number'] = epochCounter }

		--EPOCH ENDS HERE
	end
	xlua.progress(epochCounter, params.maxEpoch)
	epochCounter = epochCounter+1
end
print()
print(cmatrix)
