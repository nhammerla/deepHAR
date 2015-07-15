require 'cutorch'
require 'cunn'
require 'nngraph'
require 'mattorch'
require 'optim'
require 'dp'
LSTM = require 'LSTM'
model_utils = require 'model_utils'
require 'meanF1score'

opt={}
opt.input_size = 91
opt.rnn_size = 128
opt.output_size = 4
opt.seq_length = 64
opt.batch_size = 64
opt.maxIter = 200 --100
opt.maxEpoch = 1000
opt.inputNoise = 0--.01
opt.transitionNoise = 0 --0.01
opt.step_size = 16 
learningRate = 0.01
learningRateDecay = 0.000001
weightDecay = 0 --.00001
testEvery = 1
retainProb = 0.5

-- to save performance
epochPerformance = {}

local PDdata = require 'PDdatasource'

torch.manualSeed(122223)
torch.setnumthreads(16)

-- local weights = torch.DoubleTensor{1.0, 2.5, 1.0000, 3.0}:cuda():view(1,4)
local weights = torch.DoubleTensor{0.5, 1, 0.7, 1}:cuda():view(1,4)
--local weights = torch.DoubleTensor{0.6629, 0.8363, 0.5933, 0.9075}:cuda():view(1,4)
--weights:fill(1)

-- data
datasource = PDdata.new{path='../PD_days.mat', input_preprocess = dp.Standardize()}

-- get normalisation function
st = dp.Standardize()

-- normalise
train = datasource:trainSet():inputs()
valid = datasource:validSet():inputs()
test = datasource:testSet():inputs()
st:apply(train, true)
st:apply(valid, false)
st:apply(test, false)

-- done

D = {};
D.train_data = datasource:get('train','inputs'):t():cuda()
D.train_labels = datasource:get('train','targets'):view(1,D.train_data:size(2)):cuda()
D.test_data = datasource:get('test','inputs'):t():cuda()
D.test_labels = datasource:get('test','targets'):view(1,D.test_data:size(2)):cuda()

print('Number of samples: ' .. D.train_data:size(2))

classes = {'ASLEEP','OFF','ON','DYS'}

------------ DEFINE MODEL ---------------
-- define model prototypes for one timestep, then clone them
protos = {} 
protos.embed = nn.Sequential():add(nn.Linear(opt.input_size, opt.rnn_size)):cuda()
protos.lstmForw = LSTM.lstm(opt):cuda()
protos.lstmBack = LSTM.lstm(opt):cuda()

protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.output_size)):add(nn.LogSoftMax()):cuda()
protos.criterion = nn.ClassNLLCriterion():cuda()
dropper = nn.Dropout(retainProb):cuda()

-- get reference to params and gradparams
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.lstmForw, protos.lstmBack, protos.softmax)
params:uniform(-0.08, 0.08)
print('#parameters: ' .. params:size(1))

-- make a bunch of clones, AFTER flattening, as that reallocates memory
clones = {} -- TODO: local
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size):cuda()
local initstate_h = initstate_c:clone()
local initstateBack_c = initstate_c:clone()
local initstateBack_h = initstate_c:clone()

-- for testing
local initstateTest_c = torch.zeros(1, opt.rnn_size):cuda()
local initstateTest_h = initstateTest_c:clone()
local initstateBackTest_c = torch.zeros(1, opt.rnn_size):cuda()
local initstateBackTest_h = initstateBackTest_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, which is not ideal though
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()
local dfinalstateBack_c = initstate_c:clone()
local dfinalstateBack_h = initstate_c:clone()

local dataIter = function(data,labels) 
    --local pos = 1
    -- random start positions for a particle for each batch
    local posList = torch.rand(opt.batch_size)*(data:size(2)-opt.seq_length)
--    local posList = torch.linspace(1,data:size(2)-opt.seq_length,opt.batch_size)

    posList:ceil()

    local ndim = data:size(1)
    local nelem = opt.batch_size * opt.seq_length

    local d = torch.CudaTensor(opt.batch_size, ndim, opt.seq_length):fill(0)
    local l = torch.CudaTensor(opt.batch_size, opt.seq_length):fill(0)

    return function()
        -- generate random batch of sequences
        for i=1,opt.batch_size do
            local pos = posList[i]
            d[i] = data[{{}, {pos, pos+opt.seq_length-1}}]
            l[i] = labels[{{}, {pos, pos+opt.seq_length-1}}]
            posList[i] = posList[i] + opt.step_size 
            if posList[i] > (data:size(2)-opt.seq_length) then 
                posList[i] = 1
            end
        end
        return d,l
    end
end

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


-- train function (one epoch)
function train(data, labels)
    epoch = epoch or 1
--    local confusion = optim.ConfusionMatrix(classes)

    print('Training epoch: ' .. epoch)
    
    -- zero confusion to track performance on training set
--    confusion:zero()
    
    -- make iterator for sampling sequences
    local batchIter = dataIter(data, labels)
    
    -- reset initstate
    initstate_c:zero()
    initstate_h:zero()
    
    -- one pass through the data-set (set opt.maxIter accordingly, will just wrap around)
    local cnt = 1
    for iter=1,opt.maxIter do
        -- get next batch of data
        local x,y = batchIter()
        
        ------------------- closure for gradient calculation -------------------
        function feval(p)
            -- just in case
            collectgarbage()
            
            if p ~= params then
                params:copy(p)
            end
            grad_params:zero()
            
            ------------------- forward pass -------------------
            local embeddings = {}            -- input embeddings
            local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
            local lstm_h = {[0]=initstate_h} -- output values of LSTM
            local lstmBack_c = {[opt.seq_length+1]=torch.CudaTensor(initstate_c:size()):fill(0)}
            local lstmBack_h = {[opt.seq_length+1]=torch.CudaTensor(initstate_c:size()):fill(0)}
            local predictions = {}           -- softmax outputs
            local loss = 0
        
            -- forward pass through forward layer
            for t=1, opt.seq_length do
                embeddings[t] = clones.embed[t]:forward(x[{{}, {}, {t}}]:squeeze())
                lstm_c[t], lstm_h[t] = unpack(clones.lstmForw[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
            end
        
            -- 'forward' (i.e. right to left) pass through backward layer
            for t=opt.seq_length,1,-1 do
                lstmBack_c[t], lstmBack_h[t] = unpack(clones.lstmBack[t]:forward{lstm_h[t], lstmBack_c[t+1], lstmBack_h[t+1]})
                
                predictions[t] = clones.softmax[t]:forward(lstmBack_h[t])
                -- check if label is zero?

                loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
            end 
        
            loss = loss / opt.seq_length
        
            ------------------ backward pass -------------------
            -- complete reverse order of the above
            local dembeddings = {}                              -- d loss / d input embeddings
            local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
            local dlstm_h = {}                                  -- output values of LSTM
            local dlstmBack_c = {[0]=dfinalstateBack_c}
            local dlstmBack_h = {}
        
            -- 'backward' pass through backward layers (i.e. left to right)
            for t=1,opt.seq_length do
                local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
                
                doutput_t:cmul(weights:expandAs(doutput_t))
                dlstmBack_h[t] = clones.softmax[t]:backward(lstmBack_h[t], doutput_t)
                dlstm_h[t], dlstmBack_c[t+1], dlstmBack_h[t+1] = unpack(clones.lstmBack[t]:backward(
                    {lstm_h[t], lstmBack_c[t+1], lstmBack_h[t+1]},
                    {dlstmBack_c[t], dlstmBack_h[t]}
                ))
            end
        
            -- backward pass through forward layer (i.e. right to left)
            for t=opt.seq_length,1,-1 do
                -- backprop through loss, and softmax/linear
                dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstmForw[t]:backward(
                    {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
                    {dlstm_c[t], dlstm_h[t]}
                ))
        
                -- backprop through embeddings
                clones.embed[t]:backward(x[{{}, {}, {t}}]:squeeze(), dembeddings[t])
            end
        
            ------------------------ misc ----------------------
            -- transfer final state to initial state (BPTT)
--            if torch.rand(1)[1] <= retainProb then
--                initstate_c:copy(lstm_c[opt.step_size])
--                initstate_h:copy(lstm_h[opt.step_size])
--            end
            
            -- use a drop-out layer to drop some of the previous context
            initstate_c:copy(dropper:forward(lstm_c[opt.step_size]))
            initstate_h:copy(dropper:forward(lstm_h[opt.step_size]))

            -- clip gradient element-wise
            grad_params:clamp(-5, 5)
        
            return loss, grad_params
        end
        
        -- optimizer state
        sgdState = sgdState or {
            learningRate = learningRate,
            --momentum = momentum,
            --weightDecay = weightDecay,
            learningRateDecay = learningRateDecay
         }
         
         -- run optimizer
         optim.adagrad(feval, params, sgdState)
         
         xlua.progress(cnt, opt.maxIter)
         
         cnt = cnt + 1
    end
    
    epoch = epoch + 1
    -- done 
end

-- test network
function test(data,labels)
    -- run through test-set
    local confusion = optim.ConfusionMatrix(classes)
    local smoothConfusion = optim.ConfusionMatrix(classes)

    confusion:zero()
    initstateTest_c:zero()
    initstateTest_h:zero()
    
    local nelem = opt.seq_length
    local tt = data:size(2)-nelem
    local b = 1
    
    -- store predictions here
    local P = torch.zeros(data:size(2),4)
    
    -- run through whole set, with a batch-size of 1
    for pos=1,tt,nelem do
        xlua.progress(pos, tt)
        -- get data for this window
        local dat = data[{{},{pos+(b-1)*opt.seq_length, pos+b*opt.seq_length-1}}]
        local target = labels[{{},{pos+(b-1)*opt.seq_length, pos+b*opt.seq_length-1}}]
        
        -- do forward pass through network
        local embeddings = {}            -- input embeddings
        local lstm_c = {[0]=initstateTest_c} -- internal cell states of LSTM
        local lstm_h = {[0]=initstateTest_h} -- output values of LSTM
        local lstmBack_c = {[opt.seq_length+1]=initstateBackTest_c} -- internal cell states of LSTM
        local lstmBack_h = {[opt.seq_length+1]=initstateBackTest_h} -- output values of LSTM
        local predictions = {}           -- softmax outputs
        local loss = 0
    
        -- forward through forward layer
        for t=1, opt.seq_length do
            embeddings[t] = clones.embed[t]:forward(dat[{{}, t}])
            lstm_c[t], lstm_h[t] = unpack(clones.lstmForw[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})
        end
    
        -- forwards through backwards layer
        for t=opt.seq_length,1,-1 do
            lstmBack_c[t], lstmBack_h[t] = unpack(clones.lstmBack[t]:forward{lstm_h[t], lstmBack_c[t+1], lstmBack_h[t+1]})
            predictions[t] = clones.softmax[t]:forward(lstmBack_h[t])
            confusion:add(predictions[t],target[1][t]);
        end
    
        -- save state for next window 
        initstateTest_c:copy(lstm_c[#lstm_c])
        initstateTest_h:copy(lstm_h[#lstm_h])
        
        -- save predictions for smoothing later on
        for i=1,#predictions do
            P[pos+i-1] = predictions[i]:double()
        end
        
        -- done for this window
    end
    -- P now holds predictions for whole set
    
    local perf = {}
    perf.epoch = epoch

    -- frame by frame performance
    print('Frame-by-frame performance:')
    print(confusion)
    print('Mean f1-score: ' .. meanF1score(confusion))
    perf.raw = meanF1score(confusion)
    perf.rawConf = confusion
    
    -- calculate smoothed performance
    confusion:zero()
    local wlen = 60
    local wstep = 20
    for i=1,P:size(1)-wlen,wstep do
        local m = P[{{i,i+wlen-1},{}}]:mean(1)
        m:exp()
        m:div(m:sum())
        local l = mode(labels[{{}, {i,i+wlen-1}}]:view(wlen):double():totable())
        smoothConfusion:add(m[1], l)
    end       

    print('Smoothed performance:') 
    print(smoothConfusion)
    print('Mean f1-score: ' .. meanF1score(smoothConfusion))
    perf.smooth = meanF1score(smoothConfusion)    
    perf.smoothConf = smoothConfusion
    -- done!

    -- save progress
    table.insert(epochPerformance, perf)
end

-- training main loop
for e=1,opt.maxEpoch do
    train(D.train_data, D.train_labels)
    if e % testEvery == 0 then
        test(D.test_data, D.test_labels)
        torch.save('progress.dat', epochPerformance)
    end
end



