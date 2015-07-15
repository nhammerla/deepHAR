require 'mattorch'

local PDdata, DataSource = torch.class("dp.PDdata", "dp.DataSource")
PDdata.isPDdata = true

PDdata._name = 'PDdata'
--PDdata._image_size = {1, 96, 96}
PDdata._feature_size = 91
PDdata._data_axes = 'bf'
PDdata._target_axes = 'b'
PDdata._classes = {'ASLEEP','OFF','ON','DYS'}

function PDdata:__init(config)

    self:setInputPreprocess(dp.Standardize())
    local D = mattorch.load(config.path)

    self:loadTrain(D)
    self:loadValid(D)
    self:loadTest(D)
    
    DataSource.__init(self, {
      train_set=self:trainSet(), 
      valid_set=self:validSet(),
      test_set=self:testSet()
    })
end

function PDdata:createSet(data, which_set, removeElems)
    
    if removeElems then
        -- get indeces with label >= 0
        ind = {};
        for i=1,data:size(1) do
            if data[i][-2] >= 0  then
                table.insert(ind, i)
            end
        end
        ind = torch.LongTensor(ind)
    
        -- reduce data
        data = data:index(1,ind)
    end

    -- get labels, narrow data
    local labels = data[{{},{-2}}]:clone() + 1

    if config.cuda then
        labels = labels[{{},1}]:cuda()
        data = data[{{},{1,-3}}]:cuda()
    else
        labels = labels[{{},1}]
        data = data[{{},{1,-3}}]
    end

    -- construct inputs and targets dp.Views 
    local input_v, target_v = dp.DataView(), dp.ClassView()
    
    -- add data
    input_v:forward(self._data_axes, data)
    target_v:forward(self._target_axes, labels)
    target_v:setClasses(self._classes) 
    -- construct dataset
    return dp.DataSet{inputs=input_v,targets=target_v,which_set=which_set}
end

function PDdata:loadTrain(D)
    -- first 5 days as training data
    local data = D.day_data[{{},{},{1}}]:squeeze():t()
    for i=2, 5 do
        data = data:cat(D.day_data[{{},{},{i}}]:squeeze():t(),1)
    end
    
    self:setTrainSet(self:createSet(data, 'train', true))
    return self:trainSet()
end

function PDdata:loadTest(D)
    -- first 5 days as training data
    local data = D.day_data[{{},{},{7}}]:squeeze():t()
    self:setTestSet(self:createSet(data, 'test', false))
    return self:testSet()
end

function PDdata:loadValid(D)
   --Data will contain a tensor where each row is an example, and where
   --the last column contains the target class.
   local data = D.day_data[{{},{},{6}}]:squeeze():t()
   self:setValidSet(self:createSet(data, 'train', true))
   return self:validSet()
end

return PDdata
