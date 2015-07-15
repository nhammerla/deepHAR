-- function to read in PD data
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Read in PD data')
cmd:text()
cmd:text('Options')
cmd:option('-path', '', 'path to PD_days.mat')
cmd:option('-out', '', 'output file')
cmd:option('-normalise', false, 'whiten data?')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- we come from matlab for preparing the data-sets
require 'mattorch'

path = 'PD_days.mat'
rawData = mattorch.load(params.path)

-- helper function for normalisation
local normalise = function(data)
    local m = data.training.inputs:mean(1)
    local s = data.training.inputs:std(1)

    -- whiten data (done in-place)
    data.training.inputs:add(-m:expandAs(data.training.inputs)):cdiv(s:expandAs(data.training.inputs))
    data.test.inputs:add(-m:expandAs(data.test.inputs)):cdiv(s:expandAs(data.test.inputs))
    data.validation.inputs:add(-m:expandAs(data.validation.inputs)):cdiv(s:expandAs(data.validation.inputs))
end

-- we need to fill a table called 'data'
data = {}
data.training = {}      -- will hold input, targets, subjectIds
data.test = {}          -- will hold input, targets, subjectIds
data.validation = {}    -- will hold input, targets, subjectIds

-- training data is first 5 days of each patient
local d = rawData.day_data[{{},{},{1,5}}]:reshape(rawData.day_data:size(1), rawData.day_data:size(2)*5):t()
data.training.inputs = d[{{},{1,-3}}]
data.training.targets = d[{{},-2}]
data.training.subjectIds = d[{{},-1}]

-- validation data is 6th day of all patients
local d = rawData.day_data[{{},{},{6}}]:reshape(rawData.day_data:size(1), rawData.day_data:size(2)):t()
data.validation.inputs = d[{{},{1,-3}}]
data.validation.targets = d[{{},-2}]
data.validation.subjectIds = d[{{},-1}]

-- test data is the 7th day of all patients
local d = rawData.day_data[{{},{},{7}}]:reshape(rawData.day_data:size(1), rawData.day_data:size(2)):t()
data.test.inputs = d[{{},{1,-3}}]
data.test.targets = d[{{},-2}]
data.test.subjectIds = d[{{},-1}]

if params.normalise == true then
    normalise(data)
end

-- save to outfile
torch.save(params.out, data)
