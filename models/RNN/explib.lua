-- exp lib

-- performance metric 1
meanF1score = function(confusion)
    local C = confusion.mat

    local scores = {}
    for i=1,confusion.nclasses do
        local TP = C[i][i] -- true positives
        local FP = C[{{},i}]:sum()-TP
        local FN = C[i]:sum()-TP

        local PREC = TP / (TP + FP)
        local RECALL = TP / (TP + FN)

        local score = 2 * (PREC*RECALL) / (PREC+RECALL)

        if TP == 0 and C[i]:sum() > 0 then
            score = 0
        end

        if score == score then -- weird check for nan
            table.insert(scores, score)
        end
    end

    return torch.DoubleTensor(scores):mean()
end

-- performance metric 2
weightedF1score = function(confusion)
    local C = confusion.mat

    local scores = {}
    local nelems = {}
    local N = C:sum()

    for i=1,confusion.nclasses do
        local TP = C[i][i] -- true positives
        local FP = C[{{},i}]:sum()-TP
        local FN = C[i]:sum()-TP

        table.insert(nelems, C[i]:sum() / N)

        local PREC = TP / (TP + FP)
        local RECALL = TP / (TP + FN)

        local score = 2 * (PREC*RECALL) / (PREC+RECALL)

        if TP == 0 and C[i]:sum() > 0 then
            score = 0
        end

        if score == score then -- weird check for nan
            table.insert(scores, score*nelems[i])
        end
    end


    return torch.DoubleTensor(scores):sum()
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

-- max-norm regularisation
renormWeights = function(mod)
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
dataIter = function(data, labels)
    --local pos = 1
    -- random start positions for a particle for each batch
    local posList = torch.rand(params.batchSize)*(data:size(1)-params.sequenceLength)
    posList:ceil()
    posList = posList:long() -- need LongTensor for addressing

    local nelem = params.batchSize * params.sequenceLength

    local d
    local l
    return function()
        d = {}
        l = {}

        -- insert inputs into table
        for i=1,params.sequenceLength do
          table.insert(d, data:index(1, posList))
          table.insert(l, labels:index(1, posList))
          posList:add(1)
        end

        -- check if we have to reset positions
        for i=1,params.batchSize do
          if posList[i] + 1 > data:size(1)-params.sequenceLength then
            posList[i] = 1
          end
        end

        return d,l
    end
end

-- iterate linearly through data along first dimension
linearDataIter = function(data, labels, stepSize)
  local pos = 1
  local maxpos = data:size(1)
  local posList
  return function()
    if pos < maxpos then
      if pos+stepSize <= maxpos then
        posList = torch.range(pos,pos+stepSize-1):long()
      else
        posList = torch.range(pos,maxpos):long()
      end
      pos = pos + stepSize
      return data:index(1,posList):split(1,1), labels:index(1,posList):split(1,1)
    end
  end
end

-- check for convergence over last win epochs (on validation set)
checkConvergence = function(epoch, win)
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
