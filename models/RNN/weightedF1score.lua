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
