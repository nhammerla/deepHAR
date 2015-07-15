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

        if score == score then -- weird check for nan
            table.insert(scores, score)
        else
            table.insert(scores, 0)
        end
    end

    return torch.DoubleTensor(scores):mean()
end
