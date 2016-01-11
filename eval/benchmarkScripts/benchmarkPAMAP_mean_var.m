clear all
%For PAMAP2 dataset:
numberOfPrinComps = 56;
load('/home/shane/benchmarkScripts/pamapfrom4dec') 
addpath('/home/shane/deepHAR/data/'); 

%First half of training data for training, rest for testing
%testingData = trainingData(:,:,7001:end);
%testingLabels = trainingLabels(7001:end,:);
%trainingData = trainingData(:,:,1:7000);
%trainingLabels = trainingLabels(1:7000,:);

meanVar = @(matrix)(squeeze([mean(matrix,2); std(matrix,0,2)]));
subsample = @(matrix)(matrix(:,1:3:end,:));
%reshapeDim2 = @(matrix)(reshape(matrix, size(matrix,1)*size(matrix,2), size(matrix,3)));
process = @(matrix)(meanVar(subsample(matrix))');

trainingDataRaw = process(trainingData);
valDataRaw = process(valData);
testingDataRaw = process(testingData);

%Normalize
[trainingDataRaw, colMeans, colStds] = normalizeMatrix(trainingDataRaw);
valDataRaw = (valDataRaw - repmat(colMeans, size(valDataRaw,1),1))./repmat(colStds, size(valDataRaw,1),1);
testingDataRaw = (testingDataRaw - repmat(colMeans, size(testingDataRaw,1),1)) ./ repmat(colStds, size(testingDataRaw,1), 1);
valDataRaw(isnan(valDataRaw)) = 0;
testingDataRaw(isnan(testingDataRaw)) = 0;

%PCA:
[pc,score,latent] = princomp(trainingDataRaw);
cumsum(latent(1:numberOfPrinComps))./sum(latent(1:numberOfPrinComps))
loadings = pc(1:numberOfPrinComps,:);

trainingDataRawPCA = score(:, 1:numberOfPrinComps);
valDataRawPCA=valDataRaw*pc(:,1:numberOfPrinComps);
testingDataRawPCA=testingDataRaw*pc(:,1:numberOfPrinComps);

size(trainingDataRawPCA)
size(valDataRawPCA)
size(testingDataRawPCA)

%Apply classifiers and get confusion matrices...
mdl = fitcknn(trainingDataRawPCA, trainingLabels,'NumNeighbors',7);
%mdl = fitctree(trainingDataRawPCA, trainingLabels);
predictions = predict(mdl, testingDataRawPCA);
[C,order] = confusionmat(testingLabels,predictions)
