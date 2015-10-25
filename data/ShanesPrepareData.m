%TODO consider loading data from subjects 2,3,4 along with 1

clear;

normalize = true;
makeMeanFeatures = true;
convertToTorch = true;
%Be sure to cd into this directory first!
%base='../../dataset/';
base=('~/OpportunityUCIDataset/dataset/')
addpath(genpath('~/OpportunityUCIDataset/scripts/'))

selectedCol = [2:46 51:59 64:72 77:85 90:98 103:134 250];

% Subject 1
% training
sadl1 = load([base 'S1-ADL1.dat']);
sadl2 = load([base 'S1-ADL2.dat']);
sadl3 = load([base 'S1-ADL3.dat']);
sdrill = load([base 'S1-Drill.dat']);
% test
sadl4 = load([base 'S1-ADL4.dat']);
sadl5 = load([base 'S1-ADL5.dat']);

[training1,test1]=tarrange(4,sadl1,sadl2,sadl3,sdrill,sadl4,sadl5);
training1 = training1(:,selectedCol);
test1 = test1(:,selectedCol);

%training data without labels:
trainingData = training1(:,1:(end-1));
%labels corresponding to the training data:
trainingLabels = training1(:,end);
%Change labels into 1-18
uniques = unique(sort(trainingLabels));
trainingLabels = changem(trainingLabels, 1:length(uniques), uniques);

%test data without labels:
testingData = test1(:,1:(end-1));
%labels corresponding to the TEST data:
testingLabels = test1(:,end);
%Change labels into 1-18
uniques = unique(sort(testingLabels));
testingLabels = changem(testingLabels, 1:length(uniques), uniques);

clear base sadl1 sadl2 sadl3 sadl4 sadl5 sdrill selectedCol test1 training1 uniques

% ---CODE TO REMOVE THE NULL CLASS---
%isNULL = trainingLabels==1;
%trainingLabels(isNULL) = [];
%trainingData(isNULL, :) = []; 
%size(trainingData)

trainingData = backfillnans(trainingData); testingData = backfillnans(testingData);

if normalize 
	[trainingData, colMeans, colStds] = normalizeMatrix(trainingData);

	%Normalize the test data using the same column means and standard deviations as in the training set:
	testingData = (testingData - repmat(colMeans, size(testingData,1),1)) ./ repmat(colStds, size(testingData,1), 1);
	testingData(isnan(testingData)) = 0;
end

stepSize = 15;
windowSize = 30;

classes = unique(trainingLabels)
[trainingData, trainingLabels] = rollingWindows(trainingData, trainingLabels, stepSize, windowSize);
[testingData, testingLabels] = rollingWindows(testingData, testingLabels, stepSize, windowSize);
%trainingData = trainingData';testingData = testingData';
%trainingLabels = trainingLabels';testingLabels = testingLabels';

%Permute dimensions to match Nils' data
trainingData = permute(trainingData, [2 1 3]);
testingData = permute(testingData, [2 1 3]);

save('opp2.mat','classes', 'trainingData', 'trainingLabels', 'testingData', 'testingLabels')

%BEGINNING TESTING CODE
if makeMeanFeatures 
	%Make mean features in each sliding window:
	meanFeatures = @(matrix1) transpose(squeeze(mean(matrix1,2)));
	slidingMeanTrainData = meanFeatures(trainingData);
	slidingMeanTestData = meanFeatures(testingData);
	save('opp2MATLABMEANFEATURES.mat', 'slidingMeanTrainData', 'slidingMeanTestData', 'trainingLabels', 'testingLabels')
	%unix('Rscript replicateNearestNeighbours.R','-echo');
end
%END TESTING CODE
if convertToTorch 
	unix('th shaneMatFiles2torch.lua');
end
