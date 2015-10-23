%TODO consider loading data from subjects 2,3,4 along with 1
%TODO OPP2 permute to be same dimensions as Nils' opp1.dat

clear;

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

trainingData = backfillnans(trainingData); testingData = backfillnans(testingData);
stepSize = 15
windowSize = 30

classes = unique(trainingLabels)
[trainingData, trainingLabels] = rollingWindows(trainingData, trainingLabels, stepSize, windowSize);
[testingData, testingLabels] = rollingWindows(testingData, testingLabels, stepSize, windowSize);
%trainingData = trainingData';testingData = testingData';
%trainingLabels = trainingLabels';testingLabels = testingLabels';

%Make mean features in each sliding window:
meanFeatures = @(matrix1) squeeze(sum(matrix1,1));
slidingMeanTrainData = meanFeatures(trainingData);
slidingMeanTestData = meanFeatures(testingData);

save('opp2.mat','classes', 'trainingData', 'trainingLabels', 'testingData', 'testingLabels', 'slidingMeanTrainData', 'slidingMeanTestData')
save('opp2MATLABMEANFEATURES.mat', 'slidingMeanTrainData', 'slidingMeanTestData', 'trainingLabels', 'testingLabels')
system('th shaneMatFiles2torch.lua')
