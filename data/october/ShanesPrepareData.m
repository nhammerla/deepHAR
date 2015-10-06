clear;

%Be sure to cd into this directory first!
%base='../../dataset/';
base=('~/OpportunityUCIDataset/dataset/')
addpath(genpath('~/OpportunityUCIDataset/scripts/'))
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
selectedCol = [2:46 51:59 64:72 77:85 90:98 103:134 250];
training1 = training1(:,selectedCol);
test1 = test1(:,selectedCol);

[training1,test1]=tarrange(4,sadl1,sadl2,sadl3,sdrill,sadl4,sadl5);
selectedCol = [2:46 51:59 64:72 77:85 90:98 103:134 250];
training1 = training1(:,selectedCol);
test1 = test1(:,selectedCol);

%training data without labels:
trainingData = training1(:,1:(end-1));
%labels corresponding to the training data:
trainingLabels = training1(:,end);

%test data without labels:
testingData = test1(:,1:(end-1));
%labels corresponding to the TEST data:
testingLabels = test1(:,end);

removedNaNs = backfillnans(trainingData);removedNaNsTestset = backfillnans(testingData);
stepSize = 15
windowSize = 30
classes = unique(trainingLabels)
[trainingData, trainingLabels] = rollingWindows(removedNaNs, trainingLabels, stepSize, windowSize);
[testingData, testingLabels] = rollingWindows(removedNaNsTestset, testingLabels, stepSize, windowSize);
trainingData = trainingData';testingData = testingData';
trainingLabels = trainingLabels';testingLabels = testingLabels';

save('~/opp2.mat','classes', 'trainingData', 'trainingLabels', 'testingData', 'testingLabels')
