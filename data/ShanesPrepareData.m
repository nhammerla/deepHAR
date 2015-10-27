%TODO consider loading data from subjects 2,3,4 along with 1

clear;

normalize = true;
makeMeanFeatures = true;
convertToTorch = true;
allSubjects = true;
%Be sure to cd into this directory first!
%base='../../dataset/';
base=('~/OpportunityUCIDataset/dataset/')
addpath(genpath('~/OpportunityUCIDataset/scripts/'))

selectedCol = [2:46 51:59 64:72 77:85 90:98 103:134 250];

if allSubjects
	% training
	%Subject 1
	s1adl1 = load([base 'S1-ADL1.dat']);
	s1adl2 = load([base 'S1-ADL2.dat']);
	s1adl3 = load([base 'S1-ADL3.dat']);
	s1adl4 = load([base 'S1-ADL4.dat']);
	s1adl5 = load([base 'S1-ADL5.dat']);
	s1drill = load([base 'S1-Drill.dat']);

	%Subject 2
	s2adl1 = load([base 'S2-ADL1.dat']);
	s2adl2 = load([base 'S2-ADL2.dat']);
	s2adl3 = load([base 'S2-ADL3.dat']);
	s2drill = load([base 'S2-Drill.dat']);
	%Subject 3
	s3adl1 = load([base 'S3-ADL1.dat']);
	s3adl2 = load([base 'S3-ADL2.dat']);
	s3adl3 = load([base 'S3-ADL3.dat']);
	s3drill = load([base 'S3-ADL3.dat']);
	%Subject 4
	s4adl1 = load([base 'S4-ADL1.dat']);
	s4adl2 = load([base 'S4-ADL2.dat']);
	s4adl3 = load([base 'S4-ADL3.dat']);
	s4drill = load([base 'S4-ADL4.dat']);

	% test
	%Subject 2
	s2adl4 = load([base 'S2-ADL4.dat']);
	s2adl5 = load([base 'S2-ADL5.dat']);
	%Subject 3
	s3adl4 = load([base 'S3-ADL4.dat']);
	s3adl5 = load([base 'S3-ADL5.dat']);
	%Subject 4
	s4adl4 = load([base 'S4-ADL4.dat']);
	s4adl5 = load([base 'S4-ADL5.dat']);
	[training1,test1]=tarrange(18, s1adl1, s1adl2, s1adl3, s1adl4, s1adl5, s1drill, s2adl1, s2adl2, s2adl3, s2drill, s3adl1, s3adl2, s3adl3, s3drill, s4adl1, s4adl2, s4adl3, s4drill, s2adl4, s2adl5, s3adl4, s3adl5, s4adl4, s4adl5);

else
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
end

training1 = training1(:,selectedCol);
test1 = test1(:,selectedCol);

clear s1* s2* s3* s4* 

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

clear training1 test1

% ---CODE TO REMOVE THE NULL CLASS---
%isNULL = trainingLabels==1;
%trainingLabels(isNULL) = [];
%trainingData(isNULL, :) = []; 
%size(trainingData)

trainingData = backfillnans(trainingData); testingData = backfillnans(testingData);

if normalize 
%	[trainingData, colMeans, colStds] = normalizeMatrix(trainingData);

	%Normalize the test data using the same column means and standard deviations as in the training set:
%	testingData = (testingData - repmat(colMeans, size(testingData,1),1)) ./ repmat(colStds, size(testingData,1), 1);
%	testingData(isnan(testingData)) = 0;
	trainingData = trainingData./1000;
	testingData = testingData./1000;
end

stepSize = 15;
windowSize = 30;

classes = unique(trainingLabels);
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
