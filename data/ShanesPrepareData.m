clear;

%OPTIONS:
normalize = true; %Dividing by 1000
makeMeanFeatures = false;
convertToTorch = true;
allSubjects = true;
IMUsOnly = true;
makeSlidingWindows = false;
stepSize = 10;
windowSize = 10;

%Be sure to cd into this directory first!
%base='../../dataset/';
base=('~/OpportunityUCIDataset/dataset/')
addpath(genpath('~/OpportunityUCIDataset/scripts/'))

selectedCol = [2:46 51:59 64:72 77:85 90:98 103:134 250];

if IMUsOnly
%selectedCol = [38:48 51:59 64:72 77:85 90:98 103:134 250];
	selectedCol = [38:46 51:59 64:72 77:85 90:98 103:134 250];
end

if allSubjects
	% training
	%Subject 1
	s1adl1 = load([base 'S1-ADL1.dat']);
	s1adl5 = load([base 'S1-ADL5.dat']);
	s1adl3 = load([base 'S1-ADL3.dat']);
	s1adl4 = load([base 'S1-ADL4.dat']);
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
	s4adl4 = load([base 'S4-ADL4.dat']);
	s4adl5 = load([base 'S4-ADL5.dat']);
	s4drill = load([base 'S4-ADL4.dat']);

	%validation
	s1adl2 = load([base 'S1-ADL2.dat']);%validation
	val1 = [s1adl2];

	% test
	%Subject 2
	s2adl4 = load([base 'S2-ADL4.dat']);
	s2adl5 = load([base 'S2-ADL5.dat']);

	%Subject 3
	s3adl4 = load([base 'S3-ADL4.dat']);
	s3adl5 = load([base 'S3-ADL5.dat']);

	[training1,test1] = tarrange(19, s1adl1, s1adl5, s1adl3, s1adl4, s1drill, s2adl1, s2adl2, s2adl3, s2drill, s3adl1, s3adl2, s3adl3, s3drill, s4adl1, s4adl2, s4adl3, s4adl4, s4adl5, s4drill, s2adl4, s2adl5, s3adl4, s3adl5);
	[~,val1] = tarrange(19, s1adl1, s1adl5, s1adl3, s1adl4, s1drill, s2adl1, s2adl2, s2adl3, s2drill, s3adl1, s3adl2, s3adl3, s3drill, s4adl1, s4adl2, s4adl3, s4adl4, s4adl5, s4drill, s1adl2);


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
val1 = val1(:, selectedCol);

clear s1* s2* s3* s4* 

%training data without labels:
trainingData = training1(:,1:(end-1));
%labels corresponding to the training data:
trainingLabels = training1(:,end);
%Change labels into 1-18
uniques = unique(sort(trainingLabels));
trLabels = changeLabels(trainingLabels, unique(trainingLabels));
trainingLabels = changem(trainingLabels, 1:length(uniques), uniques);

fprintf('FOO: %.2f\n', sum(trLabels == trainingLabels))

%validation data without labels:
valData = val1(:, 1:(end-1));
%labels corresponding to the VALIDATION data:
valLabels = val1(:, end);
%Change labels into 1-18
valLabels = changem(valLabels, 1:length(uniques), uniques);

%test data without labels:
testingData = test1(:,1:(end-1));
%labels corresponding to the TEST data:
testingLabels = test1(:,end);
%Change labels into 1-18
testingLabels = changem(testingLabels, 1:length(uniques), uniques);

clear training1 test1 val1

% ---CODE TO REMOVE THE NULL CLASS---
%isNULL = trainingLabels==1;
%trainingLabels(isNULL) = [];
%trainingData(isNULL, :) = []; 
%size(trainingData)

trainingData = backfillnans(trainingData); valData = backfillnans(valData); testingData = backfillnans(testingData);

if normalize 
%	[trainingData, colMeans, colStds] = normalizeMatrix(trainingData);

	%Normalize the test data using the same column means and standard deviations as in the training set:
%	testingData = (testingData - repmat(colMeans, size(testingData,1),1)) ./ repmat(colStds, size(testingData,1), 1);
%	testingData(isnan(testingData)) = 0;
	trainingData = trainingData./1000;
	valData = valData./1000;
	testingData = testingData./1000;
end


classes = unique(trainingLabels);

if makeSlidingWindows
	[trainingData, trainingLabels] = rollingWindows(trainingData, trainingLabels, stepSize, windowSize);
	[valData, valLabels] = rollingWindows(valData, valLabels, stepSize, windowSize);
	[testingData, testingLabels] = rollingWindows(testingData, testingLabels, stepSize, windowSize);
	%Permute dimensions to match Nils' data
	trainingData = permute(trainingData, [2 1 3]);
	valData = permute(valData, [2 1 3]);
	testingData = permute(testingData, [2 1 3]);
else
	trainingData=trainingData';
	valData=valData';
	testingData=testingData';
end

filename = 'opp2.mat';
if IMUsOnly
	filename='opp2IMUsOnly.mat';
end

save(filename,'classes', 'trainingData', 'trainingLabels','valData','valLabels' ,'testingData', 'testingLabels')

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
	if IMUsOnly
		unix('th shaneMatFiles2torchIMUsOnly.lua')
	else
		unix('th shaneMatFiles2torch.lua');
	end
end
