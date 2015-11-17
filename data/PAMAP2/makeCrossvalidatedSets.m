clear;

%OPTIONS:
normalize=false;
convertToTorch=true;
makeSlidingWindows=true;
%TODO investigate which step and window sizes to use
stepSize=15;
windowSize=30;

addpath('~/deepHAR/data/')
base = ('~/PAMAP2/PAMAP2_Dataset/Protocol/')

s{1}=load([base 'subject101.dat']);
s{2}=load([base 'subject102.dat']);
s{3}=load([base 'subject103.dat']);
s{4}=load([base 'subject104.dat']);
s{5}=load([base 'subject105.dat']);
s{6}=load([base 'subject106.dat']);
s{7}=load([base 'subject107.dat']);
s{8}=load([base 'subject108.dat']);
s{9}=load([base 'subject109.dat']);

%For each of i=1:9, make s{i} a test set, and the rest to be the traning set 

for i=1:9
	valData = s{i};
	%Extract labels from validation data:
	valLabels=valData(:,2);
	%Only keep useful columns in validation data:
	valData=valData(:,3:end);

	trainingData = [];
	for j=1:9
		if i~=j
			trainingData=[trainingData; s{j}];
		end
	end

	%Extract labels from trainingData:
	trainingLabels=trainingData(:,2);
	%Only keep useful information in trainingData:
	trainingData=trainingData(:,3:end);
	
	if makeSlidingWindows
		[trainingData, trainingLabels] = rollingWindows(trainingData, trainingLabels, stepSize, windowSize);
		[valData, valLabels] = rollingWindows(valData, valLabels, stepSize, windowSize);
		trainingData = permute(trainingData, [2 1 3]);
		valData = permute(valData, [2 1 3]);
	else
		%Transpose stuff to be same as other Torch data:
		trainingData=trainingData';
		valData=valData';
	end
	matFileName = strcat('set',num2str(i));
	save(matFileName, 'trainingData', 'valData', 'trainingLabels', 'valLabels','-v7.3');
	if convertToTorch
		command = strcat('th convert2torch.lua -datafile ',{' '},matFileName,'.mat -saveAs ',{' '},matFileName,'DAT.dat' )
		unix(command{1})
	end
end

clear trainingData valData;
