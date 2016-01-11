clear;

%OPTIONS:
makeSlidingWindows=true;
windowSize=512;
stepSize=100;

changeLabels=true
ignoreSomeClasses=true
ClassToRemove=0
percentToUse=100
crossValidation=false;
folderToSaveTo = 'run5'
normalize=true
convertToTorch=true;

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

%Select columns
otherSensors=3;
startingIndices = [4,21,38];

accSensors = 2:4;
gyroSensors= 8:10;
magSensors = 11:13;

measurementsForEachSensor=[accSensors, gyroSensors, magSensors];

fun1 = @(startingIndex)(startingIndex + measurementsForEachSensor -1);
columns = cell2mat(arrayfun(fun1, startingIndices, 'UniformOutput', false));
columns=sort( [columns, otherSensors] );

%Subset data columns:
for i=1:9
	subjectsData = s{i};
	subjectsData=subjectsData(:,[1,2,columns]);
	s{i} = subjectsData;
end
%Convert to single precision
for i=1:9
	s{i}=single(s{i});
end

%Only use a certain percentage of data from each subject
if percentToUse~=100
	for i=1:9
		cutoff=ceil(size(s{i},1)*percentToUse/100);
		s{i} = s{i}(1:cutoff,:);
	end
end

%For each of i=1:9, make s{i} a test set, and the rest to be the traning set 

if crossValidation
	mkdir(folderToSaveTo)
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
		matFileName = strcat(folderToSaveTo,'/set',num2str(i));
		save(matFileName, 'trainingData', 'valData', 'trainingLabels', 'valLabels','-v7.3');
		if convertToTorch
			command = strcat('th convert2torch.lua -datafile ',{' '},matFileName,'.mat -saveAs ',{' '},matFileName,'DAT.dat' )
			unix(command{1})
		end
	end


else
	%
	trainingSubjects=[1,2,3,4,7,8,9]
	validationSubjects=5
	testSubjects=6

	trainingData=[];
	for i=trainingSubjects
		trainingData=[trainingData; s{i}];
	end
	%Extract labels from trainingData:
	trainingLabels=trainingData(:,2);
	%Change labels into 1-18
	uniques = unique(sort(trainingLabels));
	if changeLabels
		if ignoreSomeClasses
			ClassToRemove = changem(ClassToRemove, 1:length(uniques), uniques);
			trainingLabels = changem(trainingLabels, 1:length(uniques), uniques);
		end
	end
	%Only keep useful information in trainingData:
	trainingData=trainingData(:,3:end);
	
	valData=[];
	for i=validationSubjects
		valData=[valData; s{i}];
	end

	%Extract labels from validation data:
	valLabels=valData(:,2);
	%Change labels into 1-18
	if changeLabels
		if ignoreSomeClasses
			valLabels = changem(valLabels, 1:length(uniques), uniques);
		end
	end
	%Only keep useful columns in validation data:
	valData=valData(:,3:end);

	testingData=[];
	for i=testSubjects
		testingData=[testingData; s{i}];
	end

	%Extract labels from test data:
	testingLabels=testingData(:,2);
	%Change labels into 1-18
	if changeLabels
		if ignoreSomeClasses
			testingLabels = changem(testingLabels, 1:length(uniques), uniques);
		end
	end
	%Only keep useful information in testingData
	testingData=testingData(:,3:end);

	classes=unique(trainingLabels);
	
	tabulate(trainingLabels)
	tabulate(valLabels)
	tabulate(testingLabels)

	trainingData = backfillnans(trainingData); valData = backfillnans(valData); testingData = backfillnans(testingData);
	 
	if normalize
		[trainingData, colMeans, colStds] = normalizeMatrix(trainingData);
	 
	       %Normalize the test data using the same column means and standard deviations as in the training set:
		valData=(valData - repmat(colMeans, size(valData,1),1)) ./ repmat(colStds, size(valData,1), 1);
		testingData = (testingData - repmat(colMeans, size(testingData,1),1)) ./ repmat(colStds, size(testingData,1), 1);
	testingData(isnan(testingData)) = 0;
end
%Make torch file WITH sliding windows
	if makeSlidingWindows
			[trainingData, trainingLabels] = rollingWindows(trainingData, trainingLabels, stepSize, windowSize);
			[valData, valLabels] = rollingWindows(valData, valLabels, stepSize, windowSize);
			[testingData, testingLabels] = rollingWindows(testingData, testingLabels, stepSize, windowSize);
			trainingData = permute(trainingData, [2 1 3]);
			valData = permute(valData, [2 1 3]);
			testingData = permute(testingData, [2 1 3]);

			%Remove all rows which have label X:
			trainTF1 = trainingLabels==ClassToRemove;
			size(trainingData)
			trainingData(:,:,trainTF1)=[];
			trainingLabels(trainTF1)=[];

			valTF1=valLabels==ClassToRemove;
			valData(:,:,valTF1)=[];
			valLabels(valTF1)=[];

			testTF1=testingLabels==ClassToRemove;
			testingData(:,:,testTF1)=[];
			testingLabels(testTF1,:)=[];
%Make torch file WITHOUT sliding windows
	else
			%Transpose stuff to be same as other Torch data:
			trainingData=trainingData';
			valData=valData';
			testingData=testingData';
			
			%Remove all rows which have label X:
			trainTF1 = trainingLabels==ClassToRemove;
			trainingData(:,trainTF1)=[];
			trainingLabels(trainTF1)=[];

			valTF1=valLabels==ClassToRemove;
			valData(:,valTF1)=[];
			valLabels(valTF1)=[];

			testTF1=testingLabels==ClassToRemove;
			testingData(:,testTF1)=[];
			testingLabels(testTF1)=[];
	end
	matFileName='pamap2';
	save(matFileName, 'trainingData', 'valData', 'trainingLabels', 'valLabels','testingData','testingLabels','classes','-v7.3');
		if convertToTorch
			command = strcat('th convert2torch.lua -datafile ',{' '},matFileName,'.mat -saveAs ',{' '},matFileName,'DAT.dat' )
			unix(command{1})
		end
end
