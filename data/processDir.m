function processDir(filetype, labelsLoc, indicesOfSubjectNumInFileName, indicesOfRunNumInFileName, normalize, sampleBySample, subsetCols, slidWin, stepSize, windowSize,trainPart, valPart, testPart, ignoreClasses,changeLabels,doKNN,convert2torch)

	%add genpath of dependency functions to path
	addpath('~/deepHAR/data/'); 
	% Load data in folder
	files = dir(filetype);

	%For each file in dir, extract subject num and run num as indices of cell array. put the contents of each file as the contents of the cell array.
	for i=1:length(files)
		filename=getfield(files(i), 'name');
		subjectNum = num2str( filename(indicesOfSubjectNumInFileName) );
		runNum = num2str( filename(indicesOfRunNumInFileName) );
		cmd1 = ['s{' subjectNum ',' runNum '} = load(''' files(i).name ''', ''ascii'');'];
		eval(cmd1);
	end
	clear subjectNum runNum;

	training1=[];
	val1=[];
	test1=[];
	%Make train-val-test partitions
	for rowNum=1:size(trainPart,1)
		subjectNum = trainPart(rowNum, 1);
		runNum = trainPart(rowNum, 2);
		toAppend = s{subjectNum, runNum};
		if sum(size(toAppend))~=0
			toAppend = toAppend(:,[labelsLoc subsetCols]);
			training1 = [training1; toAppend];
		end
	end
	clear subjectNum runNum;
	for rowNum=1:size(valPart,1)
		subjectNum=valPart(rowNum, 1);
		runNum=valPart(rowNum, 2);
		toAppend = s{subjectNum, runNum};
		if sum(size(toAppend)) ~=0
			toAppend = toAppend(:, [labelsLoc subsetCols]);
			val1 = [val1; toAppend];
		end
	end
	clear subjectNum runNum;
	for rowNum=1:size(testPart, 1)
		subjectNum = testPart(rowNum,1);
		runNum=testPart(rowNum, 2);
		toAppend = s{subjectNum, runNum};
		if sum(size(toAppend)) ~=0
			toAppend = toAppend(:, [labelsLoc subsetCols]);
			test1 = [test1; toAppend];
		end
	end
	%Make labels vectors
	trainingLabels=training1(:,1);
	valLabels=val1(:,1);
	testingLabels=test1(:,1);
	%Extract columns of X data
	trainingData = training1(:,2:end);
	valData = val1(:, 2:end);
	testingData=test1(:, 2:end);
	classes = unique(trainingLabels);

	if changeLabels
		%Change labels to 1..18?
		uniques = unique(trainingLabels);
		trainingLabels=changem(trainingLabels, 1:length(uniques), uniques);
		valLabels=changem(valLabels, 1:length(uniques), uniques);
		testingLabels=changem(testingLabels, 1:length(uniques), uniques);
	end

	% clear unused
	clear training1 val1 test1;
	%TODO Optional: remove null class, or other vector of class titles
	ClassToRemove = ignoreClasses(1);	
	trainTF1 = trainingLabels==ClassToRemove;
	trainingData(:,:,trainTF1)=[];
	trainingLabels(trainTF1)=[];

	valTF1=valLabels==ClassToRemove;
	valData(:,:,valTF1)=[];
	valLabels(valTF1)=[];

	testTF1=testingLabels==ClassToRemove;
	testingData(:,:,testTF1)=[];
	testingLabels(testTF1,:)=[];

	clear trainTF1 valTF1 testTF1;

	%backfill nans
	trainingData=backfillnans(trainingData);
	valData=backfillnans(valData);
	testingData=backfillnans(testingData);
	%normalize
	if normalize
		[trainingData, colMeans, colStds] = normalizeMatrix(trainingData);
		valData = (valData - repmat(colMeans, size(valData,1),1))./repmat(colStds, size(valData,1),1);
		testingData = (testingData - repmat(colMeans, size(testingData,1),1)) ./ repmat(colStds, size(testingData,1), 1);
		valData(isnan(valData)) = 0;
		testingData(isnan(testingData)) = 0;
	end

	%What to do with sample-by-sample
	function saveSBS()
			%permute dimensions
			trainingData=trainingData';
			valData=valData';
			testingData=testingData';
			%save in matlab format
			filename=['sampleBySample' datestr(now,'ddd')];
			save(filename,'classes', 'trainingData', 'trainingLabels','valData','valLabels' ,'testingData', 'testingLabels')
			%launch script to convert those files into torch format
			command = strcat('th convert2torch.lua -datafile ',{' '},filename,'.mat -saveAs ',{' '},filename,'DAT.dat &' );
			unix(command{1});
	end
	%Sanity check...
	%Tabulate classes in each partition:
	disp('Training set')
	tabulate(trainingLabels)
	disp('standard deviations of columns:')
	std(trainingData)

	disp('Validation set')
	tabulate(valLabels)
	disp('standard deviations of columns:')
	std(valData)

	disp('Test set')
	tabulate(testingLabels)
	disp('standard deviations of columns:')
	std(testingData)
	
	disp('1-NN classifier confusion matrix:')
	if doKNN
		mdl = fitcknn(trainingData, trainingLabels,'NumNeighbors',1);
		predictions = predict(mdl, testingData);
		[C,order] = confusionmat(testingLabels,predictions)
	end

	if sampleBySample
		%Put this into a nested function
		saveSBS;
	end
	if slidWin
		%sliding windows
		[trainingData, trainingLabels] = rollingWindows(trainingData, trainingLabels, stepSize, windowSize);
		[valData, valLabels] = rollingWindows(valData, valLabels, stepSize, windowSize);
		[testingData, testingLabels] = rollingWindows(testingData, testingLabels, stepSize, windowSize);
		%Permute dimensions to match Nils' data
		trainingData = permute(trainingData, [2 1 3]);
		valData = permute(valData, [2 1 3]);
		testingData = permute(testingData, [2 1 3]);
		%save sliding window version in matlab format
		filename=['slidingWindows' datestr(now,'ddd')];
		save(filename,'classes', 'trainingData', 'trainingLabels','valData','valLabels' ,'testingData', 'testingLabels')
		%launch script to convert this into torch format
		command = strcat('th convert2torch.lua -datafile ',{' '},filename,'.mat -saveAs ',{' '},filename,'DAT.dat' );
		unix(command{1});
	end
end
