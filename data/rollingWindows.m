function [slidingWindows, labelsForWindows] = rollingWindows(matrix, labelsVector, stepSize, windowLength);
	% matrix function to make 3D rolling windows matrix from a 2D matrix
	
%	matrix = matrix(1:cutoff, :);
%	labelsVector= labelsVector(1:cutoff);
	disp('Input Matrix Size');size(matrix)
	disp('labelsVector');size(labelsVector)
	
	tic
	%Check dimensions of labels and training matrices:
	labelsVector = squeeze(labelsVector(:));
	if size(matrix,2)==length(labelsVector);
		matrix = matrix';
	end
	assert(size(matrix,1)==length(labelsVector), 'The labels vector should be the same size as the 1st dimension of training matrix, but is not');

	maxPossibleNumOfWindows = length(labelsVector); 

	%pre-allocate 3D array:
	B=zeros(windowLength, size(matrix,2),maxPossibleNumOfWindows);
	%NOTE: THIS IS DIFFERENT FORMmatrixT TO NILS' DATA.
	%30x113x11321
	%parfor rowNumber=1:(numberOfWindows)
	%    B(:,:,rowNumber) = matrix(rowNumber:(rowNumber+windowLength-1),:);
	%    %B(rowNumber,:,:) = matrix(rowNumber:(rowNumber+windowLength),:)
	%end
	
	%workings begin...
	%B(:,:,1) = matrix(1:(30),:);1+ (i-1)*windowSize
	%B(:,:,2) = matrix(15:(45),:);1+ (i-1)*windowSize - (i-1)*overlap
	%B(:,:,3) = matrix(31:(60),:);1+ (i-1)*windowSize - (i-1)*overlap
	%B(:,:,4) = matrix(46:(75),:);
	%workings end...

	numOfWindows = 1;
	
	disp(who)

	%pre-allocate the labels vector:
	labels = squeeze(zeros(maxPossibleNumOfWindows,1));
	
	for i=1:maxPossibleNumOfWindows
		windowBeginsAt = 1 + ( (i-1)*windowLength) - ( (i-1)*stepSize);
		windowEndsAt = windowBeginsAt + windowLength - 1;
		if windowEndsAt<=length(labelsVector)
			window = matrix(windowBeginsAt:windowEndsAt,:);
			B(:,:,i) = window;
			numOfWindows = max(numOfWindows,i);
			labels(i) = labelsVector(mode(windowBeginsAt:windowEndsAt));
		end
	end
	numOfWindows
	if size(B,3)>numOfWindows
		B = B(:,:,1:numOfWindows);
		labels = labels(1:numOfWindows);
	end

	toc

	slidingWindows = B;
	labelsForWindows = labels;
	%reshape(B, [numberOfWindows size(matrix, 2) windowLength])
end
