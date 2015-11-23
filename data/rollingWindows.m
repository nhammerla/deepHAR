function [slidingWindows, labelsForWindows] = rollingWindows(matrix, labelsVector, stepSize, windowLength);
	% matrix function to make 3D rolling windows matrix from a 2D matrix
	
	%disp('Input Matrix Size');size(matrix)
	%disp('labelsVector');size(labelsVector)
	
	%Check dimensions of labels and training matrices:
	labelsVector = squeeze(labelsVector(:));
	if size(matrix,2)==length(labelsVector);
		matrix = matrix';
	end
	if size(matrix,1)~=length(labelsVector)
		size(matrix,1)
		length(labelsVector)
	end
	assert(size(matrix,1)==length(labelsVector), 'The labels vector should be the same size as the 1st dimension of training matrix, but is not');

	%maxPossibleNumOfWindows = length(labelsVector); 
	
	numberOfRows=size(matrix,1);
	maxPossibleNumOfWindows = ceil(1+(numberOfRows/stepSize));
	%pre-allocate 3D array:
	B=zeros(windowLength, size(matrix,2),maxPossibleNumOfWindows);
	numOfWindows = 1;

	%pre-allocate the labels vector:
	labels = squeeze(zeros(maxPossibleNumOfWindows,1));
	
	for i=1:maxPossibleNumOfWindows
		windowBeginsAt = 1 + ((i-1)*stepSize);
		windowEndsAt = windowBeginsAt + windowLength - 1;
		if windowEndsAt<=length(labelsVector)
			window = matrix(windowBeginsAt:windowEndsAt,:);
			B(:,:,i) = window;
			numOfWindows = max(numOfWindows,i);
			labels(i) = labelsVector(mode(windowBeginsAt:windowEndsAt));
		end
	end
	if size(B,3)>numOfWindows
		B = B(:,:,1:numOfWindows);
		labels = labels(1:numOfWindows);
	end

	slidingWindows = B;
	labelsForWindows = labels;
	%reshape(B, [numberOfWindows size(matrix, 2) windowLength])
end
