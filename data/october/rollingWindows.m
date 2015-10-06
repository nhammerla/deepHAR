%A function to make rolling windows from a matrix
function [output,labelsOutput] = rollingWindows(matrix, labelsVector, stepSize, windowSize);
    labelsVector = transpose(labelsVector(:));
    assert(size(matrix,1)==length(labelsVector), 'The labels vector should be the same size as the first dimension of input matrix, but is not');
    transposedMatrix = transpose(matrix);
    unrolledMatrix = transpose( transposedMatrix(:) );

    labelsVector = repelem(labelsVector, size(matrix,2));

   %Multiply stepSize and windowSize by row size:
    stepSize = stepSize * size(matrix,2);
    windowSize = windowSize * size(matrix,2);


    %Creating first row of the output matrix:
    temp=unrolledMatrix(1:windowSize);
    labelsTemp = labelsVector(windowSize);

     %Setting up next rows:
    startpoint=stepSize+1;
    endpoint=startpoint+windowSize-1;
    while endpoint<=size(unrolledMatrix,2)
        rowToAdd = unrolledMatrix(startpoint:endpoint);
        temp = [temp;rowToAdd];
        labelsTemp = [labelsTemp mode(labelsVector(startpoint:endpoint))];
        %after appending this new row, set new startpoints and endpoints:
        startpoint = startpoint + stepSize;
        endpoint = startpoint+windowSize-1;
	if mod(endpoint, 1000)==0
		sprintf('%d / %d', endpoint, size(unrolledMatrix, 2))
    end
    output =temp;
    labelsOutput = transpose(labelsTemp);
end
