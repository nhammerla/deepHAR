function output = repelem(inputVector, w)
	if size(inputVector,2)==1
		inputVector = inputVector';
	end
	v = repmat(inputVector,[w 1]);
	v = v(:)';
	output = v;
