function [normalizedTrainingMatrix, colMeans, colStds] = normalizeMatrix(A)
%{    A=bsxfun(@minus,A,mean(A));
    output = A./repmat(std(A, 0, 1),size(A,1),1);
%}
	colMeans = mean(A);
	colStds = std(A,1,1);
     	temp  = (A - repmat(colMeans, size(A,1),1) ) ./ repmat( colStds, size(A,1), 1);
	temp(isnan(temp)) = 0;
	normalizedTrainingMatrix = temp;
end
