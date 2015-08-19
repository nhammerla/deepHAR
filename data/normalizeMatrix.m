function output = normalizeMatrix(A)
%{    A=bsxfun(@minus,A,mean(A));
    output = A./repmat(std(A, 0, 1),size(A,1),1);
%}
     output = A;
end
