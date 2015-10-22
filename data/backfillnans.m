function datai = backfillnans(data)
%Based on code from http://uk.mathworks.com/matlabcentral/answers/50298#answer_61414
% Dimensions
[numRow,numCol] = size(data);
% First, datai is copy of data
datai = data;
% For each column
for c = 1:numCol
    % Find first non-NaN row
    indxFirst = find(~isnan(data(:,c)),1,'first');
    if (indxFirst~=1)
            datai(1,c)=0;
    end
    if( ~isempty(indxFirst) )
        % Find all NaN rows
        indxNaN = find(isnan(data(:,c)));
        % Find NaN rows beyond first non-NaN
        indx = indxNaN(indxNaN > indxFirst);
        % For each of these, copy previous value
        for r = (indx(:))'
            datai(r,c) = datai(r-1,c);
        end
    end
end