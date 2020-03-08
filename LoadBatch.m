function B = LoadBatch(data)
%% Loads a batch of Data and normalize it
% Input:
%       data: data in matrix form
% Output:
%       B: normalized data as Batch
%%

% Normalize the data
    X = (double(data.data))';
    mean_X = mean(X,2);
    std_X = std(X,0,2);
    X = X - repmat(mean_X, [1, size(X,2)]);
    X = X ./ repmat(std_X, [1, size(X,2)]);
    
    y = data.labels;
    K = 10; % number of classes
    Y = zeros(K,length(y));
    for i=1:length(y)
        j = y(i)+1; % matlab starts counting at 1 and not zero. so class 0 corresponds to column 1
        Y(j,i)=1;
    end
    B = {X,Y,y};
end   