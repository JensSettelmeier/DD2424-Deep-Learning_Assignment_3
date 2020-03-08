function data= shuffling(data)
%% Shuffels a set of Data
% Input:
%         data: data in matrix form
% Output:
%         data: data in matrix form shuffled
%%
[n,~] = size(data);
data = data(randperm(n),:);
end