function acc = ComputeAccuracy(y,p)
%% Compute Accuracy of Network Prediction
% Input: 
%       p: prediction of the Network due softmax activation
%       y: Lables (classes)
% Output:
%       acc: Prediction Accuracy
%%

[~,k] = size(p);
[~,I] = max(p);
correct_samples = find(I' == (y+1));

acc = length(correct_samples)/k;
end