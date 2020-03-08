function J = ComputeCostDN(p,Y,lambda,W)
%% Compute Loss 
% Input: 
%       p: prediction of the Network due softmax activation
%       Y: Lables as One-Hot-Vectors
%       lambda: Ridge Regression penalty factor
% Output:
%       J: engery cost / loss
%%

% Ridge Regression part to regulate weights
l_cross = -log(diag(Y' * p));
tmp = 0;
for i=1:length(W)
    tmp = tmp +sum(sum(W{i}.^2));
end

% Energy function to minimize
J = 1/length(l_cross) * sum(l_cross) + lambda * tmp;
end