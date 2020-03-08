function [gamma,beta] = Initialize_BN_ParamsDN2(layers)
%% Initializes the scale and shift parameter for the Batch Normalisation (BN)
% Input:
%         layers: vector that specifies the number of neurones per layer
% Output:
%       gamma: Initialized gammas for BN scaling
%       beta: Initialized betas for BN shifting
%%

k = length(layers)-1;
gamma = cell(1,k-1);
beta = cell(1,k-1);

% Initialize parameters
for i = 1:k-1    
    gamma{i} = ones(layers(i+1),1);
    beta{i} = zeros(layers(i+1),1);    
end

end
