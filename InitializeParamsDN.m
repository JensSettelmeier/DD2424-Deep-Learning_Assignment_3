function [W,b] = InitializeParamsDN(layers)
%% Initializes the weight matrix and the bias vector
% Input:
%         layers: vector that specifies the number of neurones per layer
% Output:
%       W: He initialized Weights for the Network
%       b: Initialized biases for the Network
%%

k = length(layers);
W = cell(1,k-1);
b = cell(1,k-1);
   
    % He Initialisation
    for i = 2:k
        W{i-1} = sqrt(2)/sqrt(layers(i-1)) * randn(layers(i),layers(i-1));
        b{i-1} = zeros(layers(i),1);   
    end
end
