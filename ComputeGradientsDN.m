function [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsDN(S_oBN,S_BN,v,mu,X_0, Y, P, W, lambda, X_l, gamma, Batchnormalization)
%% Compute Gradients for weight update
% Input: 
%       S_oBN: un-normalised Batch, Mean mu,
%       mu (double): Mean of s
%       v (double): variance of s
%       X_0: Input data of the Network
%       Y: One-Hot Vectors 
%       P: Prediction of forward Pass
%       W (cell): Weights
%       lambda: Ridge Regression penalty factor
%       X_l (cell): Layer Activations
%       gamma: Scaling factor for the Batch Normalisation
%       Batchnormalization (logical): BN on or off  
% Output:
%       grad_W: Gradients regarding the weights W
%       grad_b: Gradients regarding the biasses b
%       grad_gamma: Gradients regarding the gammas
%       grad_betas: Gradients regarding the betas
%%
% Gradienten computation in Matrix-Vector style
G_batch = -(Y -P);
numLayers = length(W);
[~,n_b] = size(X_l{end}); %könnte auch Y size genommen werden

% Placeholders for acceleration
grad_W = cell(1,numLayers);
grad_b = cell(1,numLayers);
grad_gamma = cell(1,numLayers);
grad_beta = cell(1,numLayers);

% layer k gradients
grad_W{numLayers} = 1/n_b * G_batch * X_l{numLayers-1}' + 2*lambda * W{numLayers};
grad_b{numLayers} = 1/n_b * G_batch * ones(n_b,1);
G_batch = W{numLayers}' * G_batch;
G_batch = G_batch .* (X_l{numLayers-1}>0);

% k-1 layers gradients
for i = (numLayers-1):-1:1
    if Batchnormalization == true
        %% Gradients of shifting and scaling parameter of Batch Normalisation
        
        % 1. Compute gradient for the scale and offset parameters for layer l:
        grad_gamma{i} = 1/n_b * ( G_batch .* S_BN{i})*ones(n_b,1);
        grad_beta{i} = 1/n_b * G_batch * ones(n_b,1);
        
        % 2. Propagate the gradients through the scale and shift
        G_batch = G_batch .* (gamma{i}*ones(n_b,1)');  
        
        % 3. Propagate G_batch through the batch normalization
        G_batch = BatchNormBackPass(G_batch, S_oBN{i}, mu{i},v{i});
    else
    end        
    % 4. The gradients of J w.r.t. bias vector b_l and W_l
    if i ~=1
        tmp = X_l{i-1}';
    else
        tmp = X_0';
    end
    grad_W{i} = 1/n_b * G_batch * tmp + 2*lambda * W{i};
    grad_b{i} = 1/n_b * G_batch * ones(n_b,1);
    
    % 5. If l>1 progpagate G_{batch} to previous layer    
    if i>1
        G_batch = W{i}'*G_batch;
        G_batch = G_batch .* (X_l{i-1}>0);
    end
end
end