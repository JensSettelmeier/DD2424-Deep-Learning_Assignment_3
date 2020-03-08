function [S_oBN,S_BN, v,mu,p,X_l] = EvaluateClassifier(X,W,b,gamma,beta,Batchnormalization, leakyReLuFactor)
%% Forward the Data through the K-layer Network
% Input: 
%       X: Input data
%       W (cell): Weights of the layers
%       b (cell): Biases of the layers
%       gamma (cell): Scaling for BN
%       beta (cell): Shifting for BN
%       Batchnormalisation (logical): on/off BN
%       leakyReLuFactor: factor for the (leaky) ReLu activation
% Output:
%       S_oBN (cell): Layer-outputs w/o BN and Activation
%       S_BN (cell): Layer-outputs w/ BN and w/o Activation
%       X_l (cell): Layer-output w/ BN and Activation
%       mu (double): Mean of S_oBN
%       v (double): variance of S_oBN
%       p: prediction of the Network due softmax activation
%%

[~,N] = size(X); % N = n_b
numLayers = length(W);

% fixing a 0 index case...
X_l_dummy = cell(1,numLayers+1);
X_l_dummy{1} = X;

% Placeholders for acceleration
X_l = cell(1,numLayers-1);
S_BN = cell(1,numLayers-1);
S_oBN = cell(1,numLayers-1);
mu = cell(1,numLayers-1);
v = cell(1,numLayers-1);

%% Forward pass
% first k-1 layers
for i=1:numLayers-1
    b_tmp = repmat(b{i},1,N); 
    S = W{i}*X_l_dummy{i} + b_tmp;
    S_oBN{i} = S;
    
    if Batchnormalization==true
        %% Batchnormalization part
        
        % Mean for Normalisation
        mu_tmp = 1/N * sum(S,2);
        mu{i} = mu_tmp;
        
        % Variance for Normalisation
        v_tmp = 1/N * sum((S-mu_tmp).^2,2);
        v{i} = v_tmp;
        
        % Normalisation of the Batch
        S_hat = BatchNormalize(S,mu_tmp,v_tmp);
        S_BN{i} = S_hat;
        
        % Typical shifting and scaling during Batchnormalisierung
        [~,col_tmp] = size(S_hat);
        gamma_tmp = repmat(gamma{i},1,col_tmp);
        beta_tmp = repmat(beta{i},1,col_tmp);
        S_tilde = gamma_tmp.*S_hat + beta_tmp;

    else
        S_tilde = S;
    end
    
    % Activation with leaky ReLu and leaky factor 0.01.
    X_l{i} = max(leakyReLuFactor*S_tilde, S_tilde);
    X_l_dummy{i+1} = X_l{i};
end
% Last layer k
Sk = W{numLayers} * X_l{numLayers-1} + repmat(b{numLayers},1,N); % X_l{numLayers-1} = X_l_dummy{numLayers}

% Softmax
p = exp(Sk)./ sum(exp(Sk));  
end