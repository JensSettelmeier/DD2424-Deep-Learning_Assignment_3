function [G_batch] = BatchNormBackPass(G_batch, S_oBN, mu,v)
%% Propagate Gradient Batch trhough Batch Normalisation
% Input: 
%       S_oBN (double): Layer-outputs w/o BN and Activation
%       mu (double): Mean of S_oBN
%       v (double): variance of S_oBN
%       G_batch: Gradient batch 
% Output:
%       G_batch: Gradient batch
%%

[~,n_b] = size(G_batch);    
sigma_1 = (v+eps).^(-0.5);
sigma_2 = (v+eps).^(-1.5);

G_1 = G_batch .*(sigma_1*ones(n_b,1)');
G_2 = G_batch .*(sigma_2*ones(n_b,1)');

D = S_oBN - mu * ones(n_b,1)';
c = (G_2 .*D)*ones(n_b,1);

G_batch = G_1 - 1/n_b * (G_1 * ones(n_b,1))*ones(n_b,1)' - 1/n_b*D .*(c*ones(n_b,1)');
end