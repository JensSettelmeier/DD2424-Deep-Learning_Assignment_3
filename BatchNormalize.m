function S = BatchNormalize(s, mu, v)
%% Normalises a Batch
% Input: 
%       s (double-Mat): un-normalised Batch, Mean mu,
%       mu (double): Mean of s
%       v (double): variance of s
% Output:
%       S (double-Mat): normalized Batch
%%

% Normalisation of the Batch
S = (diag(v + eps)^(-1/2) * (s - mu));
end