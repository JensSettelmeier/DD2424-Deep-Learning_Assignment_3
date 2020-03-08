%% Assignment 3 DD2424 Deep Learning in Data Science at KTH 
% Author: Jens Settelmeier
%%

clc
clear all
%% Load Data

fprintf('Load Data\n');

% Set Data Paths
trainPath = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_1.mat');
valPath = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_2.mat');
testPath = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/test_batch.mat');

% Define how much data should be used for validation and test in percentage
validation_ratio = 20;
test_ratio = 10;

% Shuffle the loaded data to make sure it is not initilized sorted 
shuffle = true;

% Load the data
%[X_train,Y_train,y_train,X_val,Y_val,y_val,X_test,Y_test,y_test]=loadData(shuffle,trainPath,valPath,testPath);
[X_train,Y_train,y_train,X_val,Y_val,y_val,X_test,Y_test,y_test]=loadData_fast(shuffle, validation_ratio,test_ratio, trainPath);
%[X_train,Y_train,y_train,X_val,Y_val,y_val,X_test,Y_test,y_test]= loadData_all( shuffle, validation_ratio,test_ratio);
%% Network (Hyper)parameters and Initializations

fprintf('Initialize Network Parameters \n');

numberOfClasses = 10; % number of classes
[d,N] = size(X_train); % dim of each picture 32*32*3

% configure the neurons per layer and number of layers
layers = [d,50,30,20,20,10,10,10,10,numberOfClasses]; % all layers except the input layers are count as layers in a k-layer network

% Penalty factor for Ridge Regression
lambda = 0.01;

% Initialize Weights and Biasesfalse
[W, b] = InitializeParamsDN(layers);

% Initialize Shift and Scale factors for Batch Normalisation
[gamma, beta] = Initialize_BN_ParamsDN2(layers);

% Max and Min for learning rate, adjusted by circle learning method
eta_max = 1e-1;
eta_min = 1e-5;

% Number of samples per Batch (Batch Size)
Batch_size = 100;

% Number of Epochs per circle
n_epochs = 300;

% Stepsize in circle learning
[~,NumberOfSamples] = size(X_train);
mutliple_fac = 8;
n_s = mutliple_fac*floor(NumberOfSamples/Batch_size);

% Random/shuffel Batches
SGD = true;

% Batchnormalisation on/off
Batchnormalization = true;

% (leaky) ReLu as activation
leakyReLuFactor = 0.01; % default: 0

% Train the Mini Batch by Gradient Descent
[epoch_accs, epoch_losses, eta_array, W,b,gamma, beta] = MiniBatchGD_ex4(X_train,Y_train, y_train, X_val,Y_val, y_val,Batch_size, eta_max, eta_min, n_epochs,n_s,W,b,lambda, gamma, beta,SGD,Batchnormalization,leakyReLuFactor);


[max_acc,at_epoch]=max(epoch_accs(2,:))

%% Plots

% learning rate circle
figure
plot(1:length(eta_array),eta_array)
title('Check the learning rate circle');
xlabel('Iteration t');
ylabel('learning rate eta_t');
legend('Learning rate');

% Train-Loss and Val-Loss plot over epochs 
epochs = 1:length(epoch_losses(1,:));
figure
plot(epochs, epoch_losses(1,:), 'b', epochs, epoch_losses(2,:), 'r')
title('Train-Loss and Val-Loss plot over epochs');
xlabel('Epoch');
ylabel('Loss');
legend('Training','Validation');

figure
plot(epochs, epoch_accs(1,:), 'b', epochs, epoch_accs(2,:), 'r')
title('Train-Accurancy and Val-Accurancy plot over epochs');
xlabel('Epoch');
ylabel('Accuracy');
legend('Training','Validation');

%% Functionen

function [epoch_accs,epoch_losses, eta_array, Wstar, bstar, gamma_star, beta_star] = MiniBatchGD_ex4(X_train,Y_train, y_train, X_val,Y_val, y_val,Batch_size, eta_max, eta_min, n_epochs,n_s,W,b,lambda, gamma, beta,SGD,Batchnormalization,leakyReLuFactor)
fprintf('Train the network\n');
% Initialization
Wstar = W;
bstar = b;
gamma_star = gamma;
beta_star = beta;

[~,N] = size(X_train);  % number of total training samples
eta_delta = eta_max - eta_min; % n_ delta from eq (14)

epoch_loss_train =zeros(1,n_epochs);
epoch_loss_val = zeros(1,n_epochs);

epoch_acc_train = zeros(1,n_epochs);
epoch_acc_val = zeros(1,n_epochs);
eta_array = zeros(1,n_epochs * N/Batch_size);

% check if N/n_batch is integer
if N/Batch_size ~= round(N/Batch_size)
    fprintf(' N/n_batch is not an integer!')
end

%% Circle Training
eta_t = eta_min;
eta_index = 1;
t = 0;
for i=1:n_epochs 
    for j=1:N/Batch_size 
        
        %% Batch selection
        if SGD == true
            Batch_indizies = randperm(N,Batch_size);
            X_batch = X_train(:,Batch_indizies);
            Y_batch = Y_train(:,Batch_indizies);
        else
            j_start = (j-1)*Batch_size + 1;
            j_end = j*Batch_size;
            X_batch = X_train(:, j_start:j_end);
            Y_batch = Y_train(:, j_start:j_end);
        end
        
        %% Gradient evalutation for updating
        [S_oBN, S_BN, v,mu,P_batch,X_l] = EvaluateClassifier(X_batch,Wstar, bstar, gamma_star, beta_star,Batchnormalization,leakyReLuFactor);
        [grad_W, grad_b,grad_gamma,grad_beta] = ComputeGradientsDN(S_oBN,S_BN,v,mu,X_batch, Y_batch, P_batch, Wstar, lambda,X_l, gamma_star,Batchnormalization);
        
        %% weight, bias, shift and scale Parameter update: Backward pass
        for layer=1:length(Wstar)
            Wstar{layer} = Wstar{layer} - eta_t * grad_W{layer};
        end
        
        for layer=1:length(bstar)
            bstar{layer} = bstar{layer} - eta_t * grad_b{layer};
        end
        
        if Batchnormalization == true
            for layer=1:length(beta_star)
                beta_star{layer} = beta_star{layer} - eta_t * grad_beta{layer};
            end
            for layer=1:length(gamma_star)
                gamma_star{layer} = gamma_star{layer} - eta_t * grad_gamma{layer};
            end
        else
        end
        
        %% step_size regularisation
        if t<=n_s
            eta_t = eta_min + t/n_s * eta_delta;
            
        elseif t<= 2*n_s
            eta_t = eta_max - (t-n_s)/n_s * eta_delta;
        end
        
        t = mod(t+1,2*n_s);

        eta_array(eta_index) = eta_t;
        eta_index = eta_index+1;        
    end
    
    %% compute cost on whole train and test set
    % Loss on Train set
    [~, ~, ~,~,P_train,~] = EvaluateClassifier(X_train,Wstar, bstar, gamma_star, beta_star,Batchnormalization,leakyReLuFactor);
    epoch_loss_train(i) = ComputeCostDN(P_train,Y_train,lambda,Wstar);
    
    % Loss on Val set
    [~, ~, ~,~,P_val,~] = EvaluateClassifier(X_val,Wstar, bstar, gamma_star, beta_star,Batchnormalization,leakyReLuFactor);
    epoch_loss_val(i) = ComputeCostDN(P_val, Y_val,lambda,Wstar);
    
    %% compute accuracy on whole train and test set
    epoch_acc_train(i) = ComputeAccuracy(y_train,P_train);
    epoch_acc_val(i) = ComputeAccuracy(y_val, P_val);
    
end
fprintf('Testing done..\n');
%% Structure results
epoch_losses = [epoch_loss_train;epoch_loss_val];
epoch_accs = [epoch_acc_train;epoch_acc_val];
end