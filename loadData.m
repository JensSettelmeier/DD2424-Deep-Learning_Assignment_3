function [X_train,Y_train,y_train,X_val,Y_val,y_val,X_test,Y_test,y_test]=loadData(shuffel, trainPath,valPath,testPath)
%% Loads Data for the Assignment (all)
% Input:
%       trainPath: data for training
%       valPath: data for validation
%       testPath: data for testing
%       shuffle (logical): unused
% Output:
%       X_train: Train input
%       Y_train: OneHot Lable vector of train data
%       y_train: class-lable of train data
%
%       X_val: Val input
%       Y_val: OneHot Lable vector of val data
%       y_val: class-lable of val data
%
%       X_test: Test input
%       Y_test: OneHot Lable vector of test data
%       y_test: class-lable of test data
%%
B = LoadBatch(trainPath); C = LoadBatch(valPath); D = LoadBatch(testPath);

% X_train/val/test is input, Y_train/val/test is lable and y_train/val/test
% is the corresponding one-hot-vector
X_train = B{1}; Y_train = B{2}; y_train = B{3};
X_val = C{1}; Y_val = C{2}; y_val = C{3};
X_test = D{1}; Y_test = D{2}; y_test = D{3};

end