function [X_train,Y_train,y_train,X_val,Y_val,y_val,X_test,Y_test,y_test]=loadData_all( shuffle, validation_ratio,test_ratio)
%% Loads Data for the Assignment (all)
% Input:
%       Path: data 
%       validation_ratio: Percentage of the data that is used as
%                         Validation set
%       test_ratio: Percentage of the data that is used as Test set
%       shuffle (logical): on/off shuffling the data
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
Path1 = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_1.mat');
Data1 = LoadBatch(Path1);
X_train1 = Data1{1}; Y_train1 = Data1{2}; y_train1 = double(Data1{3});

Path2 = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_2.mat');
Data2 = LoadBatch(Path2);
X_train2 = Data2{1}; Y_train2 = Data2{2}; y_train2 = double(Data2{3});

Path3 = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_3.mat');
Data3 = LoadBatch(Path3);
X_train3 = Data3{1}; Y_train3 = Data3{2}; y_train3 = double(Data3{3});

Path4 = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_4.mat');
Data4 = LoadBatch(Path4);
X_train4 = Data4{1}; Y_train4 = Data4{2}; y_train4 = double(Data4{3});

Path5 = load('/media/sneey/Linux_Vol/Assignment_1/Assignment_2/DirName/Datasets/cifar-10-batches-mat/data_batch_5.mat');
Data5 = LoadBatch(Path5);
X_train5 = Data5{1}; Y_train5 = Data5{2}; y_train5 = double(Data5{3});

Input = [X_train1,X_train2,X_train3]';%,X_train4,X_train5]';
OneHot =[Y_train1,Y_train2,Y_train3]';%,Y_train4,Y_train5]';
Lable = [y_train1;y_train2;y_train3];%;y_train4;y_train5];

[~,dim_col_Input] = size(Input);
[~,dim_col_OneHot] = size(OneHot);

Data_Lable_OneHot = [Input,OneHot,Lable];

if shuffle == true
    fprintf('shuffle data...\n');
    Data_Lable_OneHot = shuffling(Data_Lable_OneHot);
else
end

[train_data, val_data, test_data] = data_divider2(Data_Lable_OneHot, validation_ratio,test_ratio);

X_train = train_data(:,1:dim_col_Input)';
Y_train = train_data(:,dim_col_Input+1:dim_col_Input + dim_col_OneHot)';
y_train = train_data(:,end);

X_val = val_data(:,1:dim_col_Input)';
Y_val = val_data(:,dim_col_Input+1:dim_col_Input + dim_col_OneHot)';
y_val = val_data(:,end);

X_test = test_data(:,1:dim_col_Input)';
Y_test = test_data(:,dim_col_Input+1:dim_col_Input + dim_col_OneHot)';
y_test = test_data(:,end);

end