function [X_train,Y_train,y_train,X_val,Y_val,y_val,X_test,Y_test,y_test]=loadData_fast( shuffle, validation_ratio,test_ratio, Path)
%% Loads Data for the Assignment (a subset)
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

Data = LoadBatch(Path);
Input = Data{1}'; OneHot = Data{2}'; Lable = double(Data{3});
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