%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 
abs_error = abs(yhat_test-ytest); % calculate absolute error
mae = sum(abs_error)/size(ytest, 1); % use absolute error to calculate MAE
i = 5;
cum_sum = sum(abs_error < i > 0)/size(ytest, 1); % use absolute error to calculate cumulative error

fprintf('MAE Linear regression = %f\n', mae);
fprintf('Cumulative error of level 5 = %f\n', cum_sum);

%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
for i = 1:15
    cum_sum(i) = sum(abs_error < i > 0)/size(ytest, 1);
end

plot (cum_sum, 'r-*')
xlabel('Error Level')
ylabel('Cumulative Score (CS)')

%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.
% partial least square regression
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(xtrain, ytrain, 10); % training

yhat_test_plsr = [ones(size(xtest,1),1) xtest]*beta; % testing

abs_error_plsr = abs(yhat_test_plsr-ytest); % calculate absolute error
mae_plsr = sum(abs_error_plsr)/size(ytest, 1); % use absolute error to calculate MAE
i=5;
cum_sum_plsr = sum(abs_error_plsr < i > 0)/size(ytest, 1); % use absolute error to calculate cumulative error

fprintf('MAE Partial Least Square Regression = %f\n', mae_plsr);
fprintf('Cumulative error of level 5 = %f\n', cum_sum_plsr);

% regression tree model
tree = fitrtree(xtrain, ytrain); % training
yhat_test_rtree = predict(tree, xtest); % testing

abs_error_rtree = abs(yhat_test_rtree-ytest); % calculate absolute error
mae_rtree = sum(abs_error_rtree)/size(ytest, 1); % use absolute error to calculate MAE
i=5;
cum_sum_rtree = sum(abs_error_rtree < i > 0)/size(ytest, 1); % use absolute error to calculate cumulative error
 
fprintf('MAE Regression Tree = %f\n', mae_rtree);
fprintf('Cumulative error of level 5 = %f\n', cum_sum_rtree);


%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox
addpath(genpath('./softwares'));

st = svmtrain(ytrain, xtrain, '-s 3 -t 0'); % training
yhat_test_svm = svmpredict(ytest, xtest, st); % testing

abs_error_svr = abs(yhat_test_svr-ytest); % calculate absolute error
mae_svr = sum(abs_error_svr)/size(ytest, 1); % use absolute error to calculate MAE
i=5;
cum_sum_svr = sum(abs_error_svr < i > 0)/ size(ytest, 1); % use absolute error to calculate cumulative error

fprintf('MAE SVR = %f\n', mae_svr);
fprintf('Cumulative error of level 5 = %f\n', cum_sum_svr);
