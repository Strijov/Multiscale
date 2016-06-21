function [testForecast, trainForecast, model] = IdentityForecast(testX, model,trainX, ~)
% Returns the features matrix as it is. Is used to test forecasting pipeline.
%
% Input:
% testX [mtest x ny]. testX is expected to be equal to testY
% model [struct] containing model and its parameters;
%   model.params stores deltaTr and deltaTp vectors 
% trainX, trainY stores training data:
%   trainX [mtrain x ny] stores the copy of target variables
%   trainY [mtrain x ny] stores target variables
%
% Output:
% testForecast  [mtest x ny] forecasted values of test y (regression of x)
% trainForecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - updated model structure

testForecast = testX;
trainForecast = trainX;
model.transform = @(X) X;
    
end