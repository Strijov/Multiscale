function [test_forecast, train_forecast, model] = VarForecast(validationX, model,trainX, trainY)
% Compute forecast using VAR model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for the forecasted period
% model [struct] containing model and its parameters;
%   model.params stores AR parameters matrix W [nx x ny] 
% trainX, trainY stores training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
%
% Output:
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - ipdated model structure

if model.unopt_flag
    regCoeff = looRegValue(trainX, trainY);
    model.params.regCoeff = regCoeff;
    model.unopt_flag = false;
end

regCoeff = model.params.regCoeff;
W = (trainX'*trainX + regCoeff*eye(size(trainX, 2)))\trainX'*trainY;
model.transform = @(x) x*W;
test_forecast = validationX*W;
train_forecast = trainX*W;


end

function regCoeff = looRegValue(X, Y)
MAX_LOO = 100;
N_TRIALS = 20;

nSamples = size(X, 1);
lstRegCoeff = [linspace(0.5, 1, N_TRIALS/2), linspace(2, 10, N_TRIALS/2)];
nLooSamples = min(nSamples, MAX_LOO);
idxLoo = randperm(nSamples, nLooSamples);
errors = zeros(nSamples, N_TRIALS);
for i = idxLoo
    trainX = X(~ismember(1:nSamples, i), :);
    trainY = Y(~ismember(1:nSamples, i), :);
    covXX = trainX'*trainX;
    covXY = trainX'*trainY;
    errors(i, :) = regErrors(covXX, covXY, X(i, :), Y(i, :), lstRegCoeff);   
end

[~, idxOpt] = min(std(errors));
regCoeff = lstRegCoeff(idxOpt);

fig = figure;
errorbar(lstRegCoeff, mean(errors), lstRegCoeff, std(errors));
xlabel('l2-regularization coefficient', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Squared loo error', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;
saveas(fig, 'var_reg_coeffs', 'epsc');
close(fig)

end

function err = regErrors(covXX, covXY, X, Y, lstRegCoeff)

err = zeros(1, numel(lstRegCoeff));
for i = 1:numel(lstRegCoeff)
    W = (covXX + lstRegCoeff(i)*eye(size(covXX)))\covXY;   
    err(i) = norm(Y - X*W);
end

end