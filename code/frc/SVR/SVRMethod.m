function [forecasted_y, train_forecast, model] = SVRMethod(validationX, model, trainX, trainY)
% Compute forecast using SVR model with fixed parameters.
%
% Input:
% validationX [mtest x nx] feature row for foreasted point
% model [struct] containing model and its parameters;
%   model.params stores parameters of SVM model, stored in structure with fields:
%   C, lambda, epsilon, kernel, kerneloption, verbose
% trainX, trainY stores training data:
%   trainX [mtrain x nx] stores features
%   trainY [mtrain x ny] stores target variables
%
% Output:
% forecast_y  [mtest x ny] forecasted values of y (regression of x)

forecasted_y = zeros(size(validationX, 1), size(trainY, 2));
train_forecast = zeros(size(trainY));

if model.unopt_flag % For now, no optimization
    model.unopt_flag = false; % AM
    model.params.C = 1e5;  
    model.params.lambda = 0.000001; 
    model.params.epsilon = .1;
    model.params.kerneloption = 1;
    model.params.kernel='gaussian';
    model.params.verbose=0;
else
    model = ExtraOptimization(trainX, trainY, model);
end

pars = model.params;
n_failed = 0; % keep track of badly converged dimensions
for i = 1:size(trainY, 2)
    train_y = trainY(:, i);
    [xsup, ~, w, b, ~, ~, obj] = svmreg(trainX, train_y, pars.C, pars.epsilon, ...
                pars.kernel, pars.kerneloption, pars.lambda, pars.verbose);
    if isempty(xsup) || any(isnan(w)) % AM
        n_failed = n_failed + 1;
    else
        forecasted_y(:, i) = svmval(validationX, xsup, w, b, pars.kernel, ...
                                                         pars.kerneloption);
        train_forecast(:, i) = svmval(trainX, xsup, w, b, pars.kernel, ...
                                                         pars.kerneloption);
    end
end
if n_failed > 0
    disp(['Warning: svmreg converged with empty set of support vectors for ', ...
        num2str(n_failed), ' target dimensions out of ', num2str(size(trainY, 2))]);
end

end