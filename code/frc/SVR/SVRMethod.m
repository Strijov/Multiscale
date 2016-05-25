function forecasted_y = SVRMethod(validation_x, model, trainX, trainY)
% Compute forecast using SVR model with fixed parameters.
%
% Input:
% x [1 x deltaTp] feature string for last period
% model [struct] containing model and its parameters;
%   model.params stores parameters of SVM model, stored in structure with fields:
%   C, lambda, epsilon, kernel, kerneloption, verbose
% trainX, trainY stores training data:
%   trainX [m x deltaTp] stores features
%   trainY [m x deltaTr] stores target variables
%
% Output:
% forecast_y  [1 x deltaTr] forecasted values of y (regression of x)

forecasted_y = zeros(1, size(trainY, 2));

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
        forecasted_y(:, i) = svmval(validation_x, xsup, w, b, pars.kernel, ...
                                                         pars.kerneloption);
    end
end
if n_failed > 0
    disp(['Warning: svmreg converged with empty set of support vectors for ', ...
        num2str(n_failed), ' target dimensions out of ', num2str(size(trainY, 2))]);
end

end