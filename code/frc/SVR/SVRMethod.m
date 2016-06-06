function [test_forecast, train_forecast, model] = SVRMethod(validationX, model, trainX, trainY)
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
% test_forecast  [mtest x ny] forecasted values of test y (regression of x)
% train_forecast  [mtrain x ny] forecasted values of train y (regression of x)
% model - ipdated model structure


if model.unopt_flag % For now, no optimization
    model.unopt_flag = false; % AM
    model.params.C = 1e5;  
    model.params.lambda = 0.000001; 
    model.params.epsilon = .1;
    model.params.kerneloption = 1;
    model.params.kernel='gaussian';
    model.params.verbose=0;
    model = train_svr(trainX, trainY, model);
else
    model = ExtraOptimization(trainX, trainY, model);
end

test_forecast = feval(model.transform, validationX);
train_forecast = feval(model.transform, trainX);


end


function model = train_svr(trainX, trainY, model)

%test_forecast = zeros(size(validationX, 1), size(trainY, 2));
%train_forecast = zeros(size(trainY));

pars = model.params;
model.params.trained = struct('w', cell(1, size(trainY, 2)), 'b', [], 'xsup', []);
for i = 1:size(trainY, 2)
    train_y = trainY(:, i);
    [xsup, ~, w, b] = svmreg(trainX, train_y, pars.C, pars.epsilon, ...
                pars.kernel, pars.kerneloption, pars.lambda, pars.verbose);
    
    model.params.trained(i).w = w; 
    model.params.trained(i).b = b; 
    model.params.trained(i).xsup = xsup;
        %test_forecast(:, i) = svmval(validationX, xsup, w, b, pars.kernel, ...
        %                                                 pars.kerneloption);
        %train_forecast(:, i) = svmval(trainX, xsup, w, b, pars.kernel, ...
        %                                                 pars.kerneloption);
        
end
model.transform = @(X) transform(X, model.params.trained, pars);


end

function frc = transform(X, model_pars, kernel_pars)

frc = zeros(size(X, 1), numel(model_pars));
n_failed = 0; % keep track of badly converged dimensions

for j = 1:numel(model_pars)
    if isempty(model_pars(j).xsup) || any(isnan(model_pars(j).w))
        n_failed = n_failed + 1;
    else
    frc(:, j) = svmval(X, model_pars(j).xsup, model_pars(j).w, model_pars(j).b, ...
                                                 kernel_pars.kernel,...
                                                 kernel_pars.kerneloption);
    end
end

if n_failed > 0
    disp(['Warning: svmreg converged with empty set of support vectors for ', ...
        num2str(n_failed), ' target dimensions out of ', num2str(size(numel(model_pars), 2))]);
end

end