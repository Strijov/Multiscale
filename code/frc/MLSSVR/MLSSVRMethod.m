function [test_forecast, train_forecast, model] = MLSSVRMethod(validationX, model, trainX, trainY)
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

N_RANDOM_SEARCH_TRIALS = 200;
if ~isfield( model.params, 'kernel_type')
    model.params.kerneloption = 'rbf';
end
if ~isfield(model.params, 'p1')
    model.params.kernel=2;
end
if ~isfield(model.params, 'p2')
    model.params.kernel=0;
end
if ~isfield(model.params, 'gamma')
    model.params.verbose=0.5;
end
if ~isfield(model.params, 'lambda')
    model.params.verbose=4;
end

model = train_svr(trainX, trainY, model);
test_forecast = feval(model.transform, validationX);
train_forecast = feval(model.transform, trainX);
end

function model = train_svr(trainX, trainY, model)

%test_forecast = zeros(size(validationX, 1), size(trainY, 2));
%train_forecast = zeros(size(trainY));

pars = model.params;
model.params.trained = struct('alpha', zeros(size(trainY)), 'b', []);

[alpha, b] = MLSSVRTrain(trainX, trainY, pars.kernel_type, pars.p1, ...
    pars.p2, pars.gamma, pars.lambda);

model.params.trained.alpha = alpha;
model.params.trained.b = b;

model.transform = @(X) transform(X, trainX, model.params.trained, pars);

end

function frc = transform(X, trainX, model_pars, kernel_pars)

frc = zeros(size(X, 1), size(model_pars.alpha, 2));

frc = MLSSVRPredict(X, trainX, kernel_pars.kernel_type, kernel_pars.p1, kernel_pars.p2, ...
    model_pars.alpha, model_pars.b, kernel_pars.lambda);
end