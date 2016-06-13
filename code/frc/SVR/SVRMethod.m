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

N_RANDOM_SEARCH_TRIALS = 200;

if model.unopt_flag % For now, no optimization
    model.unopt_flag = false;
    %model.params.C = 1e5;  
    %model.params.lambda = 0.000001; 
    %model.params.epsilon = .1;
    model.params.kerneloption = 1;
    model.params.kernel='gaussian';
    model.params.verbose=0;
    % Optimize [C, lambda, epsilon]:
    model = svr_random_search(trainX, trainY, model, N_RANDOM_SEARCH_TRIALS);
    
end

model = train_svr(trainX, trainY, model);
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
        num2str(n_failed), ' target dimensions out of ', num2str(numel(model_pars))]);
end

end


function model = svr_random_search(X, Y, model, nTrials)

cRange = [1, 6]; % min and max values (power of 10)
lamRange = [-6, -2]; % min and max values (power of 10)
epsRange = [0.25, 1];  % min and max values

cLst = rand(1, nTrials)*cRange(2) + cRange(1);
cLst = 10.^cLst;
lamLst = rand(1, nTrials)*lamRange(2) + lamRange(1);
lamLst = 10.^lamLst;
epsLst = rand(1, nTrials)*epsRange(2) + epsRange(1);
%idxSplits = randomSplit(size(X, 1), size(X, 1) - nTrials + 1);

error = zeros(1, nTrials);
for i = 1:nTrials
    [Train, Test] = crossvalind('HoldOut', size(X, 1), 0.25); 
    model.params.C = cLst(i);
    model.params.lambda = lamLst(i);
    model.params.epsilon = epsLst(i);
    model = train_svr(X(Train, :), Y(Train, :), model);
    res = feval(model.transform, X(Test, :)) - Y(Test, :);
    error(i) = mean(sqrt(mean(res.*res, 2)));
end
[~, optIdx] = min(error);
model.params.C = cLst(optIdx);  
model.params.lambda = lamLst(optIdx); 
model.params.epsilon = epsLst(optIdx);  

plotRSresuls([log10(cLst'), log10(lamLst'), epsLst'], error', ...
                                {'$C$', '$\lambda$', '$\varepsilon$'});

end

function plotRSresuls(X, Z, names)

% X = C, lambda, epsilon; z = err
rgb = [ ...
    94    79   162
    50   136   189
   102   194   165
   171   221   164
   230   245   152
   255   255   191
   254   224   139
   253   174    97
   244   109    67
   213    62    79
   158     1    66  ] / 255;

b = repmat(linspace(0,1,200),20,1);
%imshow(b,[],'InitialMagnification','fit')


figure;
colormap(rgb);
scatter3(X(:, 1),X(:, 2), X(:, 3), 400, Z-min(Z), '.');
xlabel('$C$');
ylabel('$\lambda$');
zlabel('$\varepsilon$');
colorbar;

idxPairs = combntns(1:size(X, 2), 2);
for idx = idxPairs'
    plotInterpolated(X(:, idx(1)), X(:, idx(2)), Z, names(idx));
end

end


function plotInterpolated(X, Y, Z, names)
F = scatteredInterpolant([X, Y], Z);

figure;
[Xq,Yq] = meshgrid(linspace(min(X), max(X), 100), linspace(min(Y), max(Y), 100));
Zq = F(Xq,Yq);
surf(Xq,Yq,Zq);
xlabel(names{1},'fontweight','b'), ylabel(names{2},'fontweight','b');
zlabel('MSRE','fontweight','b');


end