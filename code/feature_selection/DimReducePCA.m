function [newX, mdl] = DimReducePCA(X, mdl)
% Use PCA to reduce dimensionality of feature description
% Inputs:
% X [m x n] - feature matrix
% mdl - model selection structure with fields handle, name, pars and
% transform and res. Model parameters are the min and max number of components (the
% number of components is equal to the dimension of the resulting feature matrix),  
% and percentage of the explained variance (ranges from 1 to 100)
% Outputs:
% newX [m x nComps] - transformed feature matrix
% mdl - model structure with updated fields transform, pars and res

EXPLAINED_VAR = 99;
MAX_COMPS = 20;


if ~isfield(mdl.params, 'expVar') 
    mdl.params.expVar = EXPLAINED_VAR;
end
if ~isfield(mdl.params, 'maxComps')
    mdl.params.maxComps = MAX_COMPS;
end
if ~isfield(mdl.params, 'minComps')    
    mdl.params.minComps = 1;
end
if ~isfield(mdl.params, 'plot')
    mdl.params.plot = false;  
end



[wcoeff,~,variance,~,var_ratio] = pca(X, 'algorithm', 'svd');
if isfield(mdl.params, 'nComps')
    nComps = mdl.params.nComps;
    nComps = min([nComps, mdl.params.maxComps, size(wcoeff, 2)]);
else
    nComps = choose_n_comps(var_ratio, mdl.params.expVar);
    nComps = max([nComps, mdl.params.minComps]);
    nComps = min([nComps, mdl.params.maxComps, size(wcoeff, 2)]);
    mdl.params.nComps = nComps;
end

newX = X*wcoeff(:, 1:nComps);
mdl.transform = @(x) x*wcoeff(:, 1:nComps);
mdl.res = {};
mdl.res.var_ratio = var_ratio;
mdl.res.variane = variance;

end

function n_comps = choose_n_comps(var_ratio, intercept)

n_comps = cumsum(var_ratio) > intercept;
n_comps = n_comps(1);


end