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


if isempty(mdl.params) 
    params.expVar = EXPLAINED_VAR;
    params.maxComps = MAX_COMPS;
    params.minComps = 1;
    params.plot = false;  
    mdl.params = params;
else
    params = mdl.params;
end



[wcoeff,~,variance,~,var_ratio] = pca(X, 'algorithm', 'svd');
if isfield(params, 'nComps')
    nComps = params.nComps;
    nComps = min([nComps, maxComps, size(wcoeff, 2)]);
else
    nComps = choose_n_comps(var_ratio, params.expVar);
    nComps = max([nComps, params.minComps]);
    nComps = min([nComps, params.maxComps, size(wcoeff, 2)]);
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