function [newX, mdl] = DimReducePCA(X, mdl)

EXPLAINED_VAR = 99;
MAX_COMPS = 20;


if nargin < 2 
    pars.expVar = EXPLAINED_VAR;
    pars.maxComps = MAX_COMPS;
    pars.minComps = 1;
    pars.plot = false;
else
    pars = mdl.params;
end

if size(X, 2) > size(X, 1) || det(cov(X)) == 0
    algo = 'svd';
else
    algo = 'eig';
end

[wcoeff,~,variance,~,var_ratio] = pca(X, 'algorithm', algo);
nComps = choose_n_comps(var_ratio, pars.expVar);
nComps = max([nComps, pars.minComps]);
nComps = min([nComps, pars.maxComps, size(wcoeff, 2)]);

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