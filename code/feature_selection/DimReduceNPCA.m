function [newX, mdl] = DimReduceNPCA(X, mdl)
% Use autoencoder (nonlinear PCA by Matthias Scholz, www.nlpca.org) to reduce dimensionality of feature description
% 
% Inputs:
% X [m x n] - feature matrix
% mdl - model selection structure with fields handle, name, pars and
% transform and res. 
%
% Outputs:
% newX [m x nComps] - transformed feature matrix
% mdl - model structure with updated fields transform, pars and res

HIDDEN_SIZE = 25;
MAX_ITERATION = 500;
if ~isfield(mdl.params, 'hiddenSize')
    mdl.params.hiddenSize = HIDDEN_SIZE;
end

[newX, net, network]=nlpca(X', mdl.params.nComp,...
                                 'silence', 'yes', ...
                                 'plotting', 'no', ...
                                 'type','inverse',...
                                 'units_per_layer', [mdl.params.nComp, mdl.params.hiddenSize, size(X, 2)],... % hid=12, to be more flexible
                                 'max_iteration', MAX_ITERATION);

recX = nlpca_get_data(net, newX);
newX = newX';
%recX = nlpca_get_data(components);
mdl.res.error = mse(X-recX');
mdl.transform = @(X) nlpca_get_components(net, X')';

end