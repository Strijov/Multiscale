function  [K] = KernelFunction(kernel_type, p1, p2, X, Z)
% Kernel transformation
% Input:
% kernel_type   [str] can be {'linear', 'poly', 'rbf', 'erbf', 'sigmoid'}
% p1, p2        [1], [1] corresponding parameters in kernel function
% X             [m, p] matrix
% Z             [n, p] matrix

if size(X, 2) ~= size(Z, 2) 
    K = []; 
    display('The second dimensions for X and Z must be agree.'); 
    return; 
end 
switch lower(kernel_type)
    case 'linear'
        K = X*Z';
    case 'poly'
        K = (X*Z' + p1).^p2;
    case 'rbf'
        K = exp(-p1*(repmat(dot(X, X, 2), 1, size(Z, 1)) + ...
            repmat(dot(Z, Z, 2)', size(X, 1), 1) - 2*X*Z')); 
    case 'erbf'
        K = exp(-sqrt(repmat(dot(X, X, 2), 1, size(Z, 1)) + ...
            repmat(dot(Z, Z, 2)', size(X, 1), 1) - 2*X*Z') / (2*p1^2)) + p2; 
    case 'sigmoid'
        K = tanh(p1*X*Z'/size(X, 2)  + p2); 
    otherwise 
        K = X*Z' + p1 + p2; 
end
