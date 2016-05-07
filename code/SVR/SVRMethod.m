function y_pred = SVRMethod(X, Y, x_test)

C = 1e5;  
lambda = 0.000001; 
epsilon = .1;
kerneloption = 1;
kernel='gaussian';
verbose=0;

y_pred = zeros(size(y_test));

for i = 1:size(y_test, 2)
    y = Y(:, i);
    [xsup, ysup, w, b, newpos, alpha, obj] = svmreg(X, y, C, epsilon, kernel, kerneloption, lambda, verbose);
    y_pred(:, i) = svmval(x_test, xsup, w, b, kernel, kerneloption);
end

end