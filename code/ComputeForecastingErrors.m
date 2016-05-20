function [RMSE, model, real_y] = ComputeForecastingErrors(ts, K, alpha_coeff, model)


matrix = ts.matrix; % TODO please remove these duplicates.
deltaTp = ts.deltaTp;
deltaTr = ts.deltaTr;

model.forecasted_y = zeros(1,deltaTr*K);
real_y = zeros(1,deltaTr*K);
m = size(matrix,1);

RMSE = zeros(1, K);
for n = 1:K
    matrix_n = matrix(n:n+m-1, :);
    [trainX, trainY, testX, testY, validation_x, validation_y] = ...
                        FullSplit(matrix_n, alpha_coeff, deltaTp, deltaTr);
    
    forecast_y = feval(model.handle, validation_x, model, trainX, trainY); 
    model.forecasted_y((n-1)*deltaTr+1:n*deltaTr) = forecast_y;
    real_y((n-1)*deltaTr+1:n*deltaTr) = validation_y;
    residuals = (validation_y - forecast_y); %NOW IS MAPE!!!
    RMSE(n) = sqrt((1/deltaTr)*norm(residuals));
end
model.error = RMSE;

end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?


         
        
        
    