function [MAPE, model, real_y] = ComputeForecastingErrors(ts, K, alpha_coeff, model)


matrix = ts.matrix; % TODO please remove these duplicates.
deltaTp = size(matrix, 2) - ts.deltaTr;
deltaTr = ts.deltaTr;

model.forecasted_y = zeros(1,deltaTr*K);
real_y = zeros(1,deltaTr*K);
m = size(matrix,1);

MAPE = zeros(1, K);
for n = 1:K
    matrix_n = matrix(n:n+m-1, :);
    [trainX, trainY, testX, testY, validation_x, validation_y] = ...
                        FullSplit(matrix_n, alpha_coeff, deltaTp, deltaTr);
    
    forecast_y = feval(model.handle, validation_x, model, trainX, trainY); 
    model.forecasted_y((n-1)*deltaTr+1:n*deltaTr) = forecast_y;
    real_y((n-1)*deltaTr+1:n*deltaTr) = validation_y;
    residuals = (validation_y - forecast_y); %NOW IS MAPE!!!
    MAPE(n) = mean(abs(residuals./validation_y));
end
model.error = MAPE;

end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?


         
        
        
    