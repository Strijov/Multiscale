function [RMSE, model, real_y] = ComputeForecastingErrors(ts, K, m, alpha_coeff, model)
    matrix = ts.matrix; % TODO please remove these duplicates.
    deltaTp = ts.deltaTp;
    deltaTr = ts.deltaTr;
    n = 1;
    
    RMSE = zeros(1, K);
    model.forecasted_y = zeros(1,deltaTr*K);
    real_y = zeros(1,deltaTr*K);
    %FIXIT: first create zero-matrix for RMSE, then add values there.
    while n <= K
        matrix_n = matrix(n:n+m-1, :);
        [trainX, trainY, testX, testY, val_x, val_y] = FullSplit(matrix_n, alpha_coeff, deltaTp, deltaTr);
        if model.unopt_flag == true
            model = OptimizeModelParameters(trainX, trainY, model);
        else
            model = ExtraOptimization(trainX, trainY, model);
        end
        forecast_y = ComputeForecast(val_x, model, trainX, trainY);
        model.forecasted_y((n-1)*deltaTr+1:n*deltaTr) = forecast_y;
        real_y((n-1)*deltaTr+1:n*deltaTr) = val_y;
        epsilon = (forecast_y - val_y);%NOW IS MAPE!!!
        tmp = sqrt((1/deltaTr)*norm(epsilon));
        RMSE(n) = tmp;
        n = n + 1;
    end
end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?


         
        
        
    