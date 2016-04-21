function [RMSE] = ComputeForecastingErrors(matrix, K, m, alpha_coeff, model, deltaTp, deltaTr)
    n = 1;
    M = size(matrix, 1);
    RMSE = [];
    while n <= K
        matrix_n = matrix(n:n+m-1, :);
        [trainX, trainY, testX, testY, val_x, val_y] = FullSplit(matrix_n, alpha_coeff, deltaTp, deltaTr)
        if model.unopt_flag == true
            model = OptimizeModelParameters(trainX, trainY, model);
        else
            model = ExtraOptimization(trainX, trainY, model);
        end
        forecast_y = ComputeForecast(val_x, model);
        epsilon = forecast_y - val_y;
        error = struct('epsilon', epsilon);
        tmp = sqrt((1/deltaTr)*norm(epsilon));
        RMSE = [RMSE, tmp];
        n = n + 1;
    end
end

% DISCUSS: Compute quality of forecast in extra function? Additional quality/error
% functions?


         
        
        
    