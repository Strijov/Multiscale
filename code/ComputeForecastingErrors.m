function [RMSE] = ComputeForecastingErrors(matrix, N, m, alpha_coeff, model, deltaTp, deltaTr)
    n = 1;
    M = size(matrix, 1);
    RMSE = [];
    while n <= N
        matrix_n = matrix(n:n+m-1, :);
        [trainMatrix, testMatrix, validation] = TrainTestSplit(matrix_n, alpha_coeff);
        [trainX, trainY] = SplitIntoXY(trainMatrix, deltaTp, deltaTr);
        [testX, testY] = SplitIntoXY(testMatrix, deltaTp, deltaTr);
        [val_x, val_y] = SplitIntoXY(validation, deltaTp, deltaTr);
        model = OptimizeModelParameters(trainX, trainY, model);
        %forecast_y = model.func(val_x);
        switch model.name
            case 'VAR'
                W = model.params;
                forecast_y = val_x*W;
            case 'Neural_network'
                forecast_y = model.tuned_func(val_x');
                forecast_y = forecast_y';
        end
        epsilon = forecast_y - val_y;
        tmp = sqrt((1/deltaTr)*norm(epsilon));
        RMSE = [RMSE, tmp];
        n = n + 1;
    end
end

         
        
        
        
    