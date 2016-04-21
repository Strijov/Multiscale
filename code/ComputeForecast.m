function [forecast_y] = ComputeForecast(val_x, model)
    %Input:
    %   x              - vector [1xdeltaTp], features string for last period
    %   model          - struct containing model and it's parameters
    %Output:
    %   forecast_y     - vector [1xdeltaTr], forecast string for las period
    %Compute forecast using optimized model
    switch model.name
            case 'VAR'
                W = model.params;
                forecast_y = val_x*W;
            case 'Neural_network'
                forecast_y = model.tuned_func(val_x');
                forecast_y = forecast_y';
    end
end