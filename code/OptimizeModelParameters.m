function  model = OptimizeModelParameters(X, Y, model)
    %Input:
    %   X              - object-features matrix [MxdeltaTp]
    %   Y              - answers matrix [MxdeltaTr]
    %   model          - struct containing model and it's parameters
    %Output:
    %   model          - optimized model
    %   validation     - vector[1xN]
    %Trains model to fit provided data in X, Y.
    switch model.name
        case 'VAR'
            W = inv(X'*X)*X'*Y;
            model.params = W;
            model.tuned_func = [];
            model.unopt_flag = true;
        case 'Neural_network'
            net = fitnet(10);
            net = train(net,X',Y');
            model.params = [];
            model.tuned_func = net;
            model.unopt_flag = true; %FIXIT Should be false, but can't find function to retrain NN using old parameters.
        case 'SVR'
            model.unopt_flag = true;
            model.tuned_func = [];
            model.params = [];
    end
end


