function  model = OptimizeModelParameters(X, Y, model)
    names = ['VAR', 'Neural_network'];
    switch model.name
        case 'VAR'
            W = inv(X'*X)*X'*Y;
            model.params = W;
            model.tuned_func = [];
        case 'Neural_network'
            net = fitnet(10);
            net = train(net,X',Y');
            model.params = [];
            model.tuned_func = net;
    end
end