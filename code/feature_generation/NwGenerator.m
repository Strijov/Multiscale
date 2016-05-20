function add_features = NwGenerator(workStructTS)
    add_features = zeros(size(workStructTS.matrix,1), workStructTS.deltaTp);
    for i = [1:size(workStructTS.matrix,1)]
        x = workStructTS.matrix(i, 1:workStructTS.deltaTp);
        x_smoothed = NWSmoothing(x);
        add_features(i,:) = x_smoothed;
    end
    %workStructTS.matrix = [add_features, workStructTS.matrix(:, ...
    %    workStructTS.deltaTp + 1:workStructTS.deltaTp+workStructTS.deltaTr)];
    %add_features = [];
    
    %Sourse TS is being replaced with the smoothed one, that's why dims
    %don't change. Smoothed TS overwrites old one in X immediately. 
end