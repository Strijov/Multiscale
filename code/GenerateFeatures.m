function [workStructTS] = GenerateFeatures(workStructTS, generator)
    [X,Y] = SplitIntoXY(workStructTS.matrix, workStructTS.deltaTp, workStructTS.deltaTr);
    iterator = 0;
    add_dim_iterator = zeros(1,numel(generator));
    generator_names = {'SSA', 'NW', 'Cubic', 'Conv', 'Cluster'};
    deltaTp = workStructTS.deltaTp
    
    if ismember('SSA', generator)
        iterator = iterator + 1;
        add_features = zeros(size(workStructTS.matrix,1), 3);
        for i = [1:size(workStructTS.matrix,1)]
            x = workStructTS.matrix(i, 1:workStructTS.self_deltaTp(1));
            caterpillar_length = floor(numel(x) / 2);
            [eigenvalues, eigenvectors, principalComponents, my_mean, error] = principalComponentAnalysis(x, caterpillar_length, 0, 0);
            add_features(i, :) = eigenvalues(1:3);
        end
        add{iterator} = add_features;
        add_dim_iterator(iterator) = size(add_features, 2);
    end
    
    if ismember('Cubic', generator)
        iterator = iterator + 1;
        n = 3;
        add_features = zeros(size(workStructTS.matrix,1), n + 1);
        x = [1:size(workStructTS.matrix,2)];
        for i = [1:size(workStructTS.matrix,1)]
            y = workStructTS.matrix(i,:);
            p = polyfit(x,y,n);
            add_features(i, :) = p;
        end
        add{iterator} = add_features;
        add_dim_iterator(iterator) = size(add_features, 2);
    end
    
    if ismember('NW', generator)
        add_features = zeros(size(workStructTS.matrix,1), workStructTS.self_deltaTp(1));
        for i = [1:size(workStructTS.matrix,1)]
            x = workStructTS.matrix(i, 1:workStructTS.self_deltaTp(1));
            x_smoothed = NWSmoothing(x);
            add_features(i, :) = x_smoothed;
        end
        X(:, 1:workStructTS.self_deltaTp(1)) = add_features;
        %Sourse TS is being replaced with the smoothed one, that's why dims
        %don't change. Smoothed TS overwrites old one in X immediately. 
    end
    
    if ismember('Conv', generator)
        iterator = iterator + 1;
        n = 5;
        add_features = zeros(size(workStructTS.matrix,1), n);
        for i = [1:size(workStructTS.matrix,1)]
            x = workStructTS.matrix(i,:);
            y = [sum(x), mean(x), min(x), max(x), std(x)]; % real(fft(x)), imag(fft(x))];
            add_features(i, :) = y;
        end
        add{iterator} = add_features;
        add_dim_iterator(iterator) = size(add_features, 2);
    end
    
    Xnew = zeros(size(X,1), size(X,2) + sum(add_dim_iterator));
    Xnew(:, 1:size(X,2)) = X;
    new_begin = size(X,2);
    for i = [1:iterator]
        Xnew(:, new_begin+1:new_begin + size(add{i}, 2)) = add{i};
        new_begin = new_begin + size(add{i}, 2);
        workStructTS.self_deltaTp = [workStructTS.self_deltaTp, add_dim_iterator(iterator)];
        workStructTS.deltaTp = size(Xnew, 2);
    end
    workStructTS.matrix = [Xnew, Y];
end
