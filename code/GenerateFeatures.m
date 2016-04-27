function [matrix] = GenerateFeatures(matrix, deltaTp, deltaTr, generator)
    [X,Y] = SplitIntoXY(matrix, deltaTp, deltaTr);
    iterator = 0;
    add_dim_iterator = 0;
    generator_names = {'SSA', 'NW', 'Cubic', 'Conv', 'Cluster'};
    
%     if ismember('SSA', generator_names)
%         iterator = iterator + 1;
%         inputTS.s = matrix(1, 1:deltaTp)';
%         inputTS.t = [1:numel(matrix(1, 1:deltaTp))]';
%         idxHist = inputTS.t;
%         idxFrc = [deltaTp:1+deltaTp+deltaTr]';
%         par = []
%         [frc, par] = algSSA(inputTS, idxHist, idxFrc, par)
%         %Bad end
%     end
    
    if ismember('Cubic', generator_names)
        iterator = iterator + 1;
        n = 3;
        add_features = zeros(size(matrix,1), n + 1);
        x = [1:size(matrix,2)];
        for i = [1:size(matrix,1)]
            y = matrix(i,:);
            p = polyfit(x,y,n);
            add_features(i, :) = p;
        end
        add{iterator} = add_features;
        add_dim_iterator = add_dim_iterator + size(add_features, 2);
    end
    
    if ismember('NW', generator_names)
    end
    
    if ismember('Conv', generator_names)
        iterator = iterator + 1;
        n = 5;
        add_features = zeros(size(matrix,1), n);
        for i = [1:size(matrix,1)]
            x = matrix(i,:);
            y = [sum(x), mean(x), min(x), max(x), std(x)]; % real(fft(x)), imag(fft(x))];
            add_features(i, :) = y;
        end
        add{iterator} = add_features;
        add_dim_iterator = add_dim_iterator + size(add_features, 2);
    end
    
    Xnew = zeros(size(X,1), size(X,2)+add_dim_iterator);
    Xnew(:, 1:size(X,2)) = X;
    new_begin = size(X,2);
    for i = [1:iterator]
        Xnew(:, new_begin+1:new_begin + size(add{i}, 2)) = add{i};
        new_begin = new_begin + size(add{i}, 2);
    end
    matrix = [Xnew, Y];
end
