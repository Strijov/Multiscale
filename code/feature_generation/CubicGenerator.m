function add_features = CubicGenerator(workStructTS)
    n = 3;
    add_features = zeros(size(workStructTS.matrix,1), n + 1);
    x = [1:size(workStructTS.matrix,2)];
    for i = [1:size(workStructTS.matrix,1)]
        y = workStructTS.matrix(i,:);
        p = polyfit(x,y,n);
        add_features(i, :) = p;
    end
    
end