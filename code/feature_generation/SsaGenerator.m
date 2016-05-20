function add_features = SsaGenerator(workStructTS)
    N_COMP = 3;
    add_features = zeros(size(workStructTS.matrix, 1), N_COMP);
    for i = [1:size(workStructTS.matrix,1)]
        x = workStructTS.matrix(i, 1:workStructTS.deltaTp);
        caterpillar_length = floor(numel(x) / 2);
        eigenvalues = principalComponentAnalysis(x, caterpillar_length, 0, 0);
        add_features(i, :) = eigenvalues(1:N_COMP);
    end
 
end