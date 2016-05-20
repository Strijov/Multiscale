function add_features = ConvGenerator(workStructTS)
    n = 5;
    add_features = zeros(size(workStructTS.matrix,1), n);
    for i = [1:size(workStructTS.matrix,1)]
        x = workStructTS.matrix(i,:);
        y = [sum(x), mean(x), min(x), max(x), std(x)]; % real(fft(x)), imag(fft(x))];
        add_features(i, :) = y;
    end
end