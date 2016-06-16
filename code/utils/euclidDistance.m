function dist = euclidDistance(A, B)
%euclidDistance(a, b) computes Euclidean distance between time series in A
%and B
%Input:
% A, B - matrixes with time series
%Output:
% dist - array of Euclidean distances

    dist = sqrt(sum((A - B) .* (A - B), 2));
end

