%PRINCIPAL COMPONENT ANALYSIS
function [eigenvalues, eigenvectors, principalComponents, mean, error] = principalComponentAnalysis(timeSeries, caterpillarLength, toCenter, toNormal)
%makes covariance matrix and it's singular decomposition
%
%[eigenvalues, eigenvectors, principalComponents, mean, error] = principalComponentAnalysis(timeSeries, caterpillarLength, toCenter, toNormal)
%
%timeSeries [dimension, timeSeriesLength] historycal time series
%caterpillarLength [int] length of the caterpillar;
%toCenter [boolean] if true using centering
%toNormal [boolean] if true using normalising
%
%Example
%Arguments:
%timeSeries = [1 4 9 16 25 36]
%caterpillarLength = 3
%toCenter = 0
%toNormal = 0
%Result:
%eigenvalues = [893.4050; 4.0895; 0.0055]
%eigenvectors = [-0.3104 -0.7732 -0.5530;
%-0.5226 -0.3472 -0.7787;
%-0.7941 0.5307 -0.2963]
%principalComponents = [-9.5475 -18.6501 -31.0069 -46.6177;
%2.6142 2.2735 0.7534 -1.9461;
%-0.1053 0.0547 0.0734 -0.0492]
%mean = []
%error = []

[dimension, timeSeriesLength] = size(timeSeries);

alive = timeSeriesLength - caterpillarLength + 1; %lifetime of the caterpillar
observationLength = caterpillarLength * dimension;
observations = zeros(observationLength, alive);
for x = 1:alive
    for y = 1:dimension
        for z = 1:caterpillarLength
            observations((y - 1) * caterpillarLength + z, x) = timeSeries(y, x - 1 + z);
        end
    end
end
mean = [];
if toCenter
    mean = sum(observations, 2) ./ alive;
    observations = observations - repmat(mean, 1, alive);
end
error = [];
if toNormal
    error = sqrt(sum(observations.^2, 2) ./ alive);
    observations = observations ./ repmat(error, 1, alive);
end
covariance = observations * (observations') ./ alive;

%equalEigenvectors = eigenvectors because covariance is symmetric
[eigenvectors, diagonal, equalEigenvectors] = svd(covariance);
eigenvalues = diag(diagonal);
principalComponents = eigenvectors' * observations;
end

