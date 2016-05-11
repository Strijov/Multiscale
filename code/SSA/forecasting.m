% FORECASTING
function [newValue] = forecasting(timeSeries, eigenvectors, choosenComponents, caterpillarLength)
%makes forecasting for a new point
%
%[newTimeSeries] = forecasting(timeSeries, eigenvectors, choosenComponents, caterpillarLength)
%
%Arguments
%timeSeries [dimension, timeSeriesLength] initial time series
%eigenvectors [K, K] eigenvectors, made by principal components analysis
%choosenComponents [1, int] vector of choosen components by witch the
%   forecasting will be made
%caterpillarLength [int] length of caterpillar
%newValue [scalar] forecasting value
%
%Example
%Arguments:
%timeSeries = [1 4 9 16 25 36]
%eigenvectors = [-0.3104 -0.7732 -0.5530;
%-0.5226 -0.3472 -0.7787;
%-0.7941 0.5307 -0.2963]
%choosenComponents = [1 2]
%caterpillarLength = 3
%Result:
%newValue = 11.0050

dimension = size(eigenvectors, 1) / caterpillarLength;
timeSeriesLength = size(timeSeries, 2);
v = eigenvectors(caterpillarLength:caterpillarLength:(caterpillarLength * dimension), choosenComponents);
eigenvectors = eigenvectors(:, choosenComponents);
eigenvectors(caterpillarLength:caterpillarLength:(caterpillarLength * dimension), :) = [];
size(eigenvectors);
z = 0;
Qq = zeros((dimension - 1) * caterpillarLength, 1);
for x = 1:dimension
    for y = (timeSeriesLength - caterpillarLength + 2):timeSeriesLength
        z = z + 1;
        Qq(z, 1) = timeSeries(x, y);
    end
end
newValue = v / (eigenvectors' * eigenvectors) * eigenvectors' * Qq;
end

