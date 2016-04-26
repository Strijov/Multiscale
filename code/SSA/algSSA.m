%algSSA
function [frc, par] = algSSA(inputTS, idxHist, idxFrc, par)
%makes forecasting
%
%[frc, par] = algSSA(inputTS, idxHist, idxFrc, par)
%
%Arguments
%inputTS [ts] initial time series
%idxHist [int, 1] history indexes
%idxFrc [int, 1] forecasting indexes, max(idxFrc)-max(idxHist) < 200
%par parameters
%
%Example
%Arguments:
%inputTS.s = [1 2 0 3 1 4 5 2 3 8 7 5]'
%inputTS.t = [1:12]'
%idxHist = [1:10]'
%idxFrc = [11; 12]
%par = []
%Result:
%frc.s = [-15; 32.5]
%frc.t = [11; 12]

MAX_CATERPILLAR_LENGTH = 200;
MIN_CATERPILLAR_LENGTH = 5;
MAX_COMPONENTS_NUMBER = 10;
NO_CENTERING = 0;
NO_NORMALIZATION = 0;
if (size(idxHist, 1) < MIN_CATERPILLAR_LENGTH * 2) || (idxFrc(end) - idxHist(end) > MAX_CATERPILLAR_LENGTH)
    frc = [];
    par.errno = 1;
    return
end
par.errno = 0;
caterpillar_length = floor(size(idxHist, 1) / 2);
if caterpillar_length > MAX_CATERPILLAR_LENGTH
    caterpillar_length = MAX_CATERPILLAR_LENGTH;
end
inputTS_transposed = (inputTS.s)';
ts = inputTS_transposed(:, idxHist);
ts(any(isnan(ts), 2), :) = [];
[eigenvalues, eigenvectors, principalComponents, mean, error] = principalComponentAnalysis(ts, caterpillar_length, NO_CENTERING, NO_NORMALIZATION);
components_number = caterpillar_length;
if components_number > MAX_COMPONENTS_NUMBER
    components_number = MAX_COMPONENTS_NUMBER;
end
for forecastIndex = [1:(idxFrc(end) - idxHist(end))]
    ts = [ts forecasting(ts, eigenvectors, [1:components_number], caterpillar_length)];
end
frc.s = (ts(1, idxFrc))';
%frc.t = inputTS.t(idxFrc);
return