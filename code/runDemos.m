function runDemos(dataset, tsname)
% Since there will be several experimental designs, 
% this script will be used to run various experiments stored in 'demos/'


addpath(genpath(cd));
LoadAndSave('NNcompetition/');
if nargin == 0
    dataset = 'NNcompetition';
    tsname = 'orig_train';
end
tsStructArray  = LoadTimeSeries(dataset);
ts = tsStructArray{1};

demoCompareForeasts(tsStructArray);
demoFeatureSelection(ts);
%demoForecastHorizon(ts);


end