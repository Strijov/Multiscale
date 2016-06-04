function runDemos(dataset, tsname)
% Since there will be several experimental designs, 
% this script will be used to run various experiments stored in 'demos/'


addpath(genpath(cd));
LoadAndSave('HascData/sequence/');
if nargin == 0
    dataset = 'HascData';
    tsname = 'orig_train';
end
ts_struct_array  = LoadTimeSeries(dataset);
ts = ts_struct_array{1};

demoFeatureSelection(ts);
%demoForecastHorizon(ts);


end