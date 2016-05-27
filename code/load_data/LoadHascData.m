function ts_struct = LoadHascData(dirname)

files = dir(fullfile(dirname, '*.csv'));
ts_struct = cell(1, numel(files));

folders = strsplit(fullfile(dirname), filesep);
folder = folders(end);

readme = 'Accelerometry time series from HASC project';

for i = 1:numel(files)
   fname = files(i).name;
   data = csvread(fname);
   [~, fname, ~] = fileparts(fname);
   
   time = data(:, 1);
   % for now, use absolute value of acceleration
   ts = sqrt(sum(data(:, 2:4).^2, 2)); 
   
   approx_frequency = 1/mean(diff(time));
   freq = round(approx_frequency/10)*10;
   
   % set deltaTr to half second, deltaTp to 2 seconds
   deltaTr = freq/2; 
   deltaTp = 2*freq;
   
   ts_struct{i} = struct('x', ts, 'time', time, ...
                        'legend', folder,...
                        'deltaTp', deltaTp, 'deltaTr', deltaTr,...
                        'name', fname, 'readme', readme,...
                        'dataset', 'HascData');
   
end

end