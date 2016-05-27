function ts = LoadProcessedData(dirname)


filenames = dir(fullfile(dirname, '*.mat'));

ts = cell(1, numel(filenames));
for i = 1:numel(filenames)
   load(filenames(i).name);   
   ts{i} = ts_struct;
end

end