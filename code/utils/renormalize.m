function ts = renormalize(ts, norm_div, norm_subt)

if ~iscell(norm_div)
   norm_div = num2cell(norm_div); 
end
if ~iscell(norm_subt)
   norm_subt = num2cell(norm_subt); 
end

ts = cellfun(@(x,a,b) (x - repmat(b, size(x)))./repmat(a, size(x)), ...
        ts, norm_div, norm_subt, 'UniformOutput', 0);

end