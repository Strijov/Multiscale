function [ts, div, subt] = NormalizeTS(x)

if numel(unique(x)) == 1
    ts = ones(size(x));
    div = 1;
    subt = 0;
    return
end
subt = min(x);
tmp = x - subt;
div = max(tmp);
ts = tmp/div;

ts = ReplaceNans(ts); % Fills Nans with median; FIXIT do something more sensible
if any(isnan(ts))
    warning('regMatrixAllNans:id', ['CreateRegMatrix: normalizeTS failed, since ' ...
                        ' ts contains only nans']) 
end

end