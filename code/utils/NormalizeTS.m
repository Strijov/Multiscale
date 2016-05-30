function [ts, div, subt] = NormalizeTS(s)
    x = s.x;
    subt = min(x);
    tmp = x - subt;
    div = max(tmp);
    ts = tmp/div;
    ts = ReplaceNans(ts); % Fills Nans with median; FIXIT do something more sensible
end
