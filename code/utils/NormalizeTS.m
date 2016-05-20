function [ts, div, subt] = NormalizeTS(s)
    x = s.x;
    subt = min(x);
    tmp = x - subt;
    div = max(tmp);
    ts = tmp/div;
    ts = ReplaceNansWithZeros(ts); % FIXIT do something more sensible
end
