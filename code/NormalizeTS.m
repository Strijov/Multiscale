function ts = NormalizeTS(s)
    x = s.x;
    min_val = min(x);
    tmp = x - min(x);
    multiplier = max(tmp);
    ts = x/multiplier;
    s.normalization = [multiplier, min_val];
end