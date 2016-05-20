function ts = ReplaceNans(ts)

ts(isnan(ts)) = 0;

end