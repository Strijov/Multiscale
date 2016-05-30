function ts = ReplaceNans(ts)

ts(isnan(ts)) = median(ts(~isnan(ts)));

end