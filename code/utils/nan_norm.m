function res = nan_norm(x)

res = norm(x(~isnan(x)));

end