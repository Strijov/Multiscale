function smape = calcSymMAPE(y, forecasts)


forecasts = forecasts(:);
y = y(:);
%residuals = y - forecasts;

smape = 2*nanmean(abs((y - forecasts)./(forecasts + y)));


end