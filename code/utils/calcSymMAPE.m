function smape = calcSymMAPE(y, forecasts)


forecasts = forecasts(:);
y = y(:);
%residuals = y - forecasts;

smape = 2*mean(abs((y - forecasts)./(forecasts + y)));


end