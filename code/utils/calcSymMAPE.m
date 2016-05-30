function smape = calcSymMAPE(forecasts, y)


forecasts = forecasts(:);
y = y(:);
%residuals = y - forecasts;

smape = 2*mean(abs((y - forecasts)./(forecasts + y)));


end