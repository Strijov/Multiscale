function mase = calcMASE(y, forecasts)


forecasts = forecasts(:);
y = y(:);
%residuals = y - forecasts;

mae = mean(abs(y - forecasts));
scaling = mean(abs(diff(y)));

mase = mae/scaling;


end