function test_forecasting
timeSeries = [1 4 9 16 25 36];
eigenvectors = [-0.3104, -0.7732, -0.5530;
-0.5226, -0.3472, -0.7787;
-0.7941, 0.5307, -0.2963];
choosenComponents = [1 2];
caterpillarLength = 3;
[newValue] = forecasting(timeSeries, eigenvectors, choosenComponents, caterpillarLength);
assertEqual(newValue, 11.0050);
end