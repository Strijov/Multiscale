ts0 = csvread('\data\tsEnergyConsumption.csv', 0, 1, [0,1,365*24,1]);
deltaTp = 144;
deltaTr = 24;
time_points = linspace(8671, 8671 - 24 * 300, 301)

[X, Y, x, y] = CreateRegMatrix(ts0, time_points, deltaTp, deltaTr);

%linear VAR
W = inv(X'*X)*X'*Y;
norm(X*W - Y)
y_pred = x*W
plot(y_pred)
hold on

%neural network
net = fitnet(10);
net = train(net,X',Y');
y_pred = net(x');
perf = perform(net,y_pred,y)
plot(y_pred)
hold on

%plot results
grid on
plot(y, 'LineWidth', 1.5)
legend('VAR', 'Neural net', 'Real')