function demoFrcSimpleData(model, numTs)

FOLDER = 'fig\'; % dir to store figures
VERBOSE = true; % change to false after first run of the main loop


if nargin < 2
    numTs = 1;
end

% Experiment settings:
NOISE = 0:0.1:1;
N_HIST_POINTS = 2;
SEGM_LEN = 10;

nModels = numel(model);

ts = cell(1, numTs);
for nTs = 1:numTs
intercepts = zeros(numel(NOISE), nModels);
testError = zeros(numel(NOISE), nModels);
trainError = zeros(numel(NOISE), nModels);
for i = 1:numel(NOISE)
    ts{i} = createSimpleDataStruct(@linearSegm, NOISE(i), SEGM_LEN, N_HIST_POINTS);
end    
[test, train, bias] = demoCompareForecasts(ts, model, [], [], VERBOSE);
intercepts = intercepts + bias;
testError = testError + test;
trainError = trainError + train;

VERBOSE = false;
end

intercepts = intercepts/numTs;
testError = testError/numTs;
trainError = trainError/numTs;

fig1 = figure;
intercepts = [ones(numel(NOISE), 1)*NaN, intercepts];
h = plot(intercepts, 'linewidth', 2);
legend(['Data', {model().name}], 'Location', 'SouthWest');
xlabel('Noise level', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Mean residues', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;

fig2 = figure;
h1 = plot(testError, 'linewidth', 2);
legend({model().name}, 'Location', 'NorthWest');
hold on;
h2 = plot(trainError, '--', 'linewidth', 2);
for i = 1:nModels
   set(h1(i), 'Color', h(i+1).Color);
   set(h2(i), 'Color', h(i+1).Color); 
end
set(gca, 'Xtick', 1:numel(NOISE), 'XTickLabel', NOISE);
xlabel('Noise level', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
ylabel('Foreasting error', 'FontSize', 20, 'FontName', 'Times', 'Interpreter','latex');
set(gca, 'FontSize', 16, 'FontName', 'Times')
axis tight;

fname = fullfile(FOLDER, ts{1}.dataset, strcat('bias_', ts{1}.name, '.eps'));
saveas(fig1, fname, 'epsc');

fname = fullfile(FOLDER, ts{1}.dataset, strcat('MASE_', ts{1}.name, '.eps'));
saveas(fig2, fname, 'epsc');

close(fig1);
close(fig2);

end