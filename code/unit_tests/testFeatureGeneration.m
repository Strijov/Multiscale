function tests = testFeatureGeneration

% Test forecasting methods includes the following testing scenarios:
% - testNWreplace: checks that replace=false in NwGnerator leads to a warning
% - testNewNans: checks no new nans are added
% - testTargetInvisible: checks that the results de not depend on Y
% - testTargetUnchanged: checks that feature generation does not change Y

tests  = functiontests(localfunctions);

end

function testNWreplace(testCase)

% checks that replace=false in NwGnerator leads to a warning
ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

generators = struct('handle', @NwGenerator, 'name', 'NW', ...
                     'replace', false, 'transform', []);

warningFunc = @() GenerateFeatures(ts, generators);
verifyWarning(testCase, warningFunc, 'genFeatsReplace:id');

end


function testNewNans(testCase)

% check that generation procedures do not produce Nans from valid data
ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

generator_names = {'SSA', 'Cubic', 'Conv', 'Centroids', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @MetricGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', false, 'transform', []);
idxNW = strcmp({generators().name}, {'NW'});
generators(idxNW).replace = true; 

ts = GenerateFeatures(ts, generators);
verifyTrue(testCase, all(~isnan(ts.X(:))));

% what if the input is singular?
ts.X = ones(size(ts.X));
verifyTrue(testCase, all(~isnan(ts.X(:))));

end


function testTargetInvisible(testCase)

% checks that targets do not participate in feature generation
ts = createRandomDataStruct(3, 300); % make it even smaller that usual
ts = CreateRegMatrix(ts);

% set up generators 
generator_names = {'SSA', 'Cubic', 'Conv', 'Centroids', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @MetricGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', false, 'transform', []);
idxNW = strcmp({generators().name}, {'NW'});
generators(idxNW).replace = true; 

% Generate features for original data, then change Y randomly and see if anything changes:
N_REPEATS = 5; % it's easy to convince us
tsRY = ts;
randY = arrayfun(@(i) randn(size(ts.Y)), 1:N_REPEATS, 'UniformOutput', 0);

% Remember random state so that results do not depend on internal
% randomness
stream = RandStream.getGlobalStream;
savedState = stream.State;
for i = 1:N_REPEATS  
    % use previuosly generated Y:
    tsRY.Y = randY{i};    
    % reset state
    stream.State = savedState;
    tsOrig = GenerateFeatures(ts, generators);
    % reset state
    stream.State = savedState;
    tsRand = GenerateFeatures(tsRY, generators);
    verifyEqual(testCase, tsOrig.X, tsRand.X);
end

end

function testTargetUnchanged(testCase)

% check that target variables are not influenced

ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

generator_names = {'SSA', 'Cubic', 'Conv', 'Centroids', 'NW'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @MetricGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', false, 'transform', []);
idxNW = strcmp({generators().name}, {'NW'});
generators(idxNW).replace = true; 

oldY = ts.Y;
ts = GenerateFeatures(ts, generators);
verifyEqual(testCase, ts.Y, oldY);



end

function verifyReplaceFalse(testCase)

% checks that in false mode columns of the original matrix are always stacked 
% on the left said

ts = createRandomDataStruct(3, 500); % make it relatively small
ts = CreateRegMatrix(ts);

generator_names = {'SSA', 'Cubic', 'Conv', 'Centroids'}; %{'Identity'};
generator_handles = {@SsaGenerator, @CubicGenerator, @ConvGenerator, @MetricGenerator, @NwGenerator}; %{@IdentityGenerator};
generators = struct('handle', generator_handles, 'name', generator_names, ...
                                             'replace', false, 'transform', []);

for i = 1:numel(generators)
   tsNew = GenerateFeatures(ts, generators(i));
   verifyEqual(testCase, tsNew.X(:, sum(ts.deltaTp)), ts.X);
end

end