function StartUnitTests
%StartUnitTests launch all Unit tests
suite = TestSuite.fromName('Unit Tests');
suite.run
end