function test_principalComponentAnalysis
timeSeries = [1 4 9 16 25 36];
caterpillarLength = 3;
toCenter = 0;
toNormal = 0;
[eigenvalues, eigenvectors, principalComponents, mean, error] =...
    principalComponentAnalysis(timeSeries, caterpillarLength, toCenter, toNormal);
assertElementsAlmostEqual(eigenvalues, [893.4050; 4.0895; 0.0055],'absolute',0.01);
assertElementsAlmostEqual(eigenvectors, [-0.3104 -0.7732 -0.5530;...
-0.5226 -0.3472 -0.7787;...
-0.7941 0.5307 -0.2963;],'absolute',0.1);
assertElementsAlmostEqual(principalComponents, [-9.5475 -18.6501 -31.0069 -46.6177;
2.6142 2.2735 0.7534 -1.9461;
-0.1053 0.0547 0.0734 -0.0492],'absolute',0.1);
assertElementsAlmostEqual(mean, [],'absolute',0.1);
assertElementsAlmostEqual(error, [],'absolute',0.1);
end