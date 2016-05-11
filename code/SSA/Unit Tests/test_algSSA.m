function test_algSSA
inputTS.s = [1 2 0 3 1 4 5 2 3 8 7 5]';
inputTS.t = [1:12]';
idxHist = [1:10]';
idxFrc = [11; 12];
par = [];
[frc, par] = algSSA(inputTS, idxHist, idxFrc, par);
assertEqual(frc.s, [-38;8]);
assertEqual(frc.t, [11; 12]);
end