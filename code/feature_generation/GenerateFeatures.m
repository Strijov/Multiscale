function [workStructTS] = GenerateFeatures(workStructTS, generators)
% Generates new feature matrix using methods, specified with generators.
%
% Input:
% workStructTS	see createRegMatrix.m for explanation
% generators    cell array of feature generator handles. Options are 
% @ConvGenerator, @CubicGenerator, @NwGenerator, @SsaGenerator. When no
% feature generation is required, use {@IdentityGenerator}
%
% Output:
% workStructTS with new feature matrix

Xnew = [];    
for generator  = generators
    Xnew = [Xnew, feval(generator{1}, workStructTS)];
end
[X,Y] = SplitIntoXY(workStructTS.matrix, workStructTS.deltaTp, workStructTS.deltaTr);
workStructTS.matrix = [Xnew, Y];
workStructTS.deltaTp = size(Xnew, 2);
    
end    
   
    

    

